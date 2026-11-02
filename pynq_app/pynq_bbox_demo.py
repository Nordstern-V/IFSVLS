#!/usr/bin/env python3
"""MIX_PLUS PYNQ camera + touch + TCU single-ROI bbox application."""

from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
import threading
from collections import deque
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "touch_deploy"))
sys.path.insert(0, str(PROJECT_DIR / "tcu_deploy"))

from pefsl_touch_ui import APP_VERSION, TouchBboxUI
from touch_driver import TouchController
from tcu_bbox_backend import TcuFeatureExtractor, load_overlay
from usb_keyboard import USBKeyboardWorker


def _close_inherited_keyboard_fds():
    """A restarted touch child must not retain the parent's grabbed keyboard.

    Close only the forked descriptor copy. EVIOCGRAB must not be cleared here:
    the parent is still the active keyboard reader on the same open file.
    """
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return
    for name in names:
        if not name.isdigit() or int(name) <= 2:
            continue
        try:
            target = os.readlink("/proc/self/fd/" + name)
            if target.startswith("/dev/input/event"):
                os.close(int(name))
        except OSError:
            pass


def _dispatch_ui_inputs(ui, touch_worker, keyboard_worker, stop_event):
    """Preserve draw/type/Enter/class-click ordering across both input queues."""
    now = time.monotonic()
    pending = [(getattr(event, "timestamp", now), "touch", event)
               for event in touch_worker.drain()]
    if keyboard_worker is not None:
        pending.extend((getattr(event, "timestamp", now), "key", event)
                       for event in keyboard_worker.drain())
    pending.sort(key=lambda item: item[0])
    for _, source, event in pending:
        if source == "touch":
            ui.handle_event(event)
        elif event.kind == "interrupt":
            stop_event.set()
            return
        else:
            ui.handle_key(event)


def _touch_child_send(conn, kind, value=None):
    """Send a touch event or heartbeat from the child process."""
    try:
        conn.send((kind, value))
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _touch_child_poll(touch, conn, stop_event):
    _touch_child_send(conn, "poll_start")
    try:
        return touch.poll()
    finally:
        _touch_child_send(conn, "poll_done")


def _touch_child_main(touch, poll_interval, int_gate, release_guard, conn, stop_event):
    """Run PYNQ IIC calls in a process that the parent can terminate."""
    _close_inherited_keyboard_fds()
    try:
        _touch_child_send(conn, "starting")
        initializer = getattr(touch, "reinitialize", None)
        if not callable(initializer):
            initializer = getattr(touch, "initialize")
        info = initializer(reset=False)
        _touch_child_send(conn, "ready", info)
    except Exception as error:
        _touch_child_send(conn, "error", "initialize: {}".format(error))
        try:
            conn.close()
        except Exception:
            pass
        return

    last_int = None
    int_high_since = None
    while not stop_event.is_set():
        _touch_child_send(conn, "heartbeat")
        event = None
        try:
            level = touch.int_level if int_gate else None
            if int_gate:
                if level != last_int:
                    _touch_child_send(conn, "int", level)
                    last_int = level
                if level is not None and int(level) != 0:
                    if int_high_since is None:
                        int_high_since = time.monotonic()
                    if time.monotonic() - int_high_since >= release_guard:
                        event = touch.force_release()
                else:
                    int_high_since = None
                    event = _touch_child_poll(touch, conn, stop_event)
            else:
                event = _touch_child_poll(touch, conn, stop_event)
            if event is not None:
                _touch_child_send(conn, "event", event)
        except Exception as error:
            _touch_child_send(conn, "error", str(error))
            try:
                touch.recover_bus()
            except Exception:
                pass
            stop_event.wait(min(1.0, poll_interval))
        else:
            stop_event.wait(poll_interval)


class TouchPollWorker:
    """Poll slow touch IIC traffic outside the video frame thread."""

    def __init__(
        self,
        touch,
        poll_interval=0.10,
        max_events=64,
        int_gate=True,
        release_guard=0.25,
        watchdog_timeout=0.80,
        touch_log_interval=0.25,
    ):
        self.touch = touch
        self.poll_interval = float(poll_interval)
        self.int_gate = bool(int_gate)
        self.release_guard = max(0.05, float(release_guard))
        self.watchdog_timeout = max(0.25, float(watchdog_timeout))
        self.watchdog_interval = min(0.10, max(0.02, self.watchdog_timeout / 4.0))
        self.touch_log_interval = max(0.05, float(touch_log_interval))
        self.last_int_level = None
        self.int_high_since = None
        self.events = deque(maxlen=max_events)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="touch-poll", daemon=True
        )
        self.error = None
        self.error_count = 0
        self.retry_at = 0.0
        self.total_errors = 0
        self.recovery_count = 0
        self.recovery_failures = 0
        self.last_error_text = None
        self.last_error_at = 0.0
        self.last_error_print_at = 0.0
        self.last_poll_started = 0.0
        self.last_poll_completed = time.monotonic()
        self.in_flight = False
        self.poll_generation = 0
        self.watchdog_generation = -1
        self.watchdog_attempt_count = 0
        self.watchdog_recoveries = 0
        self.watchdog_failures = 0
        self.device_reset_count = 0
        self.device_reset_failures = 0
        self.reinitialize_count = 0
        self.reinitialize_failures = 0
        self.reinitialize_pending = False
        self.hard_reset_in_progress = False
        self.watchdog_last_print_at = 0.0
        self.last_touch_print_at = 0.0
        self.recovery_lock = threading.Lock()
        self.watchdog_stop_event = threading.Event()
        self.watchdog_max_attempts = 3
        self.watchdog_thread = threading.Thread(
            target=self._watchdog, name="touch-iic-watchdog", daemon=True
        )

    def start(self):
        self.thread.start()
        self.watchdog_thread.start()

    def _begin_transaction(self):
        with self.lock:
            self.poll_generation += 1
            self.watchdog_generation = -1
            self.watchdog_attempt_count = 0
            self.last_poll_started = time.monotonic()
            self.in_flight = True

    def _end_transaction(self):
        with self.lock:
            self.last_poll_completed = time.monotonic()
            self.in_flight = False

    def _poll_transaction(self):
        """Run one controller read while exposing its age to the watchdog."""
        self._begin_transaction()
        try:
            return self.touch.poll()
        finally:
            self._end_transaction()

    def _recover_bus(self):
        # A watchdog and the exception path can observe the same wedged
        # transaction. Serialize the MMIO SOFTR writes so they cannot race.
        with self.recovery_lock:
            try:
                return bool(self.touch.recover_bus())
            except Exception:
                return False

    def _hard_reset_device(self, schedule_reinit=True):
        """Pulse the touch reset GPIO without entering another IIC call."""
        reset = getattr(self.touch, "hard_reset", None)
        if not callable(reset):
            return False
        with self.recovery_lock:
            with self.lock:
                self.hard_reset_in_progress = True
            try:
                reset()
                with self.lock:
                    # A reset invalidates every event generated before the
                    # wedged transaction.  Leaving a queued ``down`` or
                    # ``move`` here would make the UI look touched after the
                    # controller has restarted.
                    self.events.clear()
                    self.last_int_level = None
                    self.int_high_since = None
                    self.device_reset_count += 1
                    if schedule_reinit:
                        self.reinitialize_pending = True
                return True
            except Exception:
                with self.lock:
                    self.device_reset_failures += 1
                return False
            finally:
                with self.lock:
                    self.hard_reset_in_progress = False

    def _reinitialize_if_pending(self):
        """Restore FT configuration after a watchdog reset."""
        with self.lock:
            if not self.reinitialize_pending:
                return None
            self.reinitialize_pending = False
        initializer = getattr(self.touch, "reinitialize", None)
        if not callable(initializer):
            initializer = getattr(self.touch, "initialize", None)
        if not callable(initializer):
            error = RuntimeError("touch controller cannot reinitialize")
            with self.lock:
                self.reinitialize_pending = True
                self.reinitialize_failures += 1
            return error
        self._begin_transaction()
        try:
            info = initializer(reset=False)
        except Exception as error:
            with self.lock:
                self.reinitialize_pending = True
                self.reinitialize_failures += 1
            print("[touch-iic] controller reinitialize failed: {}".format(error), flush=True)
            return error
        finally:
            self._end_transaction()
        with self.lock:
            self.reinitialize_count += 1
        print("[touch-iic] controller reinitialized: {}".format(info), flush=True)
        return None

    def _print_touch_event(self, event):
        """Keep serial logging from throttling the IIC worker during drags."""
        now = time.monotonic()
        with self.lock:
            if (
                event.kind == "move"
                and now - self.last_touch_print_at < self.touch_log_interval
            ):
                return
            self.last_touch_print_at = now
        print(
            "[touch] {}: ({}, {}) points={}".format(
                event.kind, event.x, event.y, event.points
            ), flush=True,
        )

    def _watchdog(self):
        """Reset AXI-IIC if PYNQ is stuck inside XIic_WaitBusFree()."""
        while not self.stop_event.is_set() and not self.watchdog_stop_event.is_set():
            now = time.monotonic()
            token = None
            age = 0.0
            attempt = 0
            with self.lock:
                if self.in_flight:
                    age = now - self.last_poll_started
                    token = self.poll_generation
                    if self.watchdog_generation != token:
                        self.watchdog_generation = token
                        self.watchdog_attempt_count = 0
                    attempt = self.watchdog_attempt_count + 1
                else:
                    attempt = 0
                if (
                    token is None
                    or attempt > self.watchdog_max_attempts
                    or age < self.watchdog_timeout * attempt
                ):
                    token = None
                else:
                    self.watchdog_attempt_count = attempt
            if token is not None:
                device_reset = self._hard_reset_device()
                # Reset the touch IC first so a device holding SDA/SCL is
                # released before the AXI-IIC SOFTR write wakes the blocked
                # PYNQ transaction.  Doing this in the opposite order lets
                # the poll thread race into a new transaction while the
                # controller is still being reset.
                recovered = self._recover_bus()
                with self.lock:
                    if recovered:
                        self.watchdog_recoveries += 1
                    else:
                        self.watchdog_failures += 1
                    count = self.watchdog_recoveries + self.watchdog_failures
                    should_print = (
                        count <= 3
                        or now - self.watchdog_last_print_at >= 2.0
                    )
                    if should_print:
                        self.watchdog_last_print_at = now
                if should_print:
                    print(
                        "[touch-iic] watchdog recovery={} device_reset={} attempt={} age={:.2f}s count={}".format(
                            "ok" if recovered else "unavailable",
                            "ok" if device_reset else "unavailable",
                            attempt, age, count
                        ), flush=True,
                    )
            self.watchdog_stop_event.wait(self.watchdog_interval)

    def _run(self):
        """Poll touch IIC with visible errors and active bus recovery."""
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.lock:
                retry_at = self.retry_at
            try:
                if started < retry_at:
                    self.stop_event.wait(min(self.poll_interval, retry_at - started))
                    continue
                with self.lock:
                    reset_in_progress = self.hard_reset_in_progress
                if reset_in_progress:
                    self.stop_event.wait(min(self.poll_interval, 0.05))
                    continue
                event = None
                reinitialize_error = self._reinitialize_if_pending()
                if reinitialize_error is not None:
                    raise reinitialize_error
                if self.int_gate:
                    level = self.touch.int_level
                    if level != self.last_int_level:
                        print("[touch-int] level={}".format(level), flush=True)
                        self.last_int_level = level
                    # An idle-high INT line normally lets us avoid an IIC
                    # transaction.  After an IIC exception, however, that
                    # shortcut would clear the error without ever checking
                    # whether the bus actually recovered.  Force one real
                    # poll while an error is pending; the successful poll
                    # then becomes the recovery confirmation.
                    with self.lock:
                        error_pending = self.error is not None
                    if level is not None and int(level) != 0 and not error_pending:
                        if self.int_high_since is None:
                            self.int_high_since = time.monotonic()
                        if time.monotonic() - self.int_high_since >= self.release_guard:
                            event = self.touch.force_release()
                        if event is None:
                            self._mark_success()
                            self.stop_event.wait(self.poll_interval)
                            continue
                    else:
                        self.int_high_since = None
                        event = self._poll_transaction()
                else:
                    event = self._poll_transaction()
                reinitialize_error = self._reinitialize_if_pending()
                if reinitialize_error is not None:
                    raise reinitialize_error
                if event is not None:
                    self._print_touch_event(event)
                    with self.lock:
                        self.events.append(event)
            except Exception as error:
                self._record_error(error)
                reinitialize_error = self._reinitialize_if_pending()
                if reinitialize_error is not None:
                    self._record_error(reinitialize_error)
                recovered = self._recover_bus()
                with self.lock:
                    if recovered:
                        self.recovery_count += 1
                    else:
                        self.recovery_failures += 1
                    consecutive = self.error_count
                    total = self.total_errors
                    delay = min(1.0, 0.10 * (2 ** (consecutive - 1)))
                    self.retry_at = time.monotonic() + delay
                if self._should_print_error():
                    print(
                        "[touch-iic] recovery={} errors={} consecutive={}".format(
                            "ok" if recovered else "unavailable", total, consecutive
                        ), flush=True,
                    )
                self.stop_event.wait(min(delay, self.poll_interval))
            else:
                self._mark_success()
                self.stop_event.wait(self.poll_interval)
    def _record_error(self, error):
        now = time.monotonic()
        with self.lock:
            self.error = error
            self.last_error_text = str(error)
            self.last_error_at = now
            self.error_count = min(self.error_count + 1, 5)
            self.total_errors += 1

    def _should_print_error(self):
        now = time.monotonic()
        with self.lock:
            if self.total_errors > 3 and now - self.last_error_print_at < 2.0:
                return False
            self.last_error_print_at = now
            text = self.last_error_text
            count = self.total_errors
        print("[touch-iic] error #{}: {}".format(count, text), flush=True)
        return True

    def _mark_success(self):
        with self.lock:
            had_error = self.error is not None
            previous = self.last_error_text
            self.error = None
            self.error_count = 0
            self.retry_at = 0.0
        if had_error:
            print("[touch-iic] recovered after: {}".format(previous), flush=True)

    def snapshot(self):
        """Return a thread-safe health snapshot for the main loop."""
        now = time.monotonic()
        with self.lock:
            if not self.thread.is_alive() or self.stop_event.is_set():
                state = "stopped"
            elif self.in_flight and now - self.last_poll_started > self.watchdog_timeout:
                state = "stalled"
            elif self.error is not None:
                state = "error"
            elif now < self.retry_at:
                state = "retry"
            else:
                state = "ok"
            return {
                "state": state,
                "error": str(self.error) if self.error is not None else None,
                "last_error": self.last_error_text,
                "errors": int(self.total_errors),
                "recoveries": int(self.recovery_count),
                "recovery_failures": int(self.recovery_failures),
                "watchdogs": int(self.watchdog_recoveries),
                "watchdog_failures": int(self.watchdog_failures),
                "device_resets": int(self.device_reset_count),
                "device_reset_failures": int(self.device_reset_failures),
                "reinitializes": int(self.reinitialize_count),
                "reinitialize_failures": int(self.reinitialize_failures),
                "poll_age": max(0.0, now - self.last_poll_completed),
            }

    def drain(self):
        with self.lock:
            events = list(self.events)
            self.events.clear()
            return events

    def stop(self, timeout=2.0):
        self.stop_event.set()
        self.watchdog_stop_event.set()
        # Wake a PYNQ XIic_WaitBusFree call that is already inside the C
        # wrapper.  Without this, Ctrl-C can leave the daemon thread blocked
        # until the vendor timeout and the shell reports a false shutdown
        # failure.
        try:
            # Release a touch device that may still hold SDA/SCL before
            # clearing the AXI-IIC core.  This is also safe during shutdown:
            # no new poll is started once stop_event is set.
            self._hard_reset_device(schedule_reinit=False)
        except Exception:
            pass
        try:
            self._recover_bus()
        except Exception:
            pass
        self.thread.join(timeout=float(timeout))
        self.watchdog_thread.join(timeout=min(0.5, float(timeout)))
        return not self.thread.is_alive() and not self.watchdog_thread.is_alive()


class TouchProcessWorker:
    """Supervise touch IIC in a killable Linux child process.

    PYNQ 2.7 can remain inside a C XIic polling loop after the AXI-IIC core
    has been reset. A Python thread cannot be killed while it is in that C
    call. The child process owns all touch reads; the parent only receives
    events and can terminate/recreate the child after a heartbeat timeout.
    """

    def __init__(
        self,
        touch,
        poll_interval=0.10,
        max_events=64,
        int_gate=True,
        release_guard=0.25,
        watchdog_timeout=0.80,
        touch_log_interval=0.25,
    ):
        try:
            self.context = mp.get_context("fork")
        except ValueError as error:
            raise RuntimeError("touch process requires Linux fork support") from error
        self.touch = touch
        self.poll_interval = float(poll_interval)
        self.int_gate = bool(int_gate)
        self.release_guard = max(0.05, float(release_guard))
        self.watchdog_timeout = max(0.25, float(watchdog_timeout))
        self.touch_log_interval = max(0.05, float(touch_log_interval))
        self.events = deque(maxlen=max_events)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._supervise, name="touch-process-supervisor", daemon=True
        )
        self.process = None
        self.child_stop = None
        self.connection = None
        self.last_heartbeat = time.monotonic()
        self.last_poll_started = 0.0
        self.last_poll_completed = time.monotonic()
        self.in_flight = False
        self.error = None
        self.error_count = 0
        self.total_errors = 0
        self.recovery_count = 0
        self.recovery_failures = 0
        self.watchdog_recoveries = 0
        self.watchdog_failures = 0
        self.device_reset_count = 0
        self.device_reset_failures = 0
        self.reinitialize_count = 0
        self.reinitialize_failures = 0
        self.last_error_text = None
        self.last_error_print_at = 0.0
        self.watchdog_last_print_at = 0.0
        self.last_touch_print_at = 0.0
        self.retry_at = 0.0
        self.recovery_lock = threading.Lock()
        self.restart_lock = threading.Lock()

    def start(self):
        self._spawn_child()
        self.thread.start()

    def _spawn_child(self):
        child_stop = self.context.Event()
        parent_conn, child_conn = self.context.Pipe(duplex=False)
        process = self.context.Process(
            target=_touch_child_main,
            args=(
                self.touch,
                self.poll_interval,
                self.int_gate,
                self.release_guard,
                child_conn,
                child_stop,
            ),
            name="touch-iic-child",
        )
        process.daemon = True
        try:
            process.start()
        except Exception:
            parent_conn.close()
            child_conn.close()
            raise
        child_conn.close()
        with self.lock:
            self.process = process
            self.child_stop = child_stop
            self.connection = parent_conn
            self.last_heartbeat = time.monotonic()
            self.last_poll_started = 0.0
            self.in_flight = False

    def _terminate_child(self):
        with self.lock:
            process = self.process
            child_stop = self.child_stop
            connection = self.connection
            self.process = None
            self.child_stop = None
            self.connection = None
            self.in_flight = False
        if child_stop is not None:
            child_stop.set()
        if process is not None:
            process.join(timeout=0.20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.80)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def _recover_bus(self):
        with self.recovery_lock:
            try:
                return bool(self.touch.recover_bus())
            except Exception:
                return False

    def _hard_reset(self):
        with self.recovery_lock:
            reset = getattr(self.touch, "hard_reset", None)
            if not callable(reset):
                return False
            try:
                reset()
                return True
            except Exception:
                return False

    def _restart_child(self, reason):
        if self.stop_event.is_set():
            return
        with self.restart_lock:
            if self.stop_event.is_set():
                return
            self._terminate_child()
            reset_ok = self._hard_reset()
            bus_ok = self._recover_bus()
            now = time.monotonic()
            with self.lock:
                self.events.clear()
                self.in_flight = False
                self.last_heartbeat = now
                self.retry_at = now + 0.10
                if reset_ok:
                    self.device_reset_count += 1
                else:
                    self.device_reset_failures += 1
                if bus_ok:
                    self.recovery_count += 1
                else:
                    self.recovery_failures += 1
            print(
                "[touch-iic] child restart reason={} device_reset={} recovery={}".format(
                    reason, "ok" if reset_ok else "unavailable",
                    "ok" if bus_ok else "unavailable",
                ), flush=True,
            )
            if not self.stop_event.is_set():
                self._spawn_child()

    def _record_error(self, text):
        now = time.monotonic()
        with self.lock:
            self.error = RuntimeError(str(text))
            self.last_error_text = str(text)
            self.error_count = min(self.error_count + 1, 5)
            self.total_errors += 1
            count = self.total_errors
            should_print = count <= 3 or now - self.last_error_print_at >= 2.0
            if should_print:
                self.last_error_print_at = now
        if should_print:
            print("[touch-iic] error #{}: {}".format(count, text), flush=True)

    def _mark_success(self):
        with self.lock:
            had_error = self.error is not None
            previous = self.last_error_text
            self.error = None
            self.error_count = 0
            self.retry_at = 0.0
        if had_error:
            print("[touch-iic] recovered after: {}".format(previous), flush=True)

    def _print_touch_event(self, event):
        now = time.monotonic()
        with self.lock:
            if (
                event.kind == "move"
                and now - self.last_touch_print_at < self.touch_log_interval
            ):
                return
            self.last_touch_print_at = now
        print(
            "[touch] {}: ({}, {}) points={}".format(
                event.kind, event.x, event.y, event.points
            ), flush=True,
        )

    def _drain_messages(self):
        with self.lock:
            connection = self.connection
        if connection is None:
            return False
        restart = None
        try:
            while connection.poll():
                kind, value = connection.recv()
                now = time.monotonic()
                with self.lock:
                    self.last_heartbeat = now
                if kind == "poll_start":
                    with self.lock:
                        self.in_flight = True
                        self.last_poll_started = now
                elif kind == "poll_done":
                    with self.lock:
                        self.in_flight = False
                        self.last_poll_completed = now
                elif kind == "heartbeat" or kind == "starting":
                    continue
                elif kind == "ready":
                    with self.lock:
                        self.reinitialize_count += 1
                    print("[touch-iic] child ready: {}".format(value), flush=True)
                    self._mark_success()
                elif kind == "int":
                    print("[touch-int] level={}".format(value), flush=True)
                elif kind == "event":
                    self._print_touch_event(value)
                    with self.lock:
                        self.events.append(value)
                elif kind == "error":
                    self._record_error(value)
                    restart = "child-error"
                    break
        except (EOFError, OSError) as error:
            restart = "pipe-error: {}".format(error)
        if restart is not None:
            self._restart_child(restart)
        return restart is not None

    def _supervise(self):
        while not self.stop_event.is_set():
            self._drain_messages()
            now = time.monotonic()
            with self.lock:
                process = self.process
                last_heartbeat = self.last_heartbeat
                retry_at = self.retry_at
            if process is None or not process.is_alive():
                if now >= retry_at and not self.stop_event.is_set():
                    self._restart_child("child-exited")
            elif now - last_heartbeat >= self.watchdog_timeout:
                with self.lock:
                    self.watchdog_recoveries += 1
                    count = self.watchdog_recoveries
                    should_print = count <= 3 or now - self.watchdog_last_print_at >= 2.0
                    if should_print:
                        self.watchdog_last_print_at = now
                if should_print:
                    print(
                        "[touch-iic] process watchdog timeout age={:.2f}s count={}".format(
                            now - last_heartbeat, count
                        ), flush=True,
                    )
                self._restart_child("heartbeat-timeout")
            self.stop_event.wait(0.02)

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            process = self.process
            alive = process is not None and process.is_alive()
            if self.stop_event.is_set() or not alive:
                state = "stopped"
            elif now - self.last_heartbeat >= self.watchdog_timeout:
                state = "stalled"
            elif self.error is not None:
                state = "error"
            elif now < self.retry_at:
                state = "retry"
            else:
                state = "ok"
            return {
                "state": state,
                "error": str(self.error) if self.error is not None else None,
                "last_error": self.last_error_text,
                "errors": int(self.total_errors),
                "recoveries": int(self.recovery_count),
                "recovery_failures": int(self.recovery_failures),
                "watchdogs": int(self.watchdog_recoveries),
                "watchdog_failures": int(self.watchdog_failures),
                "device_resets": int(self.device_reset_count),
                "device_reset_failures": int(self.device_reset_failures),
                "reinitializes": int(self.reinitialize_count),
                "reinitialize_failures": int(self.reinitialize_failures),
                "poll_age": max(0.0, now - self.last_poll_completed),
            }

    def drain(self):
        with self.lock:
            events = list(self.events)
            self.events.clear()
            return events

    def stop(self, timeout=2.0):
        self.stop_event.set()
        with self.restart_lock:
            self._terminate_child()
            self._hard_reset()
            self._recover_bus()
        self.thread.join(timeout=float(timeout))
        with self.lock:
            process = self.process
        return not self.thread.is_alive() and process is None


def _read_frame(video_in, stop_event):
    """Read a frame while allowing SIGINT to interrupt a stalled VDMA wait."""
    channel = getattr(getattr(video_in, "_vdma", None), "readchannel", None)
    mmio = getattr(channel, "_mmio", None)
    internal = getattr(channel, "_readframe_internal", None)
    if mmio is None or not callable(internal):
        return video_in.readframe()
    while mmio.read(0x34) & 0x1000 == 0:
        if stop_event.is_set():
            return None
        time.sleep(0.001)
    mmio.write(0x34, 0x1000)
    return internal()


def _write_frame(video_out, frame, stop_event):
    """Write a frame while allowing SIGINT to interrupt a stalled MM2S wait."""
    channel = getattr(getattr(video_out, "_vdma", None), "writechannel", None)
    mmio = getattr(channel, "_mmio", None)
    internal = getattr(channel, "_writeframe_internal", None)
    if mmio is None or not callable(internal):
        video_out.writeframe(frame)
        return True
    while mmio.read(0x04) & 0x1000 == 0:
        if stop_event.is_set():
            return False
        time.sleep(0.001)
    mmio.write(0x04, 0x1000)
    internal(frame)
    return True


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bitstream",
        default=str(PROJECT_DIR / "artifacts" / "pefsl_mix_plus_touch.bit"),
    )
    parser.add_argument(
        "--tmodel",
        default=str(PROJECT_DIR / "tcu_deploy" /
                     "resnet9_strided_16fmaps_onnx_custom_perf.tmodel"),
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_DIR / "pynq_app" / "pefsl_learned.npz"),
    )
    parser.add_argument(
        "--digit-cache",
        default=str(PROJECT_DIR / "pynq_app" / "digit_support_r17.npz"),
        help=(
            "D-mode digit support cache (default: "
            "pynq_app/digit_support_r17.npz)"
        ),
    )
    parser.add_argument(
        "--digit-min-score",
        type=float,
        default=0.25,
        help="minimum smoothed D-mode class probability (default: 0.25)",
    )
    parser.add_argument(
        "--digit-min-margin",
        type=float,
        default=0.04,
        help="minimum D-mode top-1/top-2 probability margin (default: 0.04)",
    )
    parser.add_argument(
        "--digit-history",
        type=int,
        default=5,
        help="number of D-mode probability frames to smooth (default: 5)",
    )
    parser.add_argument("--collect-frames", type=int, default=8)
    parser.add_argument("--operator-cache", default=str(APP_DIR / "operator_support_r17.npz"))
    parser.add_argument("--operator-min-score", type=float, default=0.45,
                        help="experimental operator score gate; calibrate on board (default: 0.45)")
    parser.add_argument("--operator-min-margin", type=float, default=0.08,
                        help="experimental operator top-1/top-2 margin (default: 0.08)")
    parser.add_argument("--operator-history", type=int, default=5)
    parser.add_argument("--digit-preview", choices=("on", "off"), default="on",
                        help="D-mode binary preview sidebar (default: on)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="stop after this many seconds; 0 means run until Ctrl-C",
    )
    parser.add_argument("--debug-tcu", action="store_true")
    parser.add_argument("--keyboard-device", default=None,
                        help="keyboard event path; default: discover an A-Z/Enter keyboard")
    parser.add_argument("--no-keyboard", action="store_true",
                        help="disable USB keyboard input and retain touch-only learning")
    parser.add_argument("--keyboard-no-grab", action="store_true",
                        help="diagnostic: allow the console to also receive keyboard input")
    parser.add_argument("--skip-lcd-id", action="store_true")
    parser.add_argument(
        "--touch-poll-ms",
        type=float,
        default=100.0,
        help="touch poll period in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--predict-ms",
        type=float,
        default=1000.0,
        help="B-mode prediction refresh period in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--scan-ms",
        type=float,
        default=100.0,
        help="automatic scan interval between TCU windows (default: 100)",
    )
    parser.add_argument(
        "--scan-windows",
        type=int,
        default=24,
        help="maximum windows per scan cycle (default: 24)",
    )
    parser.add_argument(
        "--scan-cos-th",
        type=float,
        default=0.56,
        help="r17 automatic scan cosine threshold (default: 0.56); manual B uses NCM",
    )
    parser.add_argument(
        "--scan-hist-th",
        type=float,
        default=0.0,
        help="optional r17 automatic scan histogram intersection gate (default: 0, disabled)",
    )
    parser.add_argument(
        "--scan-confirm-cycles", "--scan-confirm-hits",
        type=int,
        default=2,
        help="consecutive r17 scan cycles required before display (default: 2)",
    )
    parser.add_argument(
        "--scan-contrast-th",
        type=float,
        default=0.14,
        help="minimum centre/surround HSV distance for scan candidates (default: 0.14)",
    )
    parser.add_argument("--no-touch-int-gate", action="store_true", help="poll IIC even when INT is high (diagnostic)")
    parser.add_argument("--b-max-result-age", type=float, default=1.5,
                        help="accepted for C124 command compatibility; unused by restored r17 A/B")
    return parser.parse_args(argv)


def _setup_video(overlay, skip_lcd_id=False):
    import pynq.lib.ov5640 as camera_module
    import pynq.lib.video2lcd as video_module
    from pynq.lib.iic import AxiIIC
    from pynq.lib.ov5640 import OV5640

    del camera_module, video_module
    if "cam_iic" not in overlay.ip_dict:
        raise RuntimeError("overlay HWH has no cam_iic")
    if not hasattr(overlay, "video"):
        raise RuntimeError("overlay HWH has no video hierarchy")

    video = overlay.video
    video_in = video.video_in
    video_out = video.video_out
    camera_iic = AxiIIC(overlay.ip_dict["cam_iic"])
    camera = OV5640(0x3C, camera_iic)
    camera.init()
    if not skip_lcd_id:
        lcd_id = int(video_out.lcd_id_rd())
        print("LCD ID: {}".format(hex(lcd_id)), flush=True)
        if lcd_id != 0x7084:
            raise RuntimeError("expected LCD ID 0x7084, got {}".format(hex(lcd_id)))
    camera.set_outsize(800, 480, 1800, 1000)
    video_out.configure(1)
    video_in.configure(video_out.mode)
    mode = video_out.mode
    if (int(mode.width), int(mode.height), int(mode.bits_per_pixel)) != (800, 480, 24):
        raise RuntimeError("unexpected video mode: {}".format(mode))
    video_in.start()
    video_out.start()
    # Vendor notebook starts the stream with input/output channels tied.
    video_in.tie(video_out)
    return camera, video_in, video_out


def _discard_warmup_frame(video_in, stop_event):
    warmup = _read_frame(video_in, stop_event)
    if warmup is None:
        raise KeyboardInterrupt()
    try:
        print("Warm-up frame: {}".format(tuple(warmup.shape)), flush=True)
    finally:
        warmup.freebuffer()


def main(argv=None):
    args = _parse_args(argv)
    if args.collect_frames < 1:
        raise SystemExit("--collect-frames must be >= 1")
    if args.seconds < 0:
        raise SystemExit("--seconds must be >= 0")
    if args.touch_poll_ms <= 0:
        raise SystemExit("--touch-poll-ms must be > 0")
    if args.predict_ms <= 0:
        raise SystemExit("--predict-ms must be > 0")
    if args.scan_ms <= 0:
        raise SystemExit("--scan-ms must be > 0")
    if args.scan_windows < 4:
        raise SystemExit("--scan-windows must be >= 4")
    if not 0.0 <= args.scan_cos_th <= 1.0:
        raise SystemExit("--scan-cos-th must be between 0 and 1")
    if not 0.0 <= args.scan_hist_th <= 1.0:
        raise SystemExit("--scan-hist-th must be between 0 and 1")
    if args.scan_confirm_cycles < 2:
        raise SystemExit("--scan-confirm-cycles must be >= 2")
    if not 0.0 <= args.scan_contrast_th <= 1.0:
        raise SystemExit("--scan-contrast-th must be between 0 and 1")
    if not 0.0 <= args.digit_min_score <= 1.0:
        raise SystemExit("--digit-min-score must be between 0 and 1")
    if args.digit_min_margin < 0.0:
        raise SystemExit("--digit-min-margin must be >= 0")
    if args.digit_history < 1:
        raise SystemExit("--digit-history must be >= 1")
    if not 0.0 <= args.operator_min_score <= 1.0:
        raise SystemExit("--operator-min-score must be between 0 and 1")
    if not 0.0 <= args.operator_min_margin <= 1.0:
        raise SystemExit("--operator-min-margin must be between 0 and 1")
    if args.operator_history < 1:
        raise SystemExit("--operator-history must be >= 1")

    bitstream = Path(args.bitstream).resolve()
    tmodel = Path(args.tmodel).resolve()
    data_path = Path(args.data).resolve()
    required = (
        bitstream,
        bitstream.with_suffix(".hwh"),
        tmodel,
        tmodel.with_suffix(".tprog"),
        tmodel.with_suffix(".tdata"),
    )
    for path in required:
        if not path.is_file():
            raise SystemExit("missing required file: {}".format(path))
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("warning: run with sudo -E and the PYNQ virtualenv", flush=True)

    stop_event = threading.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)

    overlay = None
    touch = None
    touch_worker = None
    keyboard_worker = None
    ui = None
    video_in = None
    video_out = None
    frame = None
    touch_initialized = False
    started = time.monotonic()
    frames = 0
    last_report = started
    last_report_state = None

    try:
        import pynq
        import pynq.lib.ov5640 as camera_module
        import pynq.lib.video2lcd as video_module
        del camera_module, video_module

        print("Python: {}".format(sys.executable), flush=True)
        print("PYNQ: {}".format(getattr(pynq, "__version__", "unknown")), flush=True)
        print("Loading overlay: {}".format(bitstream), flush=True)
        overlay = load_overlay(bitstream)

        touch = TouchController.from_overlay(overlay)
        info = touch.initialize()
        touch_initialized = True
        print("Touch controller: {}".format(info), flush=True)
        print("Touch INT level: {}".format(touch.int_level), flush=True)

        # Fork the touch process before creating the UI predictor thread or
        # opening the camera. The child then inherits only the initialized
        # overlay/touch MMIO state and owns all subsequent touch IIC calls.
        touch_worker = TouchProcessWorker(
            touch,
            poll_interval=args.touch_poll_ms / 1000.0,
            int_gate=not args.no_touch_int_gate,
        )
        touch_worker.start()

        extractor = TcuFeatureExtractor(
            overlay, tmodel, tcu_path=tmodel.parent, debug=args.debug_tcu
        )
        ui = TouchBboxUI(
            extractor,
            data_path,
            collect_frames=args.collect_frames,
            predict_interval=args.predict_ms / 1000.0,
            scan_interval=args.scan_ms / 1000.0,
            scan_windows=args.scan_windows,
            scan_cos_threshold=args.scan_cos_th,
            scan_hist_threshold=args.scan_hist_th,
            scan_confirm_cycles=args.scan_confirm_cycles,
            scan_contrast_threshold=args.scan_contrast_th,
            b_max_result_age=args.b_max_result_age,
            digit_cache_path=args.digit_cache,
            digit_min_score=args.digit_min_score,
            digit_min_margin=args.digit_min_margin,
            digit_history_len=args.digit_history,
            operator_cache_path=args.operator_cache,
            operator_min_score=args.operator_min_score,
            operator_min_margin=args.operator_min_margin,
            operator_history_len=args.operator_history,
            digit_preview=args.digit_preview == "on",
        )
        print("UI version: {}".format(APP_VERSION), flush=True)
        print("Digits: {}".format(ui.digit_cache_status), flush=True)
        print("Operators: {}".format(ui.operator_cache_status), flush=True)
        print("Operator gates: score={} margin={} (experimental)".format(
            args.operator_min_score, args.operator_min_margin), flush=True)
        if not args.no_keyboard:
            # Start after the initial touch fork. The keyboard never performs
            # touch IIC or TCU work and only enqueues immutable input events.
            keyboard_worker = USBKeyboardWorker(
                device=args.keyboard_device, grab=not args.keyboard_no_grab
            )
            keyboard_worker.start()
        last_keyboard_state = None
        camera, video_in, video_out = _setup_video(overlay, args.skip_lcd_id)
        _discard_warmup_frame(video_in, stop_event)

        started = time.monotonic()
        last_report = started
        last_report_state = None
        print("Video mode: {}".format(video_out.mode), flush=True)
        print("Touch bbox UI started; draw ROI on the LCD. Ctrl-C exits.", flush=True)

        while (
            not stop_event.is_set()
            and (args.seconds == 0 or time.monotonic() - started < args.seconds)
        ):
            frame = _read_frame(video_in, stop_event)
            if frame is None:
                break
            try:
                if tuple(frame.shape) != (480, 800, 3):
                    raise RuntimeError("unexpected frame shape {}".format(frame.shape))
                _dispatch_ui_inputs(ui, touch_worker, keyboard_worker, stop_event)
                if stop_event.is_set():
                    break
                if keyboard_worker is not None:
                    keyboard_health = keyboard_worker.snapshot()
                    keyboard_state = tuple(keyboard_health.get(key) for key in
                                           ("state", "device", "name", "last_error"))
                    if keyboard_state != last_keyboard_state:
                        print("[keyboard] state={} device={} name={} error={}".format(
                            *keyboard_state), flush=True)
                        last_keyboard_state = keyboard_state
                ui.process_frame(frame)
                if not _write_frame(video_out, frame, stop_event):
                    break
                frame = None
            finally:
                if frame is not None:
                    # A keyboard Ctrl-C or interrupted VDMA write can leave
                    # this loop before ownership is transferred. Clear the
                    # reference before releasing so outer cleanup cannot
                    # release the same PYNQ allocation a second time.
                    unfinished_frame = frame
                    frame = None
                    unfinished_frame.freebuffer()

            frames += 1
            now = time.monotonic()
            if now - last_report >= 5.0:
                health = touch_worker.snapshot()
                state = (
                    ui.mode, ui.status, health["state"], health["errors"],
                    health["watchdogs"], health["device_resets"],
                    health["reinitializes"],
                )
                # Keep the normal heartbeat compact.  Once an IIC error has
                # occurred, retain its count and last text so a transient
                # error cannot disappear before the five-second report.
                if state != last_report_state or health["errors"]:
                    suffix = ""
                    if health["errors"] or health["watchdogs"]:
                        suffix = (
                            " touch_iic={} errors={} recoveries={} watchdogs={} resets={} reinit={} last={}".format(
                                health["state"], health["errors"],
                                health["recoveries"], health["watchdogs"],
                                health["device_resets"], health["reinitializes"],
                                health["last_error"]
                            )
                        )
                    print(
                        "frames={} elapsed={:.1f}s mode={} status={}{}".format(
                            frames, now - started, ui.mode, ui.status, suffix
                        ), flush=True,
                    )
                    last_report_state = state
                last_report = now

    except KeyboardInterrupt:
        stop_event.set()
        print("Stopped by Ctrl-C", flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        stop_event.set()
        if frame is not None:
            try:
                frame.freebuffer()
            except Exception:
                pass
        if keyboard_worker is not None:
            if not keyboard_worker.stop(timeout=1.0):
                print("Keyboard worker did not stop before timeout", flush=True)
        if touch_worker is not None:
            stopped = touch_worker.stop()
            if not stopped:
                print("Touch worker did not stop before timeout", flush=True)
        if ui is not None:
            stopped = ui.close()
            if not stopped:
                print("TCU prediction thread did not stop before timeout", flush=True)
        for name, channel in (("video_in", video_in), ("video_out", video_out)):
            if channel is not None:
                try:
                    channel.close()
                except Exception as error:
                    print("{} close failed: {}".format(name, error), flush=True)
        if touch_initialized:
            try:
                touch.release_reset()
            except Exception as error:
                print("touch reset release failed: {}".format(error), flush=True)
        signal.signal(signal.SIGINT, previous_sigint)

    print("PYNQ bbox app exited after {} frames".format(frames), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())













