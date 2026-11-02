#!/usr/bin/env python3
"""Small Linux evdev keyboard reader; no third-party Python packages required.

Only a device advertising all 26 alphabet keys and Enter is accepted.  The
reader owns the selected device, emits immutable editing events, and keeps all
UI work on the caller's thread.  Native input_event layout is used on ARM32.
"""
from collections import deque
from dataclasses import dataclass, field
import glob
import os
import select
import struct
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # Allows the pure decoder tests to run on a Windows host.
    fcntl = None


DRIVER_VERSION = "mix-plus-keyboard-20261101-r20"
EV_SYN, EV_KEY = 0, 1
SYN_REPORT, SYN_DROPPED = 0, 3
KEY_ENTER, KEY_KPENTER, KEY_BACKSPACE, KEY_ESC = 28, 96, 14, 1
KEY_CAPSLOCK = 58
KEY_SPACE, KEY_MINUS, KEY_NUMLOCK = 57, 12, 69
SHIFT_KEYS = frozenset((42, 54))
CTRL_KEYS = frozenset((29, 97))
ALT_KEYS = frozenset((56, 100))
LETTER_CODES = {
    30: "a", 48: "b", 46: "c", 32: "d", 18: "e", 33: "f", 34: "g",
    35: "h", 23: "i", 36: "j", 37: "k", 38: "l", 50: "m", 49: "n",
    24: "o", 25: "p", 16: "q", 19: "r", 31: "s", 20: "t", 22: "u",
    47: "v", 17: "w", 45: "x", 21: "y", 44: "z",
}
DIGIT_CODES = {code: str(code - 1) for code in range(2, 11)}
DIGIT_CODES[11] = "0"
KEYPAD_CODES = {
    71: "7", 72: "8", 73: "9", 75: "4", 76: "5",
    77: "6", 79: "1", 80: "2", 81: "3", 82: "0",
}
REQUIRED_KEYS = frozenset(LETTER_CODES) | frozenset((KEY_ENTER,))
KEY_BYTES = (0x2ff + 8) // 8
# Linux struct input_event uses two native longs, two u16s and one s32.
# On the PYNQ armv7 Python this is 16 bytes; on common 64-bit hosts it is 24.
INPUT_EVENT = struct.Struct("@llHHi")


def _ioc(direction, number, size):
    return (int(direction) << 30) | (ord("E") << 8) | int(number) | (int(size) << 16)


EVIOCGRAB = _ioc(1, 0x90, struct.calcsize("i"))
EVIOCSCLOCKID = _ioc(1, 0xa0, struct.calcsize("i"))


@dataclass(frozen=True)
class KeyEvent:
    kind: str
    text: str = ""
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class RawKeyEvent:
    event_type: int
    code: int
    value: int
    timestamp: float = field(default_factory=time.monotonic)


def bitmap_has(bitmap, code):
    index = int(code) // 8
    return index < len(bitmap) and bool(bitmap[index] & (1 << (int(code) % 8)))


def is_alphabet_keyboard(bitmap):
    return all(bitmap_has(bitmap, code) for code in REQUIRED_KEYS)


def candidate_paths(device=None):
    """Prefer stable keyboard links and deduplicate composite-device paths."""
    paths = [str(device)] if device is not None else (
        sorted(glob.glob("/dev/input/by-id/*-event-kbd")) +
        sorted(glob.glob("/dev/input/event*"), key=lambda p: (
            int(p.rsplit("event", 1)[-1]) if p.rsplit("event", 1)[-1].isdigit() else 999999,
            p,
        ))
    )
    found, seen = [], set()
    for path in paths:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            found.append(path)
    return found


class KeyboardDecoder:
    """Pure key-state decoder, independent of file descriptors or hardware."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.held = set()
        self.caps_lock = False
        self.num_lock = False
        self.dropping = False

    def synchronize(self, active_keys=(), caps_lock=False, num_lock=False):
        self.held = set(active_keys) & (SHIFT_KEYS | CTRL_KEYS | ALT_KEYS)
        self.caps_lock = bool(caps_lock)
        self.num_lock = bool(num_lock)

    def feed(self, event_type, code, value, timestamp=None):
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if event_type == EV_SYN and code == SYN_DROPPED:
            self.reset()
            self.dropping = True
            return KeyEvent("disconnected", "input events lost; retype the name", timestamp)
        if self.dropping:
            if event_type == EV_SYN and code == SYN_REPORT:
                self.dropping = False
            return None
        if event_type != EV_KEY:
            return None
        if code in SHIFT_KEYS | CTRL_KEYS | ALT_KEYS:
            if value == 1:
                self.held.add(code)
            elif value == 0:
                self.held.discard(code)
            return None
        # Ignore releases and autorepeat. One physical press produces one edit.
        if value != 1:
            return None
        if code == KEY_CAPSLOCK:
            self.caps_lock = not self.caps_lock
            return None
        if code == KEY_NUMLOCK:
            self.num_lock = not self.num_lock
            return None
        if self.held & CTRL_KEYS:
            if code == 46 and not self.held & ALT_KEYS:
                return KeyEvent("interrupt", "", timestamp)
            return None
        if self.held & ALT_KEYS:
            return None
        if code in LETTER_CODES:
            letter = LETTER_CODES[code]
            if bool(self.held & SHIFT_KEYS) != self.caps_lock:
                letter = letter.upper()
            return KeyEvent("text", letter, timestamp)
        shifted = bool(self.held & SHIFT_KEYS)
        if code in DIGIT_CODES:
            # Shift+number is punctuation on the standard keyboard layout;
            # the naming alphabet deliberately excludes those symbols.
            return None if shifted else KeyEvent("text", DIGIT_CODES[code], timestamp)
        if code == KEY_SPACE:
            return KeyEvent("text", " ", timestamp)
        if code == KEY_MINUS:
            return KeyEvent("text", "_", timestamp) if shifted else None
        if code in KEYPAD_CODES:
            # Shift reverses the keypad Num Lock/navigation meaning. Caps
            # Lock affects only alphabet keys.
            return (KeyEvent("text", KEYPAD_CODES[code], timestamp)
                    if self.num_lock != shifted else None)
        action = {
            KEY_ENTER: "enter", KEY_KPENTER: "enter",
            KEY_BACKSPACE: "backspace", KEY_ESC: "escape",
        }.get(code)
        return KeyEvent(action, "", timestamp) if action else None


class InputEventStream:
    """Decode bounded native-ABI packets, retaining an incomplete last packet."""
    def __init__(self, monotonic_clock=False):
        self.buffer = bytearray()
        self.monotonic_clock = bool(monotonic_clock)

    def feed(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > INPUT_EVENT.size * 256:
            self.buffer.clear()
            raise ValueError("oversized evdev packet batch")
        output = []
        whole = len(self.buffer) // INPUT_EVENT.size * INPUT_EVENT.size
        for offset in range(0, whole, INPUT_EVENT.size):
            seconds, microseconds, kind, code, value = INPUT_EVENT.unpack_from(self.buffer, offset)
            stamp = (float(seconds) + float(microseconds) / 1000000.0
                     if self.monotonic_clock and 0 <= microseconds < 1000000
                     else time.monotonic())
            output.append(RawKeyEvent(kind, code, value, stamp))
        del self.buffer[:whole]
        return output


class USBKeyboardWorker:
    """A bounded evdev worker with capability selection and hotplug recovery.

    start() is asynchronous. drain() returns at most 128 edit events by default.
    capture_raw=True additionally enables drain_raw() for the standalone tester.
    The device is grabbed exclusively unless grab=False is explicitly supplied.
    """
    def __init__(self, device=None, grab=True, reconnect_interval=1.0,
                 capture_raw=False, queue_limit=512):
        self.device = str(device) if device is not None else None
        self.grab = bool(grab)
        self.reconnect_interval = max(0.1, float(reconnect_interval))
        self.capture_raw = bool(capture_raw)
        self.queue_limit = max(8, int(queue_limit))
        self._events, self._raw = deque(), deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._status = {"state": "stopped", "device": None, "name": None,
                        "last_error": None, "errors": 0, "connections": 0}
        self._last_log = None

    def start(self):
        if fcntl is None or not sys.platform.startswith("linux"):
            raise RuntimeError("USB keyboard input requires Linux evdev")
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="usb-keyboard", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=1.0):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, float(timeout)))
        return thread is None or not thread.is_alive()

    def snapshot(self):
        with self._lock:
            return dict(self._status)

    def drain(self, limit=128):
        with self._lock:
            return [self._events.popleft() for _ in range(min(max(0, int(limit)), len(self._events)))]

    def drain_raw(self, limit=256):
        with self._lock:
            return [self._raw.popleft() for _ in range(min(max(0, int(limit)), len(self._raw)))]

    def _state(self, state, device=None, name=None, error=None):
        with self._lock:
            if error:
                self._status["errors"] += 1
            if state == "connected":
                self._status["connections"] += 1
            self._status.update(state=state, device=device, name=name, last_error=error)
        message = (state, device, name, error)
        if message != self._last_log:
            self._last_log = message
            print("[keyboard] {} device={} name={}{}".format(
                state, device or "-", name or "-", " error=" + str(error) if error else ""
            ), flush=True)

    def _push(self, event):
        if event is None:
            return
        with self._lock:
            if len(self._events) >= self.queue_limit:
                # Never drop only an Enter or only a letter: cancel the whole
                # pending name so the caller cannot commit a truncated word.
                self._events.clear()
                self._events.append(KeyEvent("disconnected", "keyboard queue overflow", event.timestamp))
                self._status["errors"] += 1
                self._status["last_error"] = "keyboard queue overflow; input discarded"
                return
            self._events.append(event)

    def _record_raw(self, event):
        if self.capture_raw and (event.event_type == EV_KEY or
                                 (event.event_type == EV_SYN and event.code == SYN_DROPPED)):
            with self._lock:
                if len(self._raw) >= self.queue_limit * 2:
                    self._raw.clear()
                    self._raw.append(RawKeyEvent(EV_SYN, SYN_DROPPED, 0, event.timestamp))
                self._raw.append(event)

    @staticmethod
    def _bitmap(fd, number, length):
        data = bytearray(length)
        fcntl.ioctl(fd, _ioc(2, number, length), data, True)
        return data

    def _open_keyboard(self, path):
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        try:
            caps = self._bitmap(fd, 0x20 + EV_KEY, KEY_BYTES)
            if not is_alphabet_keyboard(caps):
                os.close(fd)
                return None
            raw_name = self._bitmap(fd, 0x06, 256)
            name = bytes(raw_name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
            monotonic_clock = False
            try:
                fcntl.ioctl(fd, EVIOCSCLOCKID, struct.pack("i", time.CLOCK_MONOTONIC))
                monotonic_clock = True
            except (OSError, AttributeError):
                pass
            if self.grab:
                # A failed grab is an error, not permission to fall back to a
                # keyboard that also types into the Linux console.
                fcntl.ioctl(fd, EVIOCGRAB, 1)
            return fd, name, monotonic_clock
        except BaseException:
            os.close(fd)
            raise

    def _synchronize(self, fd, decoder):
        try:
            keys = self._bitmap(fd, 0x18, KEY_BYTES)
            leds = self._bitmap(fd, 0x19, 8)
            active = [code for code in SHIFT_KEYS | CTRL_KEYS | ALT_KEYS if bitmap_has(keys, code)]
            decoder.synchronize(active, caps_lock=bitmap_has(leds, 1),
                                num_lock=bitmap_has(leds, 0))
        except OSError:
            decoder.synchronize()

    def _close_device(self, fd):
        try:
            if self.grab:
                try:
                    fcntl.ioctl(fd, EVIOCGRAB, 0)
                except OSError:
                    pass
        finally:
            os.close(fd)

    def _run(self):
        self._state("waiting")
        try:
            while not self._stop.is_set():
                selected = None
                last_error = None
                for path in candidate_paths(self.device):
                    if self._stop.is_set():
                        break
                    try:
                        opened = self._open_keyboard(path)
                        if opened is not None:
                            selected = (path,) + opened
                            break
                    except OSError as error:
                        last_error = "{}: {}".format(path, error)
                if selected is None:
                    self._state("waiting", error=last_error)
                    self._stop.wait(self.reconnect_interval)
                    continue
                path, fd, name, monotonic_clock = selected
                decoder, stream = KeyboardDecoder(), InputEventStream(monotonic_clock)
                self._synchronize(fd, decoder)
                self._state("connected", path, name)
                try:
                    while not self._stop.is_set():
                        readable, _, _ = select.select([fd], [], [], 0.05)
                        if not readable:
                            continue
                        try:
                            data = os.read(fd, INPUT_EVENT.size * 64)
                        except BlockingIOError:
                            continue
                        if not data:
                            raise OSError("keyboard disconnected")
                        for raw in stream.feed(data):
                            self._record_raw(raw)
                            dropping = decoder.dropping
                            self._push(decoder.feed(raw.event_type, raw.code, raw.value, raw.timestamp))
                            if dropping and raw.event_type == EV_SYN and raw.code == SYN_REPORT:
                                self._synchronize(fd, decoder)
                except (OSError, ValueError) as error:
                    with self._lock:
                        self._events.clear()
                        self._raw.clear()
                    self._push(KeyEvent("disconnected", str(error)))
                    self._state("waiting", error=str(error))
                finally:
                    self._close_device(fd)
                self._stop.wait(self.reconnect_interval)
        finally:
            self._state("stopped")


# Friendly alias for integrations using normal CamelCase initialisms.
UsbKeyboardWorker = USBKeyboardWorker


def key_name(code):
    if code in LETTER_CODES:
        return "KEY_" + LETTER_CODES[code].upper()
    if code in DIGIT_CODES:
        return "KEY_" + DIGIT_CODES[code]
    if code in KEYPAD_CODES:
        return "KEY_KP" + KEYPAD_CODES[code]
    return {
        KEY_ENTER: "KEY_ENTER", KEY_KPENTER: "KEY_KPENTER",
        KEY_BACKSPACE: "KEY_BACKSPACE", KEY_ESC: "KEY_ESC", KEY_CAPSLOCK: "KEY_CAPSLOCK",
        KEY_SPACE: "KEY_SPACE", KEY_MINUS: "KEY_MINUS", KEY_NUMLOCK: "KEY_NUMLOCK",
        42: "KEY_LEFTSHIFT", 54: "KEY_RIGHTSHIFT", 29: "KEY_LEFTCTRL", 97: "KEY_RIGHTCTRL",
        56: "KEY_LEFTALT", 100: "KEY_RIGHTALT",
    }.get(code, "KEY_{}".format(code))
