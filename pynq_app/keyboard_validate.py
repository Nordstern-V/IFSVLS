#!/usr/bin/env python3
"""Validate a board-local USB keyboard without loading an FPGA overlay."""
import argparse
import time

from usb_keyboard import DRIVER_VERSION, EV_KEY, SYN_DROPPED, USBKeyboardWorker, key_name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="explicit /dev/input/eventX or by-id keyboard path")
    parser.add_argument("--seconds", type=float, default=30.0, help="0 runs until Ctrl-C")
    parser.add_argument("--no-grab", action="store_true", help="also allow the Linux console to receive keys")
    args = parser.parse_args(argv)
    if args.seconds < 0:
        parser.error("--seconds must be >= 0")
    worker = USBKeyboardWorker(args.device, grab=not args.no_grab, capture_raw=True)
    text = ""
    started = time.monotonic()
    print("Keyboard driver: {}".format(DRIVER_VERSION), flush=True)
    print("Type letters, 0-9, Space and Shift+- for _; Backspace edits, Enter confirms.", flush=True)
    print("Keypad digits follow Num Lock. Ctrl-C exits. No Overlay/TCU is loaded.", flush=True)
    try:
        worker.start()
        while args.seconds == 0 or time.monotonic() - started < args.seconds:
            for event in worker.drain_raw():
                if event.event_type == EV_KEY:
                    action = {0: "up", 1: "down", 2: "repeat"}.get(event.value, str(event.value))
                    print("raw: {} {}".format(key_name(event.code), action), flush=True)
                elif event.code == SYN_DROPPED:
                    print("raw: SYN_DROPPED; retype the pending name", flush=True)
            for event in worker.drain():
                if event.kind == "interrupt":
                    return 0
                if event.kind == "text":
                    text = (text + event.text)[:32]
                elif event.kind == "backspace":
                    text = text[:-1]
                elif event.kind == "enter":
                    print("confirmed: {!r}".format(text), flush=True)
                    text = ""
                elif event.kind in ("escape", "disconnected"):
                    text = ""
                print("edit: {} text={!r}".format(event.kind, text), flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 0
    finally:
        stopped = worker.stop()
        print("Keyboard validation stopped; device released={} status={}".format(
            stopped, worker.snapshot()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
