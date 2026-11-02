#!/usr/bin/env python3
"""Validate the MIX_PLUS touch IIC path on a PYNQ board."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from touch_driver import DRIVER_VERSION, TouchController


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    here = Path(__file__).resolve().parent
    default_bit = here.parent / "artifacts" / "pefsl_mix_plus_touch.bit"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", type=Path, default=default_bit)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--poll-ms", type=float, default=10.0)
    parser.add_argument("--out-dir", type=Path, default=here / "validation_result")
    args = parser.parse_args()
    if args.seconds <= 0 or args.poll_ms <= 0:
        parser.error("--seconds and --poll-ms must be positive")

    bitstream = args.bitstream.resolve()
    hwh = bitstream.with_suffix(".hwh")
    if not bitstream.is_file():
        parser.error("Missing bitstream: {}".format(bitstream))
    if not hwh.is_file():
        parser.error("Missing HWH beside bitstream: {}".format(hwh))
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "bitstream": str(bitstream),
        "bitstream_sha256": sha256(bitstream),
        "hwh": str(hwh),
        "hwh_sha256": sha256(hwh),
        "requested_seconds": args.seconds,
        "poll_ms": args.poll_ms,
        "events": [],
        "automated_checks_pass": False,
        "stage": "imports",
        "driver_version": DRIVER_VERSION,
    }
    exit_code = 1
    touch = None
    try:
        from pynq import Overlay

        report["stage"] = "parse_hwh"
        overlay = Overlay(str(bitstream), download=False, ignore_version=True)
        required = ("touch_iic", "touch_int_gpio", "touch_rst_gpio")
        report["ip_dict"] = sorted(overlay.ip_dict)
        missing = [name for name in required if name not in overlay.ip_dict]
        if missing:
            raise RuntimeError("HWH is missing touch IP: {}".format(", ".join(missing)))

        report["stage"] = "overlay_download"
        overlay.download()
        report["stage"] = "touch_initialize"
        touch = TouchController.from_overlay(overlay)
        report["controller"] = touch.initialize()
        report["int_level_after_init"] = touch.int_level
        print("Driver version:", DRIVER_VERSION, flush=True)
        print("Touch controller:", report["controller"], flush=True)
        print("INT level:", report["int_level_after_init"], flush=True)

        report["stage"] = "poll"
        started = time.monotonic()
        counts = {"down": 0, "move": 0, "up": 0}
        while time.monotonic() - started < args.seconds:
            event = touch.poll()
            if event is not None:
                counts[event.kind] = counts.get(event.kind, 0) + 1
                if len(report["events"]) < 200:
                    report["events"].append(event.__dict__)
                print(
                    "{}: ({}, {}) raw=({}, {}) points={}".format(
                        event.kind, event.x, event.y, event.raw_x, event.raw_y, event.points
                    ),
                    flush=True,
                )
            time.sleep(args.poll_ms / 1000.0)
        report["event_counts"] = counts
        report["automated_checks_pass"] = True
        report["stage"] = "completed"
        exit_code = 0
    except KeyboardInterrupt:
        report["error"] = "Interrupted; validation incomplete"
        report["stage"] = "interrupted"
        exit_code = 130
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        report["stage"] = "error"
        traceback.print_exc()
    finally:
        if touch is not None:
            try:
                touch.release_reset()
            except Exception:
                pass
        report_path = output / "touch_iic_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Report:", report_path, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

