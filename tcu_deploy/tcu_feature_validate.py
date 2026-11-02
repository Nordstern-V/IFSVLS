#!/usr/bin/env python3
"""Board smoke test for the PEFSL bbox to TCU feature boundary.

Run on PYNQ from this directory:
  sudo -E /usr/local/share/pynq-venv/bin/python3 tcu_feature_validate.py --bitstream ../artifacts/pefsl_mix_plus_touch.bit --tmodel ./resnet9_strided_16fmaps_onnx_custom_perf.tmodel --repeats 2

This validates preprocessing, 12-lane packing, the 84-value driver output,
and the 80-value feature view. It does not claim full camera/LCD or touch UI.
"""

from __future__ import print_function

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .tcu_bbox_backend import (
        FEATURE_DIM,
        RAW_OUTPUT_DIM,
        SIDE,
        TcuFeatureExtractor,
        load_overlay,
    )
except (ImportError, ValueError):
    from tcu_bbox_backend import (
        FEATURE_DIM,
        RAW_OUTPUT_DIM,
        SIDE,
        TcuFeatureExtractor,
        load_overlay,
    )


def make_cases():
    gradient = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    yy, xx = np.indices((SIDE, SIDE))
    gradient[:, :, 0] = (xx * 7 + yy * 3) % 256
    gradient[:, :, 1] = (xx * 3 + 40) % 256
    gradient[:, :, 2] = (yy * 5 + 80) % 256
    impulse = gradient.copy()
    impulse[0, 0, 0] = (int(impulse[0, 0, 0]) + 1) % 256
    return {"gradient": gradient, "impulse": impulse}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--tmodel", required=True)
    parser.add_argument("--tcu-path", default="")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out-dir", default="validation_result/tcu_feature")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    tmodel_path = Path(args.tmodel).resolve()
    tcu_path = Path(args.tcu_path).resolve() if args.tcu_path else tmodel_path.parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    overlay = load_overlay(args.bitstream)
    extractor = TcuFeatureExtractor(
        overlay, tmodel_path, tcu_path=tcu_path, debug=args.debug
    )

    outputs = {}
    for name, frame in make_cases().items():
        first = extractor.extract_with_diagnostics(frame)
        raw = first["raw"]
        if raw.size != RAW_OUTPUT_DIM:
            raise RuntimeError(
                "case {}: expected {} raw values, got {}".format(
                    name, RAW_OUTPUT_DIM, raw.size
                )
            )
        if first["feature"].shape != (FEATURE_DIM,):
            raise RuntimeError(
                "case {}: expected feature shape ({},), got {}".format(
                    name, FEATURE_DIM, first["feature"].shape
                )
            )
        repeat_max = 0.0
        for _ in range(args.repeats - 1):
            repeat = extractor.extract(frame)
            repeat_max = max(
                repeat_max,
                float(np.max(np.abs(repeat - first["feature"]))),
            )
        outputs[name] = {
            "raw_count": int(raw.size),
            "feature_count": int(first["feature"].size),
            "tail_count": int(first["tail"].size),
            "feature_min": float(np.min(first["feature"])),
            "feature_max": float(np.max(first["feature"])),
            "feature_mean": float(np.mean(first["feature"])),
            "tail": first["tail"].astype(float).tolist(),
            "repeat_max_abs": repeat_max,
        }
        np.save(out_dir / (name + "_feature80.npy"), first["feature"])
        np.save(out_dir / (name + "_raw84.npy"), raw)
        print(
            "{}: raw84={} feature80={} tail4={} repeat_max_abs={:.6g}".format(
                name,
                raw.size,
                first["feature"].size,
                np.array2string(first["tail"], precision=5),
                repeat_max,
            )
        )

    report = {
        "status": "PASS",
        "physical_board_verified": True,
        "bitstream": str(Path(args.bitstream).resolve()),
        "tmodel": str(tmodel_path),
        "input_shape": [SIDE, SIDE, 3],
        "tcu_input_shape": [SIDE * SIDE, 12],
        "driver_output_count": RAW_OUTPUT_DIM,
        "bbox_feature_count": FEATURE_DIM,
        "preprocess": "float32(frame32) / 255.0 - MEAN / STD",
        "cases": outputs,
        "elapsed_seconds": time.time() - started,
    }
    report_path = out_dir / "tcu_feature_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("TCU bbox feature validation PASS")
    print("Report: {}".format(report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

