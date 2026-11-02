#!/usr/bin/env python3
"""Check operator PNGs and export the exact support preprocessing output.

This diagnostic uses OpenCV/NumPy only.  It does not load an FPGA overlay,
open the camera, or access the TCU.  Camera ROI recognition must still be
checked on the board after the independent operator cache is built.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from digit_preprocess import (
    DigitPreprocessor, PREPROCESS_VERSION, opencv_available,
    preprocessing_backend,
)


APP_DIR = Path(__file__).resolve().parent
CLASSES = ("plus", "sub", "mul", "div")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("cannot write image: {}".format(path))


def _describe_binary(binary):
    ys, xs = np.where(binary == 0)
    if xs.size == 0:
        raise ValueError("preprocessed image is all white")
    if xs.size == binary.size:
        raise ValueError("preprocessed image is all black")
    result = {
        "black_pixels": int(xs.size),
        "black_ratio": float(xs.size / float(binary.size)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()),
                      int(xs.max()) + 1, int(ys.max()) + 1],
    }
    if hasattr(cv2, "connectedComponentsWithStats"):
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(binary == 0, dtype=np.uint8), connectivity=8
        )
        result["components"] = int(count - 1)
        result["component_areas"] = sorted(
            [int(area) for area in stats[1:, cv2.CC_STAT_AREA]], reverse=True
        )
    return result


def check_support(support_dir, output_dir, manifest_path):
    support_dir = Path(support_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not opencv_available():
        raise RuntimeError("required OpenCV preprocessing APIs are unavailable")
    if not support_dir.is_dir():
        raise FileNotFoundError("missing support directory: {}".format(support_dir))
    if (output_dir == support_dir or output_dir in support_dir.parents
            or support_dir in output_dir.parents):
        raise ValueError("output directory must be separate from the support directory tree")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if tuple(manifest.get("classes", ())) != CLASSES:
        raise ValueError("manifest must contain plus, sub, mul, div in that order")
    if manifest.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError("manifest preprocessing version does not match the code")
    expected = {item["path"]: item for item in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("manifest has duplicate file paths")
    actual_paths = []
    for label in CLASSES:
        class_dir = support_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError("missing operator class: {}".format(class_dir))
        files = sorted(class_dir.glob("*.png"))
        if not files:
            raise ValueError("operator class is empty: {}".format(label))
        pattern = re.compile(r"^(?:{}_[0-9]+|debug_binary_[0-9]+)\.png$".format(label))
        for path in files:
            if not pattern.match(path.name):
                raise ValueError("unexpected operator PNG name: {}".format(path.name))
            actual_paths.append((label, path))
    actual_names = {path.relative_to(support_dir).as_posix()
                    for _, path in actual_paths}
    if actual_names != set(expected):
        raise ValueError("manifest file mismatch; missing={} extra={}".format(
            sorted(set(expected) - actual_names), sorted(actual_names - set(expected))
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = DigitPreprocessor(binary_history_len=1)
    rows = []
    counts = {label: 0 for label in CLASSES}
    previews = {label: [] for label in CLASSES}
    for label, path in actual_paths:
        relative = path.relative_to(support_dir).as_posix()
        digest = _sha256(path)
        if digest != expected[relative]["sha256"]:
            raise ValueError("SHA-256 mismatch: {}".format(relative))
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None or source.size == 0:
            raise ValueError("cannot decode PNG: {}".format(path))
        if source.shape != (32, 32, 3):
            raise ValueError("expected 32x32 PNG, got {}: {}".format(source.shape, path))
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        preprocessor.reset()
        binary = preprocessor.process_support_image(source)
        if binary.shape != (32, 32) or binary.dtype != np.uint8:
            raise ValueError("invalid preprocessing shape/dtype: {}".format(relative))
        if not np.all((binary == 0) | (binary == 255)):
            raise ValueError("preprocessing output is not binary: {}".format(relative))
        if not preprocessor.last_valid:
            raise ValueError("{}: {}".format(relative, preprocessor.last_invalid_reason))
        exact_binary = bool(np.all((gray == 0) | (gray == 255)))
        if exact_binary and not np.array_equal(gray, binary):
            raise ValueError("binary support PNG was changed: {}".format(relative))
        stats = _describe_binary(binary)
        out_class_dir = output_dir / label
        out_class_dir.mkdir(parents=True, exist_ok=True)
        _write_image(out_class_dir / path.name, binary)
        row = {
            "path": relative,
            "sha256": digest,
            "input_shape": list(source.shape),
            "input_exact_binary": exact_binary,
            "input_black_pixels_lt127": int(np.sum(gray < 127)),
            "binary_unchanged": bool(np.array_equal(gray, binary)),
        }
        row.update(stats)
        rows.append(row)
        previews[label].append((path.name, gray, binary, stats))
        counts[label] += 1
        print("{}: black={} bbox={} components={}".format(
            relative, stats["black_pixels"], stats["bbox_xyxy"],
            stats.get("components", "n/a")
        ), flush=True)

    if counts != manifest["sample_counts"]:
        raise ValueError("sample counts differ from manifest")
    if len(rows) != manifest["total_images"]:
        raise ValueError("total image count differs from manifest")

    # Every cell has the source on the left and the model's binary support
    # input on the right. Nearest-neighbour enlargement exposes thin pixels.
    columns = max(counts.values())
    cell_width, cell_height, title_height = 300, 208, 36
    sheet = np.full((title_height + len(CLASSES) * cell_height,
                     columns * cell_width, 3), 255, dtype=np.uint8)
    cv2.putText(sheet, "LEFT: source gray   RIGHT: 32x32 support preprocessing",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    for row_index, label in enumerate(CLASSES):
        for column, (name, gray, binary, stats) in enumerate(previews[label]):
            x, y = column * cell_width, title_height + row_index * cell_height
            cv2.putText(sheet, "{}/{}".format(label, name), (x + 8, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 1, cv2.LINE_AA)
            for image, dx in ((gray, 10), (binary, 158)):
                enlarged = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
                sheet[y + 30:y + 158, x + dx:x + dx + 128] = np.repeat(
                    enlarged[:, :, None], 3, axis=2
                )
            cv2.putText(sheet, "black={} components={}".format(
                stats["black_pixels"], stats.get("components", "n/a")),
                (x + 8, y + 184), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    _write_image(output_dir / "operator_preprocess_contact_sheet.png", sheet)
    report = {
        "preprocess_version": PREPROCESS_VERSION,
        "preprocessing_backend": preprocessing_backend(),
        "opencv_version": cv2.__version__,
        "support_dir": str(support_dir),
        "classes": list(CLASSES),
        "sample_counts": counts,
        "total_images": len(rows),
        "files": rows,
    }
    (output_dir / "operator_preprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS: {} classes / {} images; manifest hashes and preprocessing valid".format(
        len(CLASSES), len(rows)
    ))
    print("Review minus stroke and division dots in: {}".format(
        output_dir / "operator_preprocess_contact_sheet.png"
    ))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", default=str(APP_DIR / "operator_support"))
    parser.add_argument("--output-dir", default=str(APP_DIR / "operator_support_check"))
    parser.add_argument("--manifest", default=str(APP_DIR / "operator_support_manifest.json"))
    args = parser.parse_args(argv)
    try:
        check_support(args.support_dir, args.output_dir, args.manifest)
    except Exception as error:
        print("FAIL: {}".format(error), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
