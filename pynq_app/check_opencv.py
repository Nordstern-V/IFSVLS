#!/usr/bin/env python3
"""Check the OpenCV functionality required by the r17 digit pipeline.

Run this file on the PYNQ board with the same Python interpreter used by
``pynq_bbox_demo.py``::

    python3 check_opencv.py

The check deliberately does not require a connected camera or a GUI.  r17
uses OpenCV for image processing; camera/display ownership remains with the
PYNQ video pipeline.
"""

from __future__ import print_function

import sys


REQUIRED_ATTRIBUTES = (
    "VideoCapture",
    "cvtColor",
    "COLOR_BGR2GRAY",
    "GaussianBlur",
    "adaptiveThreshold",
    "ADAPTIVE_THRESH_MEAN_C",
    "THRESH_BINARY",
    "morphologyEx",
    "MORPH_OPEN",
    "getStructuringElement",
    "MORPH_RECT",
    "threshold",
    "resize",
)


def main():
    try:
        import numpy as np
    except Exception as exc:
        print("FAIL: numpy import failed: {}".format(exc))
        return 2

    try:
        import cv2
    except Exception as exc:
        print("FAIL: cv2 import failed: {}".format(exc))
        print("Install a board-compatible OpenCV package, then rerun this check.")
        return 2

    print("cv2 version: {}".format(getattr(cv2, "__version__", "unknown")))
    print("cv2 module: {}".format(getattr(cv2, "__file__", "unknown")))

    missing = [name for name in REQUIRED_ATTRIBUTES if not hasattr(cv2, name)]
    if missing:
        print("FAIL: missing OpenCV attributes: {}".format(", ".join(missing)))
        return 3

    # Exercise the exact core operations needed by StablePreprocessor.  The
    # input is synthetic, so this test does not need a camera or a display.
    try:
        frame = np.full((48, 64, 3), 255, dtype=np.uint8)
        frame[12:36, 22:42, :] = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        _, thresholded = cv2.threshold(opened, 127, 255, cv2.THRESH_BINARY)
        resized = cv2.resize(thresholded, (32, 32), interpolation=cv2.INTER_AREA)
    except Exception as exc:
        print("FAIL: OpenCV preprocessing test failed: {}".format(exc))
        return 4

    if resized.shape != (32, 32) or resized.dtype != np.uint8:
        print(
            "FAIL: unexpected output shape/dtype: {} / {}".format(
                resized.shape, resized.dtype
            )
        )
        return 5

    print("PASS: import, required APIs, and 32x32 preprocessing pipeline")
    print("Camera probe: not run (a camera is not required to validate OpenCV itself)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
