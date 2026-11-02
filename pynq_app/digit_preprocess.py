#!/usr/bin/env python3
"""Preprocessing shared by the r17 digit support set and digit ROI path.

The existing A/B path in :mod:`pefsl_touch_ui` intentionally remains
unchanged.  Digits and operators use separate instances of this module. It follows the
PC reference's white-background/black-glyph convention while accepting the
uint8 BGR frame required by the r17 TCU extractor.

OpenCV is used when all required operations are available.  A NumPy fallback
is provided for PYNQ images that have NumPy but do not have a complete cv2
installation.  The fallback uses Otsu thresholding and a box blur in place of
OpenCV's adaptive threshold/Gaussian implementation, so it should be checked
against board samples before production use.
"""

from __future__ import print_function

from collections import deque

import numpy as np


# This version records the common support-image representation.  Valid
# images retain the r17 v2 digit algorithm; live calls additionally expose
# validity so a held previous image cannot become a new recognition result.
PREPROCESS_VERSION = "r17-symbol-preprocess-v1"


try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None


_REQUIRED_CV2 = (
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
    "resize",
    "INTER_AREA",
    "threshold",
)


def opencv_available():
    """Return whether the cv2 operations needed by this module are present."""

    return _cv2 is not None and all(hasattr(_cv2, name) for name in _REQUIRED_CV2)


def preprocessing_backend():
    """Return ``opencv`` or ``numpy`` for diagnostics and cache metadata."""

    return "opencv" if opencv_available() else "numpy"


def _as_uint8(image):
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("digit image must be 2D grayscale or 3D BGR")
    if array.ndim == 3 and array.shape[2] < 3:
        raise ValueError("digit color image must have at least 3 channels")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and float(np.max(finite)) <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _to_gray(image):
    array = _as_uint8(image)
    if array.ndim == 2:
        return array
    if opencv_available():
        return _cv2.cvtColor(array[:, :, :3], _cv2.COLOR_BGR2GRAY)

    # BGR coefficients, matching the usual OpenCV conversion closely enough
    # for the fallback path.
    b = array[:, :, 0].astype(np.float32)
    g = array[:, :, 1].astype(np.float32)
    r = array[:, :, 2].astype(np.float32)
    return np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)


def _box_blur(gray, kernel_size):
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return gray
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    padded = np.pad(gray.astype(np.float32), radius, mode="edge")
    result = np.zeros(gray.shape, dtype=np.float32)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            result += padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]]
    result /= float(kernel_size * kernel_size)
    return np.clip(result, 0, 255).astype(np.uint8)


def _otsu_threshold(gray):
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = float(gray.size)
    if total <= 0:
        return 0
    values = np.arange(256, dtype=np.float64)
    cumulative_count = np.cumsum(histogram)
    cumulative_sum = np.cumsum(histogram * values)
    total_sum = cumulative_sum[-1]
    denominator = cumulative_count * (total - cumulative_count)
    numerator = (total_sum * cumulative_count - cumulative_sum) ** 2
    score = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    score[valid] = numerator[valid] / denominator[valid]
    return int(np.argmax(score))


def _binary_threshold(gray, threshold):
    return np.where(gray > int(threshold), 255, 0).astype(np.uint8)


def _morph_open(binary, kernel_size):
    """Small square-kernel opening used only when cv2 is unavailable."""

    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return binary
    radius = kernel_size // 2
    padded = np.pad(binary, radius, mode="edge")
    eroded = np.full(binary.shape, 255, dtype=np.uint8)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            eroded = np.minimum(
                eroded,
                padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]],
            )

    padded = np.pad(eroded, radius, mode="edge")
    dilated = np.zeros(binary.shape, dtype=np.uint8)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            dilated = np.maximum(
                dilated,
                padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]],
            )
    return dilated


def _resize_nearest(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image.copy()
    ys = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    xs = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return image[np.ix_(ys, xs)]


def _force_binary(image):
    return np.where(np.asarray(image) >= 127, 255, 0).astype(np.uint8)


def gray_to_bgr(binary):
    """Convert a 32x32 grayscale binary image to contiguous uint8 BGR."""

    image = _force_binary(binary)
    if image.ndim != 2:
        raise ValueError("binary image must be 2D")
    return np.ascontiguousarray(np.repeat(image[:, :, None], 3, axis=2))


class DigitPreprocessor(object):
    """Stable fixed-ROI preprocessing for one digit or operator.

    After every call, ``last_valid`` refers to that call's image, even when
    the returned compatibility fallback contains an earlier valid image.
    A recognition caller must check it before extracting a TCU feature.
    """

    def __init__(
        self,
        target_size=32,
        pad=2,
        binary_history_len=5,
        blur_ksize=3,
        use_adaptive=True,
        block_size=21,
        C=20,
        open_ksize=2,
        min_black_ratio=0.01,
        max_black_ratio=0.70,
    ):
        self.target_size = int(target_size)
        self.pad = max(0, int(pad))
        self.blur_ksize = int(blur_ksize)
        self.use_adaptive = bool(use_adaptive)
        self.block_size = int(block_size)
        self.C = float(C)
        self.open_ksize = int(open_ksize)
        self.min_black_ratio = float(min_black_ratio)
        self.max_black_ratio = float(max_black_ratio)
        self.binary_history = deque(maxlen=max(1, int(binary_history_len)))
        self.last_good_binary = None
        self.last_valid = False
        self.last_invalid_reason = "not processed"

        if self.target_size <= 2 * self.pad:
            raise ValueError("target_size must be larger than twice pad")

    def reset(self):
        self.binary_history.clear()
        self.last_good_binary = None
        self.last_valid = False
        self.last_invalid_reason = "not processed"

    def process(self, roi_bgr):
        """Return a smoothed 2D uint8 binary image, white background."""

        self.last_valid = False
        self.last_invalid_reason = "processing"
        if roi_bgr is None:
            return self._fallback("missing ROI")
        try:
            array = np.asarray(roi_bgr)
            if array.size == 0:
                return self._fallback("empty ROI")
            gray = _to_gray(array)
        except (TypeError, ValueError) as error:
            return self._fallback("invalid ROI: {}".format(error))

        binary = self._clean_gray(gray)
        black_ratio = float(np.mean(binary == 0))
        if (
            black_ratio < self.min_black_ratio
            or black_ratio > self.max_black_ratio
        ):
            return self._fallback(
                "black ratio {:.4f} outside [{:.4f}, {:.4f}]".format(
                    black_ratio, self.min_black_ratio, self.max_black_ratio
                )
            )

        binary = self._fit_to_canvas(binary)
        if not np.any(binary == 0):
            return self._fallback("no foreground after resize")
        self.binary_history.append(binary.astype(np.float32))
        if len(self.binary_history) >= 2:
            mean_image = np.mean(np.stack(tuple(self.binary_history), axis=0), axis=0)
            binary = _force_binary(mean_image)

        if not np.any(binary == 0):
            return self._fallback("no foreground after smoothing")
        self.last_good_binary = binary.copy()
        self.last_valid = True
        self.last_invalid_reason = ""
        return binary

    def process_to_bgr(self, roi_bgr):
        """Process a live ROI and return the r17 TCU input shape/dtype."""

        return gray_to_bgr(self.process(roi_bgr))

    def process_support_image(self, image):
        """Canonicalize one support PNG without temporal smoothing.

        Formal support images are already 32x32.  Their canvas is preserved;
        non-32x32 images are fitted with the same aspect-ratio rule used for a
        live ROI.  Updated ``debug_binary_*.png`` files are already the final
        white-background/black-glyph binary output of the PC preprocessor, so
        an exact 0/255 32x32 image is preserved instead of being thresholded
        and opened a second time.
        """

        self.last_valid = False
        self.last_invalid_reason = "invalid support image"
        gray = _to_gray(image)
        if (
            gray.shape == (self.target_size, self.target_size)
            and np.all((gray == 0) | (gray == 255))
        ):
            binary = np.ascontiguousarray(gray)
        else:
            binary = self._clean_gray(gray)
            if binary.shape != (self.target_size, self.target_size):
                binary = self._fit_to_canvas(binary)
        # Support PNGs may be sparse.  Validate nonempty foreground/background
        # without imposing the live-camera black-ratio threshold on them.
        self.last_valid = bool(np.any(binary == 0) and np.any(binary == 255))
        self.last_invalid_reason = (
            "" if self.last_valid else "support image lacks foreground/background"
        )
        return binary

    def support_to_bgr(self, image):
        return gray_to_bgr(self.process_support_image(image))

    def _clean_gray(self, gray):
        """Apply the common blur, threshold, and opening stages."""

        if self.blur_ksize >= 3:
            blur_size = self.blur_ksize
            if blur_size % 2 == 0:
                blur_size += 1
            if opencv_available():
                blur = _cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            else:
                blur = _box_blur(gray, blur_size)
        else:
            blur = gray

        if self.use_adaptive and opencv_available():
            block_size = max(3, self.block_size)
            if block_size % 2 == 0:
                block_size += 1
            binary = _cv2.adaptiveThreshold(
                blur,
                255,
                _cv2.ADAPTIVE_THRESH_MEAN_C,
                _cv2.THRESH_BINARY,
                block_size,
                self.C,
            )
        else:
            # Otsu is deterministic and keeps the NumPy fallback dependency
            # small.  It is intentionally reported through preprocessing_backend().
            threshold = _otsu_threshold(blur)
            binary = _binary_threshold(blur, threshold)

        if self.open_ksize >= 2:
            if opencv_available():
                kernel = _cv2.getStructuringElement(
                    _cv2.MORPH_RECT,
                    (self.open_ksize, self.open_ksize),
                )
                binary = _cv2.morphologyEx(binary, _cv2.MORPH_OPEN, kernel)
            else:
                binary = _morph_open(binary, self.open_ksize)
        return _force_binary(binary)

    def _fit_to_canvas(self, binary):
        height, width = binary.shape[:2]
        if height <= 0 or width <= 0:
            return self._blank()

        max_dim = self.target_size - 2 * self.pad
        scale = float(max_dim) / float(max(height, width))
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        if opencv_available():
            resized = _cv2.resize(
                binary,
                (new_width, new_height),
                interpolation=_cv2.INTER_AREA,
            )
        else:
            resized = _resize_nearest(binary, new_width, new_height)
        resized = _force_binary(resized)

        canvas = np.full(
            (self.target_size, self.target_size), 255, dtype=np.uint8
        )
        px = (self.target_size - new_width) // 2
        py = (self.target_size - new_height) // 2
        canvas[py:py + new_height, px:px + new_width] = resized
        return canvas

    def _blank(self):
        return np.full(
            (self.target_size, self.target_size), 255, dtype=np.uint8
        )

    def _fallback(self, reason="invalid ROI"):
        self.last_valid = False
        self.last_invalid_reason = str(reason)
        self.binary_history.clear()
        if self.last_good_binary is not None:
            return self.last_good_binary.copy()
        return self._blank()


# The reference code calls this component StablePreprocessor.  Keep an alias
# so later migration code can use either descriptive name without touching the
# existing A/B implementation.
StablePreprocessor = DigitPreprocessor

