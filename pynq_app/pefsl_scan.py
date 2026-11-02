"""Small, bounded automatic scanner for the PYNQ bbox UI.

The scanner deliberately keeps the expensive operation outside this module:
the UI supplies one 32x32 crop and one TCU feature at a time.  This file only
does deterministic window generation, color gating, cosine matching and NMS.
"""

from __future__ import print_function

import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


FEATURE_DIM = 80
HIST_BINS = (16, 8, 4)
DEFAULT_SCAN_SCORE_MARGIN = 0.04


def _unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def color_histogram(crop):
    """Return a normalized 512-bin HSV histogram for a BGR uint8 crop."""
    image = np.asarray(crop)
    if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
        raise ValueError("crop must have shape (H,W,3)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if cv2 is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, list(HIST_BINS),
            [0, 180, 0, 256, 0, 256],
        ).reshape(-1).astype(np.float32)
    else:
        # Numpy fallback for host tests or minimal PYNQ images.  The HSV
        # conversion follows OpenCV's ranges H=0..180, S/V=0..255.
        rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)
        diff = mx - mn
        hue = np.zeros_like(mx)
        nz = diff > 1e-7
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = nz & (mx == r)
        hue[mask] = ((g[mask] - b[mask]) / diff[mask]) % 6.0
        mask = nz & (mx == g)
        hue[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
        mask = nz & (mx == b)
        hue[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
        hue = hue * 30.0
        sat = np.where(mx > 1e-7, diff / np.maximum(mx, 1e-7), 0.0) * 255.0
        val = mx * 255.0
        hist, _ = np.histogramdd(
            np.stack((hue.ravel(), sat.ravel(), val.ravel()), axis=1),
            bins=HIST_BINS,
            range=((0, 180), (0, 256), (0, 256)),
        )
        hist = hist.reshape(-1).astype(np.float32)
    total = float(hist.sum())
    return hist / max(total, 1e-8)


def _candidate_boxes(width, height, sizes, region, scales):
    x0, y0, x1, y1 = [int(v) for v in region]
    x0 = max(0, min(int(width), x0))
    y0 = max(0, min(int(height), y0))
    x1 = max(x0 + 1, min(int(width), x1))
    y1 = max(y0 + 1, min(int(height), y1))
    available_w = x1 - x0
    available_h = y1 - y0
    result = []
    seen = set()
    clean_sizes = []
    for size in sizes or ():
        try:
            sw, sh = float(size[0]), float(size[1])
        except (IndexError, TypeError, ValueError):
            continue
        if sw > 1 and sh > 1:
            clean_sizes.append((sw, sh))
    if not clean_sizes:
        clean_sizes = [(min(128, available_w), min(128, available_h))]
    # The desktop scanner uses several scales around each registered box.
    # Keep the count bounded later; preserving the registered aspect ratio is
    # more useful on the 800x480 PYNQ display than square-only windows.
    for sw, sh in clean_sizes:
        for scale in scales:
            ww = max(24, min(available_w, int(round(sw * scale))))
            hh = max(24, min(available_h, int(round(sh * scale))))
            if ww >= available_w:
                xs = [x0]
            else:
                step_x = max(12, int(round(ww * 0.60)))
                xs = list(range(x0, x1 - ww + 1, step_x))
                if xs[-1] != x1 - ww:
                    xs.append(x1 - ww)
            if hh >= available_h:
                ys = [y0]
            else:
                step_y = max(12, int(round(hh * 0.60)))
                ys = list(range(y0, y1 - hh + 1, step_y))
                if ys[-1] != y1 - hh:
                    ys.append(y1 - hh)
            for yy in ys:
                for xx in xs:
                    box = (int(xx), int(yy), int(xx + ww), int(yy + hh))
                    if box not in seen:
                        seen.add(box)
                        result.append(box)
    return result


def build_scan_boxes(width, height, region=(0, 58, 800, 422), sizes=None,
                     scales=(0.45, 0.65, 0.85, 1.0, 1.25, 1.6, 2.0),
                     max_windows=24, phase=0):
    """Build a bounded multi-scale window bank with a rotating phase.

    A full bank can be much larger than the ARM/TCU can scan in one cycle.
    ``phase`` selects a different evenly distributed subset on the next cycle,
    so the UI remains responsive while the whole image is eventually covered.
    """
    cap = max(1, int(max_windows))
    boxes = _candidate_boxes(width, height, sizes, region, scales)
    if len(boxes) <= cap:
        return boxes
    n = len(boxes)
    phase = int(phase) % n
    # Start at a rotating point, then step through the bank.  The last window
    # is always included to retain edge coverage when cap is small.
    stride = max(1, int(math.ceil(float(n) / cap)))
    selected = []
    used = set()
    for i in range(cap):
        index = (phase + i * stride) % n
        if index not in used:
            used.add(index)
            selected.append(boxes[index])
    for index in (0, n // 2, n - 1):
        if len(selected) >= cap:
            break
        if index not in used:
            used.add(index)
            selected.append(boxes[index])
    return selected[:cap]


def _hist_intersection(left, right):
    if left is None or right is None:
        return 1.0
    lhs = np.asarray(left, dtype=np.float32).reshape(-1)
    rhs = np.asarray(right, dtype=np.float32).reshape(-1)
    if lhs.size != rhs.size or lhs.size == 0:
        return 0.0
    lhs = lhs / max(float(lhs.sum()), 1e-8)
    rhs = rhs / max(float(rhs.sum()), 1e-8)
    return float(np.minimum(lhs, rhs).sum())


def histogram_distance(left, right):
    """Return the Bhattacharyya distance between two normalized histograms.

    C104 uses this distance for its centre-versus-surround gate.  Keeping the
    calculation here (rather than in the UI) also makes the gate deterministic
    and easy to exercise on a host without a PYNQ overlay.
    """
    if left is None or right is None:
        return None
    lhs = np.asarray(left, dtype=np.float32).reshape(-1)
    rhs = np.asarray(right, dtype=np.float32).reshape(-1)
    if lhs.size == 0 or lhs.size != rhs.size:
        return None
    lhs = lhs / max(float(lhs.sum()), 1e-8)
    rhs = rhs / max(float(rhs.sum()), 1e-8)
    coefficient = float(np.sqrt(np.maximum(lhs, 0.0) * np.maximum(rhs, 0.0)).sum())
    return float(math.sqrt(max(0.0, 1.0 - min(1.0, coefficient))))


def surround_histogram(frame, box, pad_fraction=0.50):
    """Build a histogram from pixels around (outside) ``box``.

    A full expanded crop would still contain much of the object.  The four
    strips below are the same cheap approximation as C104's ring gate and are
    robust to boxes close to an image edge.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] < 3 or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(1, min(width, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(1, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x = max(2, int(round((x2 - x1) * float(pad_fraction))))
    pad_y = max(2, int(round((y2 - y1) * float(pad_fraction))))
    ex1, ex2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
    ey1, ey2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
    strips = []
    if ey1 < y1:
        strips.append(image[ey1:y1, ex1:ex2, :3])
    if y2 < ey2:
        strips.append(image[y2:ey2, ex1:ex2, :3])
    if ex1 < x1:
        strips.append(image[y1:y2, ex1:x1, :3])
    if x2 < ex2:
        strips.append(image[y1:y2, x2:ex2, :3])
    pixels = [part.reshape(-1, 3) for part in strips if part.size]
    if not pixels:
        return None
    pixels = np.concatenate(pixels, axis=0)
    if pixels.shape[0] < 32:
        return None
    return color_histogram(pixels.reshape((-1, 1, 3)))


def center_surround_distance(frame, box, center_hist=None, pad_fraction=0.50):
    """Return centre/ring histogram distance, or ``None`` at an edge."""
    if center_hist is None:
        x1, y1, x2, y2 = [int(v) for v in box]
        image = np.asarray(frame)
        if image.ndim != 3:
            return None
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        center_hist = color_histogram(image[y1:y2, x1:x2, :3])
    return histogram_distance(
        center_hist,
        surround_histogram(frame, box, pad_fraction=pad_fraction),
    )


def score_window(feature, labels, shots, registered_hists=None,
                 crop_hist=None, cos_threshold=0.52, hist_threshold=0.0,
                 score_margin=DEFAULT_SCAN_SCORE_MARGIN, top_k=5):
    """Match one TCU feature against raw per-class samples.

    The desktop demo combines the strongest registered sample with a small
    top-k consensus.  The return value is ``(label, score)``; label is
    ``None`` when every class is below the cosine, optional color, or
    ambiguity threshold.
    """
    if not labels or not shots:
        return None, 0.0
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.size != FEATURE_DIM:
        raise ValueError("feature must have 80 elements")
    all_rows = [np.asarray(rows, dtype=np.float32) for rows in shots]
    all_rows = [rows.reshape((-1, FEATURE_DIM)) for rows in all_rows]
    baseline = np.concatenate(all_rows, axis=0).mean(axis=0)
    query = _unit_rows(feature[None, :] - baseline[None, :])[0]
    ranked = []
    for index, (label, rows) in enumerate(zip(labels, all_rows)):
        if rows.size == 0:
            continue
        hist = None
        if registered_hists is not None and index < len(registered_hists):
            hist = registered_hists[index]
        # Histogram metadata is useful as a tie breaker, but a hard colour
        # gate is unsafe for camera auto-exposure and for windows containing
        # background around the object.  Keep it opt-in (threshold > 0).
        if crop_hist is not None and hist is not None and float(hist_threshold) > 0:
            if _hist_intersection(crop_hist, hist) < float(hist_threshold):
                continue
        scores = np.sort(np.dot(_unit_rows(rows - baseline[None, :]), query))[::-1]
        if scores.size == 0:
            continue
        # A single accidental nearest shot should not identify a background
        # patch.  C104 gives the best shot 70% weight and the top-k consensus
        # 30% weight.
        count = max(1, min(int(top_k), int(scores.size)))
        score = float(0.7 * scores[0] + 0.3 * np.mean(scores[:count]))
        ranked.append((score, str(label)))
    if not ranked:
        return None, 0.0
    ranked.sort(reverse=True)
    best_score, best_label = ranked[0]
    if best_score < float(cos_threshold):
        return None, max(0.0, float(best_score))
    if len(ranked) > 1 and float(best_score - ranked[1][0]) < float(score_margin):
        return None, max(0.0, float(best_score))
    return best_label, float(best_score)


def _overlap(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = aa + bb - inter
    return inter / float(union), inter / float(min(aa, bb))


def suppress_detections(detections, iou_threshold=0.35,
                        contain_threshold=0.75, max_detections=4):
    """Greedy NMS with nested-window suppression.

    Windows for the same class are treated with the usual IoU rule.  Windows
    from different classes are removed only when one contains most of the
    other, which prevents duplicate labels on one object while preserving
    genuinely adjacent objects.
    """
    ordered = sorted(
        [dict(d) for d in detections],
        key=lambda d: float(d.get("score", 0.0)), reverse=True,
    )
    kept = []
    for candidate in ordered:
        box = tuple(int(v) for v in candidate["box"])
        candidate["box"] = box
        duplicate = False
        for old in kept:
            iou, containment = _overlap(box, old["box"])
            same_class = candidate.get("label") == old.get("label")
            if (same_class and iou >= float(iou_threshold)) or \
                    (not same_class and containment >= float(contain_threshold)):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        if len(kept) >= int(max_detections):
            break
    return kept

