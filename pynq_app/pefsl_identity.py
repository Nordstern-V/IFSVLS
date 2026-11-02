"""Bounded learned-ROI spatial checks for PYNQ B mode.

This is a deliberately small adaptation of C124's registered texture check.
CNN similarity, colour, and a recent tracking box do not by themselves prove
identity: current pixels must retain texture distributed across a learned ROI.
Only six references, quarter turns, and shifts of at most two reference pixels
are examined. There are no feature extractors, homography searches, or hardware
calls here, and the NumPy path works without OpenCV.
"""

import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


REFERENCE_SIDE = 48
MAX_REFERENCES = 6


def _resize(image, side=REFERENCE_SIDE):
    if cv2 is not None:
        return cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    # Bilinear fallback, bounded by the output dimensions rather than input
    # pixel count. It avoids aliasing changes caused by nearest-neighbour
    # resizing in minimal host-test environments.
    h, w = image.shape[:2]
    yy = np.linspace(0, h - 1, side, dtype=np.float32)
    xx = np.linspace(0, w - 1, side, dtype=np.float32)
    y0 = yy.astype(np.int64)
    x0 = xx.astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0)[:, None, None]
    wx = (xx - x0)[None, :, None]
    top = image[y0[:, None], x0[None, :]].astype(np.float32) * (1. - wx)
    top += image[y0[:, None], x1[None, :]].astype(np.float32) * wx
    bottom = image[y1[:, None], x0[None, :]].astype(np.float32) * (1. - wx)
    bottom += image[y1[:, None], x1[None, :]].astype(np.float32) * wx
    return np.clip(top * (1. - wy) + bottom * wy, 0, 255).astype(np.uint8)


def prepare_reference(image):
    """Return one complete ROI as a contiguous 48x48 BGR uint8 array."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("reference must be a nonempty HxWx3 BGR image")
    if not np.isfinite(image).all():
        raise ValueError("reference must contain finite pixels")
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (REFERENCE_SIDE, REFERENCE_SIDE):
        image = _resize(image)
    return np.ascontiguousarray(image).copy()


def _gray(image):
    image = prepare_reference(image).astype(np.float32)
    gray = .114 * image[:, :, 0] + .587 * image[:, :, 1] + .299 * image[:, :, 2]
    # The same small blur is applied to learned and current pixels so tiny
    # camera/compression noise is not mistaken for distributed texture.
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    horizontal = .25 * padded[:, :-2] + .5 * padded[:, 1:-1] + .25 * padded[:, 2:]
    return .25 * horizontal[:-2] + .5 * horizontal[1:-1] + .25 * horizontal[2:]


def shape_descriptor(image):
    """Return weak shape evidence: edge density and normalized gray contrast.

    Like C124's shape gate this only rejects a query with far less structure
    than the registration; it is never sufficient identity evidence.
    """
    gray = _gray(image)
    spread = float(np.percentile(gray, 90.) - np.percentile(gray, 10.))
    gain = max(1., min(3., 96. / max(1., spread)))
    normalized = np.clip((gray - float(gray.mean())) * gain + 128., 0, 255)
    dy, dx = np.gradient(normalized)
    density = float(np.mean(np.hypot(dx, dy) >= 12.))
    return np.asarray((density, float(normalized.std())), dtype=np.float32)


def _ncc(left, right):
    a = left - float(left.mean())
    b = right - float(right.mean())
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denominator <= 1e-6:
        return -1.
    return float(np.sum(a * b) / denominator)


def _distributed_support(query, reference):
    h, w = reference.shape
    textured = 0
    supported = []
    for row in range(3):
        for col in range(3):
            region = np.s_[row * h // 3:(row + 1) * h // 3,
                           col * w // 3:(col + 1) * w // 3]
            ref = reference[region]
            if float(ref.std()) < 5.:
                continue
            textured += 1
            if _ncc(query[region], ref) >= .60:
                supported.append((row, col))
    enough = (len(supported) >= max(4, int(math.ceil(.4 * textured)))
              and len(set(row for row, _ in supported)) >= 2
              and len(set(col for _, col in supported)) >= 2)
    return bool(enough), textured, len(supported)


def compile_references(references):
    """Cache learned pixels only; current query scores are never cached."""
    return tuple(_gray(view) for view in references[:MAX_REFERENCES])


def identity_evidence(image, references, prepared_references=None):
    """Verify complete learned appearance against current candidate pixels.

    ``supported=None`` means a legacy archive has no reference images; callers
    must explicitly decide their compatibility policy. A flat registered ROI
    returns ``supported=False`` with reason ``textureless reference``. A tiny
    shared logo cannot pass because at least four textured cells spanning two
    rows and two columns must agree, in addition to whole-ROI correlation.

    This limited verifier does not promise arbitrary perspective or deformable
    pose invariance; additional A-mode views should cover those appearances.
    """
    if references is None or len(references) == 0:
        return dict(supported=None, similarity=0., textured=False,
                    reason="no registration references")
    query = _gray(image)
    best = -1.
    any_textured = False
    best_cells = 0
    offsets = ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2),
               (-2, -2), (-2, 2), (2, -2), (2, 2))
    prepared = compile_references(references) if prepared_references is None else prepared_references
    for reference_index, reference in enumerate(prepared):
        if float(reference.std()) < 6.:
            continue
        any_textured = True
        for quarter_turn in range(4):
            rotated = np.rot90(reference, quarter_turn)
            for dx, dy in offsets:
                # Use at least 46/48 of each side: matching a small internal
                # texture cannot stand in for the whole learned object.
                qx, qy = max(0, dx), max(0, dy)
                rx, ry = max(0, -dx), max(0, -dy)
                w, h = REFERENCE_SIDE - abs(dx), REFERENCE_SIDE - abs(dy)
                current = query[qy:qy + h, qx:qx + w]
                target = rotated[ry:ry + h, rx:rx + w]
                score = _ncc(current, target)
                best = max(best, score)
                if score < .68:
                    continue
                distributed, _, cells = _distributed_support(current, target)
                best_cells = max(best_cells, cells)
                if distributed:
                    return dict(supported=True, similarity=score, textured=True,
                                reason="distributed registered texture",
                                cells=cells, reference=reference_index,
                                quarter_turn=quarter_turn, shift=(dx, dy))
    return dict(supported=False, similarity=max(0., best), textured=any_textured,
                cells=best_cells,
                reason=("registered texture mismatch" if any_textured else
                        "textureless reference"))
