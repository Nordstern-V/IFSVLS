#!/usr/bin/env python3
"""Touch learning, manual ROI recognition, digit mode, and bounded scanning."""

from __future__ import print_function

import threading
import time
from collections import deque
from pathlib import Path

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None
import numpy as np
from digit_calculator import DigitCalculator, MAX_NUMBERS, MAX_OPERATORS

from pefsl_scan import (
    build_scan_boxes, center_surround_distance, color_histogram,
    score_window, suppress_detections,
)
from pefsl_bmode import BModeMatcher
from pefsl_identity import prepare_reference, shape_descriptor

try:
    from digit_preprocess import DigitPreprocessor
    from digit_recognizer import DigitSupportCache
except Exception:
    # Keep the existing A/B application usable on an older deployment that
    # has not received the optional digit files/cache yet.
    DigitPreprocessor = None
    DigitSupportCache = None

APP_VERSION = "mix-plus-ab-r17-20261101-r21-keyboard-calc10"
OPERATOR_SYMBOLS = {"plus": "+", "sub": "-", "mul": "*", "div": "/"}


class _NumpyCv2Fallback:
    INTER_AREA = 0
    INTER_NEAREST = 0
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    FILLED = -1

    @staticmethod
    def resize(image, size, interpolation=0):
        height, width = int(size[1]), int(size[0])
        source = np.asarray(image)
        ys = np.linspace(0, source.shape[0] - 1, height).astype(np.int64)
        xs = np.linspace(0, source.shape[1] - 1, width).astype(np.int64)
        return source[np.ix_(ys, xs)]

    @staticmethod
    def rectangle(image, p1, p2, color, thickness=1):
        x1, y1 = [int(value) for value in p1]
        x2, y2 = [int(value) for value in p2]
        x1 = max(0, min(image.shape[1] - 1, x1))
        x2 = max(0, min(image.shape[1] - 1, x2))
        y1 = max(0, min(image.shape[0] - 1, y1))
        y2 = max(0, min(image.shape[0] - 1, y2))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if thickness == _NumpyCv2Fallback.FILLED:
            image[top:bottom + 1, left:right + 1] = color
            return image
        t = max(1, int(thickness))
        image[top:min(top + t, image.shape[0]), left:right + 1] = color
        image[max(top, bottom - t + 1):bottom + 1, left:right + 1] = color
        image[top:bottom + 1, left:min(left + t, image.shape[1])] = color
        image[top:bottom + 1, max(left, right - t + 1):right + 1] = color
        return image

    @staticmethod
    def putText(image, text, origin, font, scale, color, thickness=1, line_type=0):
        return image


cv2 = _cv2 if _cv2 is not None else _NumpyCv2Fallback()

SCREEN_W = 800
SCREEN_H = 480
TOOLBAR_H = 58
FOOTER_H = 58
D_PANEL_H = 116
FEATURE_DIM = 80
MIN_BOX = 20
MAX_NAME_LENGTH = 24
NAME_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 _")


def _valid_display_name(value):
    """Validate optional names without accepting invisible whitespace labels."""
    return (isinstance(value, str) and len(value) <= MAX_NAME_LENGTH
            and all(char in NAME_CHARACTERS for char in value)
            and (not value or bool(value.strip())))


def _unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


class LearnedData:
    """Small npz-backed feature store compatible with the PEFSL data model."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.labels = []
        self.display_names = []
        self.shots = []
        self.scan_sizes = []
        self.color_hists = []
        self.reference_views = []
        self.class_shapes = []
        self.raw_cosine = []
        self.legacy_mean = None
        self._mean = None
        self._bmode_matcher = None

    def invalidate(self):
        self._mean = None
        self._bmode_matcher = None

    def bmode_matcher(self):
        if self._bmode_matcher is None:
            self._bmode_matcher = BModeMatcher(self)
        return self._bmode_matcher

    def has_data(self):
        return bool(self.labels)

    def display_name(self, label):
        """Resolve a stable class ID only at the presentation boundary."""
        try:
            index = self.labels.index(label)
        except ValueError:
            return str(label)
        if index < len(self.display_names) and self.display_names[index]:
            return self.display_names[index]
        return str(label)

    def add_shots(self, label, features, box_size=None, hist=None,
                  reference_views=None, display_name=None):
        if display_name is not None and not _valid_display_name(display_name):
            raise ValueError("display name must use at most 24 letters, digits, spaces or underscores")
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
            raise ValueError("features must have shape (N, 80)")
        label = str(label)
        existing = label in self.labels
        index = self.labels.index(label) if existing else len(self.labels)
        views = [prepare_reference(view) for view in
                 (() if reference_views is None else reference_views)]
        legacy_mean = self.legacy_mean
        if (views and self.shots and legacy_mean is None and not all(self.raw_cosine)):
            # C124 freezes the legacy centre before upgrading/adding raw-space
            # classes. New positive samples must not recenter old identities.
            legacy_mean = self.mean_feature().copy()
        shots = np.concatenate((self.shots[index], feats), axis=0) if existing else feats.copy()
        size = self.scan_sizes[index] if existing else None
        if box_size is not None:
            size = tuple(float(v) for v in box_size)
        merged_hist = self.color_hists[index] if existing else None
        if hist is not None:
            value = np.asarray(hist, dtype=np.float32).copy()
            if merged_hist is not None:
                value = 0.5 * (merged_hist + value)
            merged_hist = value / max(float(value.sum()), 1e-8)
        bank = list(self.reference_views[index]) if existing else []
        raw_cosine = self.raw_cosine[index] if existing else False
        for view in views:
            if not bank or min(float(np.abs(view.astype(np.float32)-old).mean())
                               for old in bank) >= 4.0:
                bank.append(view.copy())
                del bank[:-4]
            raw_cosine = True
        shape = (np.mean([shape_descriptor(view) for view in bank], axis=0).astype(np.float32)
                 if bank else None)
        name = (self.display_names[index] if existing else "")
        if display_name:
            name = display_name
        # Stage all values before updating the bank: a failed reference/sample
        # conversion must not rename a class or partially append its shots.
        self.legacy_mean = legacy_mean
        if existing:
            self.shots[index] = shots
            self.scan_sizes[index] = size
            self.color_hists[index] = merged_hist
            self.reference_views[index] = bank
            self.class_shapes[index] = shape
            self.raw_cosine[index] = raw_cosine
            self.display_names[index] = name
        else:
            self.labels.append(label)
            self.shots.append(shots)
            self.scan_sizes.append(size)
            self.color_hists.append(merged_hist)
            self.reference_views.append(bank)
            self.class_shapes.append(shape)
            self.raw_cosine.append(raw_cosine)
            self.display_names.append(name)
        self.invalidate()

    def mean_feature(self):
        if not self.has_data():
            return None
        if self._mean is None:
            self._mean = np.concatenate(self.shots, axis=0).mean(axis=0)
        return self._mean

    def class_means(self):
        if not self.has_data():
            return None
        return np.stack([row.mean(axis=0) for row in self.shots], axis=0)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.has_data():
            return False, "no learned data"
        counts = np.asarray([len(row) for row in self.shots], dtype=np.int64)
        features = np.concatenate(self.shots, axis=0).astype(np.float32)
        np.savez(
            str(path),
            labels=np.asarray(self.labels),
            display_names=np.asarray(self.display_names, dtype='<U24'),
            counts=counts,
            features=features,
            bmode_schema=np.asarray(1, dtype=np.int64),
            b_legacy_mean=(self.legacy_mean if self.legacy_mean is not None else np.empty(0, np.float32)),
            b_raw_cosine=np.asarray(self.raw_cosine, dtype=np.uint8),
            b_reference_counts=np.asarray([len(v) for v in self.reference_views], dtype=np.int64),
            b_reference_images=np.asarray(
                [view for bank in self.reference_views for view in bank],
                dtype=np.uint8).reshape((-1, 48, 48, 3)),
            b_class_shapes=np.asarray([
                shape if shape is not None else (0, 0) for shape in self.class_shapes
            ], dtype=np.float32),
            scan_sizes=np.asarray([
                size if size is not None else (0, 0)
                for size in self.scan_sizes
            ], dtype=np.float32),
            color_hists=np.asarray([
                hist if hist is not None else np.zeros(512, dtype=np.float32)
                for hist in self.color_hists
            ], dtype=np.float32),
        )
        return True, "{} classes / {} shots".format(len(self.labels), len(features))

    def load(self, path):
        path = Path(path)
        if not path.is_file():
            return False, "file not found"
        with np.load(str(path), allow_pickle=False) as data:
            labels = [str(value) for value in data["labels"]]
            try:
                names = np.asarray(data["display_names"]) if "display_names" in data else None
            except (ValueError, TypeError):
                return False, "invalid class display names"
            counts = np.asarray(data["counts"], dtype=np.int64)
            features = np.asarray(data["features"], dtype=np.float32)
            sizes = np.asarray(data["scan_sizes"]) if "scan_sizes" in data else None
            hists = np.asarray(data["color_hists"]) if "color_hists" in data else None
            b_keys = ('bmode_schema', 'b_raw_cosine', 'b_reference_counts',
                      'b_reference_images', 'b_class_shapes')
            if any(key in data for key in b_keys) and not all(key in data for key in b_keys):
                return False, "incomplete B identity metadata"
            b_meta = {key: np.asarray(data[key]) for key in b_keys} if all(key in data for key in b_keys) else None
            legacy_mean = np.asarray(data['b_legacy_mean'], np.float32) if 'b_legacy_mean' in data else np.empty(0, np.float32)
        if len(labels) != len(counts) or features.ndim != 2:
            return False, "invalid npz"
        if names is not None:
            if (names.shape != (len(labels),) or names.dtype.kind != 'U'
                    or not all(_valid_display_name(str(name)) for name in names)):
                return False, "invalid class display names"
        if int(counts.sum()) != int(features.shape[0]) or features.shape[1] != FEATURE_DIM:
            return False, "invalid feature shape"
        if (counts <= 0).any() or not np.isfinite(features).all():
            return False, "invalid feature data"
        if sizes is not None and sizes.shape != (len(labels), 2):
            return False, "invalid scan sizes"
        if hists is not None and hists.shape != (len(labels), 512):
            return False, "invalid color histograms"
        if legacy_mean.shape not in ((0,), (FEATURE_DIM,)) or not np.isfinite(legacy_mean).all():
            return False, "invalid legacy feature centre"
        if b_meta is not None:
            raw = b_meta['b_raw_cosine']
            ref_counts = b_meta['b_reference_counts']
            refs = b_meta['b_reference_images']
            shapes = b_meta['b_class_shapes']
            if (b_meta['bmode_schema'].shape != () or b_meta['bmode_schema'].dtype.kind not in 'iu'
                    or int(b_meta['bmode_schema']) != 1
                    or raw.shape != (len(labels),) or not np.isin(raw, (0, 1)).all()
                    or ref_counts.shape != (len(labels),) or ref_counts.dtype.kind not in 'iu'
                    or (ref_counts < 0).any() or (ref_counts > 4).any()
                    or refs.dtype != np.uint8 or refs.shape != (int(ref_counts.sum()), 48, 48, 3)
                    or shapes.shape != (len(labels), 2) or not np.isfinite(shapes).all()
                    or ((raw != 0) & (ref_counts == 0)).any()):
                return False, "invalid B identity metadata"
        self.reset()
        self.legacy_mean = legacy_mean.copy() if legacy_mean.size else None
        offset = 0
        ref_offset = 0
        for label, count in zip(labels, counts):
            count = int(count)
            self.labels.append(label)
            self.display_names.append(str(names[len(self.labels)-1]) if names is not None else "")
            self.shots.append(features[offset:offset + count].copy())
            size = sizes[len(self.labels) - 1] if sizes is not None else None
            self.scan_sizes.append(tuple(size) if size is not None and min(size) > 0 else None)
            hist = hists[len(self.labels) - 1] if hists is not None else None
            self.color_hists.append(
                hist.astype(np.float32) / float(hist.sum())
                if hist is not None and float(hist.sum()) > 0 else None
            )
            index = len(self.labels)-1
            nrefs = int(ref_counts[index]) if b_meta is not None else 0
            self.reference_views.append([view.copy() for view in refs[ref_offset:ref_offset+nrefs]]
                                        if b_meta is not None else [])
            self.raw_cosine.append(bool(raw[index]) if b_meta is not None else False)
            self.class_shapes.append(shapes[index].astype(np.float32) if nrefs else None)
            ref_offset += nrefs
            offset += count
        self.invalidate()
        return True, "{} classes / {} shots".format(len(self.labels), len(features))


class PefslNcmClassifier:
    """NCM classifier matching few_shot_model.feature_preprocess/ncm."""

    def __init__(self):
        self.previous = None

    @staticmethod
    def preprocess(features, mean):
        features = np.asarray(features, dtype=np.float32) - mean
        return _unit_rows(features)

    def predict(self, feature, data):
        if not data.has_data():
            return None, 0.0, None
        mean = data.mean_feature()
        query = self.preprocess(np.asarray(feature)[None, :], mean)[0]
        classes = self.preprocess(data.class_means(), mean)
        distances = np.linalg.norm(classes - query[None, :], axis=1)
        logits = -20.0 * distances
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(np.sum(probabilities), 1e-8)
        self.previous = probabilities
        index = int(np.argmax(probabilities))
        return data.labels[index], float(probabilities[index]), probabilities

    def reset(self):
        self.previous = None


def clamp_box(box, width=SCREEN_W, height=SCREEN_H, footer_height=FOOTER_H):
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(TOOLBAR_H, min(height - footer_height - 1, y1))
    y2 = max(TOOLBAR_H, min(height - footer_height - 1, y2))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return left, top, right, bottom


def crop_resize(frame, box, side=32):
    x1, y1, x2, y2 = clamp_box(box, frame.shape[1], frame.shape[0])
    if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
        raise ValueError("ROI is too small")
    crop = np.asarray(frame[y1:y2, x1:x2, :3])
    if crop.size == 0:
        raise ValueError("ROI is empty")
    return cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)


class TouchBboxUI:
    """Touch state machine with a serial background TCU/scan worker."""

    def __init__(self, extractor, data_path, collect_frames=8,
                 predict_interval=1.0, scan_interval=0.10,
                 scan_windows=24, scan_cos_threshold=0.56,
                 scan_hist_threshold=0.0, scan_confirm_cycles=2,
                 scan_contrast_threshold=0.14, digit_cache_path=None,
                 digit_min_score=0.25, digit_min_margin=0.04,
                 digit_history_len=5, operator_cache_path=None,
                 operator_min_score=0.45, operator_min_margin=0.08,
                 operator_history_len=5, digit_preview=True,
                 b_max_result_age=1.5):
        self.extractor = extractor
        self.data_path = Path(data_path)
        self.collect_frames = max(1, int(collect_frames))
        self.predict_interval = max(0.25, float(predict_interval))
        self.scan_interval = max(0.05, float(scan_interval))
        self.scan_windows = max(4, int(scan_windows))
        self.scan_cos_threshold = float(scan_cos_threshold)
        self.scan_hist_threshold = float(scan_hist_threshold)
        self.scan_confirm_cycles = max(2, int(scan_confirm_cycles))
        self.scan_contrast_threshold = max(0.0, float(scan_contrast_threshold))
        self.data = LearnedData()
        self.classifier = PefslNcmClassifier()
        self.mode = "A"
        self.drawing = False
        self.draw_start = None
        self.draw_current = None
        # D mode can hold one raw camera frame while the main loop keeps
        # draining the video input.  The current input buffer is then reused
        # as the LCD output buffer, with the held image and UI overlays copied
        # into it on every iteration.
        self.freeze_active = False
        self._freeze_pending = False
        self._frozen_frame = None
        self.pending_box = None
        self.pending_name = ""
        self.name_confirmed = False
        self._name_epoch = time.monotonic()
        self.collect_display_name = None
        self.active_box = None
        self.digit_box = None
        self.collect_label = None
        self.collect_box = None
        self.collect_features = []
        self.collect_hists = []
        self.collect_views = []
        self.collect_left = 0
        self.predict_pending = False
        self.last_prediction_at = 0.0
        self.last_label = None
        self.last_score = 0.0
        self.last_probabilities = None
        self.status = "A learn: draw an ROI, then select class 1..4"
        self.last_event = None
        self.digit_cache_path = Path(digit_cache_path) if digit_cache_path else (
            Path(__file__).resolve().parent / "digit_support_r17.npz"
        )
        self.digit_data = LearnedData()
        self.digit_classifier = PefslNcmClassifier()
        self.digit_preprocessor = None
        self.digit_ready = False
        self.digit_cache_status = "digit cache unavailable"
        self.digit_min_score = max(0.0, float(digit_min_score))
        self.digit_min_margin = max(0.0, float(digit_min_margin))
        self.digit_prob_history = deque(maxlen=max(1, int(digit_history_len)))
        self.digit_last_label = None
        self.digit_last_score = 0.0
        self.digit_last_margin = 0.0
        self.digit_last_probabilities = None
        self.recognition_kind = "digits"
        self.digit_preview = bool(digit_preview)
        self.operator_cache_path = Path(operator_cache_path) if operator_cache_path else (
            Path(__file__).resolve().parent / "operator_support_r17.npz"
        )
        self.operator_data = LearnedData()
        self.operator_classifier = PefslNcmClassifier()
        self.operator_preprocessor = None
        self.operator_ready = False
        self.operator_cache_status = "operator cache unavailable"
        # Initial experimental gates; calibrate with board camera samples.
        self.operator_min_score = max(0.0, float(operator_min_score))
        self.operator_min_margin = max(0.0, float(operator_min_margin))
        self.operator_prob_history = deque(maxlen=max(1, int(operator_history_len)))
        self.debug_digit_bin = None
        self.current_candidate = None
        self.calc = DigitCalculator()
        self.calc_stable_frames = 3
        self._stable_key = None
        self._stable_count = 0
        self._candidate_epoch = 0
        self._displayed_candidate = None
        self._consumed_generation = None
        self._calc_feedback = "FREEZE, draw a digit, then store X1"
        self._calc_feedback_until = 0.0
        # Only the inference worker may reset/process these mutable instances.
        self._preprocessor_generations = {"digits": None, "operators": None}
        self._load_digit_cache()
        self._load_operator_cache()
        self.scan_enabled = False
        self.scan_detections = []
        self._scan_cycle_detections = []
        self._scan_boxes = []
        self._scan_cursor = 0
        self._scan_phase = 0
        self._scan_generation = 0
        self._scan_request = None
        self._scan_last_at = 0.0
        self._scan_examined = 0
        self._scan_cycle = 0
        # Compatibility-only option; r17 A/B has no latest-frame age gate.
        self.b_max_result_age = max(0.1, float(b_max_result_age))
        # Raw windows are hypotheses.  A label/box must survive independent
        # scan cycles before it is shown, matching C104's pending/verified
        # behaviour and preventing a single background patch from flashing on
        # the LCD.
        self._scan_tracks = []
        # TCU calls can take hundreds of milliseconds on the ARM core. Keep
        # them out of the camera/touch frame loop while B mode refreshes.
        self._prediction_lock = threading.RLock()
        self._prediction_event = threading.Event()
        self._prediction_stop = threading.Event()
        self._prediction_request = None
        self._digit_request = None
        self._collection_request = None
        self._prediction_busy = False
        self._prediction_generation = 0
        self._prediction_thread = threading.Thread(
            target=self._prediction_worker, name="tcu-predict", daemon=True
        )
        self._prediction_thread.start()

    def close(self, timeout=2.0):
        """Stop the background TCU inference worker."""
        self._prediction_stop.set()
        self._prediction_event.set()
        self._prediction_thread.join(timeout=float(timeout))
        return not self._prediction_thread.is_alive()

    def _invalidate_scan_locked(self, clear_display=False):
        self._scan_generation += 1
        self._scan_request = None
        self._scan_boxes = []
        self._scan_cursor = 0
        self._scan_cycle_detections = []
        self._scan_tracks = []
        self._scan_examined = 0
        self._scan_last_at = 0.0
        if clear_display:
            self.scan_detections = []

    def _rebuild_scan_locked(self, frame_width=SCREEN_W, frame_height=SCREEN_H):
        sizes = [size for size in self.data.scan_sizes if size is not None]
        self._scan_boxes = build_scan_boxes(
            frame_width, frame_height,
            region=(0, TOOLBAR_H, frame_width, frame_height - FOOTER_H),
            sizes=sizes,
            max_windows=self.scan_windows,
            phase=self._scan_phase,
        )
        self._scan_cursor = 0
        self._scan_cycle_detections = []
        self._scan_examined = 0
        self._scan_cycle += 1

    @staticmethod
    def _best_per_label(detections):
        best = {}
        for detection in detections:
            label = str(detection.get("label", ""))
            if not label:
                continue
            old = best.get(label)
            if old is None or float(detection.get("score", 0.0)) > float(old.get("score", 0.0)):
                best[label] = dict(detection)
        return list(best.values())

    @staticmethod
    def _box_iou(left, right):
        ix1 = max(int(left[0]), int(right[0]))
        iy1 = max(int(left[1]), int(right[1]))
        ix2 = min(int(left[2]), int(right[2]))
        iy2 = min(int(left[3]), int(right[3]))
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_left = max(1, (int(left[2]) - int(left[0])) *
                       (int(left[3]) - int(left[1])))
        area_right = max(1, (int(right[2]) - int(right[0])) *
                        (int(right[3]) - int(right[1])))
        union = area_left + area_right - inter
        return inter / float(max(1, union))

    def _confirm_scan_cycle_locked(self, detections):
        """Update label/position tracks and return only verified detections."""
        cycle = self._best_per_label(detections)
        next_tracks = []
        verified = []
        for candidate in cycle:
            match = None
            for track in self._scan_tracks:
                if track["label"] == candidate["label"] and \
                        self._box_iou(track["box"], candidate["box"]) >= 0.20:
                    match = track
                    break
            if match is None:
                track = {
                    "label": candidate["label"],
                    "box": tuple(candidate["box"]),
                    "score": float(candidate["score"]),
                    "hits": 1,
                }
            else:
                track = dict(match)
                track["box"] = tuple(candidate["box"])
                track["score"] = float(candidate["score"])
                track["hits"] = int(track.get("hits", 1)) + 1
            next_tracks.append(track)
            if int(track["hits"]) >= self.scan_confirm_cycles:
                verified.append({
                    "box": track["box"], "label": track["label"],
                    "score": track["score"],
                })
        self._scan_tracks = next_tracks
        return verified

    def _start_scan_locked(self, frame_width=SCREEN_W, frame_height=SCREEN_H):
        if not self.data.has_data():
            self.scan_enabled = False
            self._invalidate_scan_locked(clear_display=True)
            return False
        self.scan_enabled = True
        self._invalidate_scan_locked(clear_display=False)
        self._rebuild_scan_locked(frame_width, frame_height)
        return True

    def _button(self, x, y, w, h):
        return (x <= self._event_x < x + w and y <= self._event_y < y + h)

    def _set_status(self, message, d_feedback=False):
        self.status = str(message)
        # D draws its own calculator panel instead of the A/B status line.
        # Surface errors/actions there without replacing the saved result or
        # letting routine prediction updates hide the next-slot guidance.
        if d_feedback and self.mode == "D":
            self._calc_feedback = self.status
            self._calc_feedback_until = time.monotonic() + 4.0
        print("[ui] {}".format(self.status), flush=True)

    def _load_digit_cache(self):
        """Load the independent digit prototypes without touching A/B data."""
        if DigitSupportCache is None or DigitPreprocessor is None:
            self.digit_cache_status = "digit modules unavailable"
            return
        if not self.digit_cache_path.is_file():
            self.digit_cache_status = "digit cache not found: {}".format(
                self.digit_cache_path
            )
            return
        try:
            cache = DigitSupportCache.load(self.digit_cache_path)
            offset = 0
            for label, count in zip(cache.labels, cache.counts):
                count = int(count)
                self.digit_data.add_shots(
                    label,
                    cache.features[offset:offset + count],
                )
                offset += count
            self.digit_preprocessor = DigitPreprocessor(binary_history_len=5)
            self.digit_ready = True
            self.digit_cache_status = cache.summary()
        except Exception as error:
            self.digit_data.reset()
            self.digit_cache_status = "digit cache invalid: {}".format(error)

    def _reset_digit_state_locked(self):
        self._digit_request = None
        self.digit_classifier.reset()
        self.digit_prob_history.clear()
        self.operator_classifier.reset()
        self.operator_prob_history.clear()
        self.digit_last_label = None
        self.digit_last_score = 0.0
        self.digit_last_margin = 0.0
        self.digit_last_probabilities = None
        self.debug_digit_bin = None
        self.current_candidate = None
        self._displayed_candidate = None
        self._stable_key = None
        self._stable_count = 0
        self._candidate_epoch += 1
        # Preprocessing histories are reset by the worker on generation change.

    def _load_operator_cache(self):
        if DigitSupportCache is None or DigitPreprocessor is None:
            self.operator_cache_status = "operator modules unavailable"
            return
        try:
            cache = DigitSupportCache.load(self.operator_cache_path, expected_kind="operators")
            offset = 0
            for label, count in zip(cache.labels, cache.counts):
                count = int(count)
                self.operator_data.add_shots(label, cache.features[offset:offset + count])
                offset += count
            self.operator_preprocessor = DigitPreprocessor(binary_history_len=5)
            self.operator_ready = True
            self.operator_cache_status = cache.summary()
        except Exception as error:
            self.operator_data.reset()
            self.operator_cache_status = "operator cache invalid: {}".format(error)

    def _recognition_bank(self, recognition_kind):
        if recognition_kind == "operators":
            return (self.operator_data, self.operator_classifier,
                    self.operator_preprocessor, self.operator_prob_history,
                    self.operator_min_score, self.operator_min_margin,
                    self.operator_ready, self.operator_cache_status)
        if recognition_kind != "digits":
            raise ValueError("unknown recognition kind: {}".format(recognition_kind))
        return (self.digit_data, self.digit_classifier,
                self.digit_preprocessor, self.digit_prob_history,
                self.digit_min_score, self.digit_min_margin,
                self.digit_ready, self.digit_cache_status)

    def _toggle_operator_mode(self):
        with self._prediction_lock:
            if self.mode != "D":
                return
            if self.recognition_kind == "digits" and not self.operator_ready:
                self._set_status("OP unavailable: {}".format(self.operator_cache_status),
                                 d_feedback=True)
                return
            self.recognition_kind = (
                "operators" if self.recognition_kind == "digits" else "digits"
            )
            self._prediction_generation += 1
            self._reset_digit_state_locked()
            self.drawing = False
            self.draw_start = self.draw_current = None
            self.predict_pending = self.digit_box is not None
            self.last_prediction_at = 0.0
            self._set_status("D {}: draw ROI or wait for recognition".format(self.recognition_kind))

    def _clamp_ui_box(self, box, width=SCREEN_W, height=SCREEN_H):
        if self.mode == "D" and self.digit_preview:
            width = min(width, SCREEN_W - 160)
        return clamp_box(box, width, height,
                         D_PANEL_H if self.mode == "D" else FOOTER_H)

    def _candidate_store_error_locked(self, requested_kind, displayed):
        """Validate exactly the candidate the user last saw on the LCD."""
        if self.mode != "D":
            return "Store is available in D mode"
        if not self.freeze_active or self._freeze_pending or self._frozen_frame is None:
            return "Press FREEZE before storing"
        if self.drawing:
            return "Finish drawing the ROI first"
        if self.digit_box is None:
            return "Draw a character ROI first"
        if self.calc.expecting is None:
            return "Expression full: OUT, UNDO or RESET"
        if requested_kind != self.recognition_kind:
            return "Switch OP MODE to {} first".format(requested_kind)
        if requested_kind != self.calc.expecting:
            return "Next item must be {}".format(self.calc.expecting)
        current = self.current_candidate
        if not displayed or not current:
            return "Wait for a valid result on the LCD"
        keys = ("epoch", "generation", "kind", "label", "roi")
        if any(displayed.get(key) != current.get(key) for key in keys):
            return "Result changed; check the new LCD result"
        if (current.get("generation") != self._prediction_generation
                or current.get("kind") != requested_kind
                or current.get("roi") != tuple(self.digit_box)):
            return "Result expired; wait for recognition"
        if (not displayed.get("valid") or not current.get("valid")
                or displayed.get("stable_count", 0) < self.calc_stable_frames
                or current.get("stable_count", 0) < self.calc_stable_frames):
            return "Wait for READY ({} stable predictions)".format(self.calc_stable_frames)
        if (self._consumed_generation == self._prediction_generation
                or displayed.get("consumed") or current.get("consumed")):
            return "Already stored; draw a new ROI or take a new frame"
        label = current.get("label")
        if requested_kind == "digits" and label not in tuple(str(i) for i in range(10)):
            return "Result is not a single digit"
        if requested_kind == "operators" and label not in OPERATOR_SYMBOLS:
            return "Result is not an operator"
        return None

    def _calculator_action(self, action):
        with self._prediction_lock:
            if self.mode != "D":
                return False
            try:
                if action in ("number", "operator"):
                    kind = "digits" if action == "number" else "operators"
                    error = self._candidate_store_error_locked(kind, self._displayed_candidate)
                    if error:
                        raise ValueError(error)
                    label = self._displayed_candidate["label"]
                    if action == "number":
                        self.calc.append_number(label)
                        slot = "X{}".format(self.calc.number_count)
                    else:
                        self.calc.append_operator(label)
                        slot = "OP{}".format(self.calc.operator_count)
                    self._consumed_generation = self._prediction_generation
                    self.current_candidate["consumed"] = True
                    self._displayed_candidate["consumed"] = True
                    message = "Stored {}={}; {}".format(
                        slot, OPERATOR_SYMBOLS.get(label, label), self._calculator_next_hint())
                elif action in ("undo", "reset"):
                    if action == "undo":
                        if not self.calc.tokens:
                            raise ValueError("Expression is empty")
                        self.calc.undo()
                    else:
                        self.calc.reset()
                    self._prediction_generation += 1
                    self._reset_digit_state_locked()
                    self.digit_box = None
                    self.drawing = False
                    self.draw_start = self.draw_current = None
                    self.predict_pending = False
                    message = "{}; draw a new ROI".format(
                        "Last item removed" if action == "undo" else "Expression cleared")
                elif action == "output":
                    self.calc.evaluate()
                    message = "{} = {}".format(self.calc.display_expression(), self.calc.result_text)
                else:
                    raise ValueError("Unknown calculator action")
            except ValueError as error:
                self._calc_feedback = str(error)
                self._calc_feedback_until = time.monotonic() + 4.0
                self._set_status("calc: {}".format(error))
                return False
            self._calc_feedback = message
            self._calc_feedback_until = time.monotonic() + 4.0
            self._set_status("calc: {}".format(message))
            return True

    def _calculator_next_hint(self):
        if self.calc.expecting is None:
            return "Full: press OUT"
        if self.calc.expecting == "digits":
            return "Next X{}: OP MODE off".format(self.calc.number_count + 1)
        return "Next OP{}: OP MODE on, or OUT".format(self.calc.operator_count + 1)

    def _publish_candidate_locked(self, recognition_kind, generation, data,
                                  probabilities, label, score, margin, accepted,
                                  min_score, min_margin):
        # The averaged label alone can stay unchanged after a raw prediction
        # flips. Require agreement and gates on each raw result as well.
        order = np.argsort(probabilities)
        raw_index = int(order[-1])
        raw_score = float(probabilities[raw_index])
        raw_margin = raw_score - float(probabilities[order[-2]])
        consistent = (accepted and data.labels[raw_index] == label
                      and raw_score >= min_score and raw_margin >= min_margin)
        if not consistent:
            self._stable_key = None
            self._stable_count = 0
            self._candidate_epoch += 1
            self.current_candidate = None
            self._displayed_candidate = None
            return
        roi = tuple(self.digit_box) if self.digit_box is not None else None
        key = (generation, recognition_kind, roi, label)
        if key != self._stable_key:
            self._stable_key = key
            self._stable_count = 1
            self._candidate_epoch += 1
        else:
            self._stable_count += 1
        self.current_candidate = {
            "kind": recognition_kind, "generation": generation,
            "label": label, "score": score, "margin": margin, "roi": roi,
            "epoch": self._candidate_epoch, "stable_count": self._stable_count,
            "consumed": self._consumed_generation == generation, "valid": True,
        }

    def _clear_freeze_locked(self):
        """Return to live camera output and discard the held frame."""
        self.freeze_active = False
        self._freeze_pending = False
        self._frozen_frame = None

    def _toggle_digit_freeze(self):
        """Toggle the D-mode LCD frame hold without stopping recognition."""
        with self._prediction_lock:
            if self.mode != "D":
                message = "FREEZE is available in D mode"
            else:
                # Invalidate a request that may have been made from the other
                # frame state.  The next frame schedules a fresh D prediction.
                self._prediction_generation += 1
                self._prediction_request = None
                self._collection_request = None
                self._scan_request = None
                self._reset_digit_state_locked()
                self.drawing = False
                self.draw_start = self.draw_current = None
                self.predict_pending = True
                self.last_prediction_at = 0.0
                if self.freeze_active or self._freeze_pending:
                    self._clear_freeze_locked()
                    message = "D {} live: camera resumed".format(self.recognition_kind)
                else:
                    # Capture is deferred until process_frame receives the
                    # next raw camera buffer, before any overlay is drawn.
                    self._freeze_pending = True
                    message = "D freeze: capturing frame"
        self._set_status(message)

    def _top_action(self, x):
        if x < 234 or 324 <= x < 514:
            with self._prediction_lock:
                self._prediction_generation += 1
                self._reset_name_locked()
                self.collect_display_name = None
                self.pending_box = None
        if x < 78:
            with self._prediction_lock:
                self._prediction_generation += 1
                self.mode = "A"
                self._clear_freeze_locked()
                self.scan_enabled = False
                self._invalidate_scan_locked(clear_display=True)
                self.pending_box = None
                self.active_box = None
                self.digit_box = None
                self.predict_pending = False
                self.last_prediction_at = 0.0
                self.collect_label = None
                self.collect_box = None
                self.collect_features = []
                self.collect_hists = []
                self.collect_views = []
                self.collect_left = 0
                self._prediction_request = None
                self._collection_request = None
                self._reset_digit_state_locked()
            self._set_status("A learn: draw an ROI, then select class 1..4")
        elif x < 156:
            with self._prediction_lock:
                self._prediction_generation += 1
                self.mode = "B"
                self._clear_freeze_locked()
                self.pending_box = None
                self.active_box = None
                self.digit_box = None
                self.predict_pending = False
                self.last_prediction_at = 0.0
                self.collect_label = None
                self.collect_box = None
                self.collect_features = []
                self.collect_hists = []
                self.collect_views = []
                self.collect_left = 0
                self._prediction_request = None
                self._collection_request = None
                self._reset_digit_state_locked()
                ready = self._start_scan_locked()
            self._set_status(
                "B scan: searching learned objects" if ready
                else "B recognize: no learned data; use A mode first"
            )
        elif x < 234:
            with self._prediction_lock:
                self._prediction_generation += 1
                self.mode = "D"
                self.recognition_kind = "digits"
                self._clear_freeze_locked()
                self.scan_enabled = False
                self._invalidate_scan_locked(clear_display=True)
                self.pending_box = None
                self.active_box = None
                self.digit_box = None
                self.predict_pending = False
                self.last_prediction_at = 0.0
                self.collect_label = None
                self.collect_box = None
                self.collect_features = []
                self.collect_hists = []
                self.collect_views = []
                self.collect_left = 0
                self._prediction_request = None
                self._collection_request = None
                self._reset_digit_state_locked()
                ready = self.digit_ready
            self._set_status(
                "D digit: draw one digit ROI"
                if ready else "D unavailable: {}".format(self.digit_cache_status),
                d_feedback=not ready,
            )
        elif x < 324:
            with self._prediction_lock:
                ok, message = self.data.save(self.data_path)
            self._set_status(("saved: " if ok else "save: ") + message, d_feedback=True)
        elif x < 414:
            with self._prediction_lock:
                # LOAD invalidates an unfinished learning request as well as
                # its name; a new bank cannot receive stale collected shots.
                self.collect_label = None
                self.collect_box = None
                self.collect_features = []
                self.collect_hists = []
                self.collect_views = []
                self.collect_left = 0
                self._collection_request = None
                if self.mode == 'A':
                    self.active_box = None
                ok, message = self.data.load(self.data_path)
                self.classifier.reset()
                self._prediction_generation += 1
                self.last_prediction_at = 0.0
                if ok and self.mode == "B" and self.active_box is not None:
                    self.predict_pending = True
                elif ok and self.mode == "B":
                    self._start_scan_locked()
            self._set_status(("loaded: " if ok else "load: ") + message, d_feedback=True)
        elif x < 514:
            with self._prediction_lock:
                self.data.reset()
                self.classifier.reset()
                self._prediction_generation += 1
                self.scan_enabled = False
                self._invalidate_scan_locked(clear_display=True)
                self.collect_label = None
                self.collect_box = None
                self.collect_features = []
                self.collect_hists = []
                self.collect_views = []
                self.collect_left = 0
                self._prediction_request = None
                self._collection_request = None
                self._reset_digit_state_locked()
            self.pending_box = None
            self.active_box = None
            self.digit_box = None
            self.predict_pending = False
            self.last_prediction_at = 0.0
            self.last_label = None
            self._set_status("learned data cleared", d_feedback=True)
        else:
            if self.mode == "D":
                if x < 644:
                    self._toggle_operator_mode()
                else:
                    self._toggle_digit_freeze()
                return
            with self._prediction_lock:
                if self.mode != "B":
                    message = "SCAN is available in B mode"
                elif self.scan_enabled:
                    self.scan_enabled = False
                    self._invalidate_scan_locked(clear_display=False)
                    message = "scan off; draw an ROI for manual recognition"
                elif self._start_scan_locked():
                    # A manual B ROI pauses the locator.  Tapping SCAN is the
                    # explicit way back to automatic mode, so discard the
                    # manual request before enabling the scanner.
                    self._prediction_generation += 1
                    self.active_box = None
                    self.predict_pending = False
                    self.last_prediction_at = 0.0
                    message = "scan on: searching learned objects"
                else:
                    message = "scan needs learned data; use A mode first"
            self._set_status(message)

    def _reset_name_locked(self):
        self.pending_name = ""
        self.name_confirmed = False

    def handle_key(self, event):
        """Consume a queued keyboard event on the UI thread, never in IIC."""
        if event is None:
            return
        kind = getattr(event, 'kind', '')
        with self._prediction_lock:
            timestamp = getattr(event, 'timestamp', None)
            if timestamp is not None and float(timestamp) < self._name_epoch:
                return
            if kind == 'disconnected':
                if self.mode == 'A' and self.pending_box is not None and self.collect_label is None:
                    self._reset_name_locked()
                    self.pending_box = None
                    self._set_status("keyboard input interrupted: redraw the ROI to name it")
                return
            if (self.mode != 'A' or self.pending_box is None or self.drawing
                    or self.collect_label is not None):
                return
            if kind == 'escape':
                self._reset_name_locked()
                self._name_epoch = float(timestamp) if timestamp is not None else time.monotonic()
                self._set_status("name cancelled; type again or select class 1..4")
            elif kind == 'enter':
                # Preserve spaces inside a name, while an all-space draft
                # behaves like an empty Enter and keeps the existing ID/name.
                self.pending_name = self.pending_name.strip()
                self.name_confirmed = True
                self._set_status("name ready: {}; select class 1..4".format(
                    self.pending_name or "keep existing/default"))
            elif self.name_confirmed:
                return
            elif kind == 'backspace':
                self.pending_name = self.pending_name[:-1]
                self.status = "edit name; Enter confirms, Esc cancels"
            elif kind == 'text':
                text = getattr(event, 'text', '')
                characters = ''.join(char for char in text if char in NAME_CHARACTERS)
                limited = len(self.pending_name) + len(characters) > MAX_NAME_LENGTH
                self.pending_name = (self.pending_name + characters)[:MAX_NAME_LENGTH]
                self.status = ("name limit: {} characters; Enter confirms".format(MAX_NAME_LENGTH)
                               if limited else "edit name; Enter confirms, Esc cancels")

    def _footer_action(self, x):
        if self.mode == "D":
            actions = ("number", "operator", "undo", "reset", "output")
            if 0 <= x < SCREEN_W:
                self._calculator_action(actions[int(x // 160)])
            return
        if self.mode != "A":
            self._set_status(
                "D {}: draw one character ROI".format(self.recognition_kind) if self.mode == "D"
                else "B mode: draw an ROI for recognition"
            )
            return
        if x >= 360:
            self._set_status("touch class 1..4 in the bottom-left buttons")
            return
        if not self.pending_box:
            self._set_status("draw an ROI first")
            return
        if self.pending_name and not self.name_confirmed:
            self._set_status("press Enter to confirm the name before selecting a class")
            return
        class_index = min(3, max(0, int(x // 90)))
        with self._prediction_lock:
            self._prediction_generation += 1
            self.collect_label = "class{}".format(class_index + 1)
            self.collect_display_name = self.pending_name if self.name_confirmed and self.pending_name else None
            self.collect_box = tuple(self.pending_box)
            self.pending_box = None
            self._reset_name_locked()
            self.collect_features = []
            self.collect_hists = []
            self.collect_views = []
            self.collect_left = self.collect_frames
            self.active_box = self.collect_box
        self._set_status(
            "collecting {} ({})".format(
                self.collect_display_name or self.data.display_name(self.collect_label),
                self.collect_frames)
        )

    def handle_event(self, event):
        """Consume one TouchEvent. The event object is from touch_driver.py."""
        self.last_event = event
        if event is None:
            return
        self._event_x = int(event.x)
        self._event_y = int(event.y)
        points = int(getattr(event, "points", 1))
        if points > 1 and event.kind in ("down", "move"):
            self.drawing = False
            self.draw_start = None
            self.draw_current = None
            self._set_status("single touch only; release extra fingers", d_feedback=True)
            return
        if event.kind == "down":
            if self._event_y < TOOLBAR_H:
                self.drawing = False
                self.draw_start = self.draw_current = None
                self._top_action(self._event_x)
                return
            if self._event_y >= SCREEN_H - FOOTER_H:
                self._footer_action(self._event_x)
                return
            if self.collect_label is not None:
                self._set_status("wait for current collection to finish")
                return
            if self.mode == "D":
                if self._event_y >= SCREEN_H - D_PANEL_H:
                    return
                if self.digit_preview and self._event_x >= SCREEN_W - 160:
                    return
                with self._prediction_lock:
                    self._prediction_generation += 1
                    self._reset_digit_state_locked()
                    self.digit_box = None
                    self.predict_pending = False
            elif self.mode == 'A':
                with self._prediction_lock:
                    self.pending_box = None
                    self._reset_name_locked()
            self.drawing = True
            self.draw_start = (self._event_x, self._event_y)
            self.draw_current = self.draw_start
            return
        if event.kind == "move" and self.drawing:
            self.draw_current = (self._event_x, self._event_y)
            return
        if event.kind == "up" and self.drawing:
            self.draw_current = (self._event_x, self._event_y)
            self.drawing = False
            box = self._clamp_ui_box(
                (self.draw_start[0], self.draw_start[1],
                 self.draw_current[0], self.draw_current[1])
            )
            if box[2] - box[0] < MIN_BOX or box[3] - box[1] < MIN_BOX:
                self._set_status("ROI too small; draw a larger box", d_feedback=True)
                return
            if self.mode == "A":
                self.pending_box = box
                self._reset_name_locked()
                self._name_epoch = float(getattr(event, 'timestamp', time.monotonic()))
                self._set_status("type a name + Enter, or select class 1..4")
            else:
                with self._prediction_lock:
                    self._prediction_generation += 1
                    if self.mode == "D":
                        self.digit_box = box
                        self.active_box = None
                        self._reset_digit_state_locked()
                    else:
                        self.active_box = box
                    self.scan_enabled = False
                    self._invalidate_scan_locked(clear_display=False)
                    self.predict_pending = True
                    self.last_prediction_at = 0.0
                self._set_status(
                    "{}: recognizing ROI".format(self.recognition_kind) if self.mode == "D"
                    else "recognizing ROI"
                )
            return

    def _schedule_collection(self, frame):
        """Queue one immutable crop; collection must not block video/touch."""
        with self._prediction_lock:
            if (self.mode != "A" or self.collect_label is None or
                    self.collect_box is None or self._prediction_busy or
                    self._collection_request is not None or
                    self._digit_request is not None):
                return
            try:
                crop = crop_resize(frame, self.collect_box)
            except Exception as error:
                self._set_status("TCU/ROI error: {}".format(error), d_feedback=True)
                return
            self._collection_request = (
                np.array(crop, dtype=np.uint8, copy=True),
                self.collect_label,
                tuple(self.collect_box),
                self.collect_display_name,
                self._prediction_generation,
            )
            self._prediction_event.set()

    def _prediction_worker(self):
        """Run the single TCU transaction stream away from the video loop."""
        while not self._prediction_stop.is_set():
            self._prediction_event.wait(0.25)
            self._prediction_event.clear()
            if self._prediction_stop.is_set():
                return
            with self._prediction_lock:
                # Collection has priority.  A single worker is deliberate:
                # the TCU cannot safely serve concurrent feature requests.
                if self._collection_request is not None:
                    kind = "collect"
                    request = self._collection_request
                    self._collection_request = None
                elif self._digit_request is not None:
                    kind = "digit"
                    request = self._digit_request
                    self._digit_request = None
                elif self._prediction_request is not None:
                    kind = "predict"
                    request = self._prediction_request
                    self._prediction_request = None
                else:
                    kind = "scan"
                    request = self._scan_request
                    self._scan_request = None
                if request is None:
                    continue
                self._prediction_busy = True
            if kind == "collect":
                crop, label_name, box, display_name, generation = request
            elif kind == "digit":
                crop, generation, recognition_kind = request
            elif kind == "predict":
                crop, generation = request
            else:
                crop, box, crop_hist, contrast_distance, generation = request
            try:
                prepared = crop
                if kind == "digit":
                    with self._prediction_lock:
                        if (generation != self._prediction_generation or self.mode != "D"
                                or recognition_kind != self.recognition_kind):
                            continue
                        data, classifier, preprocessor, history, min_score, min_margin, ready, _ = (
                            self._recognition_bank(recognition_kind)
                        )
                    if not ready or preprocessor is None:
                        raise RuntimeError("{} preprocessor/cache unavailable".format(recognition_kind))
                    # Only this worker mutates preprocessing state. A late old
                    # request cannot contaminate the next generation's history.
                    if self._preprocessor_generations[recognition_kind] != generation:
                        preprocessor.reset()
                        self._preprocessor_generations[recognition_kind] = generation
                    prepared = preprocessor.process_to_bgr(crop)
                    with self._prediction_lock:
                        if (generation != self._prediction_generation or self.mode != "D"
                                or recognition_kind != self.recognition_kind):
                            continue
                        if not preprocessor.last_valid:
                            reason = preprocessor.last_invalid_reason
                            self._reset_digit_state_locked()
                            self.digit_last_label = "?"
                            self._set_status("{} invalid ROI: {}".format(recognition_kind, reason),
                                             d_feedback=True)
                            continue
                
                feature = np.asarray(
                    self.extractor.extract(prepared), dtype=np.float32
                ).reshape(-1)
                if feature.size != FEATURE_DIM or not np.isfinite(feature).all():
                    raise RuntimeError("TCU feature must contain 80 finite values")
                with self._prediction_lock:
                    if kind == "collect":
                        stale = (
                            generation != self._prediction_generation
                            or self.mode != "A"
                            or self.collect_label != label_name
                            or self.collect_box != box
                            or self.collect_display_name != display_name
                        )
                    elif kind == "digit":
                        stale = (
                            generation != self._prediction_generation
                            or self.mode != "D"
                            or recognition_kind != self.recognition_kind
                            or not self._recognition_bank(recognition_kind)[6]
                        )
                    elif kind == "predict":
                        stale = generation != self._prediction_generation or self.mode != "B"
                    else:
                        stale = (
                            generation != self._scan_generation
                            or self.mode != "B"
                            or not self.scan_enabled
                            or self.active_box is not None
                        )
                    if stale:
                        continue
                    if kind == "collect":
                        self.collect_features.append(feature)
                        self.collect_hists.append(color_histogram(crop))
                        self.collect_left -= 1
                        if self.collect_left > 0:
                            self._set_status(
                                "collecting {}: {} left".format(
                                    display_name or self.data.display_name(self.collect_label), self.collect_left
                                )
                            )
                        else:
                            box_width = box[2] - box[0]
                            box_height = box[3] - box[1]
                            hist = np.mean(np.stack(self.collect_hists), axis=0)
                            hist /= max(float(hist.sum()), 1e-8)
                            self.data.add_shots(
                                label_name, self.collect_features,
                                box_size=(box_width, box_height), hist=hist,
                                display_name=display_name,
                            )
                            count = len(self.data.shots[self.data.labels.index(label_name)])
                            committed_name = self.data.display_name(label_name)
                            self.collect_label = None
                            self.collect_display_name = None
                            self.collect_box = None
                            self.collect_features = []
                            self.collect_hists = []
                            self.active_box = None
                            self._set_status("learned {} ({} total shots)".format(
                                committed_name, count
                            ))
                    elif kind == "digit":
                        label, score, probabilities = classifier.predict(
                            feature, data
                        )
                        if probabilities is None:
                            self._reset_digit_state_locked()
                            self.digit_last_label = "?"
                            self.digit_last_score = 0.0
                            self.digit_last_margin = 0.0
                            self.digit_last_probabilities = None
                            self.debug_digit_bin = None
                            self.current_candidate = None
                            history.clear()
                            self._set_status("{}: no valid probabilities".format(recognition_kind),
                                             d_feedback=True)
                        else:
                            probabilities = np.asarray(
                                probabilities, dtype=np.float32
                            ).reshape(-1)
                            if (probabilities.size != len(data.labels)
                                    or not np.isfinite(probabilities).all()):
                                raise RuntimeError(
                                    "digit probability size is {}".format(
                                        probabilities.size
                                    )
                                )
                            history.append(probabilities)
                            smooth = np.mean(
                                np.stack(tuple(history), axis=0),
                                axis=0,
                            )
                            order = np.argsort(smooth)
                            top_index = int(order[-1])
                            top_score = float(smooth[top_index])
                            second_score = (
                                float(smooth[order[-2]]) if smooth.size > 1 else 0.0
                            )
                            margin = top_score - second_score
                            raw_label = data.labels[top_index]
                            accepted = (
                                top_score >= min_score
                                and margin >= min_margin
                            )
                            display_label = raw_label if accepted else "?"
                            self.digit_last_label = display_label
                            self.digit_last_score = top_score
                            self.digit_last_margin = margin
                            self.digit_last_probabilities = smooth.copy()
                            self.debug_digit_bin = prepared.copy()
                            self._publish_candidate_locked(
                                recognition_kind, generation, data, probabilities,
                                raw_label, top_score, margin, accepted, min_score, min_margin,
                            )
                            symbol = OPERATOR_SYMBOLS.get(raw_label, raw_label)
                            if accepted:
                                self._set_status(
                                    "{} {} {:.1f}% (margin {:.1f}%)".format(
                                        recognition_kind, symbol, 100.0 * top_score,
                                        100.0 * margin,
                                    )
                                )
                            else:
                                self._set_status(
                                    "{} uncertain {:.1f}% (margin {:.1f}%)".format(
                                        recognition_kind, 100.0 * top_score, 100.0 * margin,
                                    )
                                )
                    elif kind == "predict":
                        label, score, probabilities = self.classifier.predict(
                            feature, self.data
                        )
                        self.last_label = label
                        self.last_score = score
                        self.last_probabilities = probabilities
                        message = (
                            "no learned data; use A mode first" if label is None else
                            "recognized {} {:.1f}%".format(self.data.display_name(label), 100.0 * score)
                        )
                        self._set_status(message)
                    else:
                        labels = list(self.data.labels)
                        shots = [rows.copy() for rows in self.data.shots]
                        hists = list(self.data.color_hists)
                        label, score = (None, 0.0)
                        # A flat centre and its surrounding ring are usually
                        # the same background.  Reject that window before it
                        # can enter the temporal confirmation tracks.
                        contrast_ok = (
                            contrast_distance is None or
                            contrast_distance >= self.scan_contrast_threshold
                        )
                        if contrast_ok:
                            label, score = score_window(
                                feature, labels, shots, hists,
                                crop_hist=(crop_hist if self.scan_hist_threshold > 0
                                           else None),
                                cos_threshold=self.scan_cos_threshold,
                                hist_threshold=self.scan_hist_threshold,
                            )
                        if label is not None:
                            # Do not let a giant background window win merely
                            # because it contains a small object-like patch.
                            try:
                                class_index = labels.index(label)
                                ref_size = self.data.scan_sizes[class_index]
                                if ref_size is not None and min(ref_size) > 1:
                                    box_width = max(1, box[2] - box[0])
                                    box_height = max(1, box[3] - box[1])
                                    size_ratio = max(
                                        box_width / float(ref_size[0]),
                                        box_height / float(ref_size[1]),
                                    )
                                    if size_ratio > 2.30:
                                        label, score = None, 0.0
                            except (ValueError, IndexError, TypeError, ZeroDivisionError):
                                pass
                        if label is not None:
                            self._scan_cycle_detections.append({
                                "box": box, "label": label, "score": score,
                            })
                            self.last_label = label
                            self.last_score = score
                        self._scan_examined += 1
                        if self._scan_boxes and self._scan_cursor >= len(self._scan_boxes):
                            # Keep one strongest window per class before the
                            # global cap; otherwise four windows from one
                            # class could hide all other learned classes.
                            cycle_candidates = self._best_per_label(
                                self._scan_cycle_detections
                            )
                            cycle_candidates = suppress_detections(
                                cycle_candidates, max_detections=4,
                            )
                            verified = self._confirm_scan_cycle_locked(
                                cycle_candidates
                            )
                            self.scan_detections = suppress_detections(
                                verified, max_detections=4,
                            )
                            self._scan_phase += 1
                            self._rebuild_scan_locked()
                            self._set_status(
                                "scan cycle {}: {} object(s)".format(
                                    self._scan_cycle, len(self.scan_detections)
                                )
                            )
                        elif self._scan_examined % 4 == 0:
                            self._set_status(
                                "scanning {}/{}; {} object(s)".format(
                                    self._scan_examined, len(self._scan_boxes),
                                    len(self.scan_detections)
                                )
                            )
            except Exception as error:
                with self._prediction_lock:
                    valid = (
                        (kind == "collect" and generation == self._prediction_generation)
                        or (kind == "digit" and generation == self._prediction_generation
                            and self.mode == "D" and recognition_kind == self.recognition_kind)
                        or (kind == "predict" and generation == self._prediction_generation)
                        or (kind == "scan" and generation == self._scan_generation)
                    )
                    if valid and self.mode in ("A", "B", "D"):
                        self._set_status("TCU/ROI error: {}".format(error), d_feedback=True)
                    if kind == "digit" and valid:
                        self._reset_digit_state_locked()
                        self.digit_last_label = "?"
                        self.digit_last_score = 0.0
                        self.digit_last_margin = 0.0
                        self.digit_last_probabilities = None
                    if kind == "collect" and valid:
                        self.collect_label = None
                        self.collect_display_name = None
                        self.collect_box = None
                        self.collect_features = []
                        self.collect_hists = []
                        self.collect_left = 0
                        self.active_box = None
            finally:
                with self._prediction_lock:
                    self._prediction_busy = False
                    self.last_prediction_at = time.monotonic()

    def _schedule_digit_prediction(self, frame):
        """Queue one raw digit ROI; preprocessing happens in the TCU worker."""

        if self.mode != "D" or self.digit_box is None or self.drawing:
            return
        now = time.monotonic()
        with self._prediction_lock:
            ready, cache_status = self._recognition_bank(self.recognition_kind)[6:]
            if not ready:
                if self.predict_pending:
                    self.predict_pending = False
                    self._set_status(
                        "D unavailable: {}".format(cache_status), d_feedback=True,
                    )
                return
            if (self._prediction_busy or self._collection_request is not None or
                    self._digit_request is not None or
                    self._prediction_request is not None or
                    self._scan_request is not None):
                return
            if (not self.predict_pending and
                    now - self.last_prediction_at < self.predict_interval):
                return
            try:
                x1, y1, x2, y2 = self._clamp_ui_box(
                    self.digit_box, frame.shape[1], frame.shape[0]
                )
                if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
                    raise ValueError("ROI is too small")
                crop = np.asarray(
                    frame[y1:y2, x1:x2, :3], dtype=np.uint8
                ).copy()
                if crop.size == 0:
                    raise ValueError("ROI is empty")
            except Exception as error:
                self.predict_pending = False
                self._set_status("digit ROI error: {}".format(error), d_feedback=True)
                return
            self._digit_request = (
                crop,
                self._prediction_generation,
                self.recognition_kind,
            )
            self.predict_pending = False
            self._prediction_event.set()

    def _schedule_prediction(self, frame):
        if self.mode == "D":
            self._schedule_digit_prediction(frame)
            return
        if self.mode != "B" or self.active_box is None:
            return
        now = time.monotonic()
        with self._prediction_lock:
            if not self.data.has_data():
                if self.predict_pending:
                    self.predict_pending = False
                    self._set_status("no learned data; use A mode first")
                return
            if (self._prediction_busy or self._collection_request is not None or
                    self._digit_request is not None or
                    self._prediction_request is not None or self._scan_request is not None):
                return
            if (not self.predict_pending and
                    now - self.last_prediction_at < self.predict_interval):
                return
            try:
                crop = crop_resize(frame, self.active_box)
            except Exception as error:
                self.predict_pending = False
                self._set_status("TCU/ROI error: {}".format(error))
                return
            self._prediction_request = (
                np.array(crop, dtype=np.uint8, copy=True),
                self._prediction_generation,
            )
            self.predict_pending = False
            self._prediction_event.set()

    def _schedule_scan(self, frame):
        if self.mode != "B" or not self.scan_enabled or self.active_box is not None:
            return
        now = time.monotonic()
        with self._prediction_lock:
            if not self.data.has_data():
                self.scan_enabled = False
                self._invalidate_scan_locked(clear_display=True)
                self._set_status("scan needs learned data; use A mode first")
                return
            if (self._prediction_busy or self._collection_request is not None or
                    self._digit_request is not None or
                    self._prediction_request is not None or self._scan_request is not None):
                return
            if now - self._scan_last_at < self.scan_interval:
                return
            if not self._scan_boxes or self._scan_cursor >= len(self._scan_boxes):
                self._rebuild_scan_locked(frame.shape[1], frame.shape[0])
            box = self._scan_boxes[self._scan_cursor]
            self._scan_cursor += 1
            try:
                crop = crop_resize(frame, box)
            except Exception as error:
                self._set_status("scan ROI error: {}".format(error))
                return
            crop_hist = color_histogram(crop)
            contrast_distance = center_surround_distance(
                frame, box, center_hist=crop_hist,
            )
            self._scan_request = (
                np.array(crop, dtype=np.uint8, copy=True),
                box, crop_hist, contrast_distance,
                self._scan_generation,
            )
            self._scan_last_at = now
            self._prediction_event.set()

    def _run_prediction(self, frame):
        """Schedule a prediction; the worker performs it asynchronously."""
        self._schedule_prediction(frame)

    def process_frame(self, frame, event=None):
        """Handle one event, run pending TCU work, and draw the LCD overlay."""
        self.handle_event(event)
        capture_status = False
        with self._prediction_lock:
            if self.mode != "D":
                self._clear_freeze_locked()
            if self._freeze_pending:
                # ``frame`` is still the unmodified camera image here.  Keep
                # a normal NumPy copy instead of retaining a PYNQ DMA buffer,
                # which is returned to the video channel after this call.
                self._frozen_frame = np.asarray(
                    frame, dtype=np.uint8
                ).copy()
                self.freeze_active = True
                self._freeze_pending = False
                capture_status = True
            source = self._frozen_frame if self.freeze_active else frame

        if self.freeze_active:
            if self._frozen_frame is None or tuple(self._frozen_frame.shape) != tuple(frame.shape):
                raise RuntimeError("frozen frame shape does not match video frame")
            # The input buffer is still consumed every loop, but the LCD sees
            # the held image plus the current drawing/recognition overlays.
            np.copyto(frame, self._frozen_frame)
        if capture_status:
            self._set_status("D {} frozen: draw one character ROI".format(self.recognition_kind))

        if self.collect_label is not None:
            self._schedule_collection(source)
        if self.mode == "B":
            self._run_prediction(source)
            self._schedule_scan(source)
        elif self.mode == "D":
            self._run_prediction(source)
        self.draw(frame)
        return frame

    def _draw_button(self, frame, rect, label, active=False, enabled=True):
        x1, y1, x2, y2 = rect
        color = ((30, 120, 30) if active else (60, 60, 60)) if enabled else (30, 30, 30)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (190, 190, 190), 1)
        cv2.putText(frame, label, (x1 + 8, y1 + 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255) if enabled else (110, 110, 110), 1,
                    cv2.LINE_AA)

    def _calculator_snapshot_locked(self):
        candidate = dict(self.current_candidate) if self.current_candidate else None
        return {
            "candidate": candidate, "label": self.digit_last_label,
            "score": self.digit_last_score, "margin": self.digit_last_margin,
            "preview": self.debug_digit_bin,
            "kind": self.recognition_kind, "frozen": self.freeze_active,
            "expression": self.calc.display_expression(),
            "result": self.calc.result_text,
            "number_label": ("X{}".format(self.calc.number_count + 1)
                             if self.calc.number_count < MAX_NUMBERS else "X FULL"),
            "operator_label": ("OP{}".format(self.calc.operator_count + 1)
                               if self.calc.operator_count < MAX_OPERATORS else "OP FULL"),
            "number_enabled": self._candidate_store_error_locked("digits", candidate) is None,
            "operator_enabled": self._candidate_store_error_locked("operators", candidate) is None,
            "undo_enabled": bool(self.calc.tokens),
            "output_enabled": self.calc.can_evaluate,
            "hint": (self._calc_feedback if time.monotonic() < self._calc_feedback_until
                     else self._calculator_next_hint()),
        }

    def _draw_calculator_panel(self, frame, snapshot):
        height, width = frame.shape[:2]
        top = height - D_PANEL_H
        cv2.rectangle(frame, (0, top), (width - 1, height - 1),
                      (20, 20, 20), cv2.FILLED)
        expression = snapshot["expression"] or "(empty)"
        if snapshot["result"]:
            expression += " = " + snapshot["result"]
        cv2.putText(frame, expression, (8, top + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (100, 255, 255), 1, cv2.LINE_AA)
        candidate = snapshot["candidate"]
        symbol = OPERATOR_SYMBOLS.get(snapshot["label"], snapshot["label"]) or "-"
        if not snapshot["frozen"]:
            readiness = "LIVE: FREEZE to store"
        elif candidate and candidate.get("consumed"):
            readiness = "STORED: draw again"
        elif candidate and candidate["stable_count"] >= self.calc_stable_frames:
            readiness = "READY"
        elif candidate:
            readiness = "WAIT {}/{}".format(candidate["stable_count"], self.calc_stable_frames)
        else:
            readiness = "WAIT / draw ROI"
        line = "{} {} {} | {}".format(
            "NUM" if snapshot["kind"] == "digits" else "OP",
            symbol, readiness, snapshot["hint"],
        )
        cv2.putText(frame, line[:104], (8, top + 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (240, 240, 240), 1, cv2.LINE_AA)
        buttons = (
            (snapshot["number_label"], snapshot["number_enabled"]),
            (snapshot["operator_label"], snapshot["operator_enabled"]),
            ("UNDO", snapshot["undo_enabled"]), ("RESET", True),
            ("OUT", snapshot["output_enabled"]),
        )
        for index, (label, enabled) in enumerate(buttons):
            self._draw_button(frame, (index * 160, height - FOOTER_H,
                                     (index + 1) * 160 - 1, height - 1),
                              label, active=enabled and index in (0, 1, 4), enabled=enabled)

    def _draw_box_text(self, frame, box, text, color, scale=0.7):
        """Keep typed names and recognized names at the existing bbox anchor."""
        height, width = frame.shape[:2]
        measure = getattr(cv2, 'getTextSize', None)
        def text_size(value):
            if callable(measure):
                return measure(text, cv2.FONT_HERSHEY_SIMPLEX, value, 2)[0]
            return max(1, int(len(text) * 14 * value)), max(1, int(22 * value))
        tw, th = text_size(scale)
        if tw > width - 10:
            scale = max(0.25, scale * (width - 10) / float(max(1, tw)))
            tw, th = text_size(scale)
        x = max(4, min(int(box[0]), width - tw - 5))
        y = min(height - 5, max(TOOLBAR_H + 20, int(box[1]) - 8))
        cv2.rectangle(frame, (max(0, x - 3), max(TOOLBAR_H, y - th - 4)),
                      (min(width - 1, x + tw + 3), min(height - 1, y + 4)),
                      (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)

    def draw(self, frame):
        height, width = frame.shape[:2]
        snapshot = None
        if self.mode == "D":
            with self._prediction_lock:
                snapshot = self._calculator_snapshot_locked()
        cv2.rectangle(frame, (0, 0), (width - 1, TOOLBAR_H - 1),
                      (20, 20, 20), cv2.FILLED)
        if self.mode == "D":
            hold_label = "LIVE" if self.freeze_active else "FREEZE"
            hold_active = self.freeze_active
        else:
            hold_label = "SCAN"
            hold_active = self.scan_enabled
        top = [
            (0, 0, 78, TOOLBAR_H - 1, "A", self.mode == "A"),
            (78, 0, 156, TOOLBAR_H - 1, "B", self.mode == "B"),
            (156, 0, 234, TOOLBAR_H - 1, "D", self.mode == "D"),
            (234, 0, 324, TOOLBAR_H - 1, "SAVE", False),
            (324, 0, 414, TOOLBAR_H - 1, "LOAD", False),
            (414, 0, 514, TOOLBAR_H - 1, "CLEAR", False),
        ]
        if self.mode == "D":
            top.append((514, 0, 643, TOOLBAR_H - 1, "OP MODE",
                        self.recognition_kind == "operators"))
            top.append((644, 0, width - 1, TOOLBAR_H - 1, hold_label, hold_active))
        else:
            top.append((514, 0, width - 1, TOOLBAR_H - 1, hold_label, hold_active))
        for x1, y1, x2, y2, label, active in top:
            self._draw_button(frame, (x1, y1, x2, y2), label, active)
        cv2.rectangle(frame, (0, height - FOOTER_H), (width - 1, height - 1),
                      (20, 20, 20), cv2.FILLED)
        if self.mode == "D":
            self._draw_calculator_panel(frame, snapshot)
        else:
            for index in range(4):
                x1 = index * 90
                self._draw_button(
                    frame, (x1, height - FOOTER_H, x1 + 89, height - 1),
                    str(index + 1),
                    self.collect_label == "class{}".format(index + 1),
                )
            status_x = 370
            status = self.status[:92]
            cv2.putText(frame, status, (status_x, height - 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1,
                        cv2.LINE_AA)

        box = None
        color = (0, 220, 0)
        if self.drawing and self.draw_start and self.draw_current:
            box = self._clamp_ui_box(
                (self.draw_start[0], self.draw_start[1],
                 self.draw_current[0], self.draw_current[1]),
                width, height,
            )
            color = (0, 220, 220)
        elif self.pending_box is not None:
            box = self.pending_box
            color = (0, 220, 220)
        elif ((self.digit_box if self.mode == "D" else self.active_box)
              is not None):
            box = self.digit_box if self.mode == "D" else self.active_box
            active_label = (
                snapshot["label"] if self.mode == "D" else self.last_label
            )
            color = (0, 220, 0) if active_label is None else (0, 220, 255)
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            if self.mode == "D" and snapshot["label"] is not None:
                text = "{} {:.1f}%".format(
                    OPERATOR_SYMBOLS.get(snapshot["label"], snapshot["label"]),
                    100.0 * snapshot["score"]
                )
                cv2.putText(frame, text, (x1, max(TOOLBAR_H + 20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            cv2.LINE_AA)
            elif self.mode == 'A' and not self.drawing:
                if self.pending_box is not None:
                    text = (self.pending_name or "select class 1..4") if self.name_confirmed else self.pending_name + "_"
                    self._draw_box_text(frame, box, text, color)
                elif self.collect_label is not None:
                    text = self.collect_display_name or self.data.display_name(self.collect_label)
                    self._draw_box_text(frame, box, text, color)
            elif self.last_label is not None and self.mode == "B":
                text = "{} {:.1f}%".format(self.data.display_name(self.last_label), 100.0 * self.last_score)
                self._draw_box_text(frame, box, text, color)
        elif self.scan_enabled:
            for detection in self.scan_detections:
                x1, y1, x2, y2 = detection["box"]
                score = float(detection["score"])
                color = (0, 220, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                self._draw_box_text(
                    frame, detection["box"],
                    "{} {:.1f}%".format(self.data.display_name(detection["label"]), 100.0 * score),
                    color, scale=0.65,
                )
        if self.mode == "D" and self.digit_preview:
            left = width - 160
            cv2.rectangle(frame, (left, TOOLBAR_H), (width - 1, height - D_PANEL_H - 1),
                          (20, 20, 20), cv2.FILLED)
            preview = snapshot["preview"]
            if preview is not None:
                frame[TOOLBAR_H:TOOLBAR_H + 160, left:width] = cv2.resize(
                    preview, (160, 160), interpolation=cv2.INTER_NEAREST
                )
            lines = ["INPUT 32x32", snapshot["kind"].upper(),
                     "FROZEN" if snapshot["frozen"] else "LIVE",
                     "score {:.1f}%".format(100.0 * snapshot["score"]),
                     "margin {:.1f}%".format(100.0 * snapshot["margin"])]
            for index, line in enumerate(lines):
                cv2.putText(frame, line, (left + 5, 240 + index * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                            cv2.LINE_AA)
        if snapshot is not None:
            # Touch input is handled on this same thread, before the next
            # frame. Keep the candidate drawn here, never an unseen update.
            with self._prediction_lock:
                self._displayed_candidate = snapshot["candidate"]
        return frame





