#!/usr/bin/env python3
"""Build, inspect and evaluate separate r17 digit/operator support caches.

This module deliberately does not import or modify ``LearnedData``.  The
existing A/B data file and classifier remain independent.  The builder runs
one support image at a time through the current r17
``TcuFeatureExtractor.extract`` path, which preserves the deployed
uint8-BGR -> 12-lane -> raw84 -> feature80 boundary.

Typical board command (run from ``pynq_app``)::

    python3 digit_recognizer.py build

The default paths point at the checked-in r17 bitstream, tmodel, support set,
and output cache.  Use ``--limit-per-class 1`` for a one-image smoke test
before building the complete cache.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR / "tcu_deploy") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "tcu_deploy"))

from digit_preprocess import (
    DigitPreprocessor, PREPROCESS_VERSION, preprocessing_backend,
)


CACHE_SCHEMA = "digit_support_r17_v2"
OPERATOR_CACHE_SCHEMA = "operator_support_r17_v1"
FEATURE_DIM = 80
DIGIT_CLASSES = tuple(str(value) for value in range(10))
OPERATOR_CLASSES = ("plus", "sub", "mul", "div")
DEFAULT_CLASSES = DIGIT_CLASSES
CLASS_SETS = {"digits": DIGIT_CLASSES, "operators": OPERATOR_CLASSES}
CACHE_SCHEMAS = {"digits": CACHE_SCHEMA, "operators": OPERATOR_CACHE_SCHEMA}
_DEBUG_BINARY_PNG = re.compile(r"^debug_binary_[0-9]+\.png$")


def _class_set(kind):
    if kind not in CLASS_SETS:
        raise ValueError("unknown class set: {!r}".format(kind))
    return CLASS_SETS[kind]


def _default_cache(kind):
    return APP_DIR / ("operator_support_r17.npz" if kind == "operators"
                      else "digit_support_r17.npz")


def _check_output_target(cache_path, kind):
    """Catch accidental overwrites of the other classifier or A/B data."""
    _class_set(kind)
    path = Path(cache_path).resolve()
    other = "digits" if kind == "operators" else "operators"
    if path.name.lower() in (
        _default_cache(other).name.lower(), "pefsl_learned.npz",
    ):
        raise ValueError("refusing {} cache output to {}".format(kind, path))
    if path.exists():
        try:
            with np.load(str(path), allow_pickle=False) as data:
                schema = _scalar_text(data["schema"])
        except Exception as error:
            raise ValueError("existing output is not a support cache: {} ({})".format(
                path, error
            ))
        accepted = (CACHE_SCHEMAS[kind],)
        if kind == "digits":
            accepted += ("digit_support_r17_v1",)
        if schema not in accepted:
            raise ValueError("refusing to overwrite {} with {} cache".format(schema, kind))
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(value)


def _validate_feature(feature, source):
    value = np.asarray(feature, dtype=np.float32).reshape(-1)
    if value.size != FEATURE_DIM:
        raise RuntimeError(
            "{} produced {} features; expected {}".format(
                source, value.size, FEATURE_DIM
            )
        )
    if not np.isfinite(value).all():
        raise RuntimeError("{} produced non-finite features".format(source))
    return value


def _formal_images(class_dir, label):
    """Return PNGs matching the selected label or debug_binary in stable order.

    The PC reference loader accepts every PNG in a digit directory.  The
    updated support set uses ``debug_binary_*.png`` for classes 2--9, while
    classes 0 and 1 retain ``<label>_<index>.png`` names.  Accept only these
    two known formats in explicit class directories.  When both formats are present, the
    ``debug_binary`` set takes precedence so stale formal files from an older
    deployment do not get mixed into the new cache.
    """

    pattern = re.compile(r"^{}_[0-9]+\.png$".format(re.escape(label)))
    formal_files = []
    debug_files = []
    for path in sorted(class_dir.glob("*.png"), key=lambda value: value.name):
        if pattern.match(path.name):
            formal_files.append(path)
        elif _DEBUG_BINARY_PNG.match(path.name):
            debug_files.append(path)
    # If an updated directory contains debug_binary samples, prefer that
    # complete set over stale formal files left by an older deployment.
    return debug_files if debug_files else formal_files


def discover_support(support_dir, classes=DEFAULT_CLASSES):
    """Return ``[(label, [png paths...]), ...]`` for explicit known labels."""

    classes = tuple(str(label) for label in classes)
    known = set(DIGIT_CLASSES + OPERATOR_CLASSES)
    if not classes or len(set(classes)) != len(classes) or not set(classes) <= known:
        raise ValueError("support classes must be unique known labels")

    support_dir = Path(support_dir).resolve()
    if not support_dir.is_dir():
        raise FileNotFoundError("support directory not found: {}".format(support_dir))

    result = []
    for label in classes:
        class_dir = support_dir / str(label)
        if not class_dir.is_dir():
            raise FileNotFoundError("missing class directory: {}".format(class_dir))
        paths = _formal_images(class_dir, str(label))
        if not paths:
            raise ValueError("no supported PNG images for class {}".format(label))
        result.append((str(label), paths))
    return result


def _read_bgr(path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read support PNG files: {}".format(exc))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to read support image: {}".format(path))
    return image


def build_digit_cache(
    extractor,
    support_dir,
    cache_path,
    tmodel_path=None,
    bitstream_path=None,
    classes=None,
    limit_per_class=0,
    preprocessor=None,
    verbose=True,
    class_set="digits",
):
    """Extract all selected support images and atomically write an NPZ cache."""

    selected_classes = _class_set(class_set)
    if classes is not None and tuple(classes) != selected_classes:
        raise ValueError("classes must match {} in canonical order".format(class_set))
    classes = selected_classes
    if int(limit_per_class) < 0:
        raise ValueError("limit_per_class must be >= 0")
    cache_path = _check_output_target(cache_path, class_set)
    schema = CACHE_SCHEMAS[class_set]
    preprocessor = preprocessor or DigitPreprocessor(binary_history_len=1)
    discovered = discover_support(support_dir, classes=classes)
    labels = []
    features = []
    counts = []
    support_files = []

    for label, paths in discovered:
        selected = paths[: int(limit_per_class)] if int(limit_per_class) > 0 else paths
        class_features = []
        labels.append(label)
        for path in selected:
            image = _read_bgr(path)
            bgr32 = preprocessor.support_to_bgr(image)
            if not getattr(preprocessor, "last_valid", True):
                raise RuntimeError("invalid support image {}: {}".format(
                    path, getattr(preprocessor, "last_invalid_reason", "invalid input")
                ))
            if bgr32.shape != (32, 32, 3) or bgr32.dtype != np.uint8:
                raise RuntimeError(
                    "preprocessor output for {} is {}, {}".format(
                        path, bgr32.shape, bgr32.dtype
                    )
                )
            feature = _validate_feature(extractor.extract(bgr32), str(path))
            features.append(feature)
            class_features.append(feature)
            support_files.append(str(path))
            if verbose:
                print(
                    "{} {}: {} -> feature {}".format(
                        class_set, label, path.name, feature.shape
                    ),
                    flush=True,
                )
        if not class_features:
            raise RuntimeError("no selected samples for class {}".format(label))
        counts.append(len(class_features))

    feature_matrix = np.stack(features, axis=0).astype(np.float32)
    label_array = np.asarray(labels)
    count_array = np.asarray(counts, dtype=np.int64)
    if int(count_array.sum()) != int(feature_matrix.shape[0]):
        raise RuntimeError("cache count does not match feature count")
    if feature_matrix.shape[1] != FEATURE_DIM:
        raise RuntimeError("cache feature dimension is not 80")

    class_means = []
    offset = 0
    for count in count_array:
        count = int(count)
        class_means.append(feature_matrix[offset:offset + count].mean(axis=0))
        offset += count
    class_means = np.stack(class_means, axis=0).astype(np.float32)

    metadata = {
        "schema": schema,
        "class_set": class_set,
        "preprocessing_version": PREPROCESS_VERSION,
        "feature_dim": FEATURE_DIM,
        "preprocessing_backend": preprocessing_backend(),
        "support_dir": str(Path(support_dir).resolve()),
        "tmodel": str(Path(tmodel_path).resolve()) if tmodel_path else "",
        "bitstream": str(Path(bitstream_path).resolve()) if bitstream_path else "",
    }
    if tmodel_path and Path(tmodel_path).is_file():
        metadata["tmodel_sha256"] = _sha256(tmodel_path)
    if bitstream_path and Path(bitstream_path).is_file():
        metadata["bitstream_sha256"] = _sha256(bitstream_path)

    cache_path = Path(cache_path).resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary),
        schema=np.asarray(schema),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        labels=label_array,
        counts=count_array,
        features=feature_matrix,
        class_means=class_means,
        support_files=np.asarray(support_files),
        preprocessing_backend=np.asarray(preprocessing_backend()),
        feature_dim=np.asarray(FEATURE_DIM, dtype=np.int64),
    )
    os.replace(str(temporary), str(cache_path))
    if verbose:
        print(
            "wrote {}: {} classes / {} images / {}D".format(
                cache_path, len(count_array), feature_matrix.shape[0], FEATURE_DIM
            ),
            flush=True,
        )
    return cache_path


class DigitSupportCache(object):
    """Validated digit or operator cache; the expected kind is explicit."""

    def __init__(self, labels, counts, features, class_means, metadata,
                 expected_kind="digits"):
        expected_labels = _class_set(expected_kind)
        if np.asarray(labels).ndim != 1:
            raise ValueError("cache labels must be one-dimensional")
        self.labels = tuple(str(value) for value in labels)
        if self.labels != expected_labels:
            raise ValueError("{} cache labels must be {} in this order; got {}".format(
                expected_kind, expected_labels, self.labels
            ))
        raw_counts = np.asarray(counts)
        if raw_counts.ndim != 1 or raw_counts.dtype.kind not in "iu":
            raise ValueError("cache counts must be a one-dimensional integer array")
        self.counts = raw_counts.astype(np.int64)
        self.features = np.asarray(features, dtype=np.float32)
        self.class_means = np.asarray(class_means, dtype=np.float32)
        if not isinstance(metadata, dict):
            raise ValueError("cache metadata must be an object")
        self.metadata = dict(metadata)
        self.kind = expected_kind
        self.schema = CACHE_SCHEMAS[expected_kind]
        self._validate()

    def _validate(self):
        if len(self.labels) != len(self.counts):
            raise ValueError("cache labels/counts mismatch")
        if self.features.ndim != 2 or self.features.shape[1] != FEATURE_DIM:
            raise ValueError("cache features must have shape (N, 80)")
        if self.class_means.shape != (len(self.labels), FEATURE_DIM):
            raise ValueError("cache class_means shape is invalid")
        if int(self.counts.sum()) != int(self.features.shape[0]):
            raise ValueError("cache counts do not match features")
        if (self.counts <= 0).any() or not np.isfinite(self.features).all():
            raise ValueError("cache contains invalid feature data")
        if not np.isfinite(self.class_means).all():
            raise ValueError("cache contains invalid class means")
        if self.metadata.get("schema", self.schema) != self.schema:
            raise ValueError("cache metadata schema mismatch")
        if self.metadata.get("class_set", self.kind) != self.kind:
            raise ValueError("cache metadata class_set mismatch")
        if self.metadata.get("feature_dim", FEATURE_DIM) != FEATURE_DIM:
            raise ValueError("cache metadata feature dimension mismatch")
        # Legacy digit v2 caches predate explicit preprocessing versions.
        # Keep them usable; operator v1 is new and requires exact metadata.
        if self.kind == "operators":
            if self.metadata.get("preprocessing_version") != PREPROCESS_VERSION:
                raise ValueError("operator preprocessing version mismatch; rebuild cache")
            if self.metadata.get("preprocessing_backend") != preprocessing_backend():
                raise ValueError("operator preprocessing backend mismatch; rebuild cache")
        offset = 0
        for index, count in enumerate(self.counts):
            end = offset + int(count)
            mean = self.features[offset:end].mean(axis=0)
            if not np.allclose(self.class_means[index], mean, rtol=1e-5, atol=1e-6):
                raise ValueError("cache class means do not match stored features")
            offset = end

    @classmethod
    def load(cls, path, expected_kind="digits"):
        _class_set(expected_kind)
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError("{} cache not found: {}".format(expected_kind, path))
        with np.load(str(path), allow_pickle=False) as data:
            schema = _scalar_text(data["schema"])
            expected_schema = CACHE_SCHEMAS[expected_kind]
            if schema != expected_schema:
                raise ValueError(
                    "unsupported {} cache schema {!r}; expected {!r}".format(
                        expected_kind, schema, expected_schema
                    )
                )
            metadata_text = _scalar_text(data["metadata_json"])
            metadata = json.loads(metadata_text)
            if np.asarray(data["feature_dim"]).size != 1 or int(
                np.asarray(data["feature_dim"]).reshape(-1)[0]
            ) != FEATURE_DIM:
                raise ValueError("cache feature_dim must be 80")
            return cls(
                data["labels"],
                data["counts"],
                data["features"],
                data["class_means"],
                metadata,
                expected_kind=expected_kind,
            )

    def summary(self):
        return "{} classes / {} images / {}D".format(
            len(self.labels), self.features.shape[0], self.features.shape[1]
        )


def _ncm_probabilities(query, features, target_indices, class_count):
    """Match PefslNcmClassifier, deriving all statistics from this train fold."""
    mean = features.mean(axis=0)
    prototypes = np.stack([
        features[target_indices == index].mean(axis=0)
        for index in range(class_count)
    ], axis=0)
    q = np.asarray(query, dtype=np.float32)[None, :] - mean
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    centers = prototypes - mean
    centers /= np.maximum(np.linalg.norm(centers, axis=-1, keepdims=True), 1e-8)
    distances = np.linalg.norm(centers - q, axis=1)
    logits = -20.0 * distances
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    return probabilities / np.maximum(np.sum(probabilities), 1e-8)


def evaluate_cache(cache, leave_one_out=False, min_score=0.0, min_margin=0.0):
    """Evaluate cached features; neither result measures live-camera accuracy."""
    if not 0.0 <= min_score <= 1.0 or not 0.0 <= min_margin <= 1.0:
        raise ValueError("evaluation thresholds must be between 0 and 1")
    if leave_one_out and (cache.counts < 2).any():
        raise ValueError("leave-one-out requires at least 2 images per class; use full cache")
    class_count = len(cache.labels)
    # NPZ counts remain int64 on every platform. On armv7, NumPy repeat
    # requires intp (int32) repeats and rejects the implicit safe cast from
    # int64 even when all values are small. Check before narrowing, using
    # Python integers so the sum itself cannot overflow on either platform.
    counts = [int(value) for value in cache.counts]
    index_limit = int(np.iinfo(np.intp).max)
    sample_count = sum(counts)
    if (len(counts) != class_count or any(value <= 0 for value in counts)
            or any(value > index_limit for value in counts)
            or sample_count > index_limit):
        raise ValueError("cache counts exceed the platform index range or are invalid")
    if sample_count != len(cache.features):
        raise ValueError("cache counts do not match features")
    repeats = np.asarray(counts, dtype=np.intp)
    targets = np.repeat(np.arange(class_count, dtype=np.intp), repeats)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    accepted = np.zeros(class_count, dtype=np.int64)
    accepted_correct = np.zeros(class_count, dtype=np.int64)
    samples = []
    for row, query in enumerate(cache.features):
        train_mask = np.ones(len(targets), dtype=bool)
        if leave_one_out:
            train_mask[row] = False
        probabilities = _ncm_probabilities(
            query, cache.features[train_mask], targets[train_mask], class_count
        )
        order = np.argsort(probabilities)
        predicted = int(order[-1])
        truth = int(targets[row])
        score = float(probabilities[predicted])
        margin = score - float(probabilities[order[-2]])
        keep = score >= min_score and margin >= min_margin
        confusion[truth, predicted] += 1
        accepted[truth] += int(keep)
        accepted_correct[truth] += int(keep and predicted == truth)
        samples.append({
            "row": row, "label": cache.labels[truth],
            "predicted": cache.labels[predicted], "score": score,
            "margin": margin, "accepted": bool(keep),
        })
    return {
        "method": "leave-one-out" if leave_one_out else "self-match",
        "labels": list(cache.labels), "counts": cache.counts.tolist(),
        "confusion": confusion.tolist(),
        "correct": int(np.trace(confusion)), "total": len(targets),
        "accepted": accepted.tolist(),
        "accepted_correct": accepted_correct.tolist(), "samples": samples,
    }


def _print_evaluation(report, min_score, min_margin):
    print("{}: {}/{} top-1 correct ({:.1f}%)".format(
        report["method"], report["correct"], report["total"],
        100.0 * report["correct"] / report["total"],
    ))
    print("confusion rows=true, columns=predicted: {}".format(
        ", ".join(report["labels"])
    ))
    for label, row in zip(report["labels"], report["confusion"]):
        print("  {}: {}".format(label, " ".join(str(value) for value in row)))
    print("gates: score>={:.3f}, margin>={:.3f} (diagnostic only)".format(
        min_score, min_margin
    ))
    for index, label in enumerate(report["labels"]):
        total = report["counts"][index]
        accepted = report["accepted"][index]
        good = report["accepted_correct"][index]
        print("  {}: samples={} accepted_correct={} accepted_wrong={} rejected={}".format(
            label, total, good, accepted - good, total - accepted
        ))
    for sample in report["samples"]:
        print("  row={row} true={label} pred={predicted} score={score:.4f} "
              "margin={margin:.4f} accepted={accepted}".format(**sample))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build", help="build a digit or operator cache on the board")
    build.add_argument("--class-set", choices=tuple(CLASS_SETS), default="digits")
    build.add_argument(
        "--bitstream",
        default=str(PROJECT_DIR / "artifacts" / "pefsl_mix_plus_touch.bit"),
    )
    build.add_argument(
        "--tmodel",
        default=str(
            PROJECT_DIR
            / "tcu_deploy"
            / "resnet9_strided_16fmaps_onnx_custom_perf.tmodel"
        ),
    )
    build.add_argument(
        "--tcu-path", default=str(PROJECT_DIR / "tcu_deploy")
    )
    build.add_argument("--support-dir", help="default follows --class-set")
    build.add_argument("--output", help="default follows --class-set")
    build.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="use at most this many images per class; 0 means all",
    )
    build.add_argument("--debug-tcu", action="store_true")
    for name in ("inspect", "evaluate"):
        command = subparsers.add_parser(name, help="{} a cache without TCU hardware".format(name))
        command.add_argument("--class-set", choices=tuple(CLASS_SETS), default="digits")
        command.add_argument("--cache", help="default follows --class-set")
        if name == "evaluate":
            command.add_argument("--method", choices=("both", "self", "loo"), default="both")
            command.add_argument("--min-score", type=float, default=0.0)
            command.add_argument("--min-margin", type=float, default=0.0)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command is None:
        _parser().print_help()
        return 2
    if args.command in ("inspect", "evaluate"):
        cache_path = Path(args.cache or _default_cache(args.class_set)).resolve()
        cache = DigitSupportCache.load(cache_path, expected_kind=args.class_set)
        print("cache: {}".format(cache_path))
        print("schema: {}".format(cache.schema))
        print("kind: {}".format(cache.kind))
        print(cache.summary())
        print("labels: {}".format(list(cache.labels)))
        print("counts: {}".format(cache.counts.tolist()))
        print("finite features: {}".format(bool(np.isfinite(cache.features).all())))
        print("preprocessing version: {}".format(
            cache.metadata.get("preprocessing_version", "legacy digit v2 (no version field)")
        ))
        if args.command == "evaluate":
            print("Feature-cache diagnostics only; not live-camera recognition accuracy.")
            if args.method in ("both", "self"):
                report = evaluate_cache(cache, False, args.min_score, args.min_margin)
                _print_evaluation(report, args.min_score, args.min_margin)
            if args.method in ("both", "loo"):
                report = evaluate_cache(cache, True, args.min_score, args.min_margin)
                _print_evaluation(report, args.min_score, args.min_margin)
        return 0
    if args.limit_per_class < 0:
        raise SystemExit("--limit-per-class must be >= 0")

    support_dir = Path(args.support_dir or APP_DIR / (
        "operator_support" if args.class_set == "operators" else "digit_support"
    )).resolve()
    output = _check_output_target(args.output or _default_cache(args.class_set), args.class_set)
    # Fail before loading hardware if the supplied directory is incomplete.
    discover_support(support_dir, CLASS_SETS[args.class_set])

    bitstream = Path(args.bitstream).resolve()
    tmodel = Path(args.tmodel).resolve()
    tcu_path = Path(args.tcu_path).resolve()
    required = (
        bitstream,
        bitstream.with_suffix(".hwh"),
        tmodel,
        tmodel.with_suffix(".tprog"),
        tmodel.with_suffix(".tdata"),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required file(s): {}".format(", ".join(missing)))

    from tcu_bbox_backend import TcuFeatureExtractor, load_overlay

    print("Loading overlay: {}".format(bitstream), flush=True)
    overlay = load_overlay(bitstream)
    extractor = TcuFeatureExtractor(
        overlay, tmodel, tcu_path=tcu_path, debug=args.debug_tcu
    )
    print(
        "Building {} cache from {} using {}".format(
            args.class_set, support_dir, preprocessing_backend()
        ),
        flush=True,
    )
    build_digit_cache(
        extractor,
        support_dir,
        output,
        tmodel_path=tmodel,
        bitstream_path=bitstream,
        limit_per_class=args.limit_per_class,
        class_set=args.class_set,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
