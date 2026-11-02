"""Cache isolation and evaluation checks; no board or TCU is required."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import digit_recognizer

from digit_preprocess import PREPROCESS_VERSION, preprocessing_backend
from digit_recognizer import (
    APP_DIR, CACHE_SCHEMAS, CLASS_SETS, OPERATOR_CLASSES,
    DigitSupportCache, _check_output_target, build_digit_cache,
    discover_support, evaluate_cache,
)


def sample_arrays(kind="operators", shots=2):
    labels = CLASS_SETS[kind]
    counts = np.full(len(labels), shots, dtype=np.int64)
    features = np.zeros((len(labels) * shots, 80), dtype=np.float32)
    for index in range(len(labels)):
        features[index * shots:(index + 1) * shots, index] = 2.0
        features[index * shots:(index + 1) * shots, -1] = np.arange(shots) * 0.1
    means = features.reshape(len(labels), shots, 80).mean(axis=1)
    metadata = {
        "schema": CACHE_SCHEMAS[kind], "feature_dim": 80,
        "class_set": kind, "preprocessing_version": PREPROCESS_VERSION,
        "preprocessing_backend": preprocessing_backend(),
    }
    return {
        "schema": np.asarray(CACHE_SCHEMAS[kind]),
        "labels": np.asarray(labels), "counts": counts,
        "features": features, "class_means": means,
        "feature_dim": np.asarray(80, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


class OperatorCacheTests(unittest.TestCase):
    def setUp(self):
        base = (APP_DIR.parent / "tmp" / "operator_cache_tests").resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(base))
        self.folder = Path(self.temporary.name).resolve()
        if base not in self.folder.parents:
            raise RuntimeError("test temporary path escaped workspace")
        self.addCleanup(self.temporary.cleanup)

    def write_cache(self, arrays, filename="cache.npz"):
        path = self.folder / filename
        np.savez_compressed(str(path), **arrays)
        return path

    def test_operator_discovery_accepts_sub_and_excludes_unrelated_png(self):
        for label in OPERATOR_CLASSES:
            folder = self.folder / label
            folder.mkdir()
            (folder / (label + "_0.png")).touch()
            (folder / "unrelated.png").touch()
            (folder / "0_0.png").touch()
        found = discover_support(self.folder, OPERATOR_CLASSES)
        self.assertEqual([label for label, _ in found], list(OPERATOR_CLASSES))
        self.assertEqual([paths[0].name for _, paths in found],
                         [label + "_0.png" for label in OPERATOR_CLASSES])
        self.assertTrue(all(len(paths) == 1 for _, paths in found))

    def test_legacy_digits_still_load_and_cannot_be_used_as_operators(self):
        arrays = sample_arrays("digits")
        # Match the metadata produced before operator support existed.
        arrays["metadata_json"] = np.asarray(json.dumps({
            "schema": CACHE_SCHEMAS["digits"], "feature_dim": 80,
        }))
        path = self.write_cache(arrays)
        self.assertEqual(DigitSupportCache.load(path).labels, CLASS_SETS["digits"])
        with self.assertRaisesRegex(ValueError, "schema"):
            DigitSupportCache.load(path, expected_kind="operators")

    def test_operator_cache_is_separate_and_preprocessing_version_is_checked(self):
        arrays = sample_arrays()
        path = self.write_cache(arrays)
        self.assertEqual(DigitSupportCache.load(path, "operators").kind, "operators")
        with self.assertRaisesRegex(ValueError, "schema"):
            DigitSupportCache.load(path)
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["preprocessing_version"] = "obsolete"
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "preprocessing version"):
            DigitSupportCache.load(self.write_cache(arrays), "operators")

    def test_bad_labels_counts_and_features_are_rejected(self):
        mutations = (
            ("labels", np.asarray(("plus", "sub", "mul", "plus"))),
            ("counts", np.asarray([2.5, 2, 2, 2])),
            ("counts", np.asarray([2, 2, 2, 1])),
            ("features", np.full((8, 80), np.nan, dtype=np.float32)),
            ("class_means", np.ones((4, 80), dtype=np.float32)),
            ("feature_dim", np.asarray(84)),
        )
        for name, value in mutations:
            with self.subTest(field=name):
                arrays = sample_arrays()
                arrays[name] = value
                with self.assertRaises(ValueError):
                    DigitSupportCache.load(self.write_cache(arrays), "operators")

    def test_cross_kind_and_learned_output_overwrites_are_refused(self):
        for name in ("digit_support_r17.npz", "pefsl_learned.npz"):
            with self.assertRaises(ValueError):
                _check_output_target(self.folder / name, "operators")
        digit_path = self.write_cache(sample_arrays("digits"), "custom_digits.npz")
        before = digit_path.read_bytes()
        with self.assertRaises(ValueError):
            _check_output_target(digit_path, "operators")
        self.assertEqual(digit_path.read_bytes(), before)

    def test_real_support_build_preserves_four_groups_and_twenty_samples(self):
        class FakeExtractor:
            calls = 0

            def extract(self, image):
                self.calls += 1
                if image.shape != (32, 32, 3) or image.dtype != np.uint8:
                    raise AssertionError("invalid model input")
                # Test fixture only: never use these as board features.
                return np.full(80, float(self.calls), dtype=np.float32)

        extractor = FakeExtractor()
        path = build_digit_cache(
            extractor, APP_DIR / "operator_support", self.folder / "operators.npz",
            class_set="operators", verbose=False,
        )
        cache = DigitSupportCache.load(path, "operators")
        self.assertEqual(cache.labels, OPERATOR_CLASSES)
        self.assertEqual(cache.counts.tolist(), [5, 5, 5, 5])
        self.assertEqual(extractor.calls, 20)
        self.assertEqual(cache.features.shape, (20, 80))

    def test_self_match_and_leave_one_out_use_train_only_means(self):
        arrays = sample_arrays()
        # Distinct random samples make self-match and leave-one-out differ;
        # this catches accidentally retaining the held-out query in a mean.
        features = np.random.RandomState(19).normal(size=(8, 80)).astype(np.float32)
        arrays["features"] = features
        arrays["class_means"] = features.reshape(4, 2, 80).mean(axis=1)
        cache = DigitSupportCache.load(self.write_cache(arrays), "operators")
        own = evaluate_cache(cache)
        loo = evaluate_cache(cache, leave_one_out=True)
        self.assertEqual(own["total"], 8)
        self.assertEqual(loo["total"], 8)
        expected = np.zeros((4, 4), dtype=int)
        for held in range(8):
            train = [index for index in range(8) if index != held]
            center = features[train].mean(axis=0)
            query = features[held] - center
            query /= max(np.linalg.norm(query), 1e-8)
            cosines = []
            for group in range(4):
                rows = [index for index in train if index // 2 == group]
                prototype = features[rows].mean(axis=0) - center
                prototype /= max(np.linalg.norm(prototype), 1e-8)
                cosines.append(float(np.dot(query, prototype)))
            expected[held // 2, int(np.argmax(cosines))] += 1
        self.assertEqual(loo["confusion"], expected.tolist())
        self.assertNotEqual(own["confusion"], loo["confusion"])

    def test_smoke_cache_cannot_claim_leave_one_out_result(self):
        cache = DigitSupportCache.load(self.write_cache(sample_arrays(shots=1)), "operators")
        with self.assertRaisesRegex(ValueError, "at least 2"):
            evaluate_cache(cache, leave_one_out=True)

    def test_evaluation_uses_native_indices_with_32bit_repeat(self):
        cache = DigitSupportCache.load(self.write_cache(sample_arrays()), "operators")
        self.assertEqual(cache.counts.dtype, np.dtype("int64"))
        original_counts = cache.counts.copy()
        real_repeat = np.repeat

        def repeat32(values, repeats):
            # Emulate armv7's rejection without needing a 32-bit Python.
            if np.asarray(repeats).dtype != np.dtype("int32"):
                raise TypeError("Cannot safely cast repeat counts to int32")
            self.assertEqual(np.asarray(values).dtype, np.dtype("int32"))
            return real_repeat(values, repeats)

        with mock.patch.object(digit_recognizer.np, "intp", np.int32), \
                mock.patch.object(digit_recognizer.np, "repeat", side_effect=repeat32) as repeated:
            own = evaluate_cache(cache)
            loo = evaluate_cache(cache, leave_one_out=True)
        self.assertEqual(repeated.call_count, 2)
        self.assertEqual(own["correct"], 8)
        self.assertEqual(loo["correct"], 8)
        np.testing.assert_array_equal(cache.counts, original_counts)
        self.assertEqual(cache.counts.dtype, np.dtype("int64"))

    def test_evaluation_rejects_32bit_count_overflow_before_repeat(self):
        # No huge allocation: malformed stand-ins exercise the range guard.
        # The second case checks total length even when each count fits.
        for counts in ([2 ** 31, 1], [2 ** 30, 2 ** 30]):
            with self.subTest(counts=counts):
                cache = SimpleNamespace(
                    labels=("plus", "sub"), counts=np.asarray(counts, dtype=np.int64),
                    features=np.zeros((0, 80), dtype=np.float32),
                )
                with mock.patch.object(digit_recognizer.np, "intp", np.int32), \
                        mock.patch.object(digit_recognizer.np, "repeat") as repeated:
                    with self.assertRaisesRegex(ValueError, "platform index range"):
                        evaluate_cache(cache)
                    repeated.assert_not_called()

    def test_import_and_inspect_do_not_load_tcu_backend(self):
        path = self.write_cache(sample_arrays())
        code = (
            "import sys; import digit_recognizer as r; "
            "r.main(['inspect', '--class-set', 'operators', '--cache', sys.argv[1]]); "
            "assert 'tcu_bbox_backend' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(path)], cwd=str(APP_DIR),
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("4 classes / 8 images / 80D", completed.stdout)


if __name__ == "__main__":
    unittest.main()
