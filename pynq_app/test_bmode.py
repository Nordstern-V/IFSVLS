"""B-mode host regressions: real stores/matcher/worker, fake hardware only.

Run with ``python -m unittest -v test_bmode`` from this directory.
All written data lives in TemporaryDirectory; board caches are never changed.
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pefsl_identity import prepare_reference
from pefsl_scan import color_histogram
from pefsl_touch_ui import LearnedData, TouchBboxUI


def patterned(seed=4):
    cells = np.random.RandomState(seed).randint(25, 190, (12, 12)).astype(np.uint8)
    gray = np.repeat(np.repeat(cells, 4, axis=0), 4, axis=1)
    return np.repeat(gray[:, :, None], 3, axis=2)


FEATURE = np.eye(80, dtype=np.float32)[0]
BOX = (120, 120, 168, 168)


def test_directory():
    # Desktop sandbox TEMP can resolve to an AppContainer folder that permits
    # creating a directory but denies ordinary files inside it. A sibling
    # temporary directory is portable to the writable PYNQ deployment too.
    return tempfile.TemporaryDirectory(prefix="_bmode_test_", dir=str(Path(__file__).resolve().parent))


def registered(data=None):
    data = LearnedData() if data is None else data
    view = patterned()
    data.add_shots("class1", np.stack((FEATURE, FEATURE)), box_size=(48, 48),
                   hist=color_histogram(view), reference_views=[view])
    return data


class LearnedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = test_directory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_new_metadata_roundtrip_retains_reference_identity(self):
        original = registered()
        path = self.folder / "new.npz"
        self.assertTrue(original.save(path)[0])
        restored = LearnedData()
        self.assertTrue(restored.load(path)[0])
        self.assertEqual(restored.labels, original.labels)
        np.testing.assert_allclose(restored.shots[0], original.shots[0])
        np.testing.assert_array_equal(restored.reference_views[0][0], original.reference_views[0][0])
        np.testing.assert_allclose(restored.class_shapes[0], original.class_shapes[0])
        self.assertTrue(restored.raw_cosine[0])
        self.assertEqual(restored.bmode_matcher().recognize(FEATURE, patterned())["label"], "class1")

    def test_legacy_file_loads_and_can_be_resaved(self):
        path = self.folder / "legacy.npz"
        np.savez(str(path), labels=np.asarray(["class1"]), counts=np.asarray([2]),
                 features=np.stack((FEATURE, FEATURE)), scan_sizes=np.asarray([[48, 48]]),
                 color_hists=color_histogram(patterned())[None, :])
        data = LearnedData()
        self.assertTrue(data.load(path)[0])
        self.assertEqual(data.reference_views, [[]])
        self.assertEqual(data.raw_cosine, [False])
        self.assertEqual(data.bmode_matcher().recognize(FEATURE, patterned())["label"], "class1")
        self.assertTrue(data.save(self.folder / "legacy_resaved.npz")[0])
        restored = LearnedData()
        self.assertTrue(restored.load(self.folder / "legacy_resaved.npz")[0])
        self.assertEqual(restored.reference_views, [[]])

    def _valid_payload(self):
        path = self.folder / "source.npz"
        registered().save(path)
        with np.load(str(path), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    def test_malformed_optional_metadata_is_rejected_without_erasing_bank(self):
        original = self._valid_payload()
        malformed = []
        incomplete = dict(original)
        del incomplete["b_reference_images"]
        malformed.append(("incomplete", incomplete))
        for name, value in (
                ("bmode_schema", np.asarray(2)),
                ("b_reference_counts", np.asarray([5])),
                ("b_reference_counts", np.asarray([1.0])),
                ("b_reference_images", np.zeros((1, 32, 32, 3), np.uint8)),
                ("b_reference_images", np.zeros((1, 48, 48, 3), np.float32)),
                ("b_class_shapes", np.asarray([[np.nan, 0]], np.float32)),
                ("b_raw_cosine", np.asarray([7]))):
            payload = dict(original)
            payload[name] = value
            malformed.append((name + "_" + str(value.shape), payload))
        for name, payload in malformed:
            with self.subTest(name=name):
                data = registered()
                path = self.folder / "bad.npz"
                np.savez(str(path), **payload)
                ok, _ = data.load(path)
                self.assertFalse(ok)
                self.assertEqual(data.labels, ["class1"])
                self.assertEqual(len(data.reference_views[0]), 1)

    def test_schema_is_a_version_number_not_a_truncated_float(self):
        payload = self._valid_payload()
        for value in (1.5, float("nan")):
            with self.subTest(value=value):
                payload["bmode_schema"] = np.asarray(value)
                path = self.folder / "bad_schema.npz"
                np.savez(str(path), **payload)
                ok, _ = LearnedData().load(path)
                self.assertFalse(ok)

    def test_compiled_matcher_is_reused_then_refreshed_after_learning(self):
        data = registered()
        first = data.bmode_matcher()
        self.assertIs(first, data.bmode_matcher())
        data.add_shots("class2", np.eye(80, dtype=np.float32)[1:2],
                       reference_views=[patterned(12)])
        refreshed = data.bmode_matcher()
        self.assertIsNot(first, refreshed)
        self.assertEqual(refreshed.labels, ["class1", "class2"])
        self.assertEqual(first.labels, ["class1"])
        data.reset()
        self.assertEqual(data.bmode_matcher().labels, [])

    def test_legacy_center_survives_new_raw_class_and_save_reload(self):
        data = LearnedData()
        data.add_shots("legacy1", np.eye(80, dtype=np.float32)[0:1])
        data.add_shots("legacy2", np.eye(80, dtype=np.float32)[1:2])
        before = data.bmode_matcher()
        old_result = before.recognize(FEATURE, patterned())
        self.assertEqual(old_result["label"], "legacy1")
        # A very large new raw prototype would otherwise recenter the old
        # feature space, even though neither legacy identity was re-learned.
        data.add_shots("new", 100*np.eye(80, dtype=np.float32)[2:3],
                       reference_views=[patterned(12)])
        after = data.bmode_matcher()
        np.testing.assert_allclose(after.baseline, before.baseline)
        np.testing.assert_allclose(after.banks[0], before.banks[0])
        np.testing.assert_allclose(after.banks[1], before.banks[1])
        self.assertEqual(data.raw_cosine, [False, False, True])
        path = self.folder / "mixed.npz"
        self.assertTrue(data.save(path)[0])
        restored = LearnedData()
        self.assertTrue(restored.load(path)[0])
        matcher = restored.bmode_matcher()
        np.testing.assert_allclose(matcher.baseline, before.baseline)
        np.testing.assert_allclose(matcher.banks[0], before.banks[0])
        result = matcher.recognize(FEATURE, patterned())
        self.assertEqual(result["label"], old_result["label"])
        self.assertAlmostEqual(result["score"], old_result["score"], places=6)


class MatcherTests(unittest.TestCase):
    def test_single_registered_class_keeps_raw_direction_and_recognizes_view(self):
        matcher = registered().bmode_matcher()
        result = matcher.recognize(FEATURE, patterned())
        self.assertEqual(result["label"], "class1", result)
        self.assertGreater(result["score"], .9)

    def test_background_and_wrong_spatial_layout_do_not_inherit_identity(self):
        matcher = registered().bmode_matcher()
        for image in (np.full((48, 48, 3), 120, np.uint8), patterned(15)):
            with self.subTest(flat=bool(image.std() == 0)):
                # Deliberately identical CNN output: pixel evidence must veto it.
                result = matcher.recognize(FEATURE, image)
                self.assertIsNone(result["label"], result)
        result = matcher.recognize(np.eye(80, dtype=np.float32)[1], patterned())
        self.assertIsNone(result["label"], result)


class FakeExtractor:
    def __init__(self):
        self.feature = FEATURE.copy()
        self.error = None
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def extract(self, image):
        self.calls.append(np.asarray(image).copy())
        feature = self.feature.copy()
        self.entered.set()
        if not self.release.wait(4.0):
            raise RuntimeError("test extractor was not released")
        if self.error is not None:
            raise self.error
        return feature


# The C124-only UI worker expectations are archived in
# historical_tests/test_bmode_c124.py.txt. Active A/B regression coverage
# compares the restored worker against the r17 reference in
# test_ab_r17_restore.py. Storage/matcher compatibility tests above remain.


if __name__ == "__main__":
    unittest.main()
