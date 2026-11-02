#!/usr/bin/env python3
"""PC tests for D-mode routing and asynchronous inference (no PYNQ needed).

Run from pynq_app: python -m unittest -v test_operator_mode.py
The extractor below substitutes only the hardware TCU call. Preprocessing,
classification, UI events, generation checks, and the worker are real.
"""

import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from digit_preprocess import DigitPreprocessor
from pefsl_touch_ui import TouchBboxUI


class FakeExtractor:
    def __init__(self):
        self.feature = np.eye(80, dtype=np.float32)[2]
        self.images = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def extract(self, image):
        self.images.append(np.asarray(image).copy())
        feature = self.feature.copy()
        self.entered.set()
        if not self.release.wait(5.0):
            raise RuntimeError("test extractor was not released")
        return feature


class OperatorModeTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FakeExtractor()
        # These paths deliberately do not exist; the constructor only reads
        # caches. No test directories or user cache files need to be created.
        folder = Path(__file__).resolve().parent / ("_missing_test_" + uuid.uuid4().hex)
        self.ui = TouchBboxUI(
            self.extractor, folder / "learned.npz",
            digit_cache_path=folder / "missing_digits.npz",
            operator_cache_path=folder / "missing_operators.npz",
            predict_interval=60.0,
        )
        # Synthetic features have known nearest classes. These banks never
        # touch board caches, and each bank has its own feature centre.
        basis = np.eye(80, dtype=np.float32)
        with self.ui._prediction_lock:
            for index in range(10):
                self.ui.digit_data.add_shots(str(index), basis[index:index + 1])
            for index, label in enumerate(("plus", "sub", "mul", "div")):
                self.ui.operator_data.add_shots(label, basis[20 + index:21 + index])
            self.ui.digit_preprocessor = DigitPreprocessor(binary_history_len=5)
            self.ui.operator_preprocessor = DigitPreprocessor(binary_history_len=5)
            self.ui.digit_ready = True
            self.ui.operator_ready = True
            self.ui.digit_cache_status = "synthetic digit test data"
            self.ui.operator_cache_status = "synthetic operator test data"
        self.ui._top_action(180)

    def tearDown(self):
        self.extractor.release.set()
        stopped = self.ui.close(timeout=3.0)
        self.assertTrue(stopped, "worker must stop without hardware")

    @staticmethod
    def frame(blank=False):
        frame = np.full((480, 800, 3), 255, dtype=np.uint8)
        if not blank:
            frame[150:250, 195:205] = 0
            frame[195:205, 150:250] = 0
        return frame

    @staticmethod
    def touch(kind, x, y):
        return SimpleNamespace(kind=kind, x=x, y=y, points=1)

    def select_roi(self, box=(120, 120, 280, 280)):
        self.ui.handle_event(self.touch("down", box[0], box[1]))
        self.ui.handle_event(self.touch("move", box[2], box[3]))
        self.ui.handle_event(self.touch("up", box[2], box[3]))

    def wait_until(self, condition, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.ui._prediction_lock:
                if condition():
                    return
            time.sleep(0.005)
        self.fail("worker timeout; status={!r}".format(self.ui.status))

    def wait_idle(self):
        self.wait_until(lambda: (
            self.ui._digit_request is None and not self.ui._prediction_busy
        ))

    def submit(self, frame=None):
        with self.ui._prediction_lock:
            self.ui.predict_pending = True
        self.ui._schedule_digit_prediction(self.frame() if frame is None else frame)
        self.wait_idle()

    def test_manual_toggle_defaults_and_mode_reentry(self):
        self.assertEqual(self.ui.recognition_kind, "digits")
        self.ui._top_action(580)
        self.assertEqual(self.ui.recognition_kind, "operators")
        self.ui._top_action(580)
        self.assertEqual(self.ui.recognition_kind, "digits")
        self.ui._top_action(580)
        self.ui._top_action(30)
        self.assertEqual(self.ui.mode, "A")
        self.ui._top_action(180)
        self.assertEqual(self.ui.recognition_kind, "digits")

    def test_missing_operator_cache_does_not_disable_digits(self):
        self.ui.operator_ready = False
        self.ui._top_action(580)
        self.assertEqual(self.ui.recognition_kind, "digits")
        self.select_roi()
        self.submit()
        self.assertEqual(self.ui.digit_last_label, "2")

    def test_frozen_image_and_roi_survive_operator_toggle(self):
        self.ui._top_action(700)
        source = self.frame()
        self.ui.process_frame(source.copy())
        self.assertTrue(self.ui.freeze_active)
        np.testing.assert_array_equal(self.ui._frozen_frame, source)
        self.select_roi()
        box = self.ui.digit_box
        self.ui._top_action(580)
        self.assertEqual(self.ui.recognition_kind, "operators")
        self.assertTrue(self.ui.freeze_active)
        self.assertEqual(self.ui.digit_box, box)
        np.testing.assert_array_equal(self.ui._frozen_frame, source)
        self.extractor.feature = np.eye(80, dtype=np.float32)[20]
        self.ui.process_frame(np.zeros_like(source))
        self.wait_idle()
        self.assertEqual(self.ui.digit_last_label, "plus")
        np.testing.assert_array_equal(self.ui._frozen_frame, source)
        self.ui._top_action(700)
        self.assertFalse(self.ui.freeze_active)
        self.assertEqual(self.ui.recognition_kind, "operators")

    def test_independent_probability_banks_and_digit_regression(self):
        self.select_roi()
        self.submit()
        self.assertEqual(self.ui.digit_last_label, "2")
        original = self.ui.digit_last_probabilities.copy()
        self.assertEqual(original.shape, (10,))
        digit_mean = self.ui.digit_data.mean_feature().copy()
        self.assertEqual(len(self.ui.operator_prob_history), 0)

        self.ui._top_action(580)
        self.assertIsNone(self.ui.current_candidate)
        self.extractor.feature = np.eye(80, dtype=np.float32)[20]
        self.submit()
        self.assertEqual(self.ui.digit_last_label, "plus")
        self.assertEqual(self.ui.digit_last_probabilities.shape, (4,))
        self.assertTrue(all(row.shape == (4,) for row in self.ui.operator_prob_history))
        self.assertEqual(len(self.ui.digit_prob_history), 0)
        np.testing.assert_array_equal(self.ui.digit_data.mean_feature(), digit_mean)

        self.ui._top_action(580)
        self.extractor.feature = np.eye(80, dtype=np.float32)[2]
        self.submit()
        self.assertEqual(self.ui.digit_last_label, "2")
        np.testing.assert_allclose(self.ui.digit_last_probabilities, original)
        self.assertEqual(len(self.ui.operator_prob_history), 0)

    def test_invalid_roi_does_not_send_previous_good_image_to_tcu(self):
        self.select_roi()
        self.submit()
        self.assertIsNotNone(self.ui.current_candidate)
        count = len(self.extractor.images)
        self.submit(self.frame(blank=True))
        self.assertEqual(len(self.extractor.images), count)
        self.assertIsNone(self.ui.current_candidate)
        self.assertIsNone(self.ui.debug_digit_bin)
        self.assertFalse(self.ui.digit_preprocessor.last_valid)
        self.assertIn(self.ui.digit_last_label, (None, "?"))

    def test_inflight_digit_result_cannot_publish_after_switch(self):
        self.select_roi()
        self.extractor.entered.clear()
        self.extractor.release.clear()
        self.ui._schedule_digit_prediction(self.frame())
        self.assertTrue(self.extractor.entered.wait(3.0), "TCU substitute must start")
        self.ui._top_action(580)
        self.assertEqual(self.ui.recognition_kind, "operators")
        self.extractor.release.set()
        self.wait_idle()
        self.assertIsNone(self.ui.current_candidate)
        self.assertIsNone(self.ui.debug_digit_bin)
        self.assertIsNone(self.ui.digit_last_probabilities)
        self.assertEqual(len(self.ui.digit_prob_history), 0)
        self.assertEqual(len(self.ui.operator_prob_history), 0)
        self.extractor.feature = np.eye(80, dtype=np.float32)[20]
        self.submit()
        self.assertEqual(self.ui.digit_last_label, "plus")

    def test_starting_new_roi_immediately_invalidates_candidate(self):
        self.select_roi()
        self.submit()
        self.assertIsNotNone(self.ui.current_candidate)
        self.ui.handle_event(self.touch("down", 140, 140))
        self.assertTrue(self.ui.drawing)
        self.assertIsNone(self.ui.current_candidate)
        count = len(self.extractor.images)
        self.submit()
        self.assertEqual(len(self.extractor.images), count)

    def test_preview_reserves_sidebar_and_clamps_drag_outside_content(self):
        self.ui.handle_event(self.touch("down", 700, 200))
        self.assertFalse(self.ui.drawing, "preview sidebar is not a camera ROI")
        self.select_roi((600, 300, 780, 460))
        self.assertEqual(self.ui.digit_box[:2], (600, 300))
        self.assertLess(self.ui.digit_box[2], 640)
        self.assertLess(self.ui.digit_box[3], 364)

        self.ui.digit_preview = False
        self.select_roi((700, 200, 780, 300))
        self.assertEqual(self.ui.digit_box, (700, 200, 780, 300))
        self.ui._top_action(30)
        self.ui.digit_preview = True
        self.select_roi((700, 200, 780, 300))
        self.assertEqual(self.ui.pending_box, (700, 200, 780, 300))

    def test_preview_draw_does_not_cover_toolbar_and_off_restores_video(self):
        original = np.full((480, 800, 3), 17, dtype=np.uint8)
        without_image = self.ui.draw(original.copy())
        self.ui.debug_digit_bin = np.full((32, 32, 3), 239, dtype=np.uint8)
        with_image = self.ui.draw(original.copy())
        np.testing.assert_array_equal(with_image[:58], without_image[:58])
        self.assertTrue(np.all(with_image[58:218, 640:800] == 239))
        self.ui.digit_preview = False
        full_video = self.ui.draw(original.copy())
        np.testing.assert_array_equal(full_video[58:218, 640:800], original[58:218, 640:800])


if __name__ == "__main__":
    unittest.main()
