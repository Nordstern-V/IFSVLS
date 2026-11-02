#!/usr/bin/env python3
"""D calculator interaction tests using the real worker and a fake TCU.

Run: python -m unittest discover -s pynq_app -p test_calculator_ui.py -v
No board files, caches, overlays or camera devices are changed.
"""

import threading
import time
import unittest
import uuid
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import pefsl_touch_ui as ui_module
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
        self.images.append(image.copy())
        feature = self.feature.copy()
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("test extractor not released")
        return feature


class CalculatorUiTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FakeExtractor()
        missing = Path(__file__).resolve().parent / ("_absent_calc_" + uuid.uuid4().hex)
        self.ui = TouchBboxUI(
            self.extractor, missing / "learned.npz",
            digit_cache_path=missing / "digits.npz",
            operator_cache_path=missing / "operators.npz",
            predict_interval=60.0,
        )
        basis = np.eye(80, dtype=np.float32)
        with self.ui._prediction_lock:
            for index in range(10):
                self.ui.digit_data.add_shots(str(index), basis[index:index + 1])
            for index, label in enumerate(("plus", "sub", "mul", "div")):
                self.ui.operator_data.add_shots(label, basis[20 + index:21 + index])
            self.ui.digit_preprocessor = DigitPreprocessor()
            self.ui.operator_preprocessor = DigitPreprocessor()
            self.ui.digit_ready = self.ui.operator_ready = True
        self.ui._top_action(180)

    def tearDown(self):
        self.extractor.release.set()
        self.assertTrue(self.ui.close(timeout=3.0))

    @staticmethod
    def frame(blank=False):
        frame = np.full((480, 800, 3), 255, dtype=np.uint8)
        if not blank:
            frame[150:250, 195:205] = 0
            frame[195:205, 150:250] = 0
        return frame

    @staticmethod
    def touch(kind, x, y, points=1):
        return SimpleNamespace(kind=kind, x=x, y=y, points=points)

    def wait_idle(self):
        limit = time.monotonic() + 3
        while time.monotonic() < limit:
            with self.ui._prediction_lock:
                if self.ui._digit_request is None and not self.ui._prediction_busy:
                    return
            time.sleep(0.003)
        self.fail("worker timeout: {}".format(self.ui.status))

    def select_roi(self, box=(120, 120, 280, 280)):
        self.ui.handle_event(self.touch("down", box[0], box[1]))
        self.ui.handle_event(self.touch("move", box[2], box[3]))
        self.ui.handle_event(self.touch("up", box[2], box[3]))

    def freeze(self):
        if not self.ui.freeze_active:
            self.ui._top_action(700)
            self.ui.process_frame(self.frame())
            self.wait_idle()
        self.assertTrue(self.ui.freeze_active)

    def submit(self, count=1, blank=False):
        for _ in range(count):
            with self.ui._prediction_lock:
                self.ui.predict_pending = True
            self.ui._schedule_digit_prediction(self.frame(blank))
            self.wait_idle()

    def recognize(self, token, draw=True, count=3):
        kind = "digits" if isinstance(token, int) else "operators"
        if self.ui.recognition_kind != kind:
            self.ui._top_action(580)
        if kind == "digits":
            index = token
        else:
            index = 20 + ("plus", "sub", "mul", "div").index(token)
        self.extractor.feature = np.eye(80, dtype=np.float32)[index]
        self.select_roi()
        self.submit(count)
        if draw:
            self.ui.draw(self.frame())

    def store(self, token):
        self.freeze()
        self.recognize(token)
        before = self.ui.calc.tokens
        self.ui._calculator_action("number" if isinstance(token, int) else "operator")
        self.assertEqual(self.ui.calc.tokens, before + (token,))

    def test_requires_frozen_stable_and_actually_displayed_candidate(self):
        self.recognize(2)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())
        self.freeze()
        self.recognize(2, count=1)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())
        self.submit(2)
        # Worker has enough votes, but the last displayed snapshot was unstable.
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())
        self.ui.draw(self.frame())
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, (2,))

    def test_double_click_and_undo_cannot_reuse_confirmed_candidate(self):
        self.store(2)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, (2,))
        self.ui._calculator_action("undo")
        self.assertEqual(self.ui.calc.tokens, ())
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())
        self.recognize(2)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, (2,))

    def test_worker_change_after_draw_does_not_store_old_or_unseen_label(self):
        self.freeze()
        self.recognize(2)
        # Same ROI and generation: raw top changes even if the smoothed top
        # still favours the old digit. The displayed 2 must not remain storable.
        self.extractor.feature = np.eye(80, dtype=np.float32)[3]
        self.submit(1)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())
        self.submit(5)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, (), "new candidate was not drawn")
        self.ui.draw(self.frame())
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, (3,))

    def test_invalid_or_half_drawn_roi_cannot_be_saved(self):
        self.freeze()
        self.recognize(2)
        self.submit(blank=True)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())
        self.recognize(2)
        self.ui.handle_event(self.touch("down", 150, 150))
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())

    def test_wrong_slot_and_wrong_button_are_rejected_without_auto_switch(self):
        self.freeze()
        self.recognize("plus")
        self.ui._calculator_action("operator")
        self.assertEqual(self.ui.calc.tokens, ())
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, ())
        self.assertEqual(self.ui.recognition_kind, "operators")
        self.store(2)
        self.assertEqual(self.ui.recognition_kind, "digits")
        self.recognize(3)
        self.ui._calculator_action("number")
        self.assertEqual(self.ui.calc.tokens, (2,))

    def test_expression_priority_and_persistent_result(self):
        expected = (9, "sub", 8, "div", 2, "plus", 3, "mul", 4)
        for token in expected:
            self.store(token)
        self.ui._footer_action(720)  # OUT
        self.assertEqual(self.ui.calc.result, Fraction(17))
        self.assertEqual(self.ui.calc.tokens, expected)
        self.ui._set_status("operators plus refresh after output")
        with mock.patch.object(ui_module.cv2, "putText", wraps=ui_module.cv2.putText) as text:
            self.ui.draw(self.frame())
        strings = [str(call.args[1]) for call in text.call_args_list]
        self.assertTrue(any("17" in value for value in strings), strings)

    def test_ten_number_nine_operator_slots_and_full_capacity(self):
        # Store through the real worker/confirmation path beyond the old limit.
        for number in range(1, 10):
            self.store(number)
            if number != 9:
                self.store("plus")

        def displayed_strings():
            with mock.patch.object(ui_module.cv2, "putText",
                                   wraps=ui_module.cv2.putText) as text:
                self.ui.draw(self.frame())
            return [str(call.args[1]) for call in text.call_args_list]

        labels = displayed_strings()
        self.assertIn("X10", labels)
        self.assertIn("OP9", labels)
        self.store("plus")
        labels = displayed_strings()
        self.assertIn("X10", labels)
        self.assertIn("OP FULL", labels)
        self.store(0)
        self.assertEqual((self.ui.calc.number_count, self.ui.calc.operator_count), (10, 9))
        expected = self.ui.calc.tokens
        self.ui._footer_action(720)  # OUT
        self.assertEqual(self.ui.calc.result, Fraction(45))
        labels = displayed_strings()
        self.assertIn("X FULL", labels)
        self.assertIn("OP FULL", labels)
        self.assertIn("1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 0 = 45", labels)

        for token, action in (("plus", "operator"), (1, "number")):
            self.recognize(token)
            self.assertFalse(self.ui._calculator_action(action))
            self.assertEqual(self.ui.calc.tokens, expected)
            self.assertEqual(self.ui.calc.result, Fraction(45))

        # At full capacity UNDO must free X10 for a newly confirmed replacement.
        self.ui._calculator_action("undo")
        self.assertEqual((self.ui.calc.number_count, self.ui.calc.operator_count), (9, 9))
        self.assertIn("X10", displayed_strings())
        self.store(9)
        self.ui._calculator_action("output")
        self.assertEqual(self.ui.calc.result, Fraction(54))

    def test_long_expression_and_result_fit_both_layouts(self):
        # Exercise full-length arithmetic/result rendering independently of
        # recognition accuracy; neither preview layout may clip the expression.
        for operator, result_text in (("mul", "3486784401"), ("div", "~0")):
            self.ui._calculator_action("reset")
            self.ui.calc.append_number(9)
            for _ in range(9):
                self.ui.calc.append_operator(operator)
                self.ui.calc.append_number(9)
            self.assertTrue(self.ui._calculator_action("output"))
            self.assertEqual(self.ui.calc.result_text, result_text)
            expression = self.ui.calc.display_expression() + " = " + result_text
            for preview in (True, False):
                with self.subTest(operator=operator, preview=preview):
                    self.ui.digit_preview = preview
                    with mock.patch.object(ui_module.cv2, "putText",
                                           wraps=ui_module.cv2.putText) as text:
                        self.ui.draw(self.frame())
                    matches = [call for call in text.call_args_list
                               if call.args[1] == expression]
                    self.assertTrue(matches, "complete expression/result not drawn")
                    for call in matches:
                        _, rendered, origin, font, scale, _, thickness = call.args[:7]
                        (width, height), baseline = ui_module.cv2.getTextSize(
                            rendered, font, scale, thickness)
                        self.assertGreaterEqual(origin[0], 0)
                        self.assertLess(origin[0] + width, 800)
                        self.assertGreaterEqual(origin[1] - height, 364)
                        self.assertLess(origin[1] + baseline, 422)

    def test_division_by_zero_undo_and_replacement(self):
        for token in (8, "div", 0):
            self.store(token)
        self.ui._calculator_action("output")
        self.assertEqual(self.ui.calc.tokens, (8, "div", 0))
        self.assertIsNone(self.ui.calc.result)
        self.assertEqual(self.ui.calc.result_text, "DIV/0")
        self.ui._calculator_action("undo")
        self.assertEqual(self.ui.calc.tokens, (8, "div"))
        self.store(2)
        self.ui._calculator_action("output")
        self.assertEqual(self.ui.calc.result, Fraction(4))

    def test_d_errors_are_visible_without_overwriting_calculated_result(self):
        self.store(2)
        self.ui._calculator_action("output")

        def assert_error_drawn(message):
            for preview in (True, False):
                self.ui.digit_preview = preview
                with mock.patch.object(ui_module.cv2, "putText",
                                       wraps=ui_module.cv2.putText) as text:
                    self.ui.draw(self.frame())
                strings = [str(call.args[1]) for call in text.call_args_list]
                self.assertTrue(any(message in value for value in strings), strings)
                self.assertIn("2 = 2", strings)
                self.assertEqual(self.ui.calc.result, Fraction(2))
                self.assertEqual(self.ui.calc.tokens, (2,))

        # Exercise the actual unavailable-cache action, not just _set_status.
        self.ui.operator_ready = False
        self.ui.operator_cache_status = "test missing cache"
        self.ui._top_action(580)
        self.assertEqual(self.ui.recognition_kind, "digits")
        assert_error_drawn("OP unavailable: test missing cache")
        # Normal worker refreshes must not hide an important action/error.
        self.ui._set_status("digits 2: normal recognition refresh")
        assert_error_drawn("OP unavailable: test missing cache")

        # Empty-image preprocessing publishes an invalid ROI from the worker.
        self.select_roi()
        self.submit(blank=True)
        self.assertIsNone(self.ui.current_candidate)
        assert_error_drawn("invalid ROI")
        self.ui._calc_feedback_until = 0.0
        with self.ui._prediction_lock:
            self.assertEqual(self.ui._calculator_snapshot_locked()["hint"],
                             self.ui._calculator_next_hint())

    def test_freeze_mode_switch_and_legacy_clear_preserve_expression(self):
        self.store(2)
        self.ui._calculator_action("output")
        self.ui._top_action(700)  # LIVE
        self.assertFalse(self.ui.freeze_active)
        self.assertEqual(self.ui.calc.tokens, (2,))
        self.ui._top_action(580)  # OP MODE
        self.assertEqual(self.ui.calc.tokens, (2,))
        self.ui._top_action(30)   # A
        self.assertEqual(self.ui.calc.result, Fraction(2))
        self.ui._top_action(450)  # original A/B CLEAR
        self.assertEqual(self.ui.calc.tokens, (2,))
        self.ui._top_action(180)  # D defaults to digits
        self.assertEqual(self.ui.recognition_kind, "digits")
        self.assertEqual(self.ui.calc.tokens, (2,))
        self.assertEqual(self.ui.calc.result, Fraction(2))

    def test_footer_buttons_and_reset_do_not_change_recognition_kind(self):
        self.freeze()
        self.recognize(2)
        self.ui.handle_event(self.touch("down", 80, 450))
        self.assertEqual(self.ui.calc.tokens, (2,))
        self.recognize("plus")
        self.ui.handle_event(self.touch("down", 240, 450))
        self.assertEqual(self.ui.calc.tokens, (2, "plus"))
        self.assertEqual(self.ui.recognition_kind, "operators")
        self.ui.handle_event(self.touch("down", 400, 450))
        self.assertEqual(self.ui.calc.tokens, (2,))
        self.ui.handle_event(self.touch("down", 560, 450))
        self.assertEqual(self.ui.calc.tokens, ())
        self.assertEqual(self.ui.recognition_kind, "operators")
        self.assertTrue(self.ui.freeze_active)
        self.ui._calculator_action("operator")
        self.assertEqual(self.ui.calc.tokens, ())

    def test_both_layouts_exclude_calculator_panel_from_roi(self):
        for preview in (True, False):
            self.ui.digit_preview = preview
            self.ui.handle_event(self.touch("down", 100, 390))
            self.assertFalse(self.ui.drawing)
            self.select_roi((100, 300, 300, 410))
            self.assertEqual(self.ui.digit_box[3], 363)
        self.ui._top_action(30)
        self.select_roi((100, 370, 300, 410))
        self.assertEqual(self.ui.pending_box, (100, 370, 300, 410))


if __name__ == "__main__":
    unittest.main()
