"""Keyboard naming regressions using real UI/worker/storage and fake hardware."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import pefsl_touch_ui as module
from pefsl_identity import prepare_reference
from pefsl_touch_ui import LearnedData, TouchBboxUI, MAX_NAME_LENGTH
import usb_keyboard as kb


FEATURE = np.eye(80, dtype=np.float32)[0]
BOX = (120, 120, 240, 260)


def temporary():
    return tempfile.TemporaryDirectory(prefix='_keyboard_ui_test_',
                                       dir=str(Path(__file__).resolve().parent))


def patterned():
    return np.random.RandomState(17).randint(0, 240, (48, 48, 3)).astype(np.uint8)


class NamedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'learned.npz'

    def registered(self):
        data = LearnedData()
        data.add_shots('class1', FEATURE[None, :], box_size=(120, 140),
                       reference_views=[patterned()], display_name='Apple')
        return data

    def test_name_roundtrip_keeps_ids_and_c124_identity_metadata(self):
        data = self.registered()
        self.assertTrue(data.save(self.path)[0])
        restored = LearnedData()
        self.assertTrue(restored.load(self.path)[0])
        self.assertEqual(restored.labels, ['class1'])
        self.assertEqual(restored.display_name('class1'), 'Apple')
        np.testing.assert_array_equal(restored.shots[0], data.shots[0])
        np.testing.assert_array_equal(restored.reference_views[0][0], data.reference_views[0][0])
        self.assertEqual(restored.raw_cosine, data.raw_cosine)
        self.assertEqual(restored.bmode_matcher().recognize(FEATURE, patterned())['label'], 'class1')
        with np.load(str(self.path), allow_pickle=False) as saved:
            self.assertEqual(saved['display_names'].dtype.kind, 'U')
            self.assertIn('b_reference_images', saved)

    def test_extended_names_roundtrip_without_normalizing_internal_spaces(self):
        for name in ('Apple 01_test', '0123456789', 'two  spaces_9', '_'):
            with self.subTest(name=name):
                data = self.registered()
                data.add_shots('class1', FEATURE[None, :], display_name=name)
                self.assertTrue(data.save(self.path)[0])
                restored = LearnedData()
                self.assertTrue(restored.load(self.path)[0])
                self.assertEqual(restored.labels, ['class1'])
                self.assertEqual(restored.display_name('class1'), name)
                self.assertEqual(len(restored.shots[0]), 2)
                np.testing.assert_array_equal(restored.reference_views[0][0],
                                              data.reference_views[0][0])

    def test_old_npz_default_ids_and_digit_operator_labels_are_unchanged(self):
        np.savez(str(self.path), labels=np.asarray(['class1']), counts=np.asarray([1]),
                 features=FEATURE[None, :])
        data = LearnedData()
        self.assertTrue(data.load(self.path)[0])
        self.assertEqual(data.display_name('class1'), 'class1')
        data.add_shots('0', FEATURE[None, :])
        data.add_shots('plus', FEATURE[None, :])
        self.assertEqual(data.display_name('0'), '0')
        self.assertEqual(data.display_name('plus'), 'plus')
        self.assertEqual(data.display_name('missing'), 'missing')

    def test_append_without_name_preserves_name_explicit_name_renames_same_id(self):
        data = self.registered()
        data.add_shots('class1', FEATURE[None, :])
        data.add_shots('class1', FEATURE[None, :], display_name='')
        self.assertEqual(data.display_name('class1'), 'Apple')
        data.add_shots('class1', FEATURE[None, :], display_name='Pear')
        self.assertEqual(data.labels, ['class1'])
        self.assertEqual(data.display_name('class1'), 'Pear')
        self.assertEqual(len(data.shots[0]), 4)

    def test_invalid_optional_names_do_not_erase_loaded_bank(self):
        data = self.registered()
        for names in (np.asarray(['Apple', 'Pear']), np.asarray(['Apple!']),
                      np.asarray(['   ']), np.asarray(['Apple\t1']),
                      np.asarray(['Apple\n1']), np.asarray(['苹果01']),
                      np.asarray(['a' * (MAX_NAME_LENGTH + 1)]), np.asarray([123]),
                      np.asarray(['Apple'], dtype=object)):
            with self.subTest(names=repr(names)):
                np.savez(str(self.path), labels=np.asarray(['class1']), counts=np.asarray([1]),
                         features=FEATURE[None, :], display_names=names)
                self.assertFalse(data.load(self.path)[0])
                self.assertEqual(data.display_name('class1'), 'Apple')
                self.assertEqual(len(data.shots[0]), 1)

    def test_failed_sample_conversion_does_not_append_or_rename(self):
        data = self.registered()
        with self.assertRaises(ValueError):
            data.add_shots('class1', FEATURE[None, :], display_name='Pear',
                           reference_views=[np.zeros((0, 0, 3), np.uint8)])
        self.assertEqual(len(data.shots[0]), 1)
        self.assertEqual(data.display_name('class1'), 'Apple')
        with self.assertRaises(ValueError):
            data.add_shots('class1', FEATURE[None, :], display_name='invalid!')
        self.assertEqual(len(data.shots[0]), 1)

    def test_invalid_extended_name_never_appends_samples_or_renames(self):
        data = self.registered()
        for name in (' ', '  ', '\t', '\n', 'Apple\t01', 'Apple\n01',
                     '苹果', 'Apple-01', 'Apple.01', 'Apple!'):
            with self.subTest(name=repr(name)):
                with self.assertRaises(ValueError):
                    data.add_shots('class1', FEATURE[None, :], display_name=name)
                self.assertEqual(len(data.shots[0]), 1)
                self.assertEqual(data.display_name('class1'), 'Apple')


class Extractor:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def extract(self, image):
        self.entered.set()
        if not self.release.wait(3.0):
            raise RuntimeError('test extractor release timeout')
        if self.fail:
            raise RuntimeError('simulated TCU failure')
        return FEATURE.copy()


class KeyboardUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.extractor = Extractor()
        self.ui = TouchBboxUI(self.extractor, self.path / 'learned.npz', collect_frames=2,
                              digit_cache_path=self.path / 'no_digits.npz',
                              operator_cache_path=self.path / 'no_operators.npz')

    def tearDown(self):
        self.extractor.release.set()
        self.assertTrue(self.ui.close(timeout=3))

    @staticmethod
    def frame():
        return np.random.RandomState(7).randint(0, 240, (480, 800, 3)).astype(np.uint8)

    def touch(self, kind, x, y, timestamp=None):
        self.ui.handle_event(SimpleNamespace(kind=kind, x=x, y=y, points=1,
            timestamp=time.monotonic() if timestamp is None else timestamp))

    def roi(self, box=BOX):
        self.touch('down', box[0], box[1])
        self.touch('up', box[2], box[3])

    def key(self, kind, text='', timestamp=None):
        self.ui.handle_key(SimpleNamespace(kind=kind, text=text,
            timestamp=time.monotonic() if timestamp is None else timestamp))

    def type_name(self, name):
        for char in name:
            self.key('text', char)

    def wait_idle(self):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            with self.ui._prediction_lock:
                if self.ui._collection_request is None and not self.ui._prediction_busy:
                    return
            time.sleep(.002)
        self.fail('worker failed to finish: ' + self.ui.status)

    def collect(self):
        for _ in range(4):
            if self.ui.collect_label is None:
                return
            self.ui._schedule_collection(self.frame())
            self.wait_idle()
        self.fail('collection did not complete')

    def drawn_strings(self):
        calls = []
        with mock.patch.object(module.cv2, 'putText', side_effect=lambda *a, **kw: calls.append(a)):
            self.ui.draw(self.frame())
        return calls

    def test_each_letter_draws_then_enter_and_touch_commit_the_name(self):
        self.roi()
        for index, char in enumerate('apple'):
            self.key('text', char)
            self.assertIn('apple'[:index+1] + '_', [call[1] for call in self.drawn_strings()])
        self.ui._footer_action(20)
        self.assertIsNone(self.ui.collect_label)
        self.assertIn('Enter', self.ui.status)
        self.key('enter')
        self.assertIn('apple', [call[1] for call in self.drawn_strings()])
        self.ui._footer_action(20)
        self.assertEqual(self.ui.collect_label, 'class1')
        self.assertFalse(self.ui.data.has_data())
        self.assertIn('apple', [call[1] for call in self.drawn_strings()])
        self.collect()
        self.assertEqual(self.ui.data.labels, ['class1'])
        self.assertEqual(self.ui.data.display_name('class1'), 'apple')
        self.assertEqual(len(self.ui.data.shots[0]), 2)

    def test_backspace_case_ascii_filter_limit_escape_and_no_mode_shortcuts(self):
        self.roi()
        self.type_name('Abq')
        self.assertEqual(self.ui.mode, 'A')
        self.key('backspace')
        self.key('text', '你好!\t\n-')
        self.assertEqual(self.ui.pending_name, 'Ab')
        self.key('text', 'z' * 30)
        self.assertEqual(len(self.ui.pending_name), MAX_NAME_LENGTH)
        self.assertIn('limit', self.ui.status)
        self.key('escape')
        self.assertEqual(self.ui.pending_name, '')
        self.assertFalse(self.ui.name_confirmed)
        self.assertEqual(self.ui.pending_box, BOX)

    def test_physical_extended_keys_commit_save_load_and_display_in_b(self):
        self.roi()
        decoder = kb.KeyboardDecoder()
        actual_text = ''
        def feed(code, value):
            nonlocal actual_text
            event = decoder.feed(kb.EV_KEY, code, value)
            if event is not None:
                self.ui.handle_key(event)
                if event.kind == 'text':
                    actual_text += event.text
                    self.assertEqual(self.ui.pending_name, actual_text)
                    self.assertIn(actual_text + '_', [c[1] for c in self.drawn_strings()])
        def press(code):
            feed(code, 1)
            feed(code, 0)
        # Physical QWERTY key codes, including Shift+A and Shift+- for '_'.
        feed(42, 1)
        press(30)
        feed(42, 0)
        for code in (25, 25, 38, 18, 57, 11, 2):
            press(code)
        feed(42, 1)
        press(12)
        feed(42, 0)
        for code in (20, 18, 31, 20):
            press(code)
        self.assertEqual(actual_text, 'Apple 01_test')
        press(28)
        self.assertTrue(self.ui.name_confirmed)
        self.assertEqual(self.ui.pending_name, actual_text)
        self.ui._footer_action(20)
        self.assertEqual(self.ui.collect_label, 'class1')
        self.assertEqual(self.ui.collect_display_name, actual_text)
        self.collect()
        self.assertEqual(self.ui.data.labels, ['class1'])
        self.assertEqual(self.ui.data.display_name('class1'), actual_text)
        self.ui._top_action(260)  # SAVE
        self.ui._top_action(450)  # CLEAR
        self.assertFalse(self.ui.data.has_data())
        self.ui._top_action(350)  # LOAD
        self.assertEqual(self.ui.data.display_name('class1'), actual_text)
        self.ui._top_action(100)  # B: manual label and automatic scan caption.
        self.ui.active_box = BOX
        self.ui.last_label = 'class1'
        self.ui.last_score = .9
        expected_caption = actual_text + ' 90.0%'
        self.assertIn(expected_caption, [c[1] for c in self.drawn_strings()])
        self.ui.active_box = None
        self.ui.scan_detections = [dict(box=BOX, label='class1', score=.9,
                                       verified_at=time.monotonic())]
        self.assertIn(expected_caption, [c[1] for c in self.drawn_strings()])
        self.assertEqual(self.ui.scan_detections[0]['label'], 'class1')

    def test_space_trim_and_internal_double_spaces_survive_commit(self):
        self.roi()
        self.type_name('  Box  01_  ')
        self.assertEqual(self.ui.pending_name, '  Box  01_  ')
        self.key('enter')
        self.assertEqual(self.ui.pending_name, 'Box  01_')
        self.ui._footer_action(20)
        self.collect()
        self.assertEqual(self.ui.data.display_name('class1'), 'Box  01_')

    def test_space_only_enter_uses_existing_name_or_default_id(self):
        self.ui.data.add_shots('class1', FEATURE[None, :], display_name='Apple 01')
        for button_x, label, expected_name in ((20, 'class1', 'Apple 01'),
                                               (110, 'class2', 'class2')):
            with self.subTest(label=label):
                self.roi()
                self.type_name('   ')
                self.key('enter')
                self.assertEqual(self.ui.pending_name, '')
                self.assertTrue(self.ui.name_confirmed)
                self.ui._footer_action(button_x)
                self.collect()
                self.assertEqual(self.ui.data.display_name(label), expected_name)

    def test_limit_counts_digits_spaces_and_underscore_and_backspace_edits_them(self):
        self.roi()
        self.type_name('A' * (MAX_NAME_LENGTH - 2) + ' _')
        self.assertEqual(len(self.ui.pending_name), MAX_NAME_LENGTH)
        self.key('text', '0')
        self.assertEqual(self.ui.pending_name, 'A' * (MAX_NAME_LENGTH - 2) + ' _')
        self.assertIn('limit', self.ui.status)
        self.key('backspace')
        self.assertTrue(self.ui.pending_name.endswith(' '))
        self.key('backspace')
        self.key('text', '09')
        self.assertTrue(self.ui.pending_name.endswith('09'))
        self.assertEqual(len(self.ui.pending_name), MAX_NAME_LENGTH)

    def test_extended_keys_are_ignored_in_b_d_and_during_collection(self):
        self.roi()
        self.type_name('Box 01_test')
        self.key('enter')
        self.ui._footer_action(20)
        self.key('text', ' _0123456789')
        self.assertEqual(self.ui.collect_display_name, 'Box 01_test')
        self.collect()
        for x, mode in ((100, 'B'), (180, 'D')):
            self.ui._top_action(x)
            self.key('text', ' _0123456789')
            self.key('enter')
            self.assertEqual(self.ui.mode, mode)
            self.assertEqual(self.ui.pending_name, '')
        self.assertEqual(self.ui.calc.tokens, ())

    def test_unnamed_touch_or_empty_enter_preserves_existing_class_name(self):
        self.ui.data.add_shots('class1', FEATURE[None, :], display_name='apple')
        for confirm in (False, True):
            self.roi()
            if confirm:
                self.key('enter')
            self.ui._footer_action(20)
            self.collect()
            self.assertEqual(self.ui.data.display_name('class1'), 'apple')
        self.assertEqual(len(self.ui.data.shots[0]), 5)

    def test_failed_collection_does_not_rename_existing_class(self):
        self.ui.data.add_shots('class1', FEATURE[None, :], display_name='apple')
        self.roi()
        self.type_name('Pear')
        self.key('enter')
        self.ui._footer_action(20)
        self.extractor.fail = True
        self.collect()
        self.assertIn('TCU/ROI error', self.ui.status)
        self.assertEqual(self.ui.data.display_name('class1'), 'apple')
        self.assertEqual(len(self.ui.data.shots[0]), 1)
        self.assertIsNone(self.ui.collect_display_name)

    def test_new_roi_rejects_late_old_keys_and_half_drawn_names(self):
        self.roi()
        self.type_name('apple')
        # Windows host monotonic clocks can quantize several immediate calls
        # to the same tick; use a definitely older event for this regression.
        old = time.monotonic() - 1.0
        self.touch('down', 300, 150)
        self.assertIsNone(self.ui.pending_box)
        self.key('text', 'x')
        self.touch('up', 450, 300)
        self.key('text', 'old', timestamp=old)
        self.assertEqual(self.ui.pending_name, '')
        self.key('text', 'Pear')
        self.assertEqual(self.ui.pending_name, 'Pear')

    def test_disconnect_or_lost_input_cancels_whole_uncommitted_roi(self):
        for confirmed in (False, True):
            self.roi()
            self.type_name('app')
            if confirmed:
                self.key('enter')
            self.key('disconnected')
            self.assertIsNone(self.ui.pending_box)
            self.key('text', 'le')
            self.key('enter')
            self.ui._footer_action(20)
            self.assertIsNone(self.ui.collect_label)
            self.assertFalse(self.ui.data.has_data())

    def test_disconnect_during_collection_keeps_confirmed_binding(self):
        self.roi()
        self.type_name('apple')
        self.key('enter')
        self.ui._footer_action(20)
        self.key('disconnected')
        self.key('text', 'wrong')
        self.assertEqual(self.ui.collect_display_name, 'apple')
        self.collect()
        self.assertEqual(self.ui.data.display_name('class1'), 'apple')

    def test_save_load_clear_and_mode_changes_cancel_drafts_without_affecting_d(self):
        self.roi()
        self.type_name('apple')
        self.key('enter')
        self.ui._footer_action(20)
        self.collect()
        self.ui._top_action(260)  # SAVE
        self.roi()
        self.type_name('draft')
        self.ui._top_action(450)  # CLEAR
        self.assertIsNone(self.ui.pending_box)
        self.assertFalse(self.ui.data.has_data())
        self.ui._top_action(350)  # LOAD
        self.assertEqual(self.ui.data.display_name('class1'), 'apple')
        for x, mode in ((100, 'B'), (180, 'D'), (20, 'A')):
            self.ui._top_action(x)
            self.key('text', 'abq')
            self.key('enter')
            self.assertEqual(self.ui.mode, mode)
            self.assertEqual(self.ui.pending_name, '')
        self.assertEqual(self.ui.calc.tokens, ())

    def test_stale_collect_cannot_commit_old_name_into_new_collection(self):
        self.roi()
        self.type_name('apple')
        self.key('enter')
        self.ui._footer_action(20)
        self.extractor.release.clear()
        self.ui._schedule_collection(self.frame())
        self.assertTrue(self.extractor.entered.wait(1))
        self.ui._top_action(20)
        self.roi()
        self.type_name('Pear')
        self.key('enter')
        self.ui._footer_action(20)
        self.extractor.release.set()
        self.wait_idle()
        self.assertFalse(self.ui.data.has_data())
        self.collect()
        self.assertEqual(self.ui.data.display_name('class1'), 'Pear')
        self.assertEqual(len(self.ui.data.shots[0]), 2)

    def test_manual_and_scan_map_display_names_without_changing_labels(self):
        self.ui.data.add_shots('class1', FEATURE[None, :], display_name='apple')
        self.ui._top_action(100)
        self.ui.active_box = BOX
        self.ui.last_label = 'class1'
        self.ui.last_score = .9
        self.assertIn('apple 90.0%', [c[1] for c in self.drawn_strings()])
        self.ui.active_box = None
        self.ui.scan_detections = [dict(box=BOX, label='class1', score=.9, verified_at=time.monotonic())]
        self.assertIn('apple 90.0%', [c[1] for c in self.drawn_strings()])
        self.assertEqual(self.ui.scan_detections[0]['label'], 'class1')

    def test_long_name_caption_fits_screen_at_right_edge(self):
        self.ui.data.add_shots('class1', FEATURE[None, :], display_name='W' * MAX_NAME_LENGTH)
        self.ui._top_action(100)
        self.ui.active_box = (760, 100, 799, 200)
        self.ui.last_label = 'class1'
        self.ui.last_score = .9
        caption = next(c for c in self.drawn_strings() if c[1].startswith('WWW'))
        text, (x, y), _, scale = caption[1:5]
        if hasattr(module.cv2, 'getTextSize'):
            (width, height), _ = module.cv2.getTextSize(text, module.cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            self.assertLessEqual(x + width, 799)
            self.assertGreaterEqual(y - height, module.TOOLBAR_H)
        self.assertGreaterEqual(x, 0)


if __name__ == '__main__':
    unittest.main()
