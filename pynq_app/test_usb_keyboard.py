"""Decoder, packet ABI and simulated evdev lifecycle tests (no USB required)."""
import errno
import threading
import time
import unittest
from unittest.mock import patch

import usb_keyboard as kb


def bitmap(codes, length=kb.KEY_BYTES):
    data = bytearray(length)
    for code in codes:
        data[code // 8] |= 1 << (code % 8)
    return data


class DecoderTests(unittest.TestCase):
    def test_all_26_letters_and_edit_keys(self):
        decoder = kb.KeyboardDecoder()
        got = [decoder.feed(kb.EV_KEY, code, 1).text for code in kb.LETTER_CODES]
        self.assertEqual("".join(got), "abcdefghijklmnopqrstuvwxyz")
        for code, kind in ((14, "backspace"), (28, "enter"), (96, "enter"), (1, "escape")):
            event = decoder.feed(kb.EV_KEY, code, 1, timestamp=12.5)
            self.assertEqual((event.kind, event.timestamp), (kind, 12.5))
            self.assertIsNone(decoder.feed(kb.EV_KEY, code, 0))
            self.assertIsNone(decoder.feed(kb.EV_KEY, code, 2))
        self.assertEqual(decoder.feed(kb.EV_KEY, 2, 1).text, "1")

    def test_top_row_digits_space_and_underscore_are_text_once_per_press(self):
        decoder = kb.KeyboardDecoder()
        for code, expected in ((2, "1"), (3, "2"), (4, "3"), (5, "4"),
                               (6, "5"), (7, "6"), (8, "7"), (9, "8"),
                               (10, "9"), (11, "0"), (57, " ")):
            with self.subTest(code=code):
                event = decoder.feed(kb.EV_KEY, code, 1, timestamp=12.0)
                self.assertEqual((event.kind, event.text, event.timestamp), ("text", expected, 12.0))
                self.assertIsNone(decoder.feed(kb.EV_KEY, code, 2))
                self.assertIsNone(decoder.feed(kb.EV_KEY, code, 0))
        self.assertIsNone(decoder.feed(kb.EV_KEY, kb.KEY_MINUS, 1))
        decoder.feed(kb.EV_KEY, 42, 1)
        self.assertEqual(decoder.feed(kb.EV_KEY, kb.KEY_MINUS, 1).text, "_")
        self.assertIsNone(decoder.feed(kb.EV_KEY, kb.KEY_MINUS, 2))
        self.assertIsNone(decoder.feed(kb.EV_KEY, kb.KEY_MINUS, 0))
        self.assertEqual(decoder.feed(kb.EV_KEY, kb.KEY_SPACE, 1).text, " ")
        # Shift+digits denotes punctuation, which is outside the name alphabet.
        for code in range(2, 12):
            self.assertIsNone(decoder.feed(kb.EV_KEY, code, 1))
        decoder.feed(kb.EV_KEY, 42, 0)
        decoder.feed(kb.EV_KEY, kb.KEY_CAPSLOCK, 1)
        self.assertEqual(decoder.feed(kb.EV_KEY, 11, 1).text, "0")
        self.assertEqual(decoder.feed(kb.EV_KEY, 2, 1).text, "1")
        self.assertIsNone(decoder.feed(kb.EV_KEY, kb.KEY_MINUS, 1))
        decoder.feed(kb.EV_KEY, 54, 1)
        self.assertEqual(decoder.feed(kb.EV_KEY, kb.KEY_MINUS, 1).text, "_")

    def test_numeric_keypad_tracks_num_lock_shift_and_initial_led_state(self):
        decoder = kb.KeyboardDecoder()
        keypad = ((71, "7"), (72, "8"), (73, "9"), (75, "4"), (76, "5"),
                  (77, "6"), (79, "1"), (80, "2"), (81, "3"), (82, "0"))
        for code, _ in keypad:
            self.assertIsNone(decoder.feed(kb.EV_KEY, code, 1))  # navigation mode
        self.assertIsNone(decoder.feed(kb.EV_KEY, kb.KEY_NUMLOCK, 1))
        decoder.feed(kb.EV_KEY, kb.KEY_NUMLOCK, 2)  # hold does not toggle again
        decoder.feed(kb.EV_KEY, kb.KEY_NUMLOCK, 0)
        for code, expected in keypad:
            with self.subTest(code=code):
                self.assertEqual(decoder.feed(kb.EV_KEY, code, 1).text, expected)
                self.assertIsNone(decoder.feed(kb.EV_KEY, code, 0))
                self.assertIsNone(decoder.feed(kb.EV_KEY, code, 2))
        decoder.feed(kb.EV_KEY, 42, 1)
        self.assertIsNone(decoder.feed(kb.EV_KEY, 82, 1))  # shift -> navigation
        decoder.feed(kb.EV_KEY, kb.KEY_NUMLOCK, 1)  # Num Lock off, Shift held
        self.assertEqual(decoder.feed(kb.EV_KEY, 82, 1).text, "0")
        decoder.reset()
        self.assertIsNone(decoder.feed(kb.EV_KEY, 82, 1))
        decoder.synchronize(caps_lock=True, num_lock=True)
        self.assertEqual(decoder.feed(kb.EV_KEY, 79, 1).text, "1")
        decoder.synchronize([54], caps_lock=True, num_lock=False)
        self.assertEqual(decoder.feed(kb.EV_KEY, 79, 1).text, "1")

    def test_ctrl_and_alt_do_not_turn_new_keys_into_name_characters(self):
        decoder = kb.KeyboardDecoder()
        for modifier in (29, 97, 56, 100):
            with self.subTest(modifier=modifier):
                decoder.synchronize([modifier, 42], num_lock=True)
                for code in (2, 11, kb.KEY_SPACE, kb.KEY_MINUS, 82):
                    self.assertIsNone(decoder.feed(kb.EV_KEY, code, 1))

    def test_raw_key_names_identify_added_keys(self):
        for code, name in ((2, "KEY_1"), (11, "KEY_0"), (57, "KEY_SPACE"),
                           (12, "KEY_MINUS"), (69, "KEY_NUMLOCK"),
                           (71, "KEY_KP7"), (82, "KEY_KP0")):
            with self.subTest(code=code):
                self.assertEqual(kb.key_name(code), name)

    def test_shift_caps_and_two_independent_shift_keys(self):
        decoder = kb.KeyboardDecoder()
        decoder.feed(kb.EV_KEY, 42, 1)
        self.assertEqual(decoder.feed(kb.EV_KEY, 30, 1).text, "A")
        decoder.feed(kb.EV_KEY, 54, 1)
        decoder.feed(kb.EV_KEY, 42, 0)
        self.assertEqual(decoder.feed(kb.EV_KEY, 30, 1).text, "A")
        decoder.feed(kb.EV_KEY, 54, 0)
        decoder.feed(kb.EV_KEY, kb.KEY_CAPSLOCK, 1)
        decoder.feed(kb.EV_KEY, kb.KEY_CAPSLOCK, 2)  # repeat must not toggle
        self.assertEqual(decoder.feed(kb.EV_KEY, 48, 1).text, "B")
        decoder.feed(kb.EV_KEY, 42, 1)
        self.assertEqual(decoder.feed(kb.EV_KEY, 48, 1).text, "b")
        decoder.reset()
        self.assertEqual(decoder.feed(kb.EV_KEY, 30, 1).text, "a")

    def test_ctrl_c_interrupt_and_other_control_letters_ignored(self):
        decoder = kb.KeyboardDecoder()
        self.assertEqual(decoder.feed(kb.EV_KEY, 46, 1).text, "c")
        decoder.feed(kb.EV_KEY, 29, 1)
        self.assertEqual(decoder.feed(kb.EV_KEY, 46, 1).kind, "interrupt")
        self.assertIsNone(decoder.feed(kb.EV_KEY, 30, 1))
        decoder.feed(kb.EV_KEY, 29, 0)
        decoder.feed(kb.EV_KEY, 56, 1)
        self.assertIsNone(decoder.feed(kb.EV_KEY, 30, 1))
        decoder.feed(kb.EV_KEY, 56, 0)
        self.assertEqual(decoder.feed(kb.EV_KEY, 30, 1).text, "a")

    def test_syn_dropped_discards_until_report_and_can_resynchronize(self):
        decoder = kb.KeyboardDecoder()
        decoder.feed(kb.EV_KEY, 42, 1)
        self.assertEqual(decoder.feed(kb.EV_SYN, kb.SYN_DROPPED, 0).kind, "disconnected")
        self.assertTrue(decoder.dropping)
        self.assertIsNone(decoder.feed(kb.EV_KEY, 30, 1))
        self.assertIsNone(decoder.feed(kb.EV_KEY, 28, 1))
        decoder.feed(kb.EV_SYN, kb.SYN_REPORT, 0)
        decoder.synchronize([54], caps_lock=True)
        self.assertEqual(decoder.feed(kb.EV_KEY, 30, 1).text, "a")

    def test_immutable_events_default_to_monotonic_time(self):
        now = time.monotonic()
        event = kb.KeyEvent("enter")
        self.assertGreaterEqual(event.timestamp, now)
        with self.assertRaises((AttributeError, TypeError)):
            event.kind = "escape"


class StreamTests(unittest.TestCase):
    def test_native_abi_partial_chunks_and_monotonic_timestamps(self):
        self.assertIn(kb.INPUT_EVENT.size, (16, 24))
        stream = kb.InputEventStream(monotonic_clock=True)
        data = (kb.INPUT_EVENT.pack(11, 250000, kb.EV_KEY, 30, 1) +
                kb.INPUT_EVENT.pack(11, 280000, kb.EV_KEY, 30, 0))
        self.assertEqual(stream.feed(data[:3]), [])
        first = stream.feed(data[3:kb.INPUT_EVENT.size + 1])
        self.assertEqual(len(first), 1)
        self.assertEqual((first[0].code, first[0].value, first[0].timestamp), (30, 1, 11.25))
        last = stream.feed(data[kb.INPUT_EVENT.size + 1:])
        self.assertEqual((last[0].code, last[0].value, last[0].timestamp), (30, 0, 11.28))
        self.assertEqual(stream.buffer, b"")

    def test_wallclock_falls_back_to_read_monotonic_and_batches_are_bounded(self):
        stream = kb.InputEventStream()
        with patch.object(kb.time, "monotonic", return_value=5.25):
            event = stream.feed(kb.INPUT_EVENT.pack(1700000000, 0, kb.EV_KEY, 30, 1))[0]
        self.assertEqual(event.timestamp, 5.25)
        with self.assertRaisesRegex(ValueError, "oversized"):
            stream.feed(b"x" * (kb.INPUT_EVENT.size * 257))
        self.assertEqual(stream.buffer, b"")


class DeviceSelectionTests(unittest.TestCase):
    def test_composite_mouse_and_consumer_nodes_are_not_alphabet_keyboard(self):
        self.assertFalse(kb.is_alphabet_keyboard(bitmap([272, 273])))
        self.assertFalse(kb.is_alphabet_keyboard(bitmap([113, 114, 115, 116])))
        self.assertFalse(kb.is_alphabet_keyboard(bitmap(kb.LETTER_CODES)))
        self.assertTrue(kb.is_alphabet_keyboard(bitmap(kb.REQUIRED_KEYS)))

    def test_stable_alias_first_duplicate_real_paths_removed(self):
        def fake_glob(pattern):
            return (["/dev/input/by-id/cx-event-kbd"] if "by-id" in pattern else
                    ["/dev/input/event12", "/dev/input/event0", "/dev/input/event2"])
        def realpath(path):
            return "/dev/input/event0" if "by-id" in path else path
        with patch.object(kb.glob, "glob", side_effect=fake_glob), \
                patch.object(kb.os.path, "realpath", side_effect=realpath):
            self.assertEqual(kb.candidate_paths(), ["/dev/input/by-id/cx-event-kbd",
                                                   "/dev/input/event2", "/dev/input/event12"])

    def test_capability_rejection_never_grabs_and_failed_grab_closes_fd(self):
        class FakeFcntl:
            def __init__(self, capable, fail_grab=False):
                self.capable, self.fail_grab = capable, fail_grab
                self.grabs = []
            def ioctl(self, fd, request, argument, *args):
                if request == kb.EVIOCGRAB:
                    self.grabs.append(argument)
                    if self.fail_grab:
                        raise OSError(errno.EBUSY, "already grabbed")
                elif request == kb.EVIOCSCLOCKID:
                    return 0
                elif (request & 0xff) == 0x21:
                    argument[:] = bitmap(kb.REQUIRED_KEYS if self.capable else [272])
                elif (request & 0xff) == 0x06:
                    argument[:8] = b"CX keys\0"
                return 0
        worker = kb.USBKeyboardWorker()
        fake = FakeFcntl(False)
        with patch.object(kb.os, "O_NONBLOCK", 0x800, create=True), \
                patch.object(kb, "fcntl", fake), patch.object(kb.os, "open", return_value=71), \
                patch.object(kb.os, "close") as close:
            self.assertIsNone(worker._open_keyboard("/fake/mouse"))
            close.assert_called_once_with(71)
            self.assertEqual(fake.grabs, [])
        fake = FakeFcntl(True, fail_grab=True)
        with patch.object(kb.os, "O_NONBLOCK", 0x800, create=True), \
                patch.object(kb, "fcntl", fake), patch.object(kb.os, "open", return_value=72), \
                patch.object(kb.os, "close") as close:
            with self.assertRaises(OSError):
                worker._open_keyboard("/fake/keyboard")
            close.assert_called_once_with(72)
            self.assertEqual(fake.grabs, [1])


class WorkerTests(unittest.TestCase):
    def test_device_led_sync_applies_num_lock_and_caps_state(self):
        worker = kb.USBKeyboardWorker()
        decoder = kb.KeyboardDecoder()
        keys = bitmap([54])
        leds = bitmap([0, 1], length=8)  # Num Lock and Caps Lock are active.
        with patch.object(worker, "_bitmap", side_effect=[keys, leds]):
            worker._synchronize(71, decoder)
        self.assertEqual(decoder.feed(kb.EV_KEY, 30, 1).text, "a")
        self.assertIsNone(decoder.feed(kb.EV_KEY, 82, 1))  # Shift reverses Num Lock.
        decoder.feed(kb.EV_KEY, 54, 0)
        self.assertEqual(decoder.feed(kb.EV_KEY, 82, 1).text, "0")

    def test_queue_overflow_cancels_buffer_instead_of_dropping_only_enter(self):
        worker = kb.USBKeyboardWorker(queue_limit=8)
        for _ in range(8):
            worker._push(kb.KeyEvent("text", "a"))
        worker._push(kb.KeyEvent("enter"))
        events = worker.drain()
        self.assertEqual([event.kind for event in events], ["disconnected"])
        self.assertIn("overflow", worker.snapshot()["last_error"])

    def test_raw_queue_and_drain_limits_are_bounded(self):
        worker = kb.USBKeyboardWorker(queue_limit=8, capture_raw=True)
        for _ in range(30):
            worker._record_raw(kb.RawKeyEvent(kb.EV_KEY, 30, 1))
        self.assertLessEqual(len(worker._raw), 16)
        self.assertEqual(len(worker.drain_raw(limit=2)), 2)
        for _ in range(5):
            worker._push(kb.KeyEvent("text", "a"))
        self.assertEqual(len(worker.drain(limit=2)), 2)
        self.assertEqual(len(worker.drain()), 3)

    def test_stop_interrupts_long_reconnect_wait(self):
        worker = kb.USBKeyboardWorker(reconnect_interval=10.0)
        with patch.object(kb.sys, "platform", "linux"), patch.object(kb, "fcntl", object()), \
                patch.object(kb, "candidate_paths", return_value=[]):
            worker.start()
            start = time.monotonic()
            self.assertTrue(worker.stop(timeout=0.5))
            self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(worker.snapshot()["state"], "stopped")

    def test_disconnect_drops_stale_text_reconnects_and_releases_both_devices(self):
        connected_second = threading.Event()
        class FakeWorker(kb.USBKeyboardWorker):
            def __init__(self):
                super().__init__(reconnect_interval=0.1, capture_raw=True)
                self.open_count, self.closed = 0, []
            def _open_keyboard(self, path):
                self.open_count += 1
                if self.open_count == 2:
                    connected_second.set()
                return (self.open_count, "CX fake keyboard", True)
            def _synchronize(self, fd, decoder):
                decoder.synchronize()
            def _close_device(self, fd):
                self.closed.append(fd)
        worker = FakeWorker()
        reads = {1: 0, 2: 0}
        def fake_select(read, write, error, timeout):
            fd = read[0]
            if fd == 2 and reads[2] > 0:
                time.sleep(0.005)
                return [], [], []
            return read, [], []
        def fake_read(fd, length):
            reads[fd] += 1
            if fd == 1:
                if reads[fd] == 1:
                    return kb.INPUT_EVENT.pack(10, 0, kb.EV_KEY, 30, 1)
                raise OSError(errno.ENODEV, "unplugged")
            return (kb.INPUT_EVENT.pack(11, 0, kb.EV_KEY, 48, 1) +
                    kb.INPUT_EVENT.pack(11, 1000, kb.EV_KEY, 48, 0) +
                    kb.INPUT_EVENT.pack(11, 2000, kb.EV_KEY, kb.KEY_ENTER, 1))
        with patch.object(kb.sys, "platform", "linux"), patch.object(kb, "fcntl", object()), \
                patch.object(kb, "candidate_paths", return_value=["/fake/keyboard"]), \
                patch.object(kb.select, "select", side_effect=fake_select), \
                patch.object(kb.os, "read", side_effect=fake_read):
            worker.start()
            self.assertTrue(connected_second.wait(timeout=1.0))
            deadline = time.monotonic() + 1.0
            while reads[2] == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(worker.stop(timeout=0.5))
        events = worker.drain()
        self.assertEqual([(x.kind, x.text) for x in events], [
            ("disconnected", "[Errno 19] unplugged"), ("text", "b"), ("enter", "")])
        self.assertEqual(worker.closed, [1, 2])
        self.assertEqual(worker.snapshot()["connections"], 2)
        self.assertEqual([x.value for x in worker.drain_raw() if x.event_type == kb.EV_KEY], [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
