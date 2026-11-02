"""Cross-device event ordering and forked keyboard descriptor regressions."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def load_entry_point():
    touch = ModuleType("touch_driver")
    touch.TouchController = object
    tcu = ModuleType("tcu_bbox_backend")
    tcu.TcuFeatureExtractor = object
    tcu.load_overlay = lambda path: None
    spec = importlib.util.spec_from_file_location(
        "keyboard_entry_test", Path(__file__).with_name("pynq_bbox_demo.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"touch_driver": touch, "tcu_bbox_backend": tcu}):
        spec.loader.exec_module(module)
    return module


class DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entry = load_entry_point()

    def test_roi_letter_enter_then_class_order_is_preserved(self):
        events = []
        ui = SimpleNamespace(handle_event=lambda event: events.append(event.kind),
                             handle_key=lambda event: events.append(event.kind))
        touch = SimpleNamespace(drain=lambda: [
            SimpleNamespace(kind="roi-up", timestamp=1.0),
            SimpleNamespace(kind="class-down", timestamp=4.0)])
        keys = SimpleNamespace(drain=lambda: [
            SimpleNamespace(kind="text", timestamp=2.0),
            SimpleNamespace(kind="enter", timestamp=3.0)])
        self.entry._dispatch_ui_inputs(ui, touch, keys, threading.Event())
        self.assertEqual(events, ["roi-up", "text", "enter", "class-down"])

    def test_keys_before_new_roi_are_not_reordered_into_it(self):
        events = []
        ui = SimpleNamespace(handle_event=lambda event: events.append(event.kind),
                             handle_key=lambda event: events.append(event.kind))
        touch = SimpleNamespace(drain=lambda: [SimpleNamespace(kind="new-roi", timestamp=2.0)])
        keys = SimpleNamespace(drain=lambda: [SimpleNamespace(kind="old-text", timestamp=1.0)])
        self.entry._dispatch_ui_inputs(ui, touch, keys, threading.Event())
        self.assertEqual(events, ["old-text", "new-roi"])

    def test_physical_ctrl_c_stops_without_delivering_c(self):
        stop = threading.Event()
        received = []
        ui = SimpleNamespace(handle_key=received.append)
        touch = SimpleNamespace(drain=lambda: [])
        keys = SimpleNamespace(drain=lambda: [SimpleNamespace(kind="interrupt", timestamp=1.0)])
        self.entry._dispatch_ui_inputs(ui, touch, keys, stop)
        self.assertTrue(stop.is_set())
        self.assertEqual(received, [])

    def test_touch_only_mode_still_dispatches(self):
        received = []
        event = SimpleNamespace(kind="down")
        touch = SimpleNamespace(drain=lambda: [event])
        self.entry._dispatch_ui_inputs(SimpleNamespace(handle_event=received.append),
                                       touch, None, threading.Event())
        self.assertEqual(received, [event])

    def test_touch_child_closes_only_inherited_keyboard_fds(self):
        paths = {"/proc/self/fd/3": "/dev/input/event0",
                 "/proc/self/fd/4": "pipe:[123]",
                 "/proc/self/fd/5": "/dev/mem",
                 "/proc/self/fd/6": "/dev/input/event4 (deleted)"}
        with patch.object(self.entry.os, "listdir", return_value=["0", "1", "2", "3", "4", "5", "6"]), \
                patch.object(self.entry.os, "readlink", side_effect=paths.__getitem__), \
                patch.object(self.entry.os, "close") as close:
            self.entry._close_inherited_keyboard_fds()
        self.assertEqual([call.args[0] for call in close.call_args_list], [3, 6])

    def test_cli_auto_keyboard_and_diagnostic_options(self):
        args = self.entry._parse_args([])
        self.assertFalse(args.no_keyboard)
        self.assertIsNone(args.keyboard_device)
        args = self.entry._parse_args(["--keyboard-device", "/dev/input/event0", "--keyboard-no-grab"])
        self.assertEqual(args.keyboard_device, "/dev/input/event0")
        self.assertTrue(args.keyboard_no_grab)

    def _run_stop_cleanup(self, physical_interrupt):
        # Drive the actual main-loop exit path with fake hardware resources.
        # In both cases the untransferred camera buffer must be released once.
        module = self.entry
        frame = SimpleNamespace(shape=(480, 800, 3), releases=0)
        def release():
            frame.releases += 1
        frame.freebuffer = release
        ui = SimpleNamespace(digit_cache_status="test", operator_cache_status="test",
                             close=lambda: True, process_frame=lambda frame: None)
        touch = SimpleNamespace(initialize=lambda: {}, int_level=1, release_reset=lambda: None)
        touch_worker = SimpleNamespace(start=lambda: None, drain=lambda: [], stop=lambda: True)
        keyboard_worker = SimpleNamespace(
            start=lambda: None, stop=lambda timeout=1: True,
            drain=lambda: [SimpleNamespace(kind="interrupt", timestamp=1.0)],
            snapshot=lambda: {"state": "connected"})
        channel = SimpleNamespace(mode="test", close=lambda: None)
        pynq = ModuleType("pynq")
        lib = ModuleType("pynq.lib")
        camera = ModuleType("pynq.lib.ov5640")
        video = ModuleType("pynq.lib.video2lcd")
        pynq.lib = lib
        lib.ov5640, lib.video2lcd = camera, video
        modules = {"pynq": pynq, "pynq.lib": lib,
                   "pynq.lib.ov5640": camera, "pynq.lib.video2lcd": video}
        workspace = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="_key_main_test_", dir=str(workspace)) as folder:
            root = Path(folder)
            self.assertEqual(root.resolve().parent, workspace)
            for name in ("design.bit", "design.hwh", "model.tmodel", "model.tprog", "model.tdata"):
                (root / name).write_bytes(b"test")
            args = ["--bitstream", str(root / "design.bit"), "--tmodel", str(root / "model.tmodel")]
            if not physical_interrupt:
                args.append("--no-keyboard")
            with patch.dict(sys.modules, modules), \
                    patch.object(module, "load_overlay", return_value=object()), \
                    patch.object(module, "TouchController", SimpleNamespace(from_overlay=lambda overlay: touch)), \
                    patch.object(module, "TouchProcessWorker", return_value=touch_worker), \
                    patch.object(module, "USBKeyboardWorker", return_value=keyboard_worker), \
                    patch.object(module, "TcuFeatureExtractor", return_value=object()), \
                    patch.object(module, "TouchBboxUI", return_value=ui), \
                    patch.object(module, "_setup_video", return_value=(None, channel, channel)), \
                    patch.object(module, "_discard_warmup_frame"), \
                    patch.object(module, "_read_frame", return_value=frame), \
                    patch.object(module, "_write_frame", return_value=False):
                self.assertEqual(module.main(args), 0)
        self.assertEqual(frame.releases, 1)

    def test_physical_interrupt_releases_camera_frame_once(self):
        self._run_stop_cleanup(True)

    def test_interrupted_video_write_releases_camera_frame_once(self):
        self._run_stop_cleanup(False)


if __name__ == "__main__":
    unittest.main()
