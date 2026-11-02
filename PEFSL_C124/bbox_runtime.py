"""Bounded camera delivery and a single background search, without queues."""
import threading
import time
from collections import deque
from collections.abc import Mapping
import cv2
import os
import queue
import math
import numpy as np


def validate_search_result(result, width, height, classes, appearance_size):
    """Reject malformed worker payloads before they reach tracking/OpenCV."""
    required = ('elapsed_ms', 'source_time', 'kind', 'gray', 'proposals', 'stage', 'rejection')
    if any(name not in result for name in required):
        raise ValueError('search result is missing required fields')
    for name in ('elapsed_ms', 'source_time'):
        value = result[name]
        if (not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value) or (name == 'elapsed_ms' and value < 0)):
            raise ValueError('invalid search ' + name)
    if result['kind'] not in ('global', 'local'):
        raise ValueError('invalid search kind')
    gray = result['gray']
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.shape != (height, width):
        raise ValueError('invalid source grayscale frame')
    if not isinstance(result['rejection'], dict) or not isinstance(result['proposals'], list):
        raise ValueError('invalid search records')
    for proposal in result['proposals']:
        if not isinstance(proposal, dict) or any(name not in proposal for name in
                ('cls', 'box', 'points', 'appearance', 'occupancy')):
            raise ValueError('incomplete search proposal')
        ci = proposal['cls']
        if not isinstance(ci, (int, np.integer)) or not 0 <= ci < classes:
            raise ValueError('invalid proposal class')
        box = np.asarray(proposal['box'])
        if box.shape != (4,) or box.dtype.kind not in 'iuf' or not np.isfinite(box).all():
            raise ValueError('invalid proposal box')
        x, y, r, b = box
        if not (0 <= x < r <= width and 0 <= y < b <= height):
            raise ValueError('proposal outside source image')
        points = proposal['points']
        if points is not None:
            if (not isinstance(points, np.ndarray) or points.dtype != np.float32
                    or not (points.ndim == 2 and points.shape[1:] == (2,)
                            or points.ndim == 3 and points.shape[1:] == (1, 2))
                    or not np.isfinite(points).all()):
                raise ValueError('invalid proposal motion points')
        appearance = proposal['appearance']
        if appearance is not None:
            if (not isinstance(appearance, np.ndarray) or appearance.dtype.kind != 'f'
                    or appearance.shape != (appearance_size,) or not np.isfinite(appearance).all()):
                raise ValueError('invalid proposal appearance')


class WindowsRealtimePolicy:
    """Keep this real-time process at HighQoS even when its window is occluded.

    https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
    No global power plan or camera exposure settings are changed.
    """
    def __init__(self):
        self.enabled=False
        self._restore=None
        if os.name!="nt":
            return
        try:
            import ctypes
            from ctypes import wintypes
            class PowerState(ctypes.Structure):
                _fields_=[("Version",wintypes.DWORD),("ControlMask",wintypes.DWORD),("StateMask",wintypes.DWORD)]
            kernel=ctypes.WinDLL("kernel32",use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            signature=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
            kernel.GetProcessInformation.argtypes=signature
            kernel.GetProcessInformation.restype=wintypes.BOOL
            kernel.SetProcessInformation.argtypes=signature
            kernel.SetProcessInformation.restype=wintypes.BOOL
            process=kernel.GetCurrentProcess()
            original=PowerState(1,0,0)
            if not kernel.GetProcessInformation(process,4,ctypes.byref(original),ctypes.sizeof(original)):
                return
            state=PowerState(1,5,0)  # Disable execution throttling and ignored timer requests.
            ok=kernel.SetProcessInformation(process,4,ctypes.byref(state),ctypes.sizeof(state))
            if not ok:
                state.ControlMask=1  # Older Windows versions may not know the timer flag.
                ok=kernel.SetProcessInformation(process,4,ctypes.byref(state),ctypes.sizeof(state))
            self.enabled=bool(ok)
            if ok:
                self._restore=lambda: kernel.SetProcessInformation(
                    process,4,ctypes.byref(original),ctypes.sizeof(original))
        except (AttributeError,OSError):
            pass

    def close(self):
        if self._restore is not None:
            self._restore()
            self._restore=None


class LatestFrameCamera:
    """One reader owns the camera; consumers receive the newest available frame."""
    def __init__(self, cap):
        self._cap = cap
        self._condition = threading.Condition()
        self._stop = False
        self._sequence = self._consumed = 0
        self._frame = None
        self._ok = False
        self.capture_time = 0.
        self.dropped_frames = 0
        self.source_fps = 0.
        self.last_error = None
        self._capture_times = deque()
        self.properties = {}
        self._property_time = 0.
        self._thread = threading.Thread(target=self._read_loop, name="bbox-camera", daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while True:
                with self._condition:
                    if self._stop:
                        break
                error = None
                try:
                    ok, frame = self._cap.read()
                except cv2.error as exc:
                    ok, frame = False, None
                    error = str(exc)
                if ok and (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                           or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
                    ok, frame = False, None
                    error = 'camera returned an invalid BGR frame'
                if not ok:
                    frame = None
                    error = error or 'camera read returned no frame'
                with self._condition:
                    self._ok, self._frame = ok, frame
                    self.last_error = error
                    self._sequence += 1
                    self._captured = time.perf_counter()
                    if ok:
                        self._capture_times.append(self._captured)
                        while len(self._capture_times)>2 and self._captured-self._capture_times[0]>2.:
                            self._capture_times.popleft()
                        if len(self._capture_times)>1:
                            self.source_fps=(len(self._capture_times)-1)/(self._captured-self._capture_times[0])
                    else:
                        self._capture_times.clear()
                        self.source_fps = 0.
                    self._condition.notify_all()
                if not ok:
                    time.sleep(.03)
                if hasattr(self._cap,"get") and time.perf_counter()-self._property_time>2.:
                    try:
                        self.properties={name:round(float(self._cap.get(prop)),2) for name,prop in (
                            ("reported_fps",cv2.CAP_PROP_FPS),("exposure",cv2.CAP_PROP_EXPOSURE),
                            ("auto_exposure_readback",cv2.CAP_PROP_AUTO_EXPOSURE))}
                    except (cv2.error,TypeError,ValueError):
                        self.properties={}
                    self._property_time=time.perf_counter()
        finally:
            self._cap.release()
            with self._condition:
                self._stop = True
                self._condition.notify_all()

    def read(self):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._stop or self._sequence>self._consumed, timeout=.5)
            if not ready or self._stop:
                return False, None
            self.dropped_frames += max(0,self._sequence-self._consumed-1)
            self._consumed = self._sequence
            self.capture_time = self._captured
            return self._ok, self._frame

    def release(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        # The reader releases the native capture after its read completes.
        self._thread.join(timeout=2.)
        return not self._thread.is_alive()


class DiagnosticWriter:
    """Persist immutable frame snapshots without blocking camera/tracking."""
    def __init__(self, capacity=4):
        self._jobs = queue.Queue(maxsize=capacity)
        self._results = queue.SimpleQueue()
        self._closed = False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name='bbox-diagnostics', daemon=True)
        self._thread.start()

    def submit(self, directory, tag, images, text):
        with self._state_lock:
            if self._closed:
                return False
            try:
                self._jobs.put_nowait((directory, tag, images, text))
                return True
            except queue.Full:
                return False

    def poll(self):
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def _loop(self):
        while True:
            try:
                job = self._jobs.get(timeout=.1)
            except queue.Empty:
                with self._state_lock:
                    if self._closed and self._jobs.empty():
                        return
                continue
            if job is None:
                return
            directory, tag, images, text = job
            try:
                os.makedirs(directory, exist_ok=True)
                wrote = 0
                for name, image in images:
                    path = os.path.join(directory, f'{tag}_{name}.png')
                    if not cv2.imwrite(path, image):
                        raise OSError(f'cannot write {path}')
                    wrote += 1
                with open(os.path.join(directory, 'diag.txt'), 'a', encoding='utf-8') as stream:
                    stream.write(text + '\n\n')
                self._results.put(dict(tag=tag, count=wrote, directory=directory, error=None))
                print(f'[diag] saved {wrote} png -> {directory}\n{text}', flush=True)
            except Exception as error:
                self._results.put(dict(tag=tag, error=str(error)))
                print(f'[diag] save failed: {error}', flush=True)

    def close(self, timeout=5.):
        # Stop accepting work, then drain every accepted snapshot. Enqueuing a
        # sentinel here could block forever when a stalled writer fills the queue.
        with self._state_lock:
            self._closed = True
        self._thread.join(timeout=max(0., float(timeout)))
        return not self._thread.is_alive()


class SingleFlightWorker:
    """At most one running job; completed results are consumed once."""
    def __init__(self, handler):
        self._handler = handler
        self._condition = threading.Condition()
        self._job = self._result = None
        self._busy = self._stop = False
        self._thread = threading.Thread(target=self._loop, name="bbox-search", daemon=True)
        self._thread.start()

    def submit(self, job):
        if job is None:
            raise ValueError("job must not be None")
        with self._condition:
            if self._stop or self._busy or self._result is not None:
                return False
            self._busy = True
            self._job = job
            self._condition.notify_all()
            return True

    def poll(self):
        with self._condition:
            result, self._result = self._result, None
            return result

    def _loop(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stop or self._job is not None)
                if self._stop:
                    return
                job, self._job = self._job, None
            start = time.perf_counter()
            try:
                result = self._handler(job)
                if not isinstance(result, Mapping):
                    raise TypeError("search handler must return a mapping")
                result = dict(result)
            except Exception as exc:
                epoch = job.get("epoch") if isinstance(job, Mapping) else None
                result = dict(epoch=epoch, error=f"{type(exc).__name__}: {exc}")
            result["elapsed_ms"] = (time.perf_counter()-start)*1000
            with self._condition:
                self._busy = False
                if self._stop:
                    return
                self._result = result

    def close(self):
        with self._condition:
            self._stop = True
            self._job = self._result = None
            self._condition.notify_all()
        self._thread.join(timeout=2.)
