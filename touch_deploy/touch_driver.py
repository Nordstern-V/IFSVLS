"""PYNQ driver for the FT/GT touch controller on the MIX_PLUS overlay.

The hardware design exposes a standard AXI IIC at ``touch_iic`` and two
one-bit AXI GPIO peripherals named ``touch_int_gpio`` and ``touch_rst_gpio``.
The register protocol follows ``sources/touch_reference/touch_dri.v``:
FT devices use 7-bit address 0x38 and 8-bit register addresses; GT devices use
7-bit address 0x14 and 16-bit register addresses.

This module intentionally does not import PYNQ at module import time.  That
keeps the protocol and event logic testable on a host PC and lets the board
entry point provide the actual AxiIIC/AxiGPIO objects.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

DRIVER_VERSION = "mix-plus-touch-20261024-r16"


class TouchError(RuntimeError):
    """Base class for touch driver errors."""


class TouchBusError(TouchError):
    """Raised when an IIC transaction cannot be completed."""


@dataclass
class TouchCalibration:
    """Map controller coordinates to the LCD coordinate system."""

    raw_width: int = 800
    raw_height: int = 480
    screen_width: int = 800
    screen_height: int = 480
    invert_x: bool = False
    invert_y: bool = False

    @staticmethod
    def _clamp(value: int, limit: int) -> int:
        return max(0, min(int(value), max(0, limit - 1)))

    @staticmethod
    def _scale(value: int, raw_limit: int, screen_limit: int) -> int:
        value = TouchCalibration._clamp(value, raw_limit)
        if raw_limit <= 1 or screen_limit <= 1:
            return 0
        return int(round(value * (screen_limit - 1) / float(raw_limit - 1)))

    def map(self, x: int, y: int) -> Tuple[int, int]:
        x = self._clamp(x, self.raw_width)
        y = self._clamp(y, self.raw_height)
        if self.invert_x:
            x = self.raw_width - 1 - x
        if self.invert_y:
            y = self.raw_height - 1 - y
        return (
            self._scale(x, self.raw_width, self.screen_width),
            self._scale(y, self.raw_height, self.screen_height),
        )


@dataclass
class TouchSample:
    """One valid sample read from the controller."""

    raw_x: int
    raw_y: int
    x: int
    y: int
    points: int
    status: int
    controller: str
    timestamp: float


@dataclass
class TouchEvent:
    """A debounced down/move/up event."""

    kind: str
    x: int
    y: int
    raw_x: int
    raw_y: int
    points: int
    controller: str
    timestamp: float


class _AxiIICAdapter:
    """Small compatibility adapter around PYNQ's AxiIIC object."""

    def __init__(self, bus: Any, repeat_start: Optional[int] = None) -> None:
        self.bus = bus
        self.repeat_start = repeat_start
        if self.repeat_start is None:
            self.repeat_start = int(getattr(bus, "REPEAT_START", 1))
        # PYNQ 2.x/3.x backs AxiIIC with cffi and its C API expects a cffi
        # unsigned-char buffer. Generic fake/test buses can use bytearray.
        # Reuse PYNQ's own FFI object when available.  Older cffi versions
        # can reject a pointer created by a different FFI instance.
        self._ffi = getattr(bus, "_ffi", None)
        if self._ffi is None:
            self._ffi = getattr(type(bus), "_ffi", None)
        if self._ffi is None and hasattr(bus, "_virtaddr"):
            try:
                import cffi
                self._ffi = cffi.FFI()
            except ImportError:
                self._ffi = None

    def _send_buffer(self, payload: bytes) -> Any:
        if self._ffi is None:
            return payload
        buffer = self._ffi.new("unsigned char[]", len(payload))
        for index, value in enumerate(payload):
            buffer[index] = value
        return buffer

    def _receive_buffer(self, length: int) -> Any:
        if self._ffi is None:
            return bytearray(length)
        return self._ffi.new("unsigned char[]", length)

    def _buffer_bytes(self, buffer: Any, length: int) -> bytes:
        if self._ffi is None:
            return bytes(buffer[:length])
        return bytes(self._ffi.buffer(buffer, length))

    def _wait(self) -> None:
        wait = getattr(self.bus, "wait", None)
        if callable(wait):
            try:
                wait()
            except Exception as error:
                # PYNQ's AxiIIC.wait() raises RuntimeError directly when
                # XIic_WaitBusFree times out.  Normalize that path to the
                # driver's recoverable error type so _read_register() can
                # reset SOFTR and retry the same transaction.
                raise TouchBusError("IIC wait failed: {}".format(error)) from error

    def recover(self) -> bool:
        """Reset a wedged AXI-IIC core without changing the touch device.

        PYNQ's ``AxiIIC`` wrapper exposes the AXI-Lite MMIO object but no
        public reset method.  AXI-IIC PG090 defines SOFTR at offset 0x40;
        writing 0xA clears the controller state and the bus-busy flag.  The
        fallback is useful with board-specific wrappers that provide a
        reset() method.  Host fake buses simply report that recovery is not
        available.
        """
        mmio = getattr(self.bus, "mmio", None)
        if mmio is None:
            mmio = getattr(self.bus, "_mmio", None)
        if mmio is None:
            # Some board specific wrappers expose register access directly
            # on the IIC object instead of publishing ``mmio``.
            mmio = self.bus
        writer = getattr(mmio, "write", None)
        if callable(writer):
            try:
                # AXI IIC PG090: SOFTR is at 0x40 and 0x0A resets the
                # transmit/receive state machines and releases a stale bus
                # busy flag.  PYNQ's AxiIIC keeps the actual MMIO object in
                # ``mmio``; ``_virtaddr`` is only a C pointer and cannot be
                # used for register writes.
                writer(0x40, 0xA)
                time.sleep(0.005)
                return True
            except Exception:
                pass
        reset = getattr(self.bus, "reset", None)
        if callable(reset):
            try:
                reset()
                time.sleep(0.005)
                return True
            except Exception:
                pass
        return False

    def send(self, address: int, payload: bytes, repeat: bool = False) -> None:
        payload = bytes(payload)
        method = getattr(self.bus, "send", None)
        if method is None:
            method = getattr(self.bus, "write", None)
        if method is None:
            raise TouchBusError("IIC object has neither send() nor write()")
        option = self.repeat_start if repeat else 0
        data_buffer = self._send_buffer(payload)
        try:
            method(address, data_buffer, len(payload), option)
        except TypeError:
            try:
                method(address, data_buffer, len(payload))
            except TypeError:
                try:
                    method(address, data_buffer)
                except Exception as error:
                    raise TouchBusError("IIC send failed: {}".format(error)) from error
        except Exception as error:
            raise TouchBusError("IIC send failed: {}".format(error)) from error
        # REPEAT_START keeps the bus busy until receive() completes.
        if not repeat:
            self._wait()

    def receive(self, address: int, length: int) -> bytes:
        method = getattr(self.bus, "receive", None)
        if method is None:
            method = getattr(self.bus, "read", None)
        if method is None:
            raise TouchBusError("IIC object has neither receive() nor read()")
        buffer = self._receive_buffer(length)
        try:
            result = method(address, buffer, length, 0)
        except TypeError:
            try:
                result = method(address, buffer, length)
            except TypeError:
                try:
                    result = method(address, length)
                except Exception as error:
                    raise TouchBusError("IIC receive failed: {}".format(error)) from error
            except Exception as error:
                raise TouchBusError("IIC receive failed: {}".format(error)) from error
        except Exception as error:
            raise TouchBusError("IIC receive failed: {}".format(error)) from error
        self._wait()
        if isinstance(result, (bytes, bytearray, list, tuple)):
            data = bytes(result)
        else:
            data = self._buffer_bytes(buffer, length)
        if len(data) < length:
            raise TouchBusError(
                "IIC returned {} bytes, expected {}".format(len(data), length)
            )
        return data[:length]

class TouchController:
    """FT/GT controller driver and debounced event source.

    Parameters
    ----------
    iic:
        A PYNQ ``AxiIIC`` instance, or an object implementing compatible
        ``send``/``receive`` methods.
    reset_gpio:
        One-bit AxiGPIO channel.  The driver drives it low for reset and high
        for run.  It may be omitted when the panel reset is already released.
    int_gpio:
        Optional one-bit AxiGPIO input channel.  The reference RTL polls the
        status register and does not use INT as a gate, so this signal is
        exposed for diagnostics but is not required for polling.
    """

    FT_ADDR = 0x38
    GT_ADDR = 0x14
    FT_VERSION_REG = 0xA1
    FT_VERSION_VALUES = frozenset((0x3003, 0x0001, 0x0002, 0x0000))
    FT_STATUS_REG = 0x02
    FT_COORD_REG = 0x03
    GT_STATUS_REG = 0x814E
    GT_COORD_REG = 0x8150

    def __init__(
        self,
        iic: Any,
        reset_gpio: Optional[Any] = None,
        int_gpio: Optional[Any] = None,
        calibration: Optional[TouchCalibration] = None,
        release_debounce: int = 2,
        reset_low_seconds: float = 0.020,
        reset_settle_seconds: float = 0.100,
    ) -> None:
        self._iic = _AxiIICAdapter(iic)
        self.reset_gpio = reset_gpio
        self.int_gpio = int_gpio
        self.calibration = calibration or TouchCalibration()
        self.release_debounce = max(1, int(release_debounce))
        self.reset_low_seconds = float(reset_low_seconds)
        self.reset_settle_seconds = float(reset_settle_seconds)
        self.controller: Optional[str] = None
        self.address: Optional[int] = None
        self.version: Optional[int] = None
        self._last_sample: Optional[TouchSample] = None
        self._invalid_count = 0
        self._initialized = False

    @classmethod
    def from_overlay(
        cls,
        overlay: Any,
        iic_name: str = "touch_iic",
        int_name: str = "touch_int_gpio",
        reset_name: str = "touch_rst_gpio",
        **kwargs: Any,
    ) -> "TouchController":
        """Construct the driver from a downloaded PYNQ Overlay."""

        from pynq.lib import AxiGPIO
        from pynq.lib.iic import AxiIIC

        ip_dict = overlay.ip_dict
        missing = [name for name in (iic_name, int_name, reset_name) if name not in ip_dict]
        if missing:
            raise TouchError("HWH is missing touch IP: {}".format(", ".join(missing)))
        iic = AxiIIC(ip_dict[iic_name])
        int_gpio = AxiGPIO(ip_dict[int_name]).channel1
        reset_gpio = AxiGPIO(ip_dict[reset_name]).channel1
        try:
            reset_gpio.setdirection("out")
        except AttributeError:
            pass
        try:
            int_gpio.setdirection("in")
        except AttributeError:
            pass
        return cls(iic, reset_gpio=reset_gpio, int_gpio=int_gpio, **kwargs)

    @staticmethod
    def _gpio_write(gpio: Any, value: int) -> None:
        if gpio is None:
            return
        writer = getattr(gpio, "write", None)
        if callable(writer):
            try:
                writer(int(value))
            except TypeError:
                # Older PYNQ AXI GPIO channels require write(value, mask).
                writer(int(value), 1)
            return
            return
        if int(value):
            method = getattr(gpio, "on", None)
        else:
            method = getattr(gpio, "off", None)
        if callable(method):
            method()
            return
        raise TouchError("reset GPIO does not provide write()/on()/off()")

    @staticmethod
    def _gpio_read(gpio: Any) -> Optional[int]:
        if gpio is None:
            return None
        reader = getattr(gpio, "read", None)
        if callable(reader):
            return int(reader()) & 1
        return None

    def reset(self) -> None:
        """Apply the panel reset sequence from the reference RTL."""

        self._gpio_write(self.reset_gpio, 0)
        time.sleep(self.reset_low_seconds)
        self._gpio_write(self.reset_gpio, 1)
        time.sleep(self.reset_settle_seconds)

    def release_reset(self) -> None:
        """Release panel reset after an interrupted validation attempt."""

        self._gpio_write(self.reset_gpio, 1)

    def hard_reset(self) -> None:
        """Pulse the touch controller reset GPIO without using IIC.

        This method is deliberately independent from :meth:`initialize` so
        it can be called by the IIC watchdog while another thread is blocked
        inside PYNQ's ``XIic_WaitBusFree``.  The watchdog follows this pulse
        with an AXI-IIC SOFTR write; the polling thread then calls
        :meth:`reinitialize` after its in-flight transaction returns.
        """

        if self.reset_gpio is None:
            raise TouchError("touch reset GPIO is unavailable")
        self.reset()
        # A reset invalidates the current contact state even before the
        # controller is identified again.  This prevents a stale ``move`` or
        # ``up`` event from leaking into the UI after recovery.
        self._initialized = False
        self._last_sample = None
        self._invalid_count = 0

    def reinitialize(self, reset: bool = False) -> Dict[str, Any]:
        """Re-detect and configure the controller after ``hard_reset``.

        ``reset`` defaults to ``False`` because the watchdog already pulsed
        the reset GPIO.  Callers that need a complete standalone cycle can
        pass ``reset=True`` and use the regular :meth:`initialize` path.
        """

        return self.initialize(reset=reset)

    def _read_register(self, address: int, register: int, length: int, width: int) -> bytes:
        if width == 1:
            prefix = bytes((register & 0xFF,))
        elif width == 2:
            prefix = bytes(((register >> 8) & 0xFF, register & 0xFF))
        else:
            raise ValueError("register width must be 1 or 2")
        # FT/GT register reads require a repeated-start transaction: keep
        # the bus between the register pointer write and the data read.
        try:
            self._iic.send(address, prefix, repeat=True)
            return self._iic.receive(address, length)
        except TouchBusError:
            # A PYNQ 2.7 XIic_Send/XIic_Recv failure can leave the AXI-IIC
            # core's bus-busy bit set.  Reset once and retry this same read;
            # the worker keeps its normal poll period, so recovery adds only
            # a short transaction-local delay instead of a long backoff.
            if not self._iic.recover():
                raise
            self._iic.send(address, prefix, repeat=True)
            return self._iic.receive(address, length)

    def _write_register(self, address: int, register: int, data: Sequence[int], width: int) -> None:
        if width == 1:
            prefix = bytes((register & 0xFF,))
        elif width == 2:
            prefix = bytes(((register >> 8) & 0xFF, register & 0xFF))
        else:
            raise ValueError("register width must be 1 or 2")
        self._iic.send(address, prefix + bytes(data), repeat=False)

    def _detect_controller(self) -> Dict[str, Any]:
        try:
            version_bytes = self._read_register(self.FT_ADDR, self.FT_VERSION_REG, 2, 1)
            version = int.from_bytes(version_bytes, "big")
            if version in self.FT_VERSION_VALUES:
                self.controller = "FT"
                self.address = self.FT_ADDR
                self.version = version
                return {"controller": "FT", "address": self.address, "version": version}
        except TouchBusError:
            pass

        # The RTL treats an unrecognised/failed FT version read as GT.  Probe
        # GT's status register so a missing device is reported immediately.
        self._read_register(self.GT_ADDR, self.GT_STATUS_REG, 1, 2)
        self.controller = "GT"
        self.address = self.GT_ADDR
        self.version = None
        return {"controller": "GT", "address": self.address, "version": None}

    def initialize(self, reset: bool = True) -> Dict[str, Any]:
        """Reset, identify, and configure the touch controller."""

        if reset:
            self.reset()
        info = self._detect_controller()
        if self.controller == "FT":
            # Same four writes as touch_dri.v: mode, interrupt mode,
            # touch threshold, and active-period configuration.
            for register, value in ((0x00, 0), (0xA4, 0), (0x80, 22), (0x88, 12)):
                self._write_register(self.FT_ADDR, register, (value,), 1)
        self._initialized = True
        self._last_sample = None
        self._invalid_count = 0
        return info

    @property
    def initialized(self) -> bool:
        return self._initialized

    def recover_bus(self) -> bool:
        """Reset the AXI-IIC state machine after a failed transaction.

        This is intentionally separate from the panel reset: the touch
        controller keeps its configuration while the PL IIC core is cleared.
        It is safe for the poll worker to call after a send/receive failure.
        """

        return bool(self._iic.recover())

    @property
    def int_level(self) -> Optional[int]:
        """Return the raw INT GPIO level when that optional GPIO is present."""

        return self._gpio_read(self.int_gpio)

    def _read_sample(self) -> Optional[TouchSample]:
        if not self._initialized or self.address is None or self.controller is None:
            raise TouchError("call initialize() before polling")
        if self.controller == "FT":
            width = 1
            status_reg = self.FT_STATUS_REG
            coord_reg = self.FT_COORD_REG
        else:
            width = 2
            status_reg = self.GT_STATUS_REG
            coord_reg = self.GT_COORD_REG
        # FT5x06 exposes the touch count and the first point coordinates in
        # consecutive 8-bit registers (0x02..0x06).  Read them in one
        # repeated-start transaction.  The previous implementation used one
        # transaction for status and another for coordinates; at the 20 Hz
        # worker rate that doubled the AXI-IIC traffic and eventually left
        # the PYNQ 2.7 controller busy during a long drag.
        if self.controller == "FT":
            ft_payload = self._read_register(
                self.address, self.FT_STATUS_REG, 5, width
            )
            status = ft_payload[0]
            raw = ft_payload[1:5]
        else:
            status = self._read_register(self.address, status_reg, 1, width)[0]
            raw = None
        points = status & 0x0F
        if self.controller == "FT":
            valid = 0 < points <= 5
        else:
            valid = bool(status & 0x80) and 0 < points <= 5
        was_active = self._last_sample is not None
        if not valid:
            return None
        if raw is None:
            raw = self._read_register(self.address, coord_reg, 4, width)
        # Real PYNQ AxiIIC exposes _virtaddr; avoid the optional clear write there.
        if not was_active and not hasattr(self._iic.bus, "_virtaddr"):
            try:
                self._write_register(self.address, status_reg, (0,), width)
            except Exception:
                # Do not discard a valid coordinate when the optional clear
                # transaction is rejected by the vendor IIC image.
                pass
        if self.controller == "FT":
            # FT5x06 returns [XH, XL, YH, YL]. The reference RTL
            # keeps the high-byte low nibble, then swaps the axes.
            x = ((raw[2] & 0x0F) << 8) | raw[3]
            y = ((raw[0] & 0x0F) << 8) | raw[1]
        else:
            x = raw[0] | (raw[1] << 8)
            y = raw[2] | (raw[3] << 8)
        mapped_x, mapped_y = self.calibration.map(x, y)
        return TouchSample(
            raw_x=x,
            raw_y=y,
            x=mapped_x,
            y=mapped_y,
            points=points,
            status=status,
            controller=self.controller,
            timestamp=time.monotonic(),
        )

    def read_sample(self) -> Optional[TouchSample]:
        """Read one sample without event debouncing."""

        return self._read_sample()

    def force_release(self) -> Optional[TouchEvent]:
        """Finish the current contact without another IIC transaction.

        Some FT controllers drive INT as a pulse.  When the application gates
        IIC reads on that signal, the normal invalid-status polls are skipped
        while INT is high, so ``poll()`` cannot synthesize the ``up`` event.
        The worker uses this method after INT has stayed idle for the release
        guard interval.  It only updates local event state and therefore also
        works when the AXI-IIC bus is recovering from a failed transaction.
        """

        if self._last_sample is None:
            return None
        previous = self._last_sample
        self._last_sample = None
        self._invalid_count = 0
        return TouchEvent(
            kind="up",
            x=previous.x,
            y=previous.y,
            raw_x=previous.raw_x,
            raw_y=previous.raw_y,
            points=0,
            controller=previous.controller,
            timestamp=time.monotonic(),
        )

    def poll(self) -> Optional[TouchEvent]:
        """Poll the controller and return at most one debounced event."""

        sample = self._read_sample()
        if sample is not None:
            self._invalid_count = 0
            kind = "down" if self._last_sample is None else "move"
            self._last_sample = sample
            return TouchEvent(
                kind=kind,
                x=sample.x,
                y=sample.y,
                raw_x=sample.raw_x,
                raw_y=sample.raw_y,
                points=sample.points,
                controller=sample.controller,
                timestamp=sample.timestamp,
            )

        if self._last_sample is None:
            return None
        self._invalid_count += 1
        if self._invalid_count < self.release_debounce:
            return None
        previous = self._last_sample
        self._last_sample = None
        self._invalid_count = 0
        return TouchEvent(
            kind="up",
            x=previous.x,
            y=previous.y,
            raw_x=previous.raw_x,
            raw_y=previous.raw_y,
            points=0,
            controller=previous.controller,
            timestamp=time.monotonic(),
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitstream", type=str, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser


def main() -> int:
    """Minimal command-line validation entry point for board use."""

    args = _build_arg_parser().parse_args()
    from pynq import Overlay

    overlay = Overlay(args.bitstream, ignore_version=True)
    touch = TouchController.from_overlay(overlay)
    info = touch.initialize()
    print("Touch controller:", json.dumps(info, sort_keys=True))
    print("INT level:", touch.int_level)
    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        event = touch.poll()
        if event is not None:
            print(json.dumps(event.__dict__, sort_keys=True))
        time.sleep(0.005)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





















