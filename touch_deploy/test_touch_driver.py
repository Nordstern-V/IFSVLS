#!/usr/bin/env python3
"""Host-side protocol tests for touch_driver.py (no PYNQ hardware required)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from touch_driver import TouchCalibration, TouchController


class FakeGPIO:
    def __init__(self, value=0):
        self.value = value
        self.writes = []

    def write(self, value):
        self.value = int(value) & 1
        self.writes.append(self.value)

    def read(self):
        return self.value


class FakeIIC:
    REPEAT_START = 1

    def __init__(self, address, registers):
        self.address = address
        self.registers = dict(registers)
        self.pointer = 0
        self.last_option = 0

    def wait(self):
        assert self.last_option != self.REPEAT_START, "wait() called before repeated-start receive"

    def send(self, address, data, length, option=0):
        assert address == self.address
        self.last_option = option
        payload = list(bytes(data)[:length])
        width = 1 if address == 0x38 else 2
        if len(payload) == width:
            self.pointer = int.from_bytes(bytes(payload), "big")
        else:
            register = int.from_bytes(bytes(payload[:width]), "big")
            for index, value in enumerate(payload[width:]):
                self.registers[register + index] = value
            self.pointer = register
        return length

    def receive(self, address, buffer, length, option=0):
        assert address == self.address
        self.last_option = option
        for index in range(length):
            buffer[index] = self.registers.get(self.pointer + index, 0)
        return length

def test_ft():
    regs = {
        0xA1: 0x30,
        0xA2: 0x03,
        0x02: 0x01,
        # XH, XL, YH, YL.  RTL swaps X/Y after extracting 12-bit values.
        0x03: 0x01,
        0x04: 0x02,
        0x05: 0x03,
        0x06: 0x04,
    }
    bus = FakeIIC(0x38, regs)
    reset = FakeGPIO()
    touch = TouchController(
        bus,
        reset_gpio=reset,
        calibration=TouchCalibration(raw_width=4096, raw_height=4096),
        release_debounce=2,
    )
    info = touch.initialize()
    assert info["controller"] == "FT"
    assert info["version"] == 0x3003
    event = touch.poll()
    assert event is not None and event.kind == "down"
    assert (event.raw_x, event.raw_y) == (0x304, 0x102)
    bus.registers[0x02] = 0
    assert touch.poll() is None
    event = touch.poll()
    assert event is not None and event.kind == "up"
    bus.registers[0x02] = 0x01
    event = touch.poll()
    assert event is not None and event.kind == "down"
    forced = touch.force_release()
    assert forced is not None and forced.kind == "up"
    assert touch.force_release() is None
    bus.registers[0x02] = 0
    assert touch.poll() is None
    assert reset.writes == [0, 1]


def test_gt():
    regs = {
        0x814E: 0x81,
        0x8150: 0x20,
        0x8151: 0x01,
        0x8152: 0x40,
        0x8153: 0x00,
    }
    touch = TouchController(FakeIIC(0x14, regs))
    info = touch.initialize(reset=False)
    assert info["controller"] == "GT"
    sample = touch.read_sample()
    assert sample is not None
    assert (sample.raw_x, sample.raw_y) == (0x0120, 0x0040)


if __name__ == "__main__":
    test_ft()
    test_gt()
    print("touch_driver host tests: PASS")


