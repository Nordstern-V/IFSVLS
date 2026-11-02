#!/usr/bin/env python3
"""PYNQ-side TCU feature adapter for the PEFSL bbox pipeline.

The TCU model consumes a 32x32 uint8 BGR frame after the same preprocessing
used by PEFSL-main/bbox_demo.py.  The Tensil tmodel returns 84 scalars:
the first 80 are the ResNet9 feature view used by PEFSL and the final four
are diagnostic values.  This module keeps that boundary explicit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


SIDE = 32
RGB_CHANNELS = 3
TCU_LANES = 12
FEATURE_DIM = 80
RAW_OUTPUT_DIM = 84

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame32: np.ndarray) -> np.ndarray:
    """Return the exact PEFSL float32 preprocessing for one 32x32 BGR frame."""
    frame = np.asarray(frame32)
    if frame.shape != (SIDE, SIDE, RGB_CHANNELS):
        raise ValueError(
            "frame32 must have shape (32, 32, 3), got {}".format(frame.shape)
        )
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.integer):
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            raise TypeError("frame32 must be uint8")
    # Keep operator order identical to bbox_demo.py/main.py.
    return frame.astype(np.float32) / 255.0 - MEAN / STD


def pack_tcu_input(preprocessed: np.ndarray) -> np.ndarray:
    """Pad RGB pixels to the 12-lane TCU input vector."""
    pre = np.asarray(preprocessed, dtype=np.float32)
    if pre.shape == (SIDE, SIDE, RGB_CHANNELS):
        flat = pre.reshape(SIDE * SIDE, RGB_CHANNELS)
    elif pre.shape == (1, SIDE, SIDE, RGB_CHANNELS):
        flat = pre[0].reshape(SIDE * SIDE, RGB_CHANNELS)
    else:
        raise ValueError(
            "preprocessed must have shape (32,32,3) or (1,32,32,3), got {}".format(
                pre.shape
            )
        )
    vec = np.zeros((SIDE * SIDE, TCU_LANES), dtype=np.float32)
    vec[:, :RGB_CHANNELS] = flat
    return vec


def split_tcu_output(output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (feature80, diagnostic_tail4) from Driver.run output.

    Driver concatenates the two same-name Output segments as 72 + 12
    scalars. The ONNX feature view is all 72 scalars of the first segment
    followed by the first 8 scalars of the second segment.
    """
    raw = np.asarray(output, dtype=np.float32).reshape(-1)
    if raw.size != RAW_OUTPUT_DIM:
        raise ValueError(
            "TCU output has {} scalars; expected at least {}".format(
                raw.size, RAW_OUTPUT_DIM
            )
        )
    feature = np.concatenate((raw[:72], raw[72:80])).copy()
    tail = raw[80:84].copy()
    return feature, tail


class TcuFeatureExtractor:
    """Load one PYNQ overlay/model and expose an 80D feature call."""

    def __init__(
        self,
        overlay: Any,
        tmodel_path: Union[os.PathLike, str],
        tcu_path: Optional[Union[os.PathLike, str]] = None,
        debug: bool = False,
    ):
        self.overlay = overlay
        self.tmodel_path = Path(tmodel_path).resolve()
        if tcu_path is None:
            tcu_path = self.tmodel_path.parent
        tcu_root = str(Path(tcu_path).resolve())
        if tcu_root not in sys.path:
            sys.path.insert(0, tcu_root)

        from tcu_pynq.architecture import Architecture
        from tcu_pynq.data_type import DataType
        from tcu_pynq.driver import Driver

        if not hasattr(overlay, "axi_dma_0"):
            raise RuntimeError("overlay does not expose axi_dma_0")
        model = json.loads(self.tmodel_path.read_text(encoding="utf-8"))
        arch_json = dict(model["arch"])
        arch_json["data_type"] = DataType[arch_json["data_type"]]
        arch = Architecture(**arch_json)

        dma = overlay.axi_dma_0
        dma_buffer_size = self._dma_buffer_size(dma, debug)
        self.driver = Driver(
            arch, dma, dma_buffer_size=dma_buffer_size, debug=debug
        )
        self.driver.load_model(str(self.tmodel_path))
        self.input_name = model["inputs"][0]["name"]
        self.output_name = model["outputs"][0]["name"]
        self.debug = bool(debug)

    @staticmethod
    def _dma_buffer_size(dma: Any, debug: bool = False) -> int:
        """Derive the PYNQ 2.7 DMA transfer limit from the HWH parameters."""
        params = getattr(dma, "description", {}).get("parameters", {})
        width = int(
            params.get("C_M_AXI_MM2S_DATA_WIDTH", params.get("c_m_axi_mm2s_data_width", 32))
        )
        length_width = params.get(
            "C_SG_LENGTH_WIDTH", params.get("c_sg_length_width")
        )
        if length_width is None:
            value = getattr(dma, "buffer_max_size", None)
            return int(value) if value is not None else 1048576
        length_width = int(length_width)
        if length_width <= 0 or length_width > 31:
            raise RuntimeError("invalid DMA C_SG_LENGTH_WIDTH={}".format(length_width))
        max_bytes = (1 << length_width) - 1
        size = max(1, (max_bytes - 1) // max(1, width // 8))
        if debug:
            print(
                "DMA transfer limit: {} bytes, AXI word: {} bytes, "
                "Tensil buffer words: {}".format(max_bytes, width // 8, size)
            )
        return size

    def run_preprocessed(self, preprocessed: np.ndarray) -> Dict[str, np.ndarray]:
        vec = pack_tcu_input(preprocessed)
        result = self.driver.run({self.input_name: vec})
        if self.output_name not in result:
            raise RuntimeError(
                "TCU result has no output {!r}; keys={}".format(
                    self.output_name, list(result)
                )
            )
        raw = np.asarray(result[self.output_name], dtype=np.float32).reshape(-1)
        if raw.size != RAW_OUTPUT_DIM:
            raise RuntimeError(
                "TCU output has {} scalars; expected {}".format(raw.size, RAW_OUTPUT_DIM)
            )
        feature, tail = split_tcu_output(raw)
        return {
            "feature": feature,
            "tail": tail,
            "raw": raw,
        }

    def extract(self, frame32: np.ndarray) -> np.ndarray:
        """Extract the 80D feature consumed by the PEFSL classifier."""
        return self.run_preprocessed(preprocess_frame(frame32))["feature"]

    def extract_with_diagnostics(self, frame32: np.ndarray) -> Dict[str, np.ndarray]:
        return self.run_preprocessed(preprocess_frame(frame32))



class BackboneTensilWrapper:
    """Drop-in callable for PEFSL's get_model Tensil backend.

    bbox_demo.py preprocesses the image before calling the backbone. Input is
    one (1, 32, 32, 3) float32 batch and return value is one (1, 80) batch.
    """

    def __init__(self, overlay, path_tmodel, debug=False):
        self.extractor = TcuFeatureExtractor(
            overlay, path_tmodel, debug=debug
        )

    def __call__(self, single_image_batch):
        batch = np.asarray(single_image_batch, dtype=np.float32)
        if batch.ndim != 4 or batch.shape[0] != 1 or batch.shape[-1] != 3:
            raise ValueError(
                "expected one preprocessed image with shape (1,32,32,3), got {}".format(
                    batch.shape
                )
            )
        result = self.extractor.run_preprocessed(batch[0])
        return result["feature"][None, :]

def load_overlay(bitstream: Union[os.PathLike, str]) -> Any:
    """Load the overlay with compatibility for PYNQ 2.7 vendor images."""
    from pynq import Overlay

    path = str(Path(bitstream).resolve())
    try:
        return Overlay(path, ignore_version=True)
    except TypeError:
        return Overlay(path)






