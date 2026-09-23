"""OpenVINO: IR built from the ONNX graph, for Intel CPUs, GPUs and NPUs.

fp32 keeps fp32 weights; fp16 stores them at half precision; int8 comes from
the int8 ONNX graph, whose Q/DQ pairs OpenVINO turns into its own int8
FakeQuantize with the trained ranges. The sidecar sits in the IR's
``rt_info``, written before saving (an IR cannot be re-saved in place).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mayaku.inference.export.metadata import SIDECAR_KEY, sidecar_blob
from mayaku.inference.export.onnx import onnx_graph
from mayaku.model import Detector

PRECISIONS = ("fp32", "fp16", "int8")


def runnable() -> bool:
    return True


def write(model: Detector, path: Path, canvas: tuple[int, int], precision: str,
          sidecar: Mapping[str, Any]) -> None:
    import openvino as ov

    with onnx_graph(model, canvas, int8=precision == "int8") as onnx:
        ir = ov.convert_model(str(onnx))
    ir.set_rt_info(sidecar_blob(dict(sidecar)), [SIDECAR_KEY])
    ov.save_model(ir, str(path), compress_to_fp16=precision == "fp16")


class Session:
    """The OpenVINO CPU plugin; fp32 and int8 artifacts run at fp32 inference
    precision (some CPUs would otherwise pick bf16)."""

    def __init__(self, path: Path, outputs: list[str], precision: str, device: str = "auto"):
        import openvino as ov

        core = ov.Core()
        config = {} if precision == "fp16" else {"INFERENCE_PRECISION_HINT": "f32"}
        self._compiled = core.compile_model(core.read_model(str(path)), "CPU", config)
        inp = self._compiled.inputs[0]
        self._ports = [self._compiled.output(n) for n in outputs]
        self.input_hw = tuple(inp.get_shape())[-2:]
        self.batch = int(inp.get_shape()[0])

    def __call__(self, x: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        res = self._compiled([x])
        return [res[p] for p in self._ports]
