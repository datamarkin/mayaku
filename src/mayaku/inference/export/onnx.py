"""ONNX: the canonical artifact, and the source the OpenVINO and TensorRT
builds start from. fp32 is the structural five-op graph; int8 (a
quantization-aware model) is the same graph with the trained ranges as
QuantizeLinear / DequantizeLinear around every conv."""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mayaku.inference.export.metadata import embed_sidecar
from mayaku.model import Detector
from mayaku.model.contract import export_onnx

PRECISIONS = ("fp32", "int8")


def runnable() -> bool:
    return True


@contextlib.contextmanager
def onnx_graph(model: Detector, canvas: tuple[int, int], int8: bool) -> Iterator[Path]:
    """The model as a contract-checked ONNX file in a temporary directory,
    for the runtimes that build from ONNX."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        export_onnx(model, str(path), canvas, int8=int8)
        yield path


def write(model: Detector, path: Path, canvas: tuple[int, int], precision: str,
          sidecar: Mapping[str, Any]) -> None:
    export_onnx(model, str(path), canvas, int8=precision == "int8")
    embed_sidecar(path, "onnx", dict(sidecar))


class Session:
    """onnxruntime, on CUDA when asked for and available, else the CPU."""

    def __init__(self, path: Path, outputs: list[str], precision: str, device: str = "auto"):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device.startswith(("cuda", "auto")) and \
                "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._sess = ort.InferenceSession(str(path), providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input, self.outputs = inp.name, outputs
        self.input_hw = tuple(inp.shape[-2:])
        # traced at a fixed batch size (1 unless exported otherwise)
        self.batch = inp.shape[0] if isinstance(inp.shape[0], int) else None

    def __call__(self, x: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        return list(self._sess.run(self.outputs, {self._input: x}))
