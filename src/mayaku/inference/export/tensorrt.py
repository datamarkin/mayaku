"""TensorRT: a serialized engine for NVIDIA GPUs, built from the ONNX graph.

fp16 (the default) and fp32. An engine is specific to the GPU model and the
TensorRT version it was built with, so build it on the machine that serves
it. The engine format has no metadata slot, so the sidecar is length-prefixed
in front of the engine bytes (`metadata.strip_tensorrt_header` removes it).

int8 is not offered yet: TensorRT quantizes activations symmetrically only,
and the trained ranges are affine (a ReLU output's range is [0, max]).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mayaku.inference.export.metadata import embed_sidecar, strip_tensorrt_header
from mayaku.inference.export.onnx import onnx_graph
from mayaku.model import Detector

PRECISIONS = ("fp16", "fp32")


_WORKSPACE = 1 << 30


def runnable() -> bool:
    return torch.cuda.is_available()


def write(model: Detector, path: Path, canvas: tuple[int, int], precision: str,
          sidecar: Mapping[str, Any]) -> None:
    import tensorrt as trt

    if not torch.cuda.is_available():
        raise RuntimeError("building a TensorRT engine needs a CUDA GPU")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)            # explicit batch (TensorRT >= 10)
    parser = trt.OnnxParser(network, logger)
    with onnx_graph(model, canvas, int8=False) as onnx:
        if not parser.parse(onnx.read_bytes()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the ONNX graph:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)       # fp32 means fp32
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the engine; see its log above")
    path.write_bytes(bytes(engine))
    embed_sidecar(path, "tensorrt", dict(sidecar))


class Session:
    """The engine on the current CUDA device, with torch tensors as its I/O
    buffers."""

    batch = 1

    def __init__(self, path: Path, outputs: list[str], precision: str, device: str = "auto"):
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._engine = runtime.deserialize_cuda_engine(strip_tensorrt_header(path))
        self._ctx = self._engine.create_execution_context()
        # an engine runs on a CUDA device whatever `device` says
        self._device = torch.device(device if device.startswith("cuda") else "cuda")
        name = self._engine.get_tensor_name(0)
        self._input = torch.empty(tuple(self._engine.get_tensor_shape(name)),
                                  dtype=torch.float32, device=self._device)
        self._ctx.set_tensor_address(name, self._input.data_ptr())
        self.input_hw = tuple(self._input.shape[-2:])
        self._out = {n: torch.empty(tuple(self._engine.get_tensor_shape(n)),
                                    dtype=torch.float32, device=self._device)
                     for n in outputs}
        for n, t in self._out.items():
            self._ctx.set_tensor_address(n, t.data_ptr())
        self.outputs = outputs

    def __call__(self, x: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        self._input.copy_(torch.from_numpy(x))
        stream = torch.cuda.current_stream(self._device)
        self._ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return [self._out[n].cpu().numpy() for n in self.outputs]
