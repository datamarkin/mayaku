"""TensorRT: a serialized engine for NVIDIA GPUs, built from the ONNX graph.

fp16 (the default), int8 and fp32. An engine is specific to the GPU model and
the TensorRT version it was built with, so build it on the machine that
serves it. The engine format has no metadata slot, so the sidecar is
length-prefixed in front of the engine bytes (`metadata.strip_tensorrt_header`
removes it).

int8 is the explicit int8 graph (the trained ranges as Q/DQ pairs, with add
operands quantized and one scale per tensor, see `mayaku.model.quant`), so
TensorRT runs exactly what was trained; it needs symmetric activation ranges
(TensorRT has no zero point), the default since `model.qat_activations`.

TensorRT 10 builds reduced precision from builder flags. TensorRT 11 is
strongly typed: the flags are gone and the graph itself has to carry fp16, so
the ONNX is converted first with NVIDIA ModelOpt AutoCast (the
`nvidia-modelopt[onnx]` package), on a flat gray image, which converts every
node, as TensorRT 10's fp16 flag does.
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

PRECISIONS = ("fp16", "int8", "fp32")


_WORKSPACE = 1 << 30


def runnable() -> bool:
    return torch.cuda.is_available()


def write(model: Detector, path: Path, canvas: tuple[int, int], precision: str,
          sidecar: Mapping[str, Any], batch: int = 1) -> None:
    """The engine at a fixed batch (1 for `export`, which checks it against
    the model; a larger fixed batch for throughput)."""
    import tensorrt as trt

    from mayaku.model.quant import activation_scheme

    if not torch.cuda.is_available():
        raise RuntimeError("building a TensorRT engine needs a CUDA GPU")
    int8 = precision == "int8"
    if int8 and activation_scheme(model) != "symmetric":
        raise ValueError("TensorRT runs int8 with symmetric activation ranges only; this model "
                         "was trained affine (train with model.qat_activations=symmetric)")
    strongly_typed = int(trt.__version__.split(".")[0]) >= 11
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with onnx_graph(model, canvas, int8=int8, batch=batch) as onnx:
        if strongly_typed and precision != "fp32":
            onnx = _fp16_graph(onnx, canvas, batch)
        if not parser.parse(onnx.read_bytes()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the ONNX graph:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE)
    if not strongly_typed:
        if precision == "fp32":
            config.clear_flag(trt.BuilderFlag.TF32)   # fp32 means fp32
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the engine; see its log above")
    path.write_bytes(bytes(engine))
    embed_sidecar(path, "tensorrt", dict(sidecar))


def _fp16_graph(src: Path, canvas: tuple[int, int], batch: int = 1) -> Path:
    """The ONNX graph in fp16 with fp32 inputs and outputs, for a strongly
    typed build (TensorRT 11): ModelOpt AutoCast, calibrated on a flat gray
    image so that every node converts. Written next to `src`."""
    try:
        from modelopt.onnx import autocast
    except ImportError as e:
        raise ImportError("TensorRT 11 builds fp16 and int8 from an fp16 ONNX graph; "
                          "install NVIDIA ModelOpt: pip install 'nvidia-modelopt[onnx]'") from e
    import onnx

    gray = np.full((batch, 3, *canvas), 114 / 255, np.float32)
    out = src.with_suffix(".fp16.onnx")
    onnx.save(autocast.convert_to_mixed_precision(str(src), low_precision_type="fp16", keep_io_types=True,
                                                  calibration_data={"images": gray}), str(out))
    return out


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
