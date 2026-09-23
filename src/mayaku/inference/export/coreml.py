"""CoreML: an ML Program package for Apple devices, traced from the model.

fp16 (the default) runs on the Apple Neural Engine; fp32 on the GPU / CPU.
int8 is W8A8 from the trained ranges: int8 weights, and every conv input
quantized and dequantized with its observed range (iOS 17 / macOS 14 ops).
It also runs on the Neural Engine, but on an M1 Max it is slower than fp16
(3.9 against 3.4 ms for tier n at 640x640): chips before A17 Pro / M4 have
no faster int8 path, so the quantize steps are pure cost there. The sidecar
sits in ``user_defined_metadata``, written before saving.
"""

from __future__ import annotations

import platform
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mayaku.inference.export.metadata import SIDECAR_KEY, sidecar_blob
from mayaku.model import Detector
from mayaku.model.quant import deploy_mode

PRECISIONS = ("fp16", "fp32", "int8")


def runnable() -> bool:
    """Core ML runs on macOS only (a package can be written anywhere)."""
    return platform.system() == "Darwin"


def write(model: Detector, path: Path, canvas: tuple[int, int], precision: str,
          sidecar: Mapping[str, Any]) -> None:
    import coremltools as ct

    int8 = precision == "int8"
    x = torch.zeros(1, 3, *canvas)
    # coremltools converts TorchScript; torch's deprecation of it and its note
    # on the list output say nothing actionable here
    with torch.no_grad(), deploy_mode(model, int8), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(model, x)
    if int8:
        _register_int8_converters()
    ml = ct.convert(
        traced, inputs=[ct.TensorType(name="images", shape=tuple(x.shape))],
        outputs=[ct.TensorType(name=n) for n in model.out_names], convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14 if int8 else ct.target.macOS13)
    ml.user_defined_metadata[SIDECAR_KEY] = sidecar_blob(dict(sidecar))
    ml.save(str(path))


def _register_int8_converters() -> None:
    """Teach coremltools PyTorch's fake-quantize ops, which the int8 trace
    carries: per-tensor activations become quantize -> dequantize, per-channel
    weights int8 constants dequantized at load."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

    @register_torch_op(override=True)
    def fake_quantize_per_tensor_affine(context: Any, node: Any) -> None:
        x, scale, zp, _, _ = _get_inputs(context, node, expected=5)
        s, z = np.float32(scale.val), np.int8(zp.val)
        q = mb.quantize(input=x, scale=s, zero_point=z, output_dtype="int8")
        context.add(mb.dequantize(input=q, scale=s, zero_point=z, name=node.name))

    @register_torch_op(override=True)
    def fake_quantize_per_channel_affine(context: Any, node: Any) -> None:
        w, scale, _, axis, _, _ = _get_inputs(context, node, expected=6)
        s = scale.val.astype(np.float32)
        q = np.clip(np.round(w.val / s.reshape(-1, 1, 1, 1)), -128, 127).astype(np.int8)
        context.add(mb.constexpr_affine_dequantize(
            quantized_data=q, zero_point=np.int8(0), scale=s, axis=int(axis.val),
            name=node.name))


class Session:
    """Core ML; `device` "cpu" pins it to the CPU, anything else lets Core ML
    place it (the Neural Engine where it can)."""

    batch = 1

    def __init__(self, path: Path, outputs: list[str], precision: str, device: str = "auto"):
        import coremltools as ct

        units = ct.ComputeUnit.CPU_ONLY if device == "cpu" else ct.ComputeUnit.ALL
        self._model = ct.models.MLModel(str(path), compute_units=units)
        shape = self._model.get_spec().description.input[0].type.multiArrayType.shape
        self.input_hw = tuple(int(v) for v in shape[-2:])
        self.outputs = outputs

    def __call__(self, x: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
        out = self._model.predict({"images": x})
        return [np.asarray(out[n], np.float32) for n in self.outputs]
