"""Export a trained detector to a deployable artifact with its sidecar embedded.

Every target gets the same treatment: the model's deploy graph
(`Detector.for_deploy`) at its canvas, the sidecar embedded in the artifact's
own metadata slot (with the target and precision added), and the artifact run
by its own runtime and its raw output maps compared against the Torch model's
before it is returned. The targets built from ONNX check the five-op
contract on it; Core ML traces the same module.

A target module provides `PRECISIONS` (the first is the default),
`write(model, path, canvas, precision, sidecar)`, `runnable()` (whether its
runtime works on this host) and `Session(path, outputs, precision, device)`.

| target   | file         | precisions (first = default) | runtime          |
|----------|--------------|------------------------------|------------------|
| onnx     | `.onnx`      | fp32, int8                   | onnxruntime      |
| coreml   | `.mlpackage` | fp16, fp32, int8             | Core ML (macOS)  |
| openvino | `.xml`       | fp32, fp16, int8             | OpenVINO         |
| tensorrt | `.engine`    | fp16, fp32                   | TensorRT (CUDA)  |

int8 needs a quantization-aware model and is exactly the network it was
trained as: every conv's input quantized with its observed range, weights per
channel. The host decode is the same for every artifact
(`mayaku.inference.decode`).
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from mayaku.inference.export.metadata import SUFFIX_TO_TARGET, target_from_suffix
from mayaku.model import Detector
from mayaku.model.quant import deploy_mode

__all__ = ["TARGETS", "Target", "export", "open_session"]


@dataclasses.dataclass(frozen=True)
class Target:
    suffix: str
    module: ModuleType              # its `write`, `Session` and `PRECISIONS`

    @property
    def precisions(self) -> tuple[str, ...]:
        """What it exports; the first is the default."""
        return tuple(self.module.PRECISIONS)


TARGETS = {t: Target(s, importlib.import_module(f"mayaku.inference.export.{t}"))
           for s, t in SUFFIX_TO_TARGET.items()}

# Parity tolerance, relative to the largest output value: fp32 is float noise,
# fp16 its rounding over a deep stack, int8 one runtime's integer kernels
# against the simulation it was trained with.
_TOLERANCE = {"fp32": 1e-3, "fp16": 2e-2, "int8": 5e-2}


def export(model: Detector, sidecar: Mapping[str, Any], target: str, path: str | Path,
           precision: str | None = None) -> Path:
    """Write `model` (a trained `Detector`, left untouched) as a `target`
    artifact at `path` with `sidecar` embedded; `precision` defaults to the
    target's first. Raises if the artifact's raw maps disagree with the Torch
    model's. Parity is skipped, with the artifact still written, only where
    its runtime cannot run on this host (Core ML off macOS)."""
    if target not in TARGETS:
        raise ValueError(f"unknown export target {target!r}; available: {', '.join(TARGETS)}")
    t = TARGETS[target]
    precision = precision or t.precisions[0]
    if precision not in t.precisions:
        raise ValueError(f"{target} exports {', '.join(t.precisions)}, not {precision}")
    if precision == "int8" and not sidecar["quant"]["qat"]:
        raise ValueError("int8 needs a quantization-aware model (train it with model.qat)")
    path, canvas = Path(path), tuple(sidecar["canvas_hw"])
    m = model.for_deploy().cpu()
    sidecar = {**sidecar, "export": {"target": target, "precision": precision}}
    t.module.write(m, path, canvas, precision, sidecar)
    if not t.module.runnable():
        return path
    err, scale = parity(m, path, sidecar)
    if err > _TOLERANCE[precision] * max(scale, 1.0):
        raise RuntimeError(f"{path}: {target} {precision} outputs differ from the Torch model "
                           f"by {err:.3e} (largest output {scale:.3e})")
    return path


def open_session(path: Path, sidecar: Mapping[str, Any], device: str = "auto") -> Any:
    """The runtime session for an artifact: called with a (B, 3, H, W) float
    batch in [0, 1], returns the raw maps in the sidecar's output order.
    `batch` is the batch size it was traced at (None: any)."""
    t = TARGETS[target_from_suffix(path)]
    precision = sidecar["export"]["precision"]
    return t.module.Session(Path(path), list(sidecar["outputs"]), precision, device)


@torch.no_grad()
def parity(model: Detector, path: Path, sidecar: Mapping[str, Any],
           seed: int = 0) -> tuple[float, float]:
    """Max abs difference between an artifact's raw maps and the Torch deploy
    model's (its int8 simulation for an int8 artifact) on one random
    canvas-sized input, and the maps' scale."""
    session = open_session(path, sidecar, "cpu")   # a TensorRT session uses CUDA regardless
    x = torch.rand(1, 3, *sidecar["canvas_hw"], generator=torch.Generator().manual_seed(seed))
    with deploy_mode(model, sidecar["export"]["precision"] == "int8"):
        ref = model.eval()(x)
    got = session(x.numpy())
    err = max(float(abs(r.numpy() - g).max()) for r, g in zip(ref, got, strict=True))
    return err, max(float(r.abs().max()) for r in ref)
