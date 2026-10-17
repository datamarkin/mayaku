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

# Parity tolerance on the RMS error relative to the outputs' RMS: a broken
# conversion is off by about 100%, rounding is not. fp32 is float noise, fp16
# its rounding over a deep stack. int8 is chaotic in its rounding: a value that
# lands near a grid boundary flips a whole step on any summation-order change,
# and the flips compound over the stack, so the trained simulation itself run
# in float64 instead of float32 moves by about 3% (tier n); no int8 runtime
# can match it closer than that. The parity input is uniform noise, harsher on
# int8 than any image: a TensorRT 11 engine of COCO nano (fp16 graph around
# the Q/DQ pairs) sits at 10.8% there, and 0.26 AP from its fp16 engine on
# COCO val.
_TOLERANCE = {"fp32": 1e-3, "fp16": 1e-2, "int8": 1.5e-1}

# Where each target's parity runs: the compute it deploys on. Core ML pinned to
# its CPU runs fp16 with fp16 accumulation, its least accurate path (several
# percent off where the Neural Engine and the GPU are within 0.3%), so it is
# checked as deployed; a TensorRT session uses CUDA regardless.
_PARITY_DEVICE = {"coreml": "auto"}


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
    err = parity(m, path, sidecar)
    if err > _TOLERANCE[precision]:
        raise RuntimeError(f"{path}: {target} {precision} outputs differ from the Torch model "
                           f"by {err:.2%} RMS (tolerance {_TOLERANCE[precision]:.1%})")
    return path


def open_session(path: Path, sidecar: Mapping[str, Any], device: str = "auto") -> Any:
    """The runtime session for an artifact: called with a (B, 3, H, W) float
    batch in [0, 1], returns the raw maps in the sidecar's output order.
    `batch` is the batch size it was traced at (None: any)."""
    t = TARGETS[target_from_suffix(path)]
    precision = sidecar["export"]["precision"]
    return t.module.Session(Path(path), list(sidecar["outputs"]), precision, device)


@torch.no_grad()
def parity(model: Detector, path: Path, sidecar: Mapping[str, Any], seed: int = 0) -> float:
    """RMS difference between an artifact's raw maps and the Torch deploy
    model's (its int8 simulation for an int8 artifact) on one random
    canvas-sized input, relative to the maps' RMS, with the artifact run on
    the compute it deploys on (`_PARITY_DEVICE`, else the CPU)."""
    target = sidecar["export"]["target"]
    session = open_session(path, sidecar, _PARITY_DEVICE.get(target, "cpu"))
    x = torch.rand(1, 3, *sidecar["canvas_hw"], generator=torch.Generator().manual_seed(seed))
    with deploy_mode(model, sidecar["export"]["precision"] == "int8"):
        ref = model.eval()(x)
    got = session(x.numpy())
    err = sum(float(((r.double() - torch.from_numpy(g).double()) ** 2).sum())
              for r, g in zip(ref, got, strict=True))
    norm = sum(float((r.double() ** 2).sum()) for r in ref)
    return (err / max(norm, 1e-12)) ** 0.5
