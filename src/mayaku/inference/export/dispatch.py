"""Shared target dispatch for the export CLI and ``Predictor.export()``.

Both the ``mayaku export`` verb (:func:`mayaku.cli.export.run_export`) and
the in-memory :meth:`mayaku.inference.predictor.Predictor.export` need the
same thing: given a built detector, a target name, and a tracing sample,
pick the right exporter and run it. This module is that single mapping so
the two entry points can't drift on which targets exist or how each is
constructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mayaku.inference.export.base import ExportResult
from mayaku.inference.export.coreml import CoreMLExporter
from mayaku.inference.export.metadata import embed_sidecar
from mayaku.inference.export.onnx import ONNXExporter
from mayaku.inference.export.openvino import OpenVINOExporter
from mayaku.inference.export.tensorrt import TensorRTExporter

__all__ = ["AVAILABLE_TARGETS", "TARGET_SUFFIX", "build_sample", "export_detector"]

AVAILABLE_TARGETS: tuple[str, ...] = ("onnx", "coreml", "openvino", "tensorrt")

# Default artifact suffix per target — used to derive an output filename when
# the caller doesn't pass one. CoreML writes a ``.mlpackage`` directory;
# OpenVINO writes ``.xml`` (+ a sibling ``.bin``).
TARGET_SUFFIX: dict[str, str] = {
    "onnx": ".onnx",
    "coreml": ".mlpackage",
    "openvino": ".xml",
    "tensorrt": ".engine",
}


def build_sample(h: int, w: int) -> torch.Tensor:
    """Build a normalised RGB tracing sample on CPU.

    The exporter graph operates on already-normalised tensors (the detector's
    pixel mean/std subtraction lives outside the exported body). ``zeros`` is a
    fine tracing input; numerical parity is checked with random data in the
    test suite. :func:`export_detector` re-homes it to the model's device.
    """
    return torch.zeros(1, 3, h, w, dtype=torch.float32)


def export_detector(
    model: nn.Module,
    target: str,
    output: Path,
    *,
    sample: torch.Tensor,
    coreml_precision: str = "fp32",
    onnx_dynamic_input_shape: bool = True,
    sidecar: dict[str, Any] | None = None,
) -> ExportResult:
    """Dispatch ``model`` to the exporter for ``target`` and return its result.

    ``model`` must be built and in eval mode. ``sample`` is a tracing input; it
    is moved to the model's device here because the exporters trace by running
    the model forward on it and (unlike TensorRT) onnx/coreml/openvino don't
    re-home it themselves — so a model on MPS/CUDA would otherwise mismatch.

    ``sidecar`` (the :func:`mayaku.utils.build_sidecar` dict) is embedded into the
    written artifact's metadata slot so it is self-describing — ``from_pretrained``
    reconstructs config + class names from the file alone. ``None`` skips it (the
    artifact is still valid, just not self-describing).
    """
    if target not in AVAILABLE_TARGETS:
        raise ValueError(f"unknown export target {target!r}; expected one of {AVAILABLE_TARGETS}")

    sample = sample.to(next(model.parameters()).device)

    # CoreML/OpenVINO embed the sidecar inline (they hold the model in memory and
    # can't be re-saved over in place); ONNX/TensorRT embed post-hoc below.
    if target == "onnx":
        result = ONNXExporter(dynamic_input_shape=onnx_dynamic_input_shape).export(
            model, sample, output
        )
    elif target == "coreml":
        result = CoreMLExporter(compute_precision=coreml_precision).export(
            model, sample, output, sidecar=sidecar
        )
    elif target == "openvino":
        result = OpenVINOExporter().export(model, sample, output, sidecar=sidecar)
    elif target == "tensorrt":
        result = TensorRTExporter().export(model, sample, output)
    else:  # pragma: no cover — every target in AVAILABLE_TARGETS has a branch.
        raise AssertionError(
            f"unhandled export target {target!r}; AVAILABLE_TARGETS is out of sync."
        )

    if sidecar is not None and target in ("onnx", "tensorrt"):
        embed_sidecar(result.path, target, sidecar)
    return result
