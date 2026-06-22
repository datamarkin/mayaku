"""``mayaku export`` — serialise a trained detector to a deployment target.

ONNX is the required target (Step 18); CoreML / OpenVINO / TensorRT
are best-effort (Steps 19-21) and now all live.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mayaku.cli._factory import load_detector
from mayaku.inference.export import (
    CoreMLExporter,
    ONNXExporter,
    OpenVINOExporter,
    TensorRTExporter,
)
from mayaku.inference.export.base import ExportResult

__all__ = ["run_export"]

_AVAILABLE_TARGETS: tuple[str, ...] = ("onnx", "coreml", "openvino", "tensorrt")


def run_export(
    target: str,
    weights: Path | str,
    *,
    output: Path,
    sample_height: int = 640,
    sample_width: int = 640,
    coreml_precision: str = "fp32",
    onnx_dynamic_input_shape: bool = True,
) -> ExportResult:
    """Build the detector, load weights, and dispatch to the per-target
    exporter. Returns the :class:`ExportResult` for downstream use.

    ``weights`` is a trained ``.pth`` (or a bundled model name); its embedded
    sidecar defines the architecture to export.
    """
    if target not in _AVAILABLE_TARGETS:
        raise ValueError(f"unknown export target {target!r}; expected one of {_AVAILABLE_TARGETS}")

    _cfg, model = load_detector(weights)
    model.eval()

    sample = _build_sample(sample_height, sample_width)

    if target == "onnx":
        return ONNXExporter(dynamic_input_shape=onnx_dynamic_input_shape).export(
            model, sample, output
        )
    if target == "coreml":
        return CoreMLExporter(compute_precision=coreml_precision).export(model, sample, output)
    if target == "openvino":
        return OpenVINOExporter().export(model, sample, output)
    if target == "tensorrt":
        return TensorRTExporter().export(model, sample, output)

    # Defensive — every target in _AVAILABLE_TARGETS now has a branch.
    raise AssertionError(  # pragma: no cover
        f"unhandled export target {target!r}; "
        "_AVAILABLE_TARGETS is out of sync with the dispatch above."
    )


def _build_sample(h: int, w: int) -> torch.Tensor:
    """Build a normalised RGB tracing sample on CPU.

    The exporter graph operates on already-normalised tensors (the
    detector's pixel mean/std subtraction lives outside the exported
    body). ``zeros`` is a fine tracing input; numerical parity is
    checked with random data in the test suite.
    """
    return torch.zeros(1, 3, h, w, dtype=torch.float32)
