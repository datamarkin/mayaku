"""``mayaku export`` — serialise a trained detector to a deployment target.

ONNX is the required target (Step 18); CoreML / OpenVINO / TensorRT
are best-effort (Steps 19-21) and now all live.
"""

from __future__ import annotations

from pathlib import Path

from mayaku.cli._factory import load_detector
from mayaku.inference.export.base import ExportResult
from mayaku.inference.export.dispatch import (
    AVAILABLE_TARGETS,
    build_sample,
    export_detector,
    resolve_export_sample_hw,
)
from mayaku.tuning.sizing import resolve_deploy_canvas
from mayaku.utils.checkpoint import build_sidecar

__all__ = ["run_export"]


def run_export(
    target: str,
    weights: Path | str,
    *,
    output: Path,
    sample_height: int | None = None,
    sample_width: int | None = None,
    coreml_precision: str = "fp32",
    onnx_dynamic_input_shape: bool = True,
) -> ExportResult:
    """Build the detector, load weights, and dispatch to the per-target
    exporter. Returns the :class:`ExportResult` for downstream use.

    ``weights`` is a trained ``.pth`` (or a bundled model name); its embedded
    sidecar defines the architecture to export.

    ``sample_height`` / ``sample_width`` default to the checkpoint's deploy
    canvas — the geometry the model was trained to see. A full-detector graph
    refuses any other size (see :func:`resolve_export_sample_hw`).
    """
    # Validate before load_detector so a bad target fails fast (a bundled name
    # would otherwise trigger a weight download first).
    if target not in AVAILABLE_TARGETS:
        raise ValueError(f"unknown export target {target!r}; expected one of {AVAILABLE_TARGETS}")

    cfg, model, class_names = load_detector(weights)
    model.eval()

    # Make the artifact self-describing: embed the same sidecar the .pth carries
    # (config + class names), so from_pretrained loads it from the file alone.
    sidecar = build_sidecar(cfg, class_names or [])

    canvas = resolve_deploy_canvas(cfg.input.canvas_hw, cfg.input.size_budget)
    sample = build_sample(*resolve_export_sample_hw(canvas, model, sample_height, sample_width))
    return export_detector(
        model,
        target,
        output,
        sample=sample,
        coreml_precision=coreml_precision,
        onnx_dynamic_input_shape=onnx_dynamic_input_shape,
        sidecar=sidecar,
    )
