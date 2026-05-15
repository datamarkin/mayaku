"""Tests for :mod:`mayaku.inference.export.tensorrt`.

Gated by ``pytest.mark.tensorrt`` — auto-skipped on macOS / CPU-only
Linux / hosts without the ``[tensorrt]`` extra installed. Builds a
real serialised engine on the CUDA host and runs an eager-vs-TRT
parity check.

Engine builds for even a tiny ResNet-50 + FPN take a few seconds, so
each test is also marked ``slow``; users on busy CI can opt out via
``-m 'not slow'``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mayaku.config.schemas import (
    BackboneConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    RPNConfig,
)
from mayaku.inference.export import TensorRTExporter
from mayaku.inference.export.base import ExportResult, ParityResult
from mayaku.models.detectors import build_faster_rcnn

pytestmark = [pytest.mark.tensorrt, pytest.mark.slow]


def _tiny_cfg(backbone_name: str = "resnet50") -> MayakuConfig:
    """Build a minimal Faster R-CNN config wired around ``backbone_name``.

    Branches on the backbone family because the schema validator rejects
    ResNet-only fields (``norm``, ``stride_in_1x1``) on ConvNeXt.
    """
    from mayaku.models.backbones import is_convnext_variant

    if is_convnext_variant(backbone_name):
        backbone = BackboneConfig(name=backbone_name, freeze_at=0)  # type: ignore[arg-type]
    else:
        backbone = BackboneConfig(name=backbone_name, freeze_at=2, norm="FrozenBN")  # type: ignore[arg-type]
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
            backbone=backbone,
            rpn=RPNConfig(
                pre_nms_topk_train=100,
                pre_nms_topk_test=50,
                post_nms_topk_train=20,
                post_nms_topk_test=10,
                batch_size_per_image=16,
            ),
            roi_heads=ROIHeadsConfig(num_classes=2, batch_size_per_image=8),
            roi_box_head=ROIBoxHeadConfig(num_fc=1, fc_dim=32),
        ),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_tensorrt_export_writes_engine(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).cuda().eval()
    out = tmp_path / "model.engine"
    sample = torch.randn(1, 3, 96, 96)
    result = TensorRTExporter().export(model, sample, out)

    assert isinstance(result, ExportResult)
    assert result.target == "tensorrt"
    assert out.exists() and out.stat().st_size > 0
    assert result.input_names == ("image",)
    assert result.output_names == ("p2", "p3", "p4", "p5", "p6")
    assert result.extras["fp16"] == "False"


def test_tensorrt_export_fp16_flag_propagates(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).cuda().eval()
    sample = torch.randn(1, 3, 96, 96)
    out = tmp_path / "model_fp16.engine"
    result = TensorRTExporter(fp16=True).export(model, sample, out)
    assert result.extras["fp16"] == "True"
    assert out.exists()


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone_name", ["resnet50", "dinov3_convnext_tiny"])
def test_tensorrt_parity_within_tolerance(tmp_path: Path, backbone_name: str) -> None:
    """End-to-end TensorRT engine build + parity vs eager, parametrised
    over backbone family. ConvNeXt routes its LayerNorm + depthwise
    conv through the ONNX front-end the TRT builder consumes (opset 17,
    which TRT 10+ supports natively).
    """
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg(backbone_name)).cuda().eval()
    out = tmp_path / "model.engine"
    sample = torch.randn(1, 3, 96, 96)
    exporter = TensorRTExporter()
    exporter.export(model, sample, out)

    parity = exporter.parity_check(model, out, sample, atol=1e-2, rtol=1e-2)
    assert isinstance(parity, ParityResult)
    assert parity.target == "tensorrt"
    assert parity.passed, (
        f"TensorRT parity failed: max_abs={parity.max_abs_error}, "
        f"max_rel={parity.max_rel_error}, per_output={parity.per_output}"
    )
    assert set(parity.per_output) == {"p2", "p3", "p4", "p5", "p6"}


def test_tensorrt_parity_check_rejects_cpu_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()  # stays on CPU
    out = tmp_path / "model.engine"
    sample = torch.randn(1, 3, 96, 96)
    # Build the engine first using a CUDA copy of the model.
    cuda_model = build_faster_rcnn(_tiny_cfg()).cuda().eval()
    TensorRTExporter().export(cuda_model, sample, out)
    # Then attempt parity against the CPU model — should refuse.
    with pytest.raises(RuntimeError, match="CUDA model"):
        TensorRTExporter().parity_check(model, out, sample)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_cli_run_export_tensorrt(tmp_path: Path) -> None:
    from mayaku.cli.export import run_export
    from mayaku.config import dump_yaml

    cfg = _tiny_cfg()
    cfg_path = tmp_path / "cfg.yaml"
    dump_yaml(cfg, cfg_path)

    model = build_faster_rcnn(cfg).cuda().eval()
    weights = tmp_path / "model.pth"
    torch.save(model.state_dict(), weights)

    out = tmp_path / "exported.engine"
    result = run_export(
        "tensorrt",
        cfg_path,
        weights=weights,
        output=out,
        sample_height=96,
        sample_width=96,
    )
    assert isinstance(result, ExportResult)
    assert result.target == "tensorrt"
    assert out.exists()
