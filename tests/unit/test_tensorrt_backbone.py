"""Tests for :class:`mayaku.inference.export.TensorRTBackbone`.

Parity check: the runtime wrapper, when swapped into ``model.backbone``,
produces feature maps that match eager fp32 within the same tolerance
the exporter's own ``parity_check`` uses.

Auto-skipped on hosts without CUDA + tensorrt installed (see
``tests/conftest.py``).
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
from mayaku.inference.export import TensorRTBackbone, TensorRTExporter
from mayaku.models.detectors import build_faster_rcnn

pytestmark = [pytest.mark.tensorrt, pytest.mark.slow]


def _tiny_cfg() -> MayakuConfig:
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
            backbone=BackboneConfig(name="resnet50", freeze_at=2, norm="FrozenBN"),
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


def _build_engine(tmp_path: Path, h: int, w: int) -> tuple[torch.nn.Module, Path]:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).cuda().eval()
    out = tmp_path / "model.engine"
    sample = torch.randn(1, 3, h, w)
    TensorRTExporter().export(model, sample, out)
    return model, out


def test_backbone_forward_returns_expected_levels(tmp_path: Path) -> None:
    h, w = 96, 96
    _, engine_path = _build_engine(tmp_path, h, w)
    backbone = TensorRTBackbone(engine_path, pinned=(h, w))
    x = torch.zeros((1, 3, h, w), dtype=torch.float32, device="cuda")
    out = backbone(x)
    assert set(out) == {"p2", "p3", "p4", "p5", "p6"}
    # Feature spatial sizes follow the FPN strides; backbone uses
    # ceil-division so a 96-in pix input at stride 64 yields 2x2.
    assert out["p2"].shape[-2:] == (24, 24)  # 96 / 4
    assert out["p6"].shape[-2:] == (2, 2)  # ceil(96 / 64)


def test_backbone_parity_against_eager(tmp_path: Path) -> None:
    h, w = 96, 96
    model, engine_path = _build_engine(tmp_path, h, w)
    sample = torch.randn(1, 3, h, w)

    # Eager reference under strict fp32 (matches exporter's parity setup).
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            eager_out = model.backbone(sample.cuda())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn

    backbone = TensorRTBackbone(engine_path, pinned=(h, w))
    trt_out = backbone(sample.cuda())

    for name in ("p2", "p3", "p4", "p5", "p6"):
        abs_err = float((trt_out[name] - eager_out[name]).abs().max().item())
        assert abs_err <= 1e-2, f"{name}: max abs error {abs_err} > 1e-2"


def test_backbone_pads_smaller_inputs(tmp_path: Path) -> None:
    """Engine pinned at 96x96; feed an 80x80 input and expect cropped outputs."""
    h, w = 96, 96
    _build_engine(tmp_path, h, w)
    backbone = TensorRTBackbone(tmp_path / "model.engine", pinned=(h, w))
    x = torch.zeros((1, 3, 80, 80), dtype=torch.float32, device="cuda")
    out = backbone(x)
    # p2 stride=4 => ceil(80/4) = 20
    assert out["p2"].shape[-2:] == (20, 20)


def test_backbone_rejects_oversize_input(tmp_path: Path) -> None:
    h, w = 96, 96
    _build_engine(tmp_path, h, w)
    backbone = TensorRTBackbone(tmp_path / "model.engine", pinned=(h, w))
    x = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="exceeds the engine's pinned shape"):
        backbone(x)


def test_backbone_size_divisibility_property(tmp_path: Path) -> None:
    h, w = 96, 96
    _build_engine(tmp_path, h, w)
    backbone = TensorRTBackbone(tmp_path / "model.engine", pinned=(h, w), size_divisibility=64)
    assert backbone.size_divisibility == 64
