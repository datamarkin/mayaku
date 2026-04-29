"""Tests for :mod:`mayaku.models.backbones.resnet`.

Heavy modules — instantiating R-50/R-101/X-101 takes a moment and a
non-trivial chunk of memory. Where the test only checks structure or
the freezing / norm logic, we use the lightest backbone ("resnet50")
and avoid running the forward pass; where the forward shape contract
matters we still use the small (224x224 ish) input.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mayaku.config.schemas import BackboneConfig
from mayaku.models.backbones import (
    Backbone,
    FrozenBatchNorm2d,
    ResNetBackbone,
    ShapeSpec,
    build_backbone,
)

# ---------------------------------------------------------------------------
# Shape contract — parametrised over the three in-scope backbones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["resnet50", "resnet101", "resnext101_32x8d"])
def test_output_shape_contract(name: str) -> None:
    bb = ResNetBackbone(name=name)  # type: ignore[arg-type]
    spec = bb.output_shape()
    assert set(spec) == {"res2", "res3", "res4", "res5"}
    expected = {
        "res2": ShapeSpec(channels=256, stride=4),
        "res3": ShapeSpec(channels=512, stride=8),
        "res4": ShapeSpec(channels=1024, stride=16),
        "res5": ShapeSpec(channels=2048, stride=32),
    }
    assert spec == expected
    assert bb.size_divisibility == 32


def test_forward_returns_named_feature_map(device: torch.device) -> None:
    bb = ResNetBackbone(name="resnet50").to(device).eval()
    # 64-divisible HxW so every level has a clean integer size.
    x = torch.zeros(1, 3, 64, 64, device=device)
    with torch.no_grad():
        out = bb(x)
    assert set(out) == {"res2", "res3", "res4", "res5"}
    # Spec strides:
    assert out["res2"].shape[-2:] == (16, 16)  # 64/4
    assert out["res3"].shape[-2:] == (8, 8)  # 64/8
    assert out["res4"].shape[-2:] == (4, 4)  # 64/16
    assert out["res5"].shape[-2:] == (2, 2)  # 64/32
    assert out["res2"].shape[1] == 256
    assert out["res5"].shape[1] == 2048
    for v in out.values():
        assert v.dtype == torch.float32
        assert v.device.type == device.type


def test_forward_dtype_passthrough(device: torch.device) -> None:
    bb = ResNetBackbone(name="resnet50").to(device).eval()
    x = torch.zeros(1, 3, 64, 64, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = bb(x)
    for t in out.values():
        assert t.dtype == torch.float32


def test_subset_of_out_features_emits_only_those_keys() -> None:
    bb = ResNetBackbone(name="resnet50", out_features=("res4", "res5"))
    out = bb(torch.zeros(1, 3, 64, 64))
    assert set(out) == {"res4", "res5"}
    assert set(bb.output_shape()) == {"res4", "res5"}
    assert bb.size_divisibility == 32


# ---------------------------------------------------------------------------
# Freeze semantics
# ---------------------------------------------------------------------------


def _trainable(stage: nn.Module) -> bool:
    return any(p.requires_grad for p in stage.parameters())


@pytest.mark.parametrize("freeze_at", [0, 1, 2, 3, 4, 5])
def test_freeze_at_freezes_first_k_stages(freeze_at: int) -> None:
    bb = ResNetBackbone(name="resnet50", norm="BN", freeze_at=freeze_at)
    stages = [bb.stem, bb.res2, bb.res3, bb.res4, bb.res5]
    for i, stage in enumerate(stages, start=1):
        if i <= freeze_at:
            assert not _trainable(stage), f"stage {i} should be frozen"
        else:
            assert _trainable(stage), f"stage {i} should be trainable"


def test_freeze_at_converts_frozen_stages_bn_to_frozenbn() -> None:
    bb = ResNetBackbone(name="resnet50", norm="BN", freeze_at=2)
    # Stem + res2 should have FrozenBN; later stages should still have BN.
    for m in bb.stem.modules():
        assert not isinstance(m, nn.BatchNorm2d), "stem BN must be frozen"
    for m in bb.res2.modules():
        assert not isinstance(m, nn.BatchNorm2d), "res2 BN must be frozen"
    has_bn_in_res5 = any(isinstance(m, nn.BatchNorm2d) for m in bb.res5.modules())
    assert has_bn_in_res5, "res5 BN should remain trainable when freeze_at=2"


def test_freeze_at_zero_keeps_everything_trainable() -> None:
    bb = ResNetBackbone(name="resnet50", norm="BN", freeze_at=0)
    assert all(_trainable(s) for s in [bb.stem, bb.res2, bb.res3, bb.res4, bb.res5])


def test_freeze_at_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="freeze_at"):
        ResNetBackbone(freeze_at=6)
    with pytest.raises(ValueError, match="freeze_at"):
        ResNetBackbone(freeze_at=-1)


# ---------------------------------------------------------------------------
# Norm choice
# ---------------------------------------------------------------------------


def test_default_norm_is_frozen_bn() -> None:
    bb = ResNetBackbone(name="resnet50")
    has_any_trainable_bn = any(isinstance(m, nn.BatchNorm2d) for m in bb.modules())
    assert not has_any_trainable_bn
    assert any(isinstance(m, FrozenBatchNorm2d) for m in bb.modules())


def test_norm_bn_keeps_trainable_bn_in_unfrozen_stages() -> None:
    bb = ResNetBackbone(name="resnet50", norm="BN", freeze_at=2)
    # res3..res5 must still carry trainable BN so the optimiser can update them.
    has_bn_post_freeze = any(
        isinstance(m, nn.BatchNorm2d)
        for stage in [bb.res3, bb.res4, bb.res5]
        for m in stage.modules()
    )
    assert has_bn_post_freeze


def test_unsupported_norm_rejected() -> None:
    with pytest.raises(NotImplementedError, match="GN"):
        ResNetBackbone(norm="GN")


def test_unknown_out_feature_rejected() -> None:
    with pytest.raises(ValueError, match="unknown out_feature"):
        ResNetBackbone(out_features=("res2", "res99"))


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


def test_build_backbone_from_default_config() -> None:
    bb = build_backbone(BackboneConfig())
    assert isinstance(bb, ResNetBackbone)
    assert bb.name == "resnet50"
    # Default norm is FrozenBN.
    assert isinstance(bb.stem[1], FrozenBatchNorm2d)


def test_build_backbone_honours_freeze_at_and_name() -> None:
    cfg = BackboneConfig(name="resnet101", freeze_at=4, norm="BN")
    bb = build_backbone(cfg)
    assert bb.name == "resnet101"
    assert not _trainable(bb.res4)
    assert _trainable(bb.res5)


def test_build_backbone_invalid_weights_rejected() -> None:
    bb = ResNetBackbone(name="resnet50")
    assert isinstance(bb, Backbone)
    # Direct invalid-weights path on the constructor.
    with pytest.raises(ValueError, match="weights"):
        ResNetBackbone(name="resnet50", weights="bogus")  # type: ignore[arg-type]


def test_stride_in_3x3_default_matches_torchvision_layout() -> None:
    """torchvision builds Bottleneck with stride on the 3x3 conv. Mayaku's
    default must preserve that so torchvision pretrained weights load
    cleanly and behave as torchvision intends."""
    bb = ResNetBackbone(name="resnet50")
    for stage in (bb.res3, bb.res4, bb.res5):
        b0 = stage[0]
        assert b0.conv1.stride == (1, 1)
        assert b0.conv2.stride == (2, 2)


def test_stride_in_1x1_relocates_downsample_to_first_conv() -> None:
    """``stride_in_1x1=True`` is required to load Detectron2's MSRA-pretrained
    R50 model-zoo weights (e.g. ``faster_rcnn_R_50_FPN_3x``). Without it the
    same kernels produce different activations from res3 onward; the resulting
    model loads silently with wrong outputs. Regression for ADR 003 attempt 5."""
    bb = ResNetBackbone(name="resnet50", stride_in_1x1=True)
    for stage in (bb.res3, bb.res4, bb.res5):
        b0 = stage[0]
        assert b0.conv1.stride == (2, 2), f"{stage} first block conv1 should carry the stride"
        assert b0.conv2.stride == (1, 1)
        # downsample shortcut still owns the spatial stride either way.
        assert b0.downsample[0].stride == (2, 2)
    # res2's first block has no spatial stride to relocate; must stay (1, 1).
    b0 = bb.res2[0]
    assert b0.conv1.stride == (1, 1)
    assert b0.conv2.stride == (1, 1)


def test_stride_in_1x1_threads_through_build_backbone() -> None:
    cfg = BackboneConfig(stride_in_1x1=True)
    bb = build_backbone(cfg)
    assert bb.res3[0].conv1.stride == (2, 2)
    assert bb.res3[0].conv2.stride == (1, 1)


# ---------------------------------------------------------------------------
# Backbone protocol
# ---------------------------------------------------------------------------


def test_dummy_input_lives_on_backbone_device(device: torch.device) -> None:
    bb = ResNetBackbone(name="resnet50").to(device)
    x = bb.dummy_input(batch=2, h=64, w=64)
    assert x.shape == (2, 3, 64, 64)
    assert x.device.type == device.type
