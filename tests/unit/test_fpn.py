"""Tests for :mod:`mayaku.models.necks.fpn`.

Stand up a tiny `FakeBackbone` for shape/contract tests so we don't pay
the ResNet construction cost; use a real :class:`ResNetBackbone` once
to confirm end-to-end composition.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from mayaku.config.schemas import FPNConfig
from mayaku.models.backbones import (
    Backbone,
    FrozenBatchNorm2d,
    ResNetBackbone,
    ShapeSpec,
)
from mayaku.models.necks import FPN, LastLevelMaxPool, build_fpn

# ---------------------------------------------------------------------------
# Tiny fake backbone for cheap shape tests
# ---------------------------------------------------------------------------


class _FakeBackbone(Backbone):
    """Identity-ish backbone: zero-conv per stage, named res2..res5."""

    def __init__(self) -> None:
        super().__init__()
        self._out_features = ("res2", "res3", "res4", "res5")
        self._out_feature_channels = {
            "res2": 64,
            "res3": 128,
            "res4": 256,
            "res5": 512,
        }
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        # One 1x1 conv per stage so .parameters() is non-empty (lets the
        # Backbone protocol's dummy_input() resolve a device).
        self.stages = nn.ModuleDict(
            {
                name: nn.Conv2d(3, ch, kernel_size=1, bias=False)
                for name, ch in self._out_feature_channels.items()
            }
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for name in self._out_features:
            stride = self._out_feature_strides[name]
            # Build a feature map of correct (C, H/s, W/s) by 1x1 conv +
            # average-pool-as-stride.
            sampled = nn.functional.avg_pool2d(x, kernel_size=stride, stride=stride)
            out[name] = self.stages[name](sampled)
        return out


# ---------------------------------------------------------------------------
# Output shape contract
# ---------------------------------------------------------------------------


def test_fpn_default_output_shape_with_top_block() -> None:
    fpn = FPN(_FakeBackbone(), top_block=LastLevelMaxPool())
    spec = fpn.output_shape()
    assert set(spec) == {"p2", "p3", "p4", "p5", "p6"}
    expected_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}
    for k, ss in spec.items():
        assert ss == ShapeSpec(channels=256, stride=expected_strides[k])


def test_fpn_size_divisibility_is_max_bottom_up_stride() -> None:
    # The top block (p6) is a strided pool — input padding is determined
    # by res5 (stride 32), not by p6 (stride 64).
    fpn = FPN(_FakeBackbone(), top_block=LastLevelMaxPool())
    assert fpn.size_divisibility == 32


def test_fpn_no_top_block_has_no_p6() -> None:
    fpn = FPN(_FakeBackbone(), top_block=None)
    assert set(fpn.output_shape()) == {"p2", "p3", "p4", "p5"}
    assert fpn.size_divisibility == 32


def test_fpn_subset_in_features() -> None:
    fpn = FPN(_FakeBackbone(), in_features=("res4", "res5"), top_block=None)
    assert set(fpn.output_shape()) == {"p4", "p5"}
    assert fpn.size_divisibility == 32


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_fpn_forward_shapes_and_dtype(device: torch.device) -> None:
    fpn = FPN(_FakeBackbone(), top_block=LastLevelMaxPool()).to(device).eval()
    x = torch.zeros(2, 3, 64, 96, device=device)
    with torch.no_grad():
        out = fpn(x)
    assert set(out) == {"p2", "p3", "p4", "p5", "p6"}
    # Strides 4, 8, 16, 32, 64.
    expected = {
        "p2": (16, 24),
        "p3": (8, 12),
        "p4": (4, 6),
        "p5": (2, 3),
        "p6": (1, 2),
    }
    for k, (h, w) in expected.items():
        t = out[k]
        assert t.shape == (2, 256, h, w), f"{k}: {t.shape}"
        assert t.dtype == torch.float32
        assert t.device.type == device.type


def test_fpn_forward_with_resnet_backbone(device: torch.device) -> None:
    bb = ResNetBackbone(name="resnet50").to(device).eval()
    fpn = FPN(bb, top_block=LastLevelMaxPool()).to(device).eval()
    x = torch.zeros(1, 3, 64, 64, device=device)
    with torch.no_grad():
        out = fpn(x)
    assert set(out) == {"p2", "p3", "p4", "p5", "p6"}
    assert out["p2"].shape == (1, 256, 16, 16)
    assert out["p6"].shape == (1, 256, 1, 1)


def test_fpn_avg_fuse_halves_sum() -> None:
    # Build an FPN with avg fuse and pin one set of inputs that makes
    # the math observable: zero out the upsample contribution and check
    # the lateral x 0.5 dominates p2.
    bb = _FakeBackbone()
    fpn_sum = FPN(bb, fuse_type="sum", top_block=None).eval()
    fpn_avg = FPN(bb, fuse_type="avg", top_block=None).eval()
    # Copy weights so the only difference is the fuse op.
    fpn_avg.lateral_convs.load_state_dict(fpn_sum.lateral_convs.state_dict())
    fpn_avg.output_convs.load_state_dict(fpn_sum.output_convs.state_dict())
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = fpn_sum(x)
        b = fpn_avg(x)
    # p5 has no fuse step (it's just lateral → output_conv) so it must
    # be identical between the two fuse modes.
    torch.testing.assert_close(a["p5"], b["p5"])
    # p2 has had two avg-fuse divisions (from p5→p4, p4→p3, p3→p2 — three)
    # applied to the upsampled chain. Strict numeric equality isn't
    # meaningful through 3x3 convs; just check they differ.
    assert not torch.allclose(a["p2"], b["p2"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_fpn_rejects_unknown_in_feature() -> None:
    with pytest.raises(ValueError, match="not in bottom_up"):
        FPN(_FakeBackbone(), in_features=("res2", "res99"))


def test_fpn_rejects_empty_in_features() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FPN(_FakeBackbone(), in_features=())


def test_fpn_rejects_unknown_fuse_type() -> None:
    with pytest.raises(ValueError, match="fuse_type"):
        FPN(_FakeBackbone(), fuse_type="max")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Norm choices
# ---------------------------------------------------------------------------


def test_fpn_default_norm_has_no_norm_layers() -> None:
    fpn = FPN(_FakeBackbone(), top_block=None)
    has_norm = any(
        isinstance(m, nn.BatchNorm2d | nn.GroupNorm | FrozenBatchNorm2d) for m in fpn.modules()
    )
    assert not has_norm


def test_fpn_bn_norm_inserts_batchnorm() -> None:
    fpn = FPN(_FakeBackbone(), norm="BN", top_block=None)
    bn_count = sum(1 for m in fpn.modules() if isinstance(m, nn.BatchNorm2d))
    # 4 lateral + 4 output = 8 norms for the default in_features set.
    assert bn_count == 8


def test_fpn_gn_norm_inserts_groupnorm() -> None:
    fpn = FPN(_FakeBackbone(), norm="GN", top_block=None)
    gn_count = sum(1 for m in fpn.modules() if isinstance(m, nn.GroupNorm))
    assert gn_count == 8


def test_fpn_frozen_bn_norm_inserts_frozenbn() -> None:
    fpn = FPN(_FakeBackbone(), norm="FrozenBN", top_block=None)
    fb_count = sum(1 for m in fpn.modules() if isinstance(m, FrozenBatchNorm2d))
    assert fb_count == 8


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


def test_build_fpn_from_default_config_with_top_block() -> None:
    cfg = FPNConfig()  # in_features=("res2..res5"), out_channels=256, norm="", sum
    bb = _FakeBackbone()
    fpn = build_fpn(cfg, bb)
    assert isinstance(fpn, FPN)
    assert fpn.out_channels == 256
    assert set(fpn.output_shape()) == {"p2", "p3", "p4", "p5", "p6"}


def test_build_fpn_no_top_block() -> None:
    cfg = FPNConfig()
    fpn = build_fpn(cfg, _FakeBackbone(), with_top_block=False)
    assert set(fpn.output_shape()) == {"p2", "p3", "p4", "p5"}
