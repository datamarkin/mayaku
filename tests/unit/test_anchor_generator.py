"""Tests for :mod:`mayaku.models.proposals.anchor_generator`."""

from __future__ import annotations

import pytest
import torch

from mayaku.config.schemas import AnchorGeneratorConfig
from mayaku.models.proposals.anchor_generator import (
    DefaultAnchorGenerator,
    _generate_cell_anchors,
    build_anchor_generator,
)
from mayaku.structures.boxes import Boxes

# ---------------------------------------------------------------------------
# Cell-anchor math
# ---------------------------------------------------------------------------


def test_cell_anchor_square_aspect_ratio() -> None:
    # ratio=1, size=32 → w = h = 32; box centred at origin = (-16, -16, 16, 16).
    cell = _generate_cell_anchors([32], [1.0])
    torch.testing.assert_close(cell, torch.tensor([[-16.0, -16.0, 16.0, 16.0]]))


def test_cell_anchor_aspect_ratio_preserves_area() -> None:
    cell = _generate_cell_anchors([32], [0.5, 1.0, 2.0])
    # All three should have area = 32**2 = 1024.
    for row in cell:
        w = (row[2] - row[0]).item()
        h = (row[3] - row[1]).item()
        assert abs(w * h - 1024.0) < 1e-3
        # Centred at origin.
        torch.testing.assert_close(row[0] + row[2], torch.tensor(0.0))
        torch.testing.assert_close(row[1] + row[3], torch.tensor(0.0))
    # h/w ratios match the input.
    ratios = [(row[3] - row[1]) / (row[2] - row[0]) for row in cell]
    expected = [0.5, 1.0, 2.0]
    for r, e in zip(ratios, expected, strict=True):
        assert abs(r.item() - e) < 1e-3


def test_cell_anchor_multiple_sizes() -> None:
    cell = _generate_cell_anchors([32, 64], [1.0])
    assert cell.shape == (2, 4)
    # Areas 32**2 and 64**2.
    assert abs((cell[0, 2] - cell[0, 0]).item() ** 2 - 1024.0) < 1e-3
    assert abs((cell[1, 2] - cell[1, 0]).item() ** 2 - 4096.0) < 1e-3


# ---------------------------------------------------------------------------
# Generator forward
# ---------------------------------------------------------------------------


def test_generator_per_level_shape(device: torch.device) -> None:
    sizes = [(32,), (64,)]
    aspect_ratios = [(0.5, 1.0, 2.0)]  # shared across both levels
    strides = (4, 8)
    gen = DefaultAnchorGenerator(sizes, aspect_ratios, strides=strides).to(device)
    feats = [torch.zeros(1, 8, 4, 5, device=device), torch.zeros(1, 8, 2, 3, device=device)]
    out = gen(feats)
    assert isinstance(out, list) and len(out) == 2
    # A = len(sizes) * len(aspect_ratios) = 1 * 3 = 3
    assert isinstance(out[0], Boxes)
    assert out[0].tensor.shape == (4 * 5 * 3, 4)
    assert out[1].tensor.shape == (2 * 3 * 3, 4)


def test_generator_anchor_centres_match_grid_stride(device: torch.device) -> None:
    # Single anchor (1 size, 1 ratio) → A=1; cell anchor = (-w/2, -h/2, w/2, h/2)
    # so when shifted by (i*stride, j*stride) the centre is exactly (i*stride, j*stride).
    gen = DefaultAnchorGenerator([(8,)], [(1.0,)], strides=(8,)).to(device)
    feats = [torch.zeros(1, 1, 2, 3, device=device)]
    out = gen(feats)
    boxes = out[0].tensor
    # 6 anchors total. Check the (i=0, j=0) anchor.
    centre_x = (boxes[0, 0] + boxes[0, 2]) / 2
    centre_y = (boxes[0, 1] + boxes[0, 3]) / 2
    torch.testing.assert_close(centre_x, torch.tensor(0.0, device=device))
    torch.testing.assert_close(centre_y, torch.tensor(0.0, device=device))
    # (i=0, j=2) anchor: stride 8, so centre x = 16.
    centre_x2 = (boxes[2, 0] + boxes[2, 2]) / 2
    torch.testing.assert_close(centre_x2, torch.tensor(16.0, device=device))


def test_generator_offset_shifts_grid() -> None:
    gen = DefaultAnchorGenerator([(8,)], [(1.0,)], strides=(8,), offset=0.5)
    feats = [torch.zeros(1, 1, 2, 2)]
    out = gen(feats)
    boxes = out[0].tensor
    # (i=0, j=0) centre = ((0+0.5)*8, (0+0.5)*8) = (4, 4)
    centre_x = (boxes[0, 0] + boxes[0, 2]) / 2
    centre_y = (boxes[0, 1] + boxes[0, 3]) / 2
    torch.testing.assert_close(centre_x, torch.tensor(4.0))
    torch.testing.assert_close(centre_y, torch.tensor(4.0))


def test_generator_validates_inputs() -> None:
    with pytest.raises(ValueError, match="strides"):
        DefaultAnchorGenerator([(32,), (64,)], [(1.0,)], strides=(4,))
    with pytest.raises(ValueError, match="aspect_ratios"):
        DefaultAnchorGenerator(
            [(32,), (64,)],
            [(0.5, 1.0), (1.0, 2.0), (1.0,)],  # 3 ratio tuples for 2 levels
            strides=(4, 8),
        )
    with pytest.raises(ValueError, match="offset"):
        DefaultAnchorGenerator([(32,)], [(1.0,)], strides=(4,), offset=1.0)


def test_generator_forward_rejects_feature_count_mismatch() -> None:
    gen = DefaultAnchorGenerator([(32,), (64,)], [(1.0,)], strides=(4, 8))
    with pytest.raises(ValueError, match="feature maps"):
        gen([torch.zeros(1, 1, 2, 2)])


def test_num_anchors_per_cell_matches_config() -> None:
    gen = DefaultAnchorGenerator([(32,), (64,)], [(0.5, 1.0, 2.0)], strides=(4, 8))
    assert gen.num_anchors_per_cell == (3, 3)


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


def test_build_anchor_generator_from_default_config() -> None:
    cfg = AnchorGeneratorConfig()
    gen = build_anchor_generator(cfg, strides=(4, 8, 16, 32, 64))
    assert gen.num_anchors_per_cell == (3, 3, 3, 3, 3)  # 1 size * 3 ratios per level
    feats = [torch.zeros(1, 1, 64 // s, 64 // s) for s in (4, 8, 16, 32, 64)]
    out = gen(feats)
    assert len(out) == 5
