"""Tests for :mod:`mayaku.models.poolers`."""

from __future__ import annotations

import pytest
import torch

from mayaku.models.poolers import ROIPooler, assign_boxes_to_levels
from mayaku.structures.boxes import Boxes

# ---------------------------------------------------------------------------
# Level assignment math
# ---------------------------------------------------------------------------


def test_assign_canonical_box_lands_on_canonical_level() -> None:
    # 224x224 box, canonical_box_size=224, canonical_level=4 → level 4.
    # Levels are clamped to (min_level, max_level) and returned as
    # 0-based indices into the input list. With 4 input levels (p2..p5),
    # min_level=2, max_level=5; canonical level 4 → index 2.
    box = Boxes(torch.tensor([[0.0, 0.0, 224.0, 224.0]]))
    level = assign_boxes_to_levels(
        [box],
        min_level=2,
        max_level=5,
        canonical_box_size=224,
        canonical_level=4,
    )
    assert level.tolist() == [2]


def test_assign_tiny_box_clamps_to_min_level() -> None:
    box = Boxes(torch.tensor([[0.0, 0.0, 4.0, 4.0]]))
    level = assign_boxes_to_levels(
        [box], min_level=2, max_level=5, canonical_box_size=224, canonical_level=4
    )
    assert level.tolist() == [0]  # clamped to min_level=2 → index 0


def test_assign_huge_box_clamps_to_max_level() -> None:
    box = Boxes(torch.tensor([[0.0, 0.0, 2048.0, 2048.0]]))
    level = assign_boxes_to_levels(
        [box], min_level=2, max_level=5, canonical_box_size=224, canonical_level=4
    )
    assert level.tolist() == [3]  # clamped to max_level=5 → index 3


def test_assign_handles_empty_boxes() -> None:
    out = assign_boxes_to_levels(
        [Boxes(torch.zeros(0, 4))],
        min_level=2,
        max_level=5,
        canonical_box_size=224,
        canonical_level=4,
    )
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# ROIPooler forward
# ---------------------------------------------------------------------------


def test_pooler_output_shape_and_device(device: torch.device) -> None:
    pooler = ROIPooler(output_size=7, scales=(1 / 4, 1 / 8, 1 / 16, 1 / 32))
    feats = [
        torch.zeros(1, 8, 64, 64, device=device),
        torch.zeros(1, 8, 32, 32, device=device),
        torch.zeros(1, 8, 16, 16, device=device),
        torch.zeros(1, 8, 8, 8, device=device),
    ]
    box_lists = [
        Boxes(
            torch.tensor(
                [[0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 224.0, 224.0]],
                device=device,
            )
        )
    ]
    out = pooler(feats, box_lists)
    assert out.shape == (2, 8, 7, 7)
    assert out.device.type == device.type


def test_pooler_routes_boxes_to_correct_level(device: torch.device) -> None:
    # Make each level's feature uniformly equal to its index so we can
    # tell which level a pooled output came from.
    pooler = ROIPooler(output_size=4, scales=(1 / 4, 1 / 8, 1 / 16, 1 / 32))
    feats = [torch.full((1, 1, 64, 64), float(i), device=device) for i in range(4)]
    box_lists = [
        Boxes(
            torch.tensor(
                [
                    [0.0, 0.0, 16.0, 16.0],  # tiny → level 0
                    [0.0, 0.0, 224.0, 224.0],  # canonical → level 2
                    [0.0, 0.0, 1000.0, 1000.0],  # huge → level 3
                ],
                device=device,
            )
        )
    ]
    out = pooler(feats, box_lists)
    # All cells are constant for any given RoI, so mean() reveals the level.
    assert abs(out[0].mean().item() - 0.0) < 1e-3
    assert abs(out[1].mean().item() - 2.0) < 1e-3
    assert abs(out[2].mean().item() - 3.0) < 1e-3


def test_pooler_empty_boxes_returns_empty_tensor(device: torch.device) -> None:
    pooler = ROIPooler(output_size=4, scales=(1 / 4, 1 / 8))
    feats = [
        torch.zeros(1, 4, 16, 16, device=device),
        torch.zeros(1, 4, 8, 8, device=device),
    ]
    out = pooler(feats, [Boxes(torch.zeros(0, 4, device=device))])
    assert out.shape == (0, 4, 4, 4)


def test_pooler_validates_feature_count() -> None:
    pooler = ROIPooler(output_size=4, scales=(1 / 4, 1 / 8))
    with pytest.raises(ValueError, match="feature maps"):
        pooler([torch.zeros(1, 4, 8, 8)], [Boxes(torch.zeros(0, 4))])


def test_pooler_int_output_size_is_square() -> None:
    pooler = ROIPooler(output_size=5, scales=(1 / 4,))
    assert pooler.output_size == (5, 5)
