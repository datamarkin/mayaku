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


# ---------------------------------------------------------------------------
# Deploy one-pass  ↔  training roi_align parity (the #3 numerical core)
#
# The model trains with torchvision roi_align (pooler.train()) and ships with
# the single-pass grid_sample pooler (pooler.eval()). These must compute the
# same features or the exported model degrades silently. The cases below pin
# that, including the two defects #3 fixes: edge boxes (border clamp) and large
# boxes (matched sample count).
# ---------------------------------------------------------------------------

_SCALES = (1 / 4, 1 / 8, 1 / 16, 1 / 32)
_TOL = {"atol": 2e-4, "rtol": 2e-4}


def _pyramid(device: torch.device, channels: int = 4) -> list[torch.Tensor]:
    """Random p2..p5 feature maps for a ~256px image (seeded, reproducible)."""
    gen = torch.Generator().manual_seed(0)
    sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]
    return [torch.rand(1, channels, h, w, generator=gen).to(device) for h, w in sizes]


def _train_vs_deploy(
    pooler: ROIPooler, feats: list[torch.Tensor], boxes: list[Boxes]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool the same RoIs through the train (roi_align) and deploy (one-pass)
    paths; return ``(reference, deploy)``."""
    with torch.no_grad():
        pooler.train()
        ref = pooler(feats, boxes)
        pooler.eval()
        got = pooler(feats, boxes)
    return ref, got


@pytest.mark.parametrize(
    ("sampling_ratio", "box_rows"),
    [
        # interior boxes spanning fine→coarse levels (baseline parity)
        pytest.param(
            2,
            [[10.0, 10.0, 40.0, 40.0], [20.0, 20.0, 200.0, 200.0], [5.0, 5.0, 250.0, 250.0]],
            id="interior",
        ),
        # Defect (a): boxes hugging the borders. The old zeros-padding dimmed
        # these toward 0 (max |Δ| ~0.9); the border clamp matches roi_align.
        pytest.param(
            2,
            [[0.0, 0.0, 30.0, 30.0], [226.0, 226.0, 256.0, 256.0], [0.0, 100.0, 256.0, 130.0]],
            id="edge_boxes",
        ),
        # Defect (b): sampling_ratio=0 resolves to a fixed count on BOTH paths,
        # else a large box averages adaptive-many samples in train but 2 at
        # deploy. With the fix, train roi_align also uses the fixed count.
        pytest.param(0, [[0.0, 0.0, 250.0, 250.0]], id="large_box_sr0"),
    ],
)
def test_deploy_matches_train(
    device: torch.device, sampling_ratio: int, box_rows: list[list[float]]
) -> None:
    pooler = ROIPooler(output_size=7, scales=_SCALES, sampling_ratio=sampling_ratio)
    feats = _pyramid(device)
    boxes = [Boxes(torch.tensor(box_rows, device=device))]
    ref, got = _train_vs_deploy(pooler, feats, boxes)
    torch.testing.assert_close(got, ref, **_TOL)


def test_eff_sampling_ratio_resolves_nonpositive_to_two() -> None:
    # sampling_ratio<=0 is the fixed deploy default (2), shared by both paths —
    # never torchvision's per-box adaptive count (the export one-pass can't do it).
    assert ROIPooler(output_size=7, scales=_SCALES, sampling_ratio=0)._eff_sampling_ratio == 2
    assert ROIPooler(output_size=7, scales=_SCALES, sampling_ratio=3)._eff_sampling_ratio == 3
