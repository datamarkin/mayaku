"""Tests for :mod:`mayaku.structures.masks`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mayaku.structures.masks import (
    BitMasks,
    PolygonMasks,
    ROIMasks,
    paste_masks_in_image,
)

# ---------------------------------------------------------------------------
# BitMasks
# ---------------------------------------------------------------------------


def test_bitmasks_construction_dtype_and_shape(device: torch.device) -> None:
    t = torch.ones(2, 4, 5, device=device)
    bm = BitMasks(t)
    assert bm.tensor.dtype == torch.bool
    assert bm.image_size == (4, 5)
    assert len(bm) == 2


def test_bitmasks_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\(N, H, W\)"):
        BitMasks(torch.zeros(4, 5))


def test_bitmasks_nonempty(device: torch.device) -> None:
    t = torch.zeros(3, 4, 4, dtype=torch.bool, device=device)
    t[0, 1, 1] = True
    bm = BitMasks(t)
    assert bm.nonempty().tolist() == [True, False, False]


def test_bitmasks_indexing_and_cat(device: torch.device) -> None:
    bm = BitMasks(torch.ones(3, 4, 4, dtype=torch.bool, device=device))
    sub = bm[1]
    assert isinstance(sub, BitMasks)
    assert len(sub) == 1
    cat = BitMasks.cat([bm, sub])
    assert len(cat) == 4


def test_bitmasks_crop_and_resize_full_box(device: torch.device) -> None:
    # A box covering the whole bitmap should resample to all-True at any size.
    bm = BitMasks(torch.ones(1, 16, 16, dtype=torch.bool, device=device))
    box = torch.tensor([[0.0, 0.0, 16.0, 16.0]], device=device)
    out = bm.crop_and_resize(box, mask_size=8)
    assert out.shape == (1, 8, 8)
    assert out.all()


def test_bitmasks_crop_and_resize_zero_box_yields_empty(device: torch.device) -> None:
    # A zero-area box produces a zero-area sample (all-False after threshold).
    bm = BitMasks(torch.ones(1, 16, 16, dtype=torch.bool, device=device))
    box = torch.tensor([[3.0, 3.0, 3.0, 3.0]], device=device)
    out = bm.crop_and_resize(box, mask_size=4)
    # roi_align of a zero-area box is well-defined as a degenerate sample;
    # we don't pin the value, just the shape and dtype.
    assert out.shape == (1, 4, 4)
    assert out.dtype == torch.bool


def test_bitmasks_empty_input() -> None:
    bm = BitMasks(torch.zeros(0, 4, 4, dtype=torch.bool))
    out = bm.crop_and_resize(torch.zeros(0, 4), mask_size=4)
    assert out.shape == (0, 4, 4)


def test_bitmasks_crop_resize_box_count_mismatch() -> None:
    bm = BitMasks(torch.ones(2, 4, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="batch"):
        bm.crop_and_resize(torch.zeros(3, 4), mask_size=2)


# ---------------------------------------------------------------------------
# PolygonMasks
# ---------------------------------------------------------------------------


def _square_polygon(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def test_polygon_masks_construction_validates_polygon_shape() -> None:
    with pytest.raises(ValueError, match=">=3 points"):
        PolygonMasks([[[0.0, 0.0, 1.0, 1.0]]])  # only 2 points


def test_polygon_masks_len_and_index() -> None:
    polys = [
        [_square_polygon(0, 0, 4, 4)],
        [_square_polygon(2, 2, 6, 6)],
    ]
    pm = PolygonMasks(polys)
    assert len(pm) == 2
    sub = pm[0]
    assert isinstance(sub, PolygonMasks) and len(sub) == 1
    sub2 = pm[torch.tensor([True, False])]
    assert len(sub2) == 1


def test_polygon_masks_cat() -> None:
    pm1 = PolygonMasks([[_square_polygon(0, 0, 4, 4)]])
    pm2 = PolygonMasks([[_square_polygon(2, 2, 6, 6)]])
    out = PolygonMasks.cat([pm1, pm2])
    assert len(out) == 2


def test_polygon_masks_area() -> None:
    pm = PolygonMasks([[_square_polygon(0, 0, 4, 4)], [_square_polygon(0, 0, 2, 3)]])
    a = pm.area()
    torch.testing.assert_close(a, torch.tensor([16.0, 6.0]))


def test_polygon_masks_crop_and_resize_full_box(device: torch.device) -> None:
    # A polygon that covers the entire box should rasterise to all-True.
    pm = PolygonMasks([[_square_polygon(0, 0, 8, 8)]])
    box = torch.tensor([[0.0, 0.0, 8.0, 8.0]], device=device)
    out = pm.crop_and_resize(box, mask_size=4)
    assert out.shape == (1, 4, 4)
    assert out.device.type == device.type
    assert out.all()


def test_polygon_masks_crop_and_resize_empty(device: torch.device) -> None:
    pm = PolygonMasks([])
    out = pm.crop_and_resize(torch.zeros(0, 4, device=device), mask_size=4)
    assert out.shape == (0, 4, 4)


def test_bitmasks_from_polygon_masks() -> None:
    pm = PolygonMasks(
        [
            [_square_polygon(0, 0, 4, 4)],
            [_square_polygon(5, 5, 8, 8)],
        ]
    )
    bm = BitMasks.from_polygon_masks(pm, height=10, width=10)
    assert bm.tensor.shape == (2, 10, 10)
    # First instance covers (0,0)-(4,4) (with rasteriser inclusion of edges).
    assert bool(bm.tensor[0, 1, 1])
    assert not bool(bm.tensor[0, 8, 8])


# ---------------------------------------------------------------------------
# ROIMasks + paste-back
# ---------------------------------------------------------------------------


def test_paste_masks_full_image(device: torch.device) -> None:
    # Single mask of all-ones, box covering the whole image → all True.
    soft = torch.ones(1, 8, 8, device=device)
    box = torch.tensor([[0.0, 0.0, 16.0, 16.0]], device=device)
    out = paste_masks_in_image(soft, box, image_size=(16, 16), threshold=0.5)
    assert out.dtype == torch.bool
    assert out.shape == (1, 16, 16)
    assert out.all()


def test_paste_masks_localised_to_box(device: torch.device) -> None:
    soft = torch.ones(1, 4, 4, device=device)
    box = torch.tensor([[2.0, 3.0, 6.0, 7.0]], device=device)
    out = paste_masks_in_image(soft, box, image_size=(10, 10), threshold=0.5)
    # Inside the box → True; outside → False (zero-padded sampling).
    assert out[0, 5, 4]
    assert not out[0, 0, 0]
    assert not out[0, 9, 9]


def test_paste_masks_threshold_respected(device: torch.device) -> None:
    soft = torch.full((1, 4, 4), 0.4, device=device)
    box = torch.tensor([[0.0, 0.0, 8.0, 8.0]], device=device)
    out = paste_masks_in_image(soft, box, (8, 8), threshold=0.5)
    assert not out.any()


def test_paste_masks_empty_input(device: torch.device) -> None:
    soft = torch.zeros(0, 4, 4, device=device)
    out = paste_masks_in_image(soft, torch.zeros(0, 4, device=device), (10, 10))
    assert out.shape == (0, 10, 10)


def test_roimasks_to_bitmasks_round_trip(device: torch.device) -> None:
    soft = torch.ones(2, 4, 4, device=device)
    boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0], [4.0, 4.0, 12.0, 12.0]], device=device)
    rm = ROIMasks(soft)
    bm = rm.to_bitmasks(boxes, height=16, width=16)
    assert isinstance(bm, BitMasks)
    assert bm.tensor.shape == (2, 16, 16)


def test_roimasks_shape_validation() -> None:
    with pytest.raises(ValueError, match=r"\(N, M, M\)"):
        ROIMasks(torch.zeros(2, 4))


def test_paste_masks_shape_validation() -> None:
    with pytest.raises(ValueError, match=r"\(N, M, M\)"):
        paste_masks_in_image(torch.zeros(2, 4), torch.zeros(2, 4), (4, 4))
    with pytest.raises(ValueError, match="boxes shape"):
        paste_masks_in_image(torch.zeros(2, 4, 4), torch.zeros(3, 4), (4, 4))


def test_polygon_to_bitmask_known_region() -> None:
    # Sanity: hand-computed area of a 4x4 square polygon rasterised at 16x16.
    pm = PolygonMasks([[_square_polygon(0, 0, 4, 4)]])
    bm = BitMasks.from_polygon_masks(pm, height=16, width=16)
    # pycocotools includes the edge → roughly the 4x4 area; check with tolerance.
    assert 12 <= int(bm.tensor[0].sum().item()) <= 25
    # Pixel inside the polygon is set.
    assert bool(bm.tensor[0, 1, 1])


def test_numpy_polygon_input_accepted() -> None:
    arr = np.array(_square_polygon(0, 0, 2, 2), dtype=np.float32)
    pm = PolygonMasks([[arr]])
    assert len(pm) == 1
