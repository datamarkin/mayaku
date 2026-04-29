"""Tests for :mod:`mayaku.structures.boxes`."""

from __future__ import annotations

import pytest
import torch

from mayaku.structures.boxes import (
    Boxes,
    BoxMode,
    pairwise_intersection,
    pairwise_ioa,
    pairwise_iou,
)

# ---------------------------------------------------------------------------
# Boxes container
# ---------------------------------------------------------------------------


def test_construction_and_repr(device: torch.device) -> None:
    t = torch.tensor([[0.0, 0.0, 4.0, 4.0]], device=device)
    b = Boxes(t)
    assert len(b) == 1
    assert b.tensor.shape == (1, 4)
    assert b.device.type == device.type
    assert "Boxes" in repr(b)


def test_construction_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        Boxes(torch.zeros(3, 5))


def test_construction_normalizes_empty() -> None:
    b = Boxes(torch.zeros(0))
    assert b.tensor.shape == (0, 4)


def test_area_and_clip(device: torch.device) -> None:
    t = torch.tensor([[0.0, 0.0, 4.0, 6.0], [-1.0, -1.0, 5.0, 5.0]], device=device)
    b = Boxes(t)
    torch.testing.assert_close(b.area(), torch.tensor([24.0, 36.0], device=device))
    b.clip((4, 4))
    torch.testing.assert_close(
        b.tensor,
        torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0]], device=device),
    )


def test_negative_box_area_clamps_to_zero(device: torch.device) -> None:
    # Inverted box (x1 < x0) should report zero area, not negative.
    b = Boxes(torch.tensor([[5.0, 5.0, 2.0, 2.0]], device=device))
    torch.testing.assert_close(b.area(), torch.zeros(1, device=device))


def test_scale_centers_nonempty_inside_box(device: torch.device) -> None:
    t = torch.tensor([[1.0, 2.0, 5.0, 6.0], [0.0, 0.0, 0.0, 0.0]], device=device)
    b = Boxes(t.clone())
    b.scale(2.0, 0.5)
    torch.testing.assert_close(
        b.tensor,
        torch.tensor([[2.0, 1.0, 10.0, 3.0], [0.0, 0.0, 0.0, 0.0]], device=device),
    )
    centers = Boxes(t).get_centers()
    torch.testing.assert_close(centers, torch.tensor([[3.0, 4.0], [0.0, 0.0]], device=device))
    nonempty = Boxes(t).nonempty()
    assert nonempty.tolist() == [True, False]
    inside = Boxes(t).inside_box((10, 10))
    assert inside.tolist() == [True, True]


def test_indexing_and_iteration(device: torch.device) -> None:
    t = torch.arange(12, device=device, dtype=torch.float32).view(3, 4)
    b = Boxes(t)
    # int → 1-row Boxes
    sub = b[1]
    assert isinstance(sub, Boxes)
    assert sub.tensor.shape == (1, 4)
    torch.testing.assert_close(sub.tensor[0], t[1])
    # bool mask
    mask = torch.tensor([True, False, True], device=device)
    sub = b[mask]
    assert sub.tensor.shape == (2, 4)
    # iteration yields rows
    rows = list(b)
    assert len(rows) == 3 and rows[0].shape == (4,)


def test_cat_and_to(device: torch.device) -> None:
    a = Boxes(torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    b = Boxes(torch.tensor([[2.0, 2.0, 3.0, 3.0]]))
    c = Boxes.cat([a, b]).to(device)
    assert c.tensor.shape == (2, 4)
    assert c.device.type == device.type
    empty = Boxes.cat([])
    assert empty.tensor.shape == (0, 4)


# ---------------------------------------------------------------------------
# BoxMode conversions
# ---------------------------------------------------------------------------


def test_xyxy_xywh_round_trip() -> None:
    box = torch.tensor([[10.0, 20.0, 30.0, 80.0]])
    xywh = BoxMode.convert(box, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    torch.testing.assert_close(xywh, torch.tensor([[10.0, 20.0, 20.0, 60.0]]))
    back = BoxMode.convert(xywh, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    torch.testing.assert_close(back, box)


def test_rel_modes_require_image_size() -> None:
    box = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
    with pytest.raises(ValueError, match="image_size"):
        BoxMode.convert(box, BoxMode.XYXY_REL, BoxMode.XYXY_ABS)


def test_rel_round_trip_through_xyxy_abs() -> None:
    box = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
    abs_box = BoxMode.convert(box, BoxMode.XYXY_REL, BoxMode.XYXY_ABS, image_size=(100, 200))
    torch.testing.assert_close(abs_box, torch.tensor([[20.0, 20.0, 100.0, 60.0]]))
    rel_again = BoxMode.convert(abs_box, BoxMode.XYXY_ABS, BoxMode.XYXY_REL, image_size=(100, 200))
    torch.testing.assert_close(rel_again, box)


def test_xywh_rel_to_xywh_abs() -> None:
    box = torch.tensor([[0.1, 0.2, 0.4, 0.4]])  # rel x0, y0, w, h
    abs_box = BoxMode.convert(box, BoxMode.XYWH_REL, BoxMode.XYWH_ABS, image_size=(100, 200))
    torch.testing.assert_close(abs_box, torch.tensor([[20.0, 20.0, 80.0, 40.0]]))


def test_same_mode_clones() -> None:
    box = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = BoxMode.convert(box, BoxMode.XYXY_ABS, BoxMode.XYXY_ABS)
    assert out is not box
    torch.testing.assert_close(out, box)


# ---------------------------------------------------------------------------
# Pairwise IoU / IoA
# ---------------------------------------------------------------------------


def test_pairwise_iou_basic(device: torch.device) -> None:
    a = Boxes(torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device))
    b = Boxes(
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],  # identical → 1.0
                [5.0, 5.0, 15.0, 15.0],  # 25 / 175 = 0.142857
                [20.0, 20.0, 30.0, 30.0],  # disjoint → 0.0
            ],
            device=device,
        )
    )
    iou = pairwise_iou(a, b)
    expected = torch.tensor([[1.0, 25.0 / 175.0, 0.0]], device=device)
    torch.testing.assert_close(iou, expected, atol=1e-6, rtol=1e-6)


def test_pairwise_intersection_matches_iou_numerator(device: torch.device) -> None:
    a = Boxes(torch.tensor([[0.0, 0.0, 4.0, 4.0]], device=device))
    b = Boxes(torch.tensor([[2.0, 2.0, 6.0, 6.0]], device=device))
    inter = pairwise_intersection(a, b)
    torch.testing.assert_close(inter, torch.tensor([[4.0]], device=device))


def test_pairwise_iou_empty_boxes_yield_zero(device: torch.device) -> None:
    a = Boxes(torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device))
    b = Boxes(torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device))
    iou = pairwise_iou(a, b)
    torch.testing.assert_close(iou, torch.zeros(1, 1, device=device))


def test_pairwise_ioa(device: torch.device) -> None:
    # boxes2 area = 4. intersection = 2. ioa = 0.5.
    a = Boxes(torch.tensor([[0.0, 0.0, 4.0, 1.0]], device=device))
    b = Boxes(torch.tensor([[2.0, 0.0, 4.0, 2.0]], device=device))
    ioa = pairwise_ioa(a, b)
    torch.testing.assert_close(ioa, torch.tensor([[0.5]], device=device))


def test_pairwise_iou_empty_input_returns_empty(device: torch.device) -> None:
    a = Boxes(torch.zeros(0, 4, device=device))
    b = Boxes(torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device))
    iou = pairwise_iou(a, b)
    assert iou.shape == (0, 1)
