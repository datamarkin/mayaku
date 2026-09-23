"""Box, mask and keypoint maths shared by the loss, the evaluator and the host decode."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mayaku.data.geometry import unletterbox_maps
from mayaku.model.aux import KERNEL_PARAMS, MASK_CH
from mayaku.model.box import (
    anchor_grid,
    decode_head,
    dfl_expectation,
    iou_xyxy,
    ltrb_to_xyxy,
    xyxy_to_ltrb,
)
from mayaku.model.kpt import decode, encode, focal, gaussian_targets, oks_loss, sigmas, snap
from mayaku.model.mask import (
    COORD_SCALE,
    assemble,
    dice,
    dyn_conv,
    rel_coords,
    split_kernels,
    to_canvas,
)

# --------------------------------------------------------------------------
# box
# --------------------------------------------------------------------------


def test_anchor_grid() -> None:
    shapes, strides = [(80, 80), (40, 40), (20, 20)], [8, 16, 32]
    points, stride = anchor_grid(shapes, strides)
    assert points.shape == (8400, 2) and stride.shape == (8400, 1)
    assert torch.equal(points[0], torch.tensor([4.0, 4.0]))
    assert torch.equal(points[-1], torch.tensor([624.0, 624.0]))
    assert stride[0] == 8 and stride[-1] == 32
    # shared, not rebuilt; shape is the only thing it depends on
    assert anchor_grid(shapes, strides)[0] is points
    assert anchor_grid([(40, 40), (20, 20), (10, 10)], strides)[0] is not points


def test_anchor_grid_rectangular() -> None:
    points, _ = anchor_grid([(4, 6)], [8])
    assert points.shape == (24, 2)
    assert torch.equal(points[5], torch.tensor([44.0, 4.0]))   # last column, first row


def test_dfl_expectation_uniform() -> None:
    reg_max = 16
    flat = torch.zeros(1, 5, 4 * reg_max)
    assert torch.allclose(dfl_expectation(flat, reg_max), torch.full((1, 5, 4), (reg_max - 1) / 2))


def test_ltrb_round_trip_and_clamp() -> None:
    reg_max = 16
    box = torch.tensor([[[100.0, 120.0, 180.0, 200.0]]])
    pt = torch.tensor([[[140.0, 160.0]]])
    st = torch.tensor([[[8.0]]])
    d = xyxy_to_ltrb(box, pt, st, reg_max)
    assert torch.allclose(ltrb_to_xyxy(d, pt, st), box, atol=1e-4)
    far = torch.tensor([[[0.0, 0.0, 640.0, 640.0]]])
    assert xyxy_to_ltrb(far, pt, st, reg_max).max() <= reg_max - 1


def test_iou() -> None:
    box = torch.tensor([[[100.0, 120.0, 180.0, 200.0]]])
    assert torch.allclose(iou_xyxy(box, box), torch.ones(1, 1))
    off = torch.tensor([[[400.0, 400.0, 480.0, 480.0]]])
    assert iou_xyxy(box, off).item() == 0.0
    assert iou_xyxy(box, off, ciou=True).item() < 0.0
    ref = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    half = torch.tensor([[[5.0, 0.0, 15.0, 10.0]]])
    assert abs(iou_xyxy(ref, half).item() - 1 / 3) < 1e-6


def test_iou_matches_torchvision() -> None:
    from torchvision.ops import box_iou, complete_box_iou

    torch.manual_seed(0)
    p1 = torch.rand(64, 4) * 300
    p1[:, 2:] += p1[:, :2] + 1
    p2 = torch.rand(64, 4) * 300
    p2[:, 2:] += p2[:, :2] + 1
    assert (iou_xyxy(p1, p2) - box_iou(p1, p2).diagonal()).abs().max() < 1e-6
    assert (iou_xyxy(p1, p2, ciou=True) - complete_box_iou(p1, p2).diagonal()).abs().max() < 1e-6


def test_decode_head() -> None:
    nc, reg_max = 3, 8
    preds = []
    for h in (8, 4, 2):
        preds += [torch.full((2, nc, h, h), -5.0), torch.zeros(2, 4 * reg_max, h, h)]
    cls, dist, boxes, _, _, shp = decode_head(preds, nc, reg_max)
    n = 64 + 16 + 4
    assert cls.shape == (2, n, nc) and dist.shape == (2, n, 4 * reg_max)
    assert boxes.shape == (2, n, 4) and shp == [(8, 8), (4, 4), (2, 2)]
    # flat bin logits are uniform, so every side decodes to (reg_max-1)/2 strides
    assert torch.allclose(boxes[0, 0], torch.tensor([4.0 - 28, 4.0 - 28, 4.0 + 28, 4.0 + 28]))


# --------------------------------------------------------------------------
# mask
# --------------------------------------------------------------------------


def test_kernel_split_round_trips() -> None:
    torch.manual_seed(0)
    k = torch.randn(3, KERNEL_PARAMS)
    parts = split_kernels(k)
    flat = torch.cat([torch.cat((w.reshape(3, -1), b), 1) for w, b in parts], 1)
    assert torch.equal(flat, k)
    assert sum(w[0].numel() + b[0].numel() for w, b in parts) == KERNEL_PARAMS


def test_dyn_conv_matches_hand_computation() -> None:
    torch.manual_seed(0)
    p, h, w = 3, 10, 12
    k = torch.randn(p, KERNEL_PARAMS)
    feat = torch.randn(p, MASK_CH, h * w)
    pts = torch.tensor([[12.0, 20.0], [44.0, 36.0], [80.0, 8.0]])
    st = torch.tensor([[8.0], [16.0], [8.0]])
    coords = rel_coords(pts, st, h, w)
    j, i = int(pts[0, 0] // 8), int(pts[0, 1] // 8)
    assert coords[0, :, i * w + j].abs().max() < 0.5 / (COORD_SCALE * 8) * 8 + 1e-6
    out = dyn_conv(feat, coords, k)
    x = torch.cat((coords, feat), 1)[0]
    (w1, b1), (w2, b2), (w3, b3) = [(a[0], b[0]) for a, b in split_kernels(k)]
    ref = w3 @ F.relu(w2 @ F.relu(w1 @ x + b1[:, None]) + b2[:, None]) + b3[:, None]
    assert torch.allclose(out[0], ref[0], atol=1e-5)


def test_dice_bounds() -> None:
    assert dice(torch.full((2, 5), 20.0), torch.ones(2, 5)).max() < 1e-4
    assert dice(torch.full((1, 5), 20.0), torch.zeros(1, 5)).min() > 0.99


def test_assemble_crops_and_maps_back_to_the_image() -> None:
    torch.manual_seed(0)
    h, w = 10, 12
    k = torch.randn(1, KERNEL_PARAMS)
    lg = assemble(torch.randn(MASK_CH, h, w), k, torch.tensor([[12.0, 20.0]]),
                  torch.tensor([[8.0]]), torch.tensor([[8.0, 8.0, 24.0, 24.0]]))
    inside = torch.isfinite(lg[0])
    assert inside.sum() == 4 and inside[1:3, 1:3].all()
    up = to_canvas(lg)
    assert up.shape == (1, 1, h * 8, w * 8)
    m = unletterbox_maps(up, {"pad": (0, 0), "ratio": 1.0, "shape": (72, 96)})[:, 0] > 0
    assert m.shape == (1, 72, 96) and not m[0, :6, :6].any()


# --------------------------------------------------------------------------
# keypoints
# --------------------------------------------------------------------------


def test_keypoint_encode_decode_round_trip() -> None:
    torch.manual_seed(0)
    k = 17
    pts = torch.tensor([[12.0, 20.0], [44.0, 36.0]])
    st = torch.tensor([[8.0], [16.0]])
    xy = torch.rand(2, k, 2) * 90
    dec, _ = decode(torch.cat((encode(xy, pts, st), torch.zeros(2, k, 1)), -1), pts, st)
    assert torch.allclose(dec, xy, atol=1e-5)


def test_heatmap_targets_focal_snap_oks() -> None:
    k, h, w = 17, 10, 12
    # one labelled keypoint at pixel (20, 28): cell (2, 3), offset 0
    gxy = torch.zeros(1, 1, k, 2)
    gxy[0, 0, 0] = torch.tensor([20.0, 28.0])
    vis = torch.zeros(1, 1, k, dtype=torch.bool)
    vis[0, 0, 0] = True
    boxes = torch.tensor([[[0.0, 0.0, 96.0, 80.0]]])
    heat, off, at = gaussian_targets(gxy, vis, boxes, k, h, w)
    assert heat[0, 0, 3, 2] == 1.0 and heat[0, 1:].max() == 0 and at[0, 3, 2]
    assert heat[0, 0].sum() > 1.5
    assert off[0, :, 3, 2].abs().max() < 1e-6
    lg = torch.full((1, k, h, w), -8.0)
    lg[0, 0, 3, 2] = 8.0
    assert focal(lg, heat) < focal(torch.zeros(1, k, h, w), heat)
    # snapping moves a point 1.5 cells off onto the confident peak
    guess = torch.zeros(1, k, 2)
    guess[0, 0] = torch.tensor([32.0, 36.0])
    snapped = snap(guess, torch.ones(1, k), lg[0], off[0], boxes[0])
    assert torch.allclose(snapped[0, 0], torch.tensor([20.0, 28.0]), atol=1e-4)
    assert torch.equal(snapped[0, 1:], guess[0, 1:])
    assert oks_loss(gxy[0], gxy[0], vis[0], torch.tensor([96.0 * 80]), sigmas(k)) < 1e-6
