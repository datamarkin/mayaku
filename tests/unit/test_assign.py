"""TaskAligned and ATSS assignment, conflict resolution, and the sub-stride floor."""

from __future__ import annotations

import torch

from mayaku.engine.assign import (
    AssignInput,
    ATSSAssigner,
    TaskAlignedAssigner,
    candidates_in_gt,
    masked_mean_std,
    resolve_conflicts,
)
from mayaku.model.blocks import STRIDES
from mayaku.model.box import anchor_grid, iou_xyxy, ltrb_to_xyxy

NC = 80
BUCKETS = [(0, 8), (8, 32), (32, 96), (96, 1e9)]


def _synthetic(n_img=16, imgsz=640, seed=0):
    """`n_img` images, each holding one object from every size bucket."""
    g = torch.Generator().manual_seed(seed)
    sides = []
    for lo, hi in BUCKETS:
        hi = min(hi, imgsz * 0.9)
        sides.append(torch.rand(n_img, 1, generator=g) * (hi - lo) + lo)
    side = torch.cat(sides, 1)
    cx = torch.rand(n_img, len(BUCKETS), generator=g) * (imgsz - side) + side / 2
    cy = torch.rand(n_img, len(BUCKETS), generator=g) * (imgsz - side) + side / 2
    boxes = torch.stack((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), -1)
    labels = torch.randint(0, NC, (n_img, len(BUCKETS), 1), generator=g).float()
    return boxes, labels


def _oracle(points, gt_boxes, gt_labels, n):
    """What a converged model would emit: every anchor predicts the smallest
    object whose box contains it, at high confidence."""
    b = gt_boxes.shape[0]
    boxes = torch.zeros(b, n, 4)
    scores = torch.full((b, n, NC), 0.01)
    inside = candidates_in_gt(points, gt_boxes)
    area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
    for i in range(b):
        for j in area[i].argsort(descending=True).tolist():  # small overwrites large
            hit = inside[i, j]
            boxes[i][hit] = gt_boxes[i, j]
            scores[i, :, int(gt_labels[i, j, 0])][hit] = 0.9
    return scores, boxes


def _setup():
    torch.manual_seed(0)
    shapes = [(640 // s, 640 // s) for s in STRIDES]
    n_per_level = tuple(h * w for h, w in shapes)
    points, stride = anchor_grid(shapes, STRIDES)
    gt_boxes, gt_labels = _synthetic()
    b, m = gt_boxes.shape[:2]
    n = len(points)

    def inputs(scores, boxes):
        return AssignInput(scores, boxes, points, stride, n_per_level,
                           gt_labels, gt_boxes, torch.ones(b, m, 1))

    noise = inputs(torch.rand(b, n, NC) * 0.1,
                   ltrb_to_xyxy(torch.rand(b, n, 4) * 4, points, stride))
    good = inputs(*_oracle(points, gt_boxes, gt_labels, n))
    return points, stride, n_per_level, noise, good


def test_atss_reads_no_prediction() -> None:
    _, _, _, noise, good = _setup()
    assert torch.equal(ATSSAssigner(NC)(noise).fg, ATSSAssigner(NC)(good).fg)


def test_tal_targets_are_well_formed() -> None:
    _, _, _, _, good = _setup()
    out = TaskAlignedAssigner(NC)(good)
    b, n = good.pd_boxes.shape[:2]
    assert out.scores.shape == (b, n, NC) and out.boxes.shape == (b, n, 4)
    assert out.labels.shape == (b, n) and out.fg.shape == (b, n)
    assert (out.scores.sum(-1) > 0).sum() == out.fg.sum()
    assert out.scores.max() <= 1.0 + 1e-5
    # starvation is sub-stride only: every object above one stride is served
    assert not out.starved[:, 1:].any()


def test_every_positive_anchor_has_one_owner() -> None:
    _, _, _, _, good = _setup()
    b, m = good.gt_boxes.shape[:2]
    n = good.pd_boxes.shape[1]
    ov = iou_xyxy(good.gt_boxes[:, :, None, :], good.pd_boxes[:, None, :, :]).clamp_(0)
    mp = resolve_conflicts((torch.rand(b, m, n) > 0.5).float(), ov)
    assert mp.sum(-2).max() <= 1


def test_sub_stride_box_is_reachable() -> None:
    points, stride, n_per_level, noise, _ = _setup()
    thin = torch.tensor([[[100.0, 100.0, 103.0, 103.0]]])
    assert not candidates_in_gt(points, thin, finest=0.0).any()
    assert candidates_in_gt(points, thin).any()
    pd_boxes = noise.pd_boxes[:1].clone()
    pd_boxes[:] = thin[0, 0]
    one = AssignInput(noise.pd_scores[:1], pd_boxes, points, stride, n_per_level,
                      torch.zeros(1, 1, 1), thin, torch.ones(1, 1, 1))
    out = TaskAlignedAssigner(NC)(one)
    assert not out.starved.any() and out.fg.sum() > 0


def test_atss_threshold_guard() -> None:
    x = torch.tensor([[[0.0, 1.0, 1.0]]])
    mean, std = masked_mean_std(x, torch.ones_like(x, dtype=torch.bool))
    raw = (mean + std).item()
    assert raw > 1.0 and min(raw, x.max().item()) == 1.0
