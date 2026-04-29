"""Tests for :mod:`mayaku.backends.ops.nms`.

Coverage:
* Hand-computed two-box overlap → only the higher-score box survives.
* Class independence in ``batched_nms``.
* The ``boxes.float()`` cast: feeding fp16 boxes whose per-class offset
  would overflow must still produce correct results
  (``DETECTRON2_TECHNICAL_SPEC.md`` §7.2 / report §3).
* Pure-PyTorch fallback parity with torchvision on a known case.
"""

from __future__ import annotations

import torch
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import nms as tv_nms

from mayaku.backends.ops.nms import (
    _batched_nms_fallback,
    _nms_fallback,
    batched_nms,
    nms,
)


def test_nms_suppresses_overlapping_lower_score(device: torch.device) -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]],
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.95], device=device)
    keep = nms(boxes, scores, iou_threshold=0.5)
    # Box 1 (IoU ~0.68 with box 0) should be dropped.
    assert sorted(keep.tolist()) == [0, 2]


def test_nms_keeps_disjoint_boxes(device: torch.device) -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0], [20.0, 20.0, 21.0, 21.0]],
        device=device,
    )
    scores = torch.tensor([0.5, 0.6, 0.7], device=device)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]


def test_nms_returns_descending_score_order(device: torch.device) -> None:
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0], [20.0, 20.0, 21.0, 21.0]],
        device=device,
    )
    scores = torch.tensor([0.3, 0.9, 0.6], device=device)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [1, 2, 0]


def test_nms_empty(device: torch.device) -> None:
    out = nms(
        torch.zeros((0, 4), device=device),
        torch.zeros((0,), device=device),
        iou_threshold=0.5,
    )
    assert out.shape == (0,)
    assert out.dtype == torch.int64


def test_batched_nms_class_independence(device: torch.device) -> None:
    """Boxes with the same coordinates but different class indices must
    both survive — that's the whole point of per-class NMS."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    idxs = torch.tensor([0, 1], device=device)
    keep = batched_nms(boxes, scores, idxs, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_batched_nms_within_class_suppression(device: torch.device) -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # IoU > 0.5 with box 0, same class
            [0.0, 0.0, 10.0, 10.0],  # different class, must survive
        ],
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    idxs = torch.tensor([0, 0, 1], device=device)
    keep = batched_nms(boxes, scores, idxs, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_batched_nms_matches_torchvision(device: torch.device) -> None:
    torch.manual_seed(0)
    n = 50
    boxes = torch.rand(n, 4, device=device) * 100
    boxes[:, 2:] = boxes[:, :2] + torch.rand(n, 2, device=device) * 50 + 1
    scores = torch.rand(n, device=device)
    idxs = torch.randint(0, 5, (n,), device=device)
    ours = batched_nms(boxes, scores, idxs, iou_threshold=0.5)
    theirs = tv_batched_nms(boxes.float(), scores, idxs, iou_threshold=0.5)
    assert ours.tolist() == theirs.tolist()


def test_batched_nms_fp16_offset_safety(device: torch.device) -> None:
    """The ``.float()`` cast must protect ``idxs * (max_coord+1)`` from
    fp16 overflow on large class indices and large image coordinates."""
    if device.type == "cpu":
        # fp16 on CPU is supported but sluggish; the algorithm is
        # device-agnostic, run anyway.
        pass
    boxes = torch.tensor(
        [[0.0, 0.0, 1000.0, 1000.0], [10.0, 10.0, 1010.0, 1010.0]],
        device=device,
        dtype=torch.float16,
    )
    scores = torch.tensor([0.9, 0.8], device=device, dtype=torch.float16)
    # Large class indices: idxs * (max_coord + 1) ≈ 80 * 1001 = 80,080
    # which exceeds fp16's 65,504 max — without the cast this corrupts.
    idxs = torch.tensor([80, 81], device=device)
    keep = batched_nms(boxes, scores, idxs, iou_threshold=0.5)
    # Despite high IoU, different classes ⇒ both survive.
    assert sorted(keep.tolist()) == [0, 1]


def test_nms_fallback_matches_torchvision() -> None:
    torch.manual_seed(3)
    n = 30
    boxes = torch.rand(n, 4) * 100
    boxes[:, 2:] = boxes[:, :2] + torch.rand(n, 2) * 30 + 1
    scores = torch.rand(n)
    ours = _nms_fallback(boxes, scores, iou_threshold=0.4)
    theirs = tv_nms(boxes, scores, iou_threshold=0.4)
    # Sets must match; order may differ in ties (none expected here).
    assert sorted(ours.tolist()) == sorted(theirs.tolist())


def test_batched_nms_fallback_matches_torchvision() -> None:
    torch.manual_seed(4)
    n = 40
    boxes = torch.rand(n, 4) * 100
    boxes[:, 2:] = boxes[:, :2] + torch.rand(n, 2) * 30 + 1
    scores = torch.rand(n)
    idxs = torch.randint(0, 4, (n,))
    ours = _batched_nms_fallback(boxes, scores, idxs, iou_threshold=0.5)
    theirs = tv_batched_nms(boxes.float(), scores, idxs, iou_threshold=0.5)
    assert sorted(ours.tolist()) == sorted(theirs.tolist())
