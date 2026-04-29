"""Tests for :mod:`mayaku.models.heads.mask_head`."""

from __future__ import annotations

import pytest
import torch

from mayaku.config.schemas import ROIMaskHeadConfig
from mayaku.models.backbones._base import ShapeSpec
from mayaku.models.heads import (
    MaskRCNNConvUpsampleHead,
    build_mask_head,
    mask_rcnn_inference,
    mask_rcnn_loss,
    select_foreground_proposals,
)
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances
from mayaku.structures.masks import BitMasks, PolygonMasks

# ---------------------------------------------------------------------------
# Head shape contract
# ---------------------------------------------------------------------------


def test_mask_head_doubles_spatial_dim_to_28_from_14(device: torch.device) -> None:
    head = MaskRCNNConvUpsampleHead(
        input_shape=ShapeSpec(channels=8, stride=1),
        num_classes=3,
        num_conv=2,
        conv_dim=8,
    ).to(device)
    x = torch.randn(5, 8, 14, 14, device=device)
    out = head(x)
    assert out.shape == (5, 3, 28, 28)


def test_mask_head_cls_agnostic_emits_single_channel(device: torch.device) -> None:
    head = MaskRCNNConvUpsampleHead(
        input_shape=ShapeSpec(channels=8, stride=1),
        num_classes=4,
        num_conv=1,
        conv_dim=8,
        cls_agnostic_mask=True,
    ).to(device)
    x = torch.randn(2, 8, 14, 14, device=device)
    out = head(x)
    assert out.shape == (2, 1, 28, 28)


def test_mask_head_rejects_zero_conv() -> None:
    with pytest.raises(ValueError, match="num_conv"):
        MaskRCNNConvUpsampleHead(
            input_shape=ShapeSpec(channels=8, stride=1),
            num_classes=2,
            num_conv=0,
        )


def test_build_mask_head_from_config() -> None:
    cfg = ROIMaskHeadConfig(num_conv=2, conv_dim=16)
    head = build_mask_head(cfg, ShapeSpec(channels=16, stride=1), num_classes=4)
    assert isinstance(head, MaskRCNNConvUpsampleHead)
    assert head.num_classes == 4


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _fg_proposals_with_polygon(device: torch.device) -> list[Instances]:
    inst = Instances(image_size=(64, 64))
    inst.proposal_boxes = Boxes(
        torch.tensor([[0.0, 0.0, 16.0, 16.0], [4.0, 4.0, 20.0, 20.0]], device=device)
    )
    inst.gt_classes = torch.tensor([0, 1], device=device)
    # Two square polygons covering the proposal regions roughly.
    inst.gt_masks = PolygonMasks(
        [
            [[0, 0, 16, 0, 16, 16, 0, 16]],
            [[4, 4, 20, 4, 20, 20, 4, 20]],
        ]
    )
    return [inst]


def test_mask_loss_basic_value_finite_and_scalar(device: torch.device) -> None:
    fg = _fg_proposals_with_polygon(device)
    logits = torch.randn(2, 3, 28, 28, device=device)
    loss = mask_rcnn_loss(logits, fg)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_mask_loss_zero_for_empty_input() -> None:
    logits = torch.zeros(0, 3, 28, 28, requires_grad=True)
    loss = mask_rcnn_loss(logits, [])
    assert loss.item() == 0.0
    # backward must succeed (zero-grad sentinel).
    loss.backward()


def test_mask_loss_uses_per_class_channel(device: torch.device) -> None:
    # If we set the gt-class channel to match the target exactly and the
    # other channels to garbage, BCE on those channels still drops to ~0
    # because we only score the gt-class channel.
    fg = _fg_proposals_with_polygon(device)
    target_bitmask = fg[0].gt_masks.crop_and_resize(fg[0].proposal_boxes.tensor, 28).to(device)
    logits = torch.full((2, 3, 28, 28), -10.0, device=device)
    # Drive gt-class channel to match the target.
    for i, c in enumerate(fg[0].gt_classes.tolist()):
        logits[i, c] = torch.where(
            target_bitmask[i], torch.tensor(10.0, device=device), torch.tensor(-10.0, device=device)
        )
    loss = mask_rcnn_loss(logits, fg)
    assert loss.item() < 0.1


def test_mask_loss_cls_agnostic_path(device: torch.device) -> None:
    fg = _fg_proposals_with_polygon(device)
    logits = torch.randn(2, 1, 28, 28, device=device)
    loss = mask_rcnn_loss(logits, fg, cls_agnostic=True)
    assert torch.isfinite(loss).item()


def test_mask_loss_works_with_bitmasks(device: torch.device) -> None:
    inst = Instances(image_size=(32, 32))
    inst.proposal_boxes = Boxes(torch.tensor([[0.0, 0.0, 16.0, 16.0]], device=device))
    inst.gt_classes = torch.tensor([0], device=device)
    bitmap = torch.zeros(1, 32, 32, dtype=torch.bool, device=device)
    bitmap[0, 4:12, 4:12] = True
    inst.gt_masks = BitMasks(bitmap)
    logits = torch.randn(1, 2, 28, 28, device=device)
    loss = mask_rcnn_loss(logits, [inst])
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_mask_inference_attaches_pred_masks(device: torch.device) -> None:
    inst = Instances(image_size=(64, 64))
    inst.pred_boxes = Boxes(torch.zeros(3, 4, device=device))
    inst.scores = torch.zeros(3, device=device)
    inst.pred_classes = torch.tensor([0, 1, 0], device=device)
    logits = torch.randn(3, 2, 28, 28, device=device)
    mask_rcnn_inference(logits, [inst])
    assert inst.has("pred_masks")
    assert inst.pred_masks.shape == (3, 1, 28, 28)
    # Sigmoid output: in [0, 1].
    assert (inst.pred_masks >= 0).all()
    assert (inst.pred_masks <= 1).all()


def test_mask_inference_cls_agnostic(device: torch.device) -> None:
    inst = Instances(image_size=(64, 64))
    inst.pred_boxes = Boxes(torch.zeros(2, 4, device=device))
    inst.scores = torch.zeros(2, device=device)
    inst.pred_classes = torch.tensor([0, 0], device=device)
    logits = torch.randn(2, 1, 28, 28, device=device)
    mask_rcnn_inference(logits, [inst], cls_agnostic=True)
    assert inst.pred_masks.shape == (2, 1, 28, 28)


# ---------------------------------------------------------------------------
# select_foreground_proposals
# ---------------------------------------------------------------------------


def test_select_foreground_proposals_drops_background() -> None:
    inst = Instances(image_size=(8, 8))
    inst.proposal_boxes = Boxes(torch.zeros(4, 4))
    # Class 5 = num_classes (background); -1 = ignore.
    inst.gt_classes = torch.tensor([0, 1, 5, -1])
    fg, masks = select_foreground_proposals([inst], num_classes=5)
    assert len(fg[0]) == 2
    assert masks[0].tolist() == [True, True, False, False]
