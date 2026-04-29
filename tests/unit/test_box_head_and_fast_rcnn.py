"""Tests for box head + FastRCNNOutputLayers + fast_rcnn_inference."""

from __future__ import annotations

import pytest
import torch

from mayaku.models.backbones._base import ShapeSpec
from mayaku.models.heads import (
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    fast_rcnn_inference_single_image,
)
from mayaku.models.proposals.box_regression import Box2BoxTransform
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances

# ---------------------------------------------------------------------------
# Box head
# ---------------------------------------------------------------------------


def test_box_head_default_output_dim() -> None:
    head = FastRCNNConvFCHead(
        input_shape=ShapeSpec(channels=256, stride=1),
        pooler_resolution=7,
    )
    # Default num_fc=2, fc_dim=1024 → output is 1024.
    assert head.output_dim == 1024
    x = torch.randn(3, 256, 7, 7)
    out = head(x)
    assert out.shape == (3, 1024)


def test_box_head_with_conv_layers() -> None:
    head = FastRCNNConvFCHead(
        input_shape=ShapeSpec(channels=64, stride=1),
        pooler_resolution=4,
        num_conv=2,
        conv_dim=128,
        num_fc=1,
        fc_dim=256,
    )
    x = torch.randn(2, 64, 4, 4)
    out = head(x)
    assert out.shape == (2, 256)


def test_box_head_rejects_zero_layers() -> None:
    with pytest.raises(ValueError, match="num_conv > 0"):
        FastRCNNConvFCHead(
            input_shape=ShapeSpec(channels=8, stride=1),
            pooler_resolution=4,
            num_conv=0,
            num_fc=0,
        )


# ---------------------------------------------------------------------------
# FastRCNNOutputLayers shapes
# ---------------------------------------------------------------------------


def _output_layers(*, num_classes: int = 5) -> FastRCNNOutputLayers:
    return FastRCNNOutputLayers(
        input_dim=32,
        num_classes=num_classes,
        box_transform=Box2BoxTransform(weights=(10.0, 10.0, 5.0, 5.0)),
    )


def test_output_layers_shapes(device: torch.device) -> None:
    layers = _output_layers().to(device)
    x = torch.randn(7, 32, device=device)
    scores, deltas = layers(x)
    assert scores.shape == (7, 6)  # K + 1
    assert deltas.shape == (7, 20)  # K * 4


def test_output_layers_cls_agnostic_emits_4_deltas() -> None:
    layers = FastRCNNOutputLayers(
        input_dim=32,
        num_classes=5,
        box_transform=Box2BoxTransform(weights=(10.0, 10.0, 5.0, 5.0)),
        cls_agnostic_bbox_reg=True,
    )
    x = torch.randn(3, 32)
    _, deltas = layers(x)
    assert deltas.shape == (3, 4)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def _toy_proposals(device: torch.device) -> list[Instances]:
    inst = Instances(image_size=(32, 32))
    inst.proposal_boxes = Boxes(
        torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0], [0.0, 0.0, 8.0, 8.0]],
            device=device,
        )
    )
    inst.gt_classes = torch.tensor([1, 5, 5], device=device, dtype=torch.long)  # 1=fg, 5=bg
    inst.gt_boxes = torch.tensor(
        [[1.0, 1.0, 11.0, 11.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        device=device,
    )
    return [inst]


def test_output_layers_losses_have_expected_keys(device: torch.device) -> None:
    layers = _output_layers(num_classes=5).to(device)
    x = torch.randn(3, 32, device=device)
    pred = layers(x)
    losses = layers.losses(pred, _toy_proposals(device))
    assert set(losses) == {"loss_cls", "loss_box_reg"}
    for v in losses.values():
        assert v.ndim == 0
        assert torch.isfinite(v).item()


def test_output_layers_losses_handle_no_foreground(device: torch.device) -> None:
    # All bg → loss_box_reg is exactly zero.
    layers = _output_layers(num_classes=5).to(device)
    inst = Instances(image_size=(32, 32))
    inst.proposal_boxes = Boxes(torch.tensor([[0.0, 0.0, 4.0, 4.0]], device=device))
    inst.gt_classes = torch.tensor([5], device=device)  # bg
    inst.gt_boxes = torch.zeros(1, 4, device=device)
    x = torch.randn(1, 32, device=device)
    losses = layers.losses(layers(x), [inst])
    assert losses["loss_box_reg"].item() == 0.0


# ---------------------------------------------------------------------------
# fast_rcnn_inference
# ---------------------------------------------------------------------------


def test_fast_rcnn_inference_single_image_basic() -> None:
    # Two RoIs, three classes (+ background).
    boxes = torch.tensor(
        [
            [0, 0, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10],  # roi 0
            [5, 5, 15, 15, 5, 5, 15, 15, 5, 5, 15, 15],  # roi 1
        ],
        dtype=torch.float32,
    )
    probs = torch.tensor(
        [
            [0.1, 0.2, 0.6, 0.1],  # class 2 strong
            [0.7, 0.05, 0.2, 0.05],  # class 0 strong
        ]
    )
    inst = fast_rcnn_inference_single_image(
        boxes,
        probs,
        image_shape=(20, 20),
        score_thresh=0.05,
        nms_thresh=0.5,
        topk_per_image=10,
    )
    # Both rois keep their best class above the threshold; per-class NMS
    # keeps each.
    assert len(inst) >= 2
    # All scores >= threshold.
    assert (inst.scores >= 0.05).all()
    # Predicted classes are valid indices in [0, K).
    assert (inst.pred_classes >= 0).all()
    assert (inst.pred_classes < 3).all()


def test_fast_rcnn_inference_filters_below_threshold() -> None:
    boxes = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.float32)  # 1 roi, K=1
    probs = torch.tensor([[0.01, 0.99]])  # class 0 below thresh, class 1 = bg dropped
    inst = fast_rcnn_inference_single_image(
        boxes, probs, (4, 4), score_thresh=0.05, nms_thresh=0.5, topk_per_image=10
    )
    assert len(inst) == 0


def test_fast_rcnn_inference_clips_to_image() -> None:
    # Single roi/class with box extending way past the image bounds.
    boxes = torch.tensor([[-10, -10, 100, 100]], dtype=torch.float32)
    probs = torch.tensor([[0.9, 0.1]])
    inst = fast_rcnn_inference_single_image(
        boxes, probs, (10, 10), score_thresh=0.5, nms_thresh=0.5, topk_per_image=10
    )
    assert len(inst) == 1
    b = inst.pred_boxes.tensor
    assert b[0, 0].item() == 0.0 and b[0, 1].item() == 0.0
    assert b[0, 2].item() == 10.0 and b[0, 3].item() == 10.0


def test_fast_rcnn_inference_class_agnostic_box_form() -> None:
    boxes = torch.tensor([[0, 0, 5, 5]], dtype=torch.float32)  # 4-tuple → cls-agnostic
    probs = torch.tensor([[0.1, 0.4, 0.5]])  # K=2 + bg
    inst = fast_rcnn_inference_single_image(
        boxes, probs, (10, 10), score_thresh=0.05, nms_thresh=0.5, topk_per_image=10
    )
    # Both classes survive the threshold; same box, different class+score.
    assert len(inst) >= 2
