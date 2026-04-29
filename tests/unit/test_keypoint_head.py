"""Tests for :mod:`mayaku.models.heads.keypoint_head`."""

from __future__ import annotations

import pytest
import torch

from mayaku.config.schemas import ROIKeypointHeadConfig
from mayaku.models.backbones._base import ShapeSpec
from mayaku.models.heads import (
    KRCNNConvDeconvUpsampleHead,
    build_keypoint_head,
    keypoint_rcnn_inference,
    keypoint_rcnn_loss,
    select_proposals_with_visible_keypoints,
)
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances
from mayaku.structures.keypoints import Keypoints

# ---------------------------------------------------------------------------
# Head shape contract — pooler 14 → deconv 28 → bilinear 56
# ---------------------------------------------------------------------------


def test_head_outputs_56x56_logits(device: torch.device) -> None:
    head = KRCNNConvDeconvUpsampleHead(
        input_shape=ShapeSpec(channels=8, stride=1),
        num_keypoints=4,
        conv_dims=(8, 8),
    ).to(device)
    x = torch.randn(3, 8, 14, 14, device=device)
    out = head(x)
    assert out.shape == (3, 4, 56, 56)


def test_head_rejects_zero_conv_dims() -> None:
    with pytest.raises(ValueError, match="at least one"):
        KRCNNConvDeconvUpsampleHead(
            input_shape=ShapeSpec(channels=8, stride=1),
            num_keypoints=4,
            conv_dims=(),
        )


def test_build_keypoint_head_from_config() -> None:
    cfg = ROIKeypointHeadConfig(num_keypoints=4, conv_dims=(8, 8))
    head = build_keypoint_head(cfg, ShapeSpec(channels=8, stride=1))
    assert isinstance(head, KRCNNConvDeconvUpsampleHead)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _proposals_with_visible_keypoints(device: torch.device) -> list[Instances]:
    inst = Instances(image_size=(64, 64))
    inst.proposal_boxes = Boxes(
        torch.tensor([[0.0, 0.0, 16.0, 16.0], [4.0, 4.0, 20.0, 20.0]], device=device)
    )
    # 4 keypoints per instance: (x, y, v); v=2 means visible.
    kp = torch.tensor(
        [
            [[2.0, 2.0, 2.0], [10.0, 4.0, 2.0], [4.0, 10.0, 2.0], [12.0, 12.0, 2.0]],
            [[6.0, 6.0, 2.0], [14.0, 8.0, 2.0], [8.0, 14.0, 2.0], [16.0, 16.0, 1.0]],
        ],
        device=device,
    )
    inst.gt_keypoints = Keypoints(kp)
    return [inst]


def test_keypoint_loss_basic_finite_scalar(device: torch.device) -> None:
    fg = _proposals_with_visible_keypoints(device)
    logits = torch.randn(2, 4, 56, 56, device=device)
    loss = keypoint_rcnn_loss(logits, fg)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_keypoint_loss_zero_for_empty_input() -> None:
    logits = torch.zeros(0, 4, 56, 56, requires_grad=True)
    loss = keypoint_rcnn_loss(logits, [])
    assert loss.item() == 0.0
    loss.backward()  # zero-grad sentinel must be valid


def test_keypoint_loss_zero_when_no_visible_keypoints() -> None:
    inst = Instances(image_size=(32, 32))
    inst.proposal_boxes = Boxes(torch.tensor([[0.0, 0.0, 16.0, 16.0]]))
    # All visibility-0 → no supervision signal → zero loss.
    inst.gt_keypoints = Keypoints(torch.zeros(1, 4, 3))
    logits = torch.randn(1, 4, 56, 56, requires_grad=True)
    loss = keypoint_rcnn_loss(logits, [inst])
    assert loss.item() == 0.0
    loss.backward()


def test_keypoint_loss_static_normalizer_requires_constant() -> None:
    fg = _proposals_with_visible_keypoints(torch.device("cpu"))
    logits = torch.randn(2, 4, 56, 56)
    with pytest.raises(ValueError, match="static_normalizer_constant"):
        keypoint_rcnn_loss(logits, fg, normalizer="static")


def test_keypoint_loss_static_normalizer_path() -> None:
    fg = _proposals_with_visible_keypoints(torch.device("cpu"))
    logits = torch.randn(2, 4, 56, 56)
    loss = keypoint_rcnn_loss(logits, fg, normalizer="static", static_normalizer_constant=4.0)
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_keypoint_inference_attaches_predictions(device: torch.device) -> None:
    inst = Instances(image_size=(64, 64))
    inst.pred_boxes = Boxes(
        torch.tensor([[0.0, 0.0, 32.0, 32.0], [10.0, 10.0, 50.0, 50.0]], device=device)
    )
    inst.scores = torch.tensor([0.9, 0.8], device=device)
    inst.pred_classes = torch.tensor([0, 0], device=device)
    logits = torch.randn(2, 4, 56, 56, device=device)
    keypoint_rcnn_inference(logits, [inst])
    assert inst.has("pred_keypoints")
    assert inst.has("pred_keypoint_heatmaps")
    assert inst.pred_keypoints.shape == (2, 4, 3)
    assert inst.pred_keypoint_heatmaps.shape == (2, 4, 56, 56)


def test_keypoint_inference_planted_peak_decodes_inside_box(device: torch.device) -> None:
    """Synthetic-landmark accuracy gate (spec §2.6 round-trip).

    Plant a sharp peak at heatmap cell ``(cy, cx) = (28, 28)`` for a
    single keypoint, with a single ROI covering a 56-pixel-wide box.
    The decoded keypoint must land roughly in the centre of the box."""
    inst = Instances(image_size=(64, 64))
    inst.pred_boxes = Boxes(torch.tensor([[0.0, 0.0, 56.0, 56.0]], device=device))
    inst.scores = torch.tensor([1.0], device=device)
    inst.pred_classes = torch.tensor([0], device=device)
    logits = torch.full((1, 1, 56, 56), -10.0, device=device)
    logits[0, 0, 28, 28] = 10.0
    keypoint_rcnn_inference(logits, [inst])
    decoded_xy = inst.pred_keypoints[0, 0, :2]
    # Bicubic on a 56→56 input is identity, so decoded x ≈ 28.5 (cell + 0.5).
    assert abs(decoded_xy[0].item() - 28.5) < 2.0
    assert abs(decoded_xy[1].item() - 28.5) < 2.0


# ---------------------------------------------------------------------------
# select_proposals_with_visible_keypoints
# ---------------------------------------------------------------------------


def test_visible_kp_selector_drops_invisible_only_proposals() -> None:
    inst = Instances(image_size=(8, 8))
    inst.proposal_boxes = Boxes(torch.zeros(3, 4))
    inst.gt_classes = torch.tensor([0, 0, 0])
    kp = torch.zeros(3, 4, 3)
    kp[0, 0, 2] = 2.0  # one visible
    # instance 1: all invisible
    kp[2, 1, 2] = 1.0  # one occluded → still > 0, so visible
    inst.gt_keypoints = Keypoints(kp)
    out, masks = select_proposals_with_visible_keypoints([inst])
    assert masks[0].tolist() == [True, False, True]
    assert len(out[0]) == 2
