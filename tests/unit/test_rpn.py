"""Tests for :mod:`mayaku.models.proposals.rpn`."""

from __future__ import annotations

import torch

from mayaku.config.schemas import AnchorGeneratorConfig, RPNConfig
from mayaku.models.backbones._base import ShapeSpec
from mayaku.models.proposals import (
    RPN,
    StandardRPNHead,
    build_rpn,
    find_top_rpn_proposals,
)
from mayaku.models.proposals.box_regression import Box2BoxTransform
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances

# ---------------------------------------------------------------------------
# StandardRPNHead
# ---------------------------------------------------------------------------


def test_standard_rpn_head_per_level_shapes(device: torch.device) -> None:
    head = StandardRPNHead(in_channels=32, num_anchors=3).to(device)
    feats = [
        torch.randn(2, 32, 8, 12, device=device),
        torch.randn(2, 32, 4, 6, device=device),
    ]
    cls, reg = head(feats)
    assert len(cls) == len(reg) == 2
    assert cls[0].shape == (2, 3, 8, 12)
    assert cls[1].shape == (2, 3, 4, 6)
    assert reg[0].shape == (2, 12, 8, 12)  # A*4 = 12
    assert reg[1].shape == (2, 12, 4, 6)


# ---------------------------------------------------------------------------
# Full RPN forward / loss
# ---------------------------------------------------------------------------


def _build_tiny_rpn(device: torch.device) -> RPN:
    """A two-level RPN with strides (4, 8) and 3 anchors per cell."""
    in_shapes = {
        "p2": ShapeSpec(channels=32, stride=4),
        "p3": ShapeSpec(channels=32, stride=8),
    }
    rpn_cfg = RPNConfig(
        in_features=("p2", "p3"),
        pre_nms_topk_train=20,
        pre_nms_topk_test=10,
        post_nms_topk_train=10,
        post_nms_topk_test=5,
        batch_size_per_image=8,
    )
    anchor_cfg = AnchorGeneratorConfig(
        sizes=((32,), (64,)),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    rpn = build_rpn(rpn_cfg, anchor_cfg, in_shapes).to(device)
    return rpn


def test_rpn_inference_returns_proposals(device: torch.device) -> None:
    torch.manual_seed(0)
    rpn = _build_tiny_rpn(device).eval()
    features = {
        "p2": torch.randn(2, 32, 8, 12, device=device),
        "p3": torch.randn(2, 32, 4, 6, device=device),
    }
    image_sizes = [(32, 48), (32, 48)]
    with torch.no_grad():
        proposals, losses = rpn(image_sizes, features)
    assert losses == {}
    assert len(proposals) == 2
    for inst in proposals:
        assert isinstance(inst, Instances)
        assert inst.has("proposal_boxes")
        assert inst.has("objectness_logits")
        assert isinstance(inst.proposal_boxes, Boxes)
        # ≤ post_nms_topk_test = 5
        assert len(inst) <= 5


def test_rpn_training_returns_loss_dict(device: torch.device) -> None:
    torch.manual_seed(0)
    rpn = _build_tiny_rpn(device).train()
    features = {
        "p2": torch.randn(1, 32, 8, 12, device=device),
        "p3": torch.randn(1, 32, 4, 6, device=device),
    }
    image_sizes = [(32, 48)]
    gt = Instances(image_size=(32, 48))
    gt.gt_boxes = torch.tensor([[2.0, 3.0, 22.0, 28.0]], device=device)
    proposals, losses = rpn(image_sizes, features, gt_instances=[gt])
    assert set(losses) == {"loss_rpn_cls", "loss_rpn_loc"}
    for v in losses.values():
        assert v.ndim == 0
        assert torch.isfinite(v).item()
    assert len(proposals) == 1


def test_rpn_training_handles_image_with_no_gt(device: torch.device) -> None:
    rpn = _build_tiny_rpn(device).train()
    features = {
        "p2": torch.randn(1, 32, 8, 12, device=device),
        "p3": torch.randn(1, 32, 4, 6, device=device),
    }
    image_sizes = [(32, 48)]
    gt = Instances(image_size=(32, 48))
    gt.gt_boxes = torch.zeros(0, 4, device=device)
    _proposals, losses = rpn(image_sizes, features, gt_instances=[gt])
    # Loc loss collapses to zero when there are no foreground anchors.
    assert torch.isfinite(losses["loss_rpn_cls"]).item()
    assert losses["loss_rpn_loc"].item() == 0.0


def test_rpn_loss_grad_flows_to_head_params(device: torch.device) -> None:
    rpn = _build_tiny_rpn(device).train()
    features = {
        "p2": torch.randn(1, 32, 8, 12, device=device),
        "p3": torch.randn(1, 32, 4, 6, device=device),
    }
    gt = Instances(image_size=(32, 48))
    gt.gt_boxes = torch.tensor([[2.0, 3.0, 22.0, 28.0]], device=device)
    _, losses = rpn([(32, 48)], features, gt_instances=[gt])
    total = losses["loss_rpn_cls"] + losses["loss_rpn_loc"]
    total.backward()
    # All head conv weights should have non-None grads after backward.
    for name, p in rpn.head.named_parameters():
        assert p.grad is not None, f"head parameter {name} got no grad"


def test_rpn_training_proposals_are_detached_from_autograd(device: torch.device) -> None:
    """Proposals returned during training must not carry RPN-regression
    gradients into downstream consumers (the ROI heads).

    Without the no_grad wrapper around `find_top_rpn_proposals`, a
    downstream loss that uses `proposal_boxes` (e.g. via
    `box_transform.get_deltas(proposal_boxes, gt_boxes)` in the ROI box
    head) will backprop through the decoded proposal coordinates and
    pollute the RPN's regression gradient. Mirrors detectron2's
    `predict_proposals` contract.
    """
    rpn = _build_tiny_rpn(device).train()
    features = {
        "p2": torch.randn(1, 32, 8, 12, device=device),
        "p3": torch.randn(1, 32, 4, 6, device=device),
    }
    gt = Instances(image_size=(32, 48))
    gt.gt_boxes = torch.tensor([[2.0, 3.0, 22.0, 28.0]], device=device)
    proposals, _ = rpn([(32, 48)], features, gt_instances=[gt])
    for inst in proposals:
        assert not inst.proposal_boxes.tensor.requires_grad, (
            "proposal_boxes leak RPN regression gradient — wrap "
            "find_top_rpn_proposals in torch.no_grad()"
        )
        assert not inst.objectness_logits.requires_grad, "objectness_logits leak RPN cls gradient"


# ---------------------------------------------------------------------------
# find_top_rpn_proposals
# ---------------------------------------------------------------------------


def test_find_top_rpn_proposals_clips_and_ranks(device: torch.device) -> None:
    box_transform = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
    # Single level, single image, 3 anchors.
    anchors = [
        Boxes(
            torch.tensor(
                [[0.0, 0.0, 10.0, 10.0], [4.0, 4.0, 14.0, 14.0], [8.0, 8.0, 18.0, 18.0]],
                device=device,
            )
        )
    ]
    cls = [torch.tensor([[0.5, 0.9, 0.1]], device=device)]
    deltas = [torch.zeros(1, 3, 4, device=device)]
    proposals = find_top_rpn_proposals(
        cls,
        deltas,
        anchors,
        image_sizes=[(20, 20)],
        box_transform=box_transform,
        pre_nms_topk=3,
        post_nms_topk=2,
        nms_thresh=0.7,
    )
    inst = proposals[0]
    # Highest-scoring anchor (1) survives; others have heavy IoU with it
    # so NMS may drop them.
    scores = inst.objectness_logits.tolist()
    # Scores must be returned in descending order.
    assert scores == sorted(scores, reverse=True)
    # No more than the post-NMS cap.
    assert len(inst) <= 2
    # All boxes lie inside the (20, 20) image.
    boxes = inst.proposal_boxes.tensor
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
    assert (boxes[:, 2] <= 20).all() and (boxes[:, 3] <= 20).all()
