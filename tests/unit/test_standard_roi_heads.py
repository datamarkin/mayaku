"""Tests for :class:`mayaku.models.roi_heads.StandardROIHeads`."""

from __future__ import annotations

import torch

from mayaku.config.schemas import ROIBoxHeadConfig, ROIHeadsConfig
from mayaku.models.backbones._base import ShapeSpec
from mayaku.models.roi_heads import build_standard_roi_heads
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances


def _in_shapes() -> dict[str, ShapeSpec]:
    return {
        "p2": ShapeSpec(channels=8, stride=4),
        "p3": ShapeSpec(channels=8, stride=8),
        "p4": ShapeSpec(channels=8, stride=16),
        "p5": ShapeSpec(channels=8, stride=32),
    }


def _features(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "p2": torch.randn(1, 8, 16, 16, device=device),
        "p3": torch.randn(1, 8, 8, 8, device=device),
        "p4": torch.randn(1, 8, 4, 4, device=device),
        "p5": torch.randn(1, 8, 2, 2, device=device),
    }


def _proposals(device: torch.device, k: int = 6) -> list[Instances]:
    inst = Instances(image_size=(64, 64))
    inst.proposal_boxes = Boxes(
        torch.tensor(
            [[float(i), float(i), float(i + 20), float(i + 20)] for i in range(k)],
            device=device,
        )
    )
    inst.objectness_logits = torch.linspace(1.0, 0.1, k, device=device)
    return [inst]


def _gt_instances(device: torch.device) -> list[Instances]:
    gt = Instances(image_size=(64, 64))
    gt.gt_boxes = torch.tensor([[2.0, 3.0, 25.0, 25.0]], device=device)
    gt.gt_classes = torch.tensor([0], dtype=torch.long, device=device)
    return [gt]


def test_label_and_sample_assigns_classes_and_subsamples(device: torch.device) -> None:
    cfg = ROIHeadsConfig(num_classes=2, batch_size_per_image=4, positive_fraction=0.5)
    head_cfg = ROIBoxHeadConfig(num_fc=1, fc_dim=16)
    rh = build_standard_roi_heads(cfg, head_cfg, _in_shapes()).to(device)
    sampled = rh.label_and_sample_proposals(_proposals(device), _gt_instances(device))
    inst = sampled[0]
    assert len(inst) <= 4  # subsample cap
    # Background class is encoded as num_classes (= 2).
    assert ((inst.gt_classes >= 0) & (inst.gt_classes <= cfg.num_classes)).all()
    # Each sampled proposal has matching gt_boxes.
    assert inst.gt_boxes.shape == (len(inst), 4)


def test_label_and_sample_appends_gt_boxes(device: torch.device) -> None:
    cfg = ROIHeadsConfig(num_classes=2, batch_size_per_image=64, positive_fraction=0.5)
    head_cfg = ROIBoxHeadConfig(num_fc=1, fc_dim=16)
    rh = build_standard_roi_heads(cfg, head_cfg, _in_shapes()).to(device)
    sampled = rh.label_and_sample_proposals(_proposals(device, k=2), _gt_instances(device))
    # With proposal_append_gt=True (default), the GT box is added to the
    # proposal pool so a fg sample is virtually guaranteed.
    fg = sampled[0].gt_classes < cfg.num_classes
    assert fg.any().item()


def test_inference_returns_pred_boxes_and_scores(device: torch.device) -> None:
    torch.manual_seed(0)
    cfg = ROIHeadsConfig(num_classes=2, batch_size_per_image=64)
    head_cfg = ROIBoxHeadConfig(num_fc=1, fc_dim=16)
    rh = build_standard_roi_heads(cfg, head_cfg, _in_shapes()).to(device).eval()
    with torch.no_grad():
        out, losses = rh(_features(device), _proposals(device, k=4), targets=None)
    assert losses == {}
    assert len(out) == 1
    inst = out[0]
    assert inst.has("pred_boxes")
    assert inst.has("scores")
    assert inst.has("pred_classes")


def test_training_returns_loss_dict(device: torch.device) -> None:
    cfg = ROIHeadsConfig(num_classes=2, batch_size_per_image=8, positive_fraction=0.5)
    head_cfg = ROIBoxHeadConfig(num_fc=1, fc_dim=16)
    rh = build_standard_roi_heads(cfg, head_cfg, _in_shapes()).to(device).train()
    _, losses = rh(_features(device), _proposals(device), targets=_gt_instances(device))
    assert set(losses) == {"loss_cls", "loss_box_reg"}
    for v in losses.values():
        assert torch.isfinite(v).item()
