"""Smoke tests for QueryRCNN detector."""

from __future__ import annotations

import torch
import pytest

from mayaku.config.schemas import MayakuConfig
from mayaku.models.detectors.query_rcnn import QueryRCNN, build_query_rcnn
from mayaku.models.heads.query_head import QueryHead
from mayaku.models.heads.query_stage import DynamicConv, QueryStage
from mayaku.models.losses.set_criterion import SetCriterion, generalized_box_iou
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances


def _make_cfg(num_proposals: int = 10, num_stages: int = 2, num_classes: int = 5) -> MayakuConfig:
    return MayakuConfig(
        model={
            "meta_architecture": "query_rcnn",
            "backbone": {"name": "resnet50"},
            "query_rcnn_head": {
                "num_proposals": num_proposals,
                "num_stages": num_stages,
            },
            "roi_heads": {"num_classes": num_classes},
        }
    )


def _make_batch(
    batch_size: int = 1, h: int = 256, w: int = 256, num_gt: int = 2, num_classes: int = 5
) -> list[dict]:
    batch = []
    for _ in range(batch_size):
        img = torch.randn(3, h, w)
        boxes = torch.rand(num_gt, 4)
        boxes[:, 2:] = boxes[:, :2] + torch.rand(num_gt, 2) * 0.3 + 0.05
        boxes = boxes.clamp(0, 1)
        boxes[:, 0] *= w
        boxes[:, 1] *= h
        boxes[:, 2] *= w
        boxes[:, 3] *= h
        gt_boxes = Boxes(boxes)
        gt_classes = torch.randint(0, num_classes, (num_gt,))
        instances = Instances(image_size=(h, w), gt_boxes=gt_boxes, gt_classes=gt_classes)
        batch.append({"image": img, "instances": instances})
    return batch


class TestSetCriterion:
    def test_loss_keys(self) -> None:
        criterion = SetCriterion(num_classes=5)
        outputs_list = [
            {"pred_logits": torch.randn(1, 10, 5), "pred_boxes": torch.rand(1, 10, 4) * 200}
            for _ in range(3)
        ]
        targets = [{
            "labels": torch.tensor([0, 2]),
            "boxes_xyxy": torch.tensor([[10., 10., 50., 50.], [100., 100., 200., 200.]]),
            "image_size_xyxy": torch.tensor([256., 256., 256., 256.]),
            "image_size_xyxy_tgt": torch.tensor([[256., 256., 256., 256.], [256., 256., 256., 256.]]),
        }]
        losses = criterion(outputs_list, targets)
        expected = {"loss_ce_0", "loss_bbox_0", "loss_giou_0",
                    "loss_ce_1", "loss_bbox_1", "loss_giou_1",
                    "loss_ce_2", "loss_bbox_2", "loss_giou_2"}
        assert set(losses.keys()) == expected
        assert all(v.isfinite() for v in losses.values())

    def test_empty_targets(self) -> None:
        criterion = SetCriterion(num_classes=5)
        outputs = [{"pred_logits": torch.randn(1, 10, 5), "pred_boxes": torch.rand(1, 10, 4) * 200}]
        targets = [{
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes_xyxy": torch.zeros(0, 4),
            "image_size_xyxy": torch.tensor([256., 256., 256., 256.]),
            "image_size_xyxy_tgt": torch.zeros(0, 4),
        }]
        losses = criterion(outputs, targets)
        assert all(v.isfinite() for v in losses.values())


class TestGIoU:
    def test_identity(self) -> None:
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 200.0, 200.0]])
        giou = generalized_box_iou(boxes, boxes)
        assert torch.allclose(torch.diag(giou), torch.ones(2), atol=1e-5)

    def test_non_overlapping(self) -> None:
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
        giou = generalized_box_iou(boxes1, boxes2)
        assert giou.item() < 0


class TestDynamicConv:
    def test_shapes(self) -> None:
        dc = DynamicConv(hidden_dim=256, dim_dynamic=64, pooler_resolution=7)
        pro = torch.randn(1, 20, 256)  # (1, N*B, d)
        roi = torch.randn(49, 20, 256)  # (P*P, N*B, d)
        out = dc(pro, roi)
        assert out.shape == (20, 256)


class TestQueryRCNN:
    def test_forward_train(self) -> None:
        cfg = _make_cfg()
        model = build_query_rcnn(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=2)
        losses = model(batch)
        assert isinstance(losses, dict)
        assert len(losses) == 6  # 2 stages × 3 loss types
        assert all(v.isfinite() for v in losses.values())
        assert all(v.requires_grad for v in losses.values())

    def test_forward_inference(self) -> None:
        cfg = _make_cfg(num_proposals=20)
        model = build_query_rcnn(cfg, backbone_weights=None)
        model.eval()
        with torch.no_grad():
            results = model([{"image": torch.randn(3, 256, 256)}])
        assert len(results) == 1
        inst = results[0]["instances"]
        assert hasattr(inst, "pred_boxes")
        assert hasattr(inst, "scores")
        assert hasattr(inst, "pred_classes")

    def test_gradient_flow(self) -> None:
        cfg = _make_cfg(num_proposals=5, num_stages=3)
        model = build_query_rcnn(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=1, num_gt=1)
        losses = model(batch)
        total = sum(losses.values())
        total.backward()
        assert model.head.init_proposal_features.weight.grad is not None
        assert model.head.init_proposal_features.weight.grad.norm() > 0
        assert model.head.init_proposal_boxes.weight.grad is not None

    def test_loss_balance(self) -> None:
        """Verify weight_dict is applied and losses are balanced."""
        cfg = _make_cfg(num_proposals=50, num_stages=2, num_classes=10)
        model = build_query_rcnn(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=1, num_gt=5, num_classes=10)
        losses = model(batch)
        # All three loss types should have non-trivial magnitude
        ce = sum(v.item() for k, v in losses.items() if "loss_ce" in k)
        bbox = sum(v.item() for k, v in losses.items() if "loss_bbox" in k)
        giou = sum(v.item() for k, v in losses.items() if "loss_giou" in k)
        total = ce + bbox + giou
        assert ce > 0
        assert bbox > 0
        assert giou > 0
        # With weights 2/5/2, box losses should be substantial (not <1% of total)
        assert (bbox + giou) / total > 0.05

    def test_config_from_yaml_schema(self) -> None:
        cfg = MayakuConfig(
            model={
                "meta_architecture": "query_rcnn",
                "query_rcnn_head": {"num_proposals": 300, "num_stages": 6},
            }
        )
        assert cfg.model.query_rcnn_head is not None
        assert cfg.model.query_rcnn_head.num_proposals == 300

    def test_config_rejects_invalid_combination(self) -> None:
        with pytest.raises(Exception):
            MayakuConfig(
                model={
                    "meta_architecture": "query_rcnn",
                    "mask_on": True,
                    "query_rcnn_head": {},
                }
            )

    def test_cascade_iou_training(self) -> None:
        cfg = MayakuConfig(
            model={
                "meta_architecture": "query_rcnn",
                "backbone": {"name": "resnet50"},
                "query_rcnn_head": {
                    "num_proposals": 10,
                    "num_stages": 3,
                    "cascade_iou_thresholds": [0.0, 0.3, 0.5],
                },
                "roi_heads": {"num_classes": 5},
            }
        )
        model = build_query_rcnn(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=1, num_gt=3)
        losses = model(batch)
        assert len(losses) == 9  # 3 stages × 3 loss types
        assert all(v.isfinite() for v in losses.values())

    def test_cascade_iou_config_validation(self) -> None:
        with pytest.raises(Exception):
            MayakuConfig(
                model={
                    "meta_architecture": "query_rcnn",
                    "query_rcnn_head": {
                        "num_stages": 6,
                        "cascade_iou_thresholds": [0.0, 0.3],
                    },
                }
            )
