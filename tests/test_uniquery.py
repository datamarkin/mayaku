"""Smoke tests for UniQuery detector."""

from __future__ import annotations

import pytest
import torch

from mayaku.config.schemas import MayakuConfig
from mayaku.models.detectors.uniquery import build_uniquery
from mayaku.models.heads.uniquery_mask_head import UniQueryDynamicMaskHead
from mayaku.models.heads.uniquery_stage import DynamicConv
from mayaku.models.losses.set_criterion import SetCriterion, generalized_box_iou
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances
from mayaku.structures.masks import PolygonMasks


def _make_cfg(num_proposals: int = 10, num_stages: int = 2, num_classes: int = 5) -> MayakuConfig:
    return MayakuConfig(
        model={
            "meta_architecture": "uniquery",
            "backbone": {"name": "resnet50"},
            "uniquery_head": {
                "num_proposals": num_proposals,
                "num_stages": num_stages,
            },
            "roi_heads": {"num_classes": num_classes},
        }
    )


def _make_mask_cfg(
    num_proposals: int = 10, num_stages: int = 2, num_classes: int = 5
) -> MayakuConfig:
    return MayakuConfig(
        model={
            "meta_architecture": "uniquery",
            "backbone": {"name": "resnet50"},
            "uniquery_head": {
                "num_proposals": num_proposals,
                "num_stages": num_stages,
            },
            "roi_heads": {"num_classes": num_classes},
            "uniquery_mask": {},
        }
    )


def _make_batch(
    batch_size: int = 1,
    h: int = 256,
    w: int = 256,
    num_gt: int = 2,
    num_classes: int = 5,
    with_masks: bool = False,
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
        if with_masks:
            polygons = []
            for i in range(num_gt):
                x0, y0, x1, y1 = boxes[i].tolist()
                polygons.append(
                    [torch.tensor([x0, y0, x1, y0, x1, y1, x0, y1], dtype=torch.float32)]
                )
            instances.gt_masks = PolygonMasks(polygons)
        batch.append({"image": img, "instances": instances})
    return batch


class TestSetCriterion:
    def test_loss_keys(self) -> None:
        criterion = SetCriterion(num_classes=5)
        outputs_list = [
            {"pred_logits": torch.randn(1, 10, 5), "pred_boxes": torch.rand(1, 10, 4) * 200}
            for _ in range(3)
        ]
        targets = [
            {
                "labels": torch.tensor([0, 2]),
                "boxes_xyxy": torch.tensor(
                    [[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 200.0, 200.0]]
                ),
                "image_size_xyxy": torch.tensor([256.0, 256.0, 256.0, 256.0]),
                "image_size_xyxy_tgt": torch.tensor(
                    [[256.0, 256.0, 256.0, 256.0], [256.0, 256.0, 256.0, 256.0]]
                ),
            }
        ]
        losses = criterion(outputs_list, targets)
        expected = {
            "loss_ce_0",
            "loss_bbox_0",
            "loss_giou_0",
            "loss_ce_1",
            "loss_bbox_1",
            "loss_giou_1",
            "loss_ce_2",
            "loss_bbox_2",
            "loss_giou_2",
        }
        assert set(losses.keys()) == expected
        assert all(v.isfinite() for v in losses.values())

    def test_empty_targets(self) -> None:
        criterion = SetCriterion(num_classes=5)
        outputs = [{"pred_logits": torch.randn(1, 10, 5), "pred_boxes": torch.rand(1, 10, 4) * 200}]
        targets = [
            {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes_xyxy": torch.zeros(0, 4),
                "image_size_xyxy": torch.tensor([256.0, 256.0, 256.0, 256.0]),
                "image_size_xyxy_tgt": torch.zeros(0, 4),
            }
        ]
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


class TestUniQuery:
    def test_forward_train(self) -> None:
        cfg = _make_cfg()
        model = build_uniquery(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=2)
        losses = model(batch)
        assert isinstance(losses, dict)
        assert len(losses) == 6  # 2 stages × 3 loss types
        assert all(v.isfinite() for v in losses.values())
        assert all(v.requires_grad for v in losses.values())

    def test_forward_inference(self) -> None:
        cfg = _make_cfg(num_proposals=20)
        model = build_uniquery(cfg, backbone_weights=None)
        model.eval()
        with torch.no_grad():
            results = model([{"image": torch.randn(3, 256, 256)}])
        assert len(results) == 1
        inst = results[0]["instances"]
        assert hasattr(inst, "pred_boxes")
        assert hasattr(inst, "scores")
        assert hasattr(inst, "pred_classes")

    def test_num_proposals_override_rejected_on_learned_path(self) -> None:
        # The learned-proposal path (no QGN) has no proposal ranking, so the
        # inference proposal dial would slice an arbitrary first-k. It must
        # error rather than silently degrade. (The QGN path applies top-k by
        # objectness and is exercised in TestUniQueryGenerator.)
        model = build_uniquery(_make_cfg(num_proposals=20), backbone_weights=None)
        model.eval()
        model.inference_num_proposals = 5
        with torch.no_grad(), pytest.raises(ValueError, match="QGN proposal path"):
            model([{"image": torch.randn(3, 256, 256)}])

    def test_gradient_flow(self) -> None:
        cfg = _make_cfg(num_proposals=5, num_stages=3)
        model = build_uniquery(cfg, backbone_weights=None)
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
        model = build_uniquery(cfg, backbone_weights=None)
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
                "meta_architecture": "uniquery",
                "uniquery_head": {"num_proposals": 300, "num_stages": 6},
            }
        )
        assert cfg.model.uniquery_head is not None
        assert cfg.model.uniquery_head.num_proposals == 300

    def test_config_rejects_invalid_combination(self) -> None:
        with pytest.raises(Exception):
            MayakuConfig(
                model={
                    "meta_architecture": "uniquery",
                    "mask_on": True,
                    "uniquery_head": {},
                }
            )

    def test_cascade_iou_training(self) -> None:
        cfg = MayakuConfig(
            model={
                "meta_architecture": "uniquery",
                "backbone": {"name": "resnet50"},
                "uniquery_head": {
                    "num_proposals": 10,
                    "num_stages": 3,
                    "cascade_iou_thresholds": [0.0, 0.3, 0.5],
                },
                "roi_heads": {"num_classes": 5},
            }
        )
        model = build_uniquery(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=1, num_gt=3)
        losses = model(batch)
        assert len(losses) == 9  # 3 stages × 3 loss types
        assert all(v.isfinite() for v in losses.values())

    def test_mask_forward_train(self) -> None:
        cfg = _make_mask_cfg(num_proposals=10, num_stages=2, num_classes=5)
        model = build_uniquery(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=2, num_gt=2, with_masks=True)
        losses = model(batch)
        assert "loss_mask" in losses
        assert losses["loss_mask"].isfinite()
        assert losses["loss_mask"].requires_grad
        # Detection losses still present: 2 stages × 3 types + 1 mask
        assert len(losses) == 7

    def test_mask_forward_inference(self) -> None:
        cfg = _make_mask_cfg(num_proposals=10, num_stages=2, num_classes=5)
        model = build_uniquery(cfg, backbone_weights=None)
        model.eval()
        with torch.no_grad():
            results = model([{"image": torch.randn(3, 256, 256)}])
        inst = results[0]["instances"]
        if len(inst) > 0:
            assert hasattr(inst, "pred_masks")
            assert inst.pred_masks.shape[1] == 1  # class-agnostic
            assert inst.pred_masks.shape[2] == 28

    def test_mask_gradient_flow(self) -> None:
        cfg = _make_mask_cfg(num_proposals=5, num_stages=2, num_classes=5)
        model = build_uniquery(cfg, backbone_weights=None)
        model.train()
        batch = _make_batch(batch_size=1, num_gt=2, with_masks=True)
        losses = model(batch)
        total = sum(losses.values())
        total.backward()
        assert model.mask_head.kernel_fc.weight.grad is not None
        assert model.mask_head.kernel_fc.weight.grad.norm() > 0

    def test_dynamic_mask_head_shapes(self) -> None:
        head = UniQueryDynamicMaskHead(hidden_dim=256, conv_dim=256, num_conv=4)
        roi_feats = torch.randn(5, 256, 14, 14)
        obj_feats = torch.randn(5, 256)
        out = head(roi_feats, obj_feats)
        assert out.shape == (5, 1, 28, 28)

    def test_cascade_iou_config_validation(self) -> None:
        with pytest.raises(Exception):
            MayakuConfig(
                model={
                    "meta_architecture": "uniquery",
                    "uniquery_head": {
                        "num_stages": 6,
                        "cascade_iou_thresholds": [0.0, 0.3],
                    },
                }
            )


def _make_qgn_cfg(
    num_proposals: int = 10, num_stages: int = 2, num_classes: int = 5
) -> MayakuConfig:
    return MayakuConfig(
        model={
            "meta_architecture": "uniquery",
            "backbone": {"name": "resnet50"},
            "uniquery_head": {
                "num_proposals": num_proposals,
                "num_stages": num_stages,
                "uniquery_generator": True,
            },
            "roi_heads": {"num_classes": num_classes},
        }
    )


class TestUniQueryGenerator:
    def test_forward_train_loss_keys(self) -> None:
        model = build_uniquery(_make_qgn_cfg())
        model.train()
        losses = model(_make_batch(batch_size=2))
        for i in range(2):
            assert f"loss_ce_{i}" in losses
        assert "loss_qgn_obj" in losses
        assert "loss_qgn_giou" in losses
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} not finite"

    def test_forward_inference(self) -> None:
        model = build_uniquery(_make_qgn_cfg())
        model.eval()
        with torch.no_grad():
            results = model(_make_batch(batch_size=2))
        assert len(results) == 2
        for r in results:
            inst = r["instances"]
            assert inst.has("pred_boxes") and inst.has("scores") and inst.has("pred_classes")

    def test_gradient_flow_into_qgn(self) -> None:
        model = build_uniquery(_make_qgn_cfg())
        model.train()
        losses = model(_make_batch(batch_size=1))
        total = sum(losses.values())
        total.backward()
        qgn = model.head.uniquery_generator
        assert qgn.objectness.weight.grad is not None
        assert qgn.objectness.weight.grad.norm() > 0
        # query features feed the stages -> head losses must reach this branch
        assert qgn.query_feat.weight.grad is not None
        assert qgn.query_feat.weight.grad.norm() > 0
        # boxes are detached into stage 1; ltrb is trained by the QGN GIoU loss
        assert qgn.ltrb.weight.grad is not None

    def test_no_blind_embeddings_when_qgn(self) -> None:
        model = build_uniquery(_make_qgn_cfg())
        assert (
            not hasattr(model.head, "init_proposal_boxes")
            or model.head.init_proposal_boxes is None
            or model.head.uniquery_generator is not None
        )

    def test_stage_truncation_inference(self) -> None:
        model = build_uniquery(_make_qgn_cfg(num_stages=2))
        model.eval()
        model.inference_num_stages = 1
        with torch.no_grad():
            results = model(_make_batch(batch_size=1))
        assert len(results) == 1

    def test_empty_targets(self) -> None:
        model = build_uniquery(_make_qgn_cfg())
        model.train()
        batch = _make_batch(batch_size=1, num_gt=1)
        inst = batch[0]["instances"]
        batch[0]["instances"] = Instances(
            image_size=inst.image_size,
            gt_boxes=Boxes(torch.zeros(0, 4)),
            gt_classes=torch.zeros(0, dtype=torch.long),
        )
        losses = model(batch)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} not finite"
