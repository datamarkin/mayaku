"""End-to-end tests for the Mask R-CNN detector + toy training step."""

from __future__ import annotations

import torch

from mayaku.config.schemas import (
    BackboneConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    ROIMaskHeadConfig,
    RPNConfig,
)
from mayaku.models.detectors import build_mask_rcnn
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances
from mayaku.structures.masks import PolygonMasks


def _tiny_mask_config() -> MayakuConfig:
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="mask_rcnn",
            mask_on=True,
            backbone=BackboneConfig(name="resnet50", freeze_at=2, norm="FrozenBN"),
            rpn=RPNConfig(
                pre_nms_topk_train=200,
                pre_nms_topk_test=100,
                post_nms_topk_train=50,
                post_nms_topk_test=20,
                batch_size_per_image=32,
            ),
            roi_heads=ROIHeadsConfig(num_classes=2, batch_size_per_image=16),
            roi_box_head=ROIBoxHeadConfig(num_fc=1, fc_dim=64),
            roi_mask_head=ROIMaskHeadConfig(num_conv=2, conv_dim=32),
        ),
    )


def _toy_batch(device: torch.device, h: int = 96, w: int = 96):  # type: ignore[no-untyped-def]
    image = torch.rand(3, h, w, device=device) * 255.0
    inst = Instances(image_size=(h, w))
    inst.gt_boxes = Boxes(torch.tensor([[10.0, 10.0, 60.0, 60.0]], device=device))
    inst.gt_classes = torch.tensor([0], dtype=torch.long, device=device)
    inst.gt_masks = PolygonMasks([[[10, 10, 60, 10, 60, 60, 10, 60]]])
    return [{"image": image, "instances": inst, "height": h, "width": w}]


def test_build_mask_rcnn_constructs_module(device: torch.device) -> None:
    model = build_mask_rcnn(_tiny_mask_config()).to(device)
    # Mask path enabled on the ROI heads.
    assert model.roi_heads.mask_on


def test_mask_rcnn_training_returns_mask_loss(device: torch.device) -> None:
    torch.manual_seed(0)
    model = build_mask_rcnn(_tiny_mask_config()).to(device).train()
    losses = model(_toy_batch(device))
    assert isinstance(losses, dict)
    assert {"loss_rpn_cls", "loss_rpn_loc", "loss_cls", "loss_box_reg", "loss_mask"} <= set(losses)
    for v in losses.values():
        assert torch.isfinite(v).item()


def test_mask_rcnn_inference_attaches_pred_masks(device: torch.device) -> None:
    torch.manual_seed(0)
    model = build_mask_rcnn(_tiny_mask_config()).to(device).eval()
    with torch.no_grad():
        out = model(_toy_batch(device))
    assert isinstance(out, list) and len(out) == 1
    inst = out[0]["instances"]
    assert inst.has("pred_masks")
    # When there are detections, pred_masks should be (R, 1, M, M).
    if len(inst) > 0:
        assert inst.pred_masks.shape[1] == 1
        assert inst.pred_masks.ndim == 4


def test_mask_loss_decreases_over_a_handful_of_sgd_steps(
    device: torch.device,
) -> None:
    """Spec gate (Step 11): toy-dataset training run shows mask loss
    decreases on every backend. Same recipe as the Faster R-CNN gate
    (low LR, momentum off, grad-norm clip) so RPN init noise can't
    poison the run."""
    torch.manual_seed(0)
    model = build_mask_rcnn(_tiny_mask_config()).to(device).train()
    batch = _toy_batch(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, momentum=0.0
    )

    def avg_mask_loss(n: int = 5) -> float:
        with torch.no_grad():
            vals: list[float] = []
            for _ in range(n):
                losses = model(batch)
                assert isinstance(losses, dict)
                vals.append(float(losses["loss_mask"].item()))
        return sum(vals) / len(vals)

    initial = avg_mask_loss()
    for _ in range(20):
        optimizer.zero_grad()
        losses = model(batch)
        assert isinstance(losses, dict)
        total = sum(losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=10.0
        )
        optimizer.step()
    final = avg_mask_loss()
    assert torch.isfinite(torch.tensor(final)).item(), f"mask loss diverged to {final}"
    assert final < initial, f"mask loss did not decrease: {initial:.4f} → {final:.4f}"
