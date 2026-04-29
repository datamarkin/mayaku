"""End-to-end tests for the Keypoint R-CNN detector + toy training step."""

from __future__ import annotations

import torch

from mayaku.config.schemas import (
    BackboneConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    ROIKeypointHeadConfig,
    RPNConfig,
)
from mayaku.models.detectors import build_keypoint_rcnn
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances
from mayaku.structures.keypoints import Keypoints


def _tiny_keypoint_config() -> MayakuConfig:
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="keypoint_rcnn",
            keypoint_on=True,
            backbone=BackboneConfig(name="resnet50", freeze_at=2, norm="FrozenBN"),
            rpn=RPNConfig(
                pre_nms_topk_train=200,
                pre_nms_topk_test=100,
                post_nms_topk_train=50,
                post_nms_topk_test=20,
                batch_size_per_image=32,
            ),
            roi_heads=ROIHeadsConfig(num_classes=1, batch_size_per_image=16),
            roi_box_head=ROIBoxHeadConfig(num_fc=1, fc_dim=64, smooth_l1_beta=0.5),
            roi_keypoint_head=ROIKeypointHeadConfig(
                num_keypoints=4,
                conv_dims=(32, 32),
                flip_indices=(0, 2, 1, 3),
            ),
        ),
    )


def _toy_keypoint_batch(device: torch.device, h: int = 96, w: int = 96):  # type: ignore[no-untyped-def]
    image = torch.rand(3, h, w, device=device) * 255.0
    inst = Instances(image_size=(h, w))
    inst.gt_boxes = Boxes(torch.tensor([[10.0, 10.0, 60.0, 60.0]], device=device))
    inst.gt_classes = torch.tensor([0], dtype=torch.long, device=device)
    # 4 visible keypoints inside the GT box.
    kp = torch.tensor(
        [[[20.0, 20.0, 2.0], [40.0, 20.0, 2.0], [20.0, 40.0, 2.0], [40.0, 40.0, 2.0]]],
        device=device,
    )
    inst.gt_keypoints = Keypoints(kp, flip_indices=torch.tensor([0, 2, 1, 3]))
    return [{"image": image, "instances": inst, "height": h, "width": w}]


def test_build_keypoint_rcnn_constructs_module(device: torch.device) -> None:
    model = build_keypoint_rcnn(_tiny_keypoint_config()).to(device)
    assert model.roi_heads.keypoint_on
    # Make sure both paths are present (box + keypoint), no mask path.
    assert not model.roi_heads.mask_on


def test_keypoint_rcnn_training_returns_keypoint_loss(device: torch.device) -> None:
    torch.manual_seed(0)
    model = build_keypoint_rcnn(_tiny_keypoint_config()).to(device).train()
    losses = model(_toy_keypoint_batch(device))
    assert isinstance(losses, dict)
    assert {
        "loss_rpn_cls",
        "loss_rpn_loc",
        "loss_cls",
        "loss_box_reg",
        "loss_keypoint",
    } <= set(losses)
    for v in losses.values():
        assert torch.isfinite(v).item()


def test_keypoint_rcnn_inference_attaches_pred_keypoints(device: torch.device) -> None:
    torch.manual_seed(0)
    model = build_keypoint_rcnn(_tiny_keypoint_config()).to(device).eval()
    with torch.no_grad():
        out = model(_toy_keypoint_batch(device))
    assert isinstance(out, list) and len(out) == 1
    inst = out[0]["instances"]
    assert inst.has("pred_keypoints")
    assert inst.has("pred_keypoint_heatmaps")
    if len(inst) > 0:
        assert inst.pred_keypoints.shape[1:] == (4, 3)
        assert inst.pred_keypoint_heatmaps.shape[1:] == (4, 56, 56)


def test_keypoint_loss_decreases_over_a_handful_of_sgd_steps(
    device: torch.device,
) -> None:
    """Spec gate (Step 12): toy-dataset training run shows the keypoint
    loss decreases on every backend. Same recipe as Steps 10/11
    (low LR, no momentum, grad clip)."""
    torch.manual_seed(0)
    model = build_keypoint_rcnn(_tiny_keypoint_config()).to(device).train()
    batch = _toy_keypoint_batch(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, momentum=0.0
    )

    def avg_keypoint_loss(n: int = 5) -> float:
        with torch.no_grad():
            vals: list[float] = []
            for _ in range(n):
                losses = model(batch)
                assert isinstance(losses, dict)
                vals.append(float(losses["loss_keypoint"].item()))
        return sum(vals) / len(vals)

    initial = avg_keypoint_loss()
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
    final = avg_keypoint_loss()
    assert torch.isfinite(torch.tensor(final)).item(), f"keypoint loss diverged to {final}"
    assert final < initial, f"keypoint loss did not decrease: {initial:.4f} → {final:.4f}"
