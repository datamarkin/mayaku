"""End-to-end tests for the Faster R-CNN detector + toy training step."""

from __future__ import annotations

import torch

from mayaku.config.schemas import (
    BackboneConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    RPNConfig,
)
from mayaku.models.detectors import FasterRCNN, build_faster_rcnn
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances


def _tiny_config() -> MayakuConfig:
    """A tiny but legal Faster R-CNN config: ResNet-50, 2 classes,
    small RPN sampling so a few-image training step actually runs."""
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
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
        ),
    )


def _toy_batch(device: torch.device, h: int = 96, w: int = 96):  # type: ignore[no-untyped-def]
    """One image + one GT box; deterministic for reproducible loss decrease."""
    image = torch.rand(3, h, w, device=device) * 255.0
    inst = Instances(image_size=(h, w))
    inst.gt_boxes = Boxes(torch.tensor([[10.0, 10.0, 60.0, 60.0]], device=device))
    inst.gt_classes = torch.tensor([0], dtype=torch.long, device=device)
    return [{"image": image, "instances": inst, "height": h, "width": w}]


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------


def test_build_faster_rcnn_constructs_module(device: torch.device) -> None:
    model = build_faster_rcnn(_tiny_config()).to(device)
    assert isinstance(model, FasterRCNN)
    # Pixel mean/std loaded as buffers, not parameters.
    assert "pixel_mean" in dict(model.named_buffers())
    assert "pixel_std" in dict(model.named_buffers())


def test_faster_rcnn_inference_returns_per_image_dicts(device: torch.device) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_config()).to(device).eval()
    batch = _toy_batch(device)
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, list)
    assert len(out) == 1
    assert "instances" in out[0]
    inst = out[0]["instances"]
    assert inst.has("pred_boxes") and inst.has("scores") and inst.has("pred_classes")


def test_faster_rcnn_training_returns_loss_dict(device: torch.device) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_config()).to(device).train()
    losses = model(_toy_batch(device))
    assert isinstance(losses, dict)
    assert set(losses) >= {"loss_rpn_cls", "loss_rpn_loc", "loss_cls", "loss_box_reg"}
    for v in losses.values():
        assert torch.isfinite(v).item()


# ---------------------------------------------------------------------------
# The "toy training step" gate from the implementation prompt
# ---------------------------------------------------------------------------


def test_loss_decreases_over_a_handful_of_sgd_steps(device: torch.device) -> None:
    """Spec gate (Step 10): 'Toy-dataset training run; loss decreases on
    CPU and on accelerator.'  We use a single fixed batch and a handful
    of plain-SGD steps — enough to overfit it. We average the loss over
    a few measurement passes because the RPN does stochastic
    label-and-sample, so a single forward isn't deterministic.

    LR is intentionally small (``1e-4``) and momentum is off: the
    network is randomly initialised, so the first few RPN proposals
    decode anchors against arbitrary deltas. Larger LRs blow the
    smooth-L1 box loss past fp32's representable range within one
    update."""
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_config()).to(device).train()
    batch = _toy_batch(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, momentum=0.0
    )

    def average_loss(n: int = 5) -> float:
        with torch.no_grad():
            vals: list[float] = []
            for _ in range(n):
                losses = model(batch)
                assert isinstance(losses, dict)
                vals.append(float(sum(losses.values()).item()))
        return sum(vals) / len(vals)

    initial = average_loss()
    for _ in range(20):
        optimizer.zero_grad()
        losses = model(batch)
        assert isinstance(losses, dict)
        total = sum(losses.values())
        total.backward()
        # Defensive clipping so a single bad iteration can't poison the run.
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=10.0
        )
        optimizer.step()
    final = average_loss()
    assert torch.isfinite(torch.tensor(final)).item(), f"loss diverged to {final}"
    assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"
