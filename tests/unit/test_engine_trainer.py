"""Tests for :mod:`mayaku.engine.trainer`.

Includes an integration test that wires :class:`SimpleTrainer` with
:class:`LRScheduler` + :class:`IterationTimer` against a tiny
:class:`FasterRCNN` and a one-image data loader to confirm the loop
runs end-to-end and the loss decreases.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mayaku.backends.device import Device
from mayaku.config.schemas import (
    BackboneConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    RPNConfig,
    SolverConfig,
)
from mayaku.engine import (
    AMPTrainer,
    IterationTimer,
    LRScheduler,
    PeriodicCheckpointer,
    SimpleTrainer,
    TrainerBase,
    build_lr_scheduler,
    build_optimizer,
)
from mayaku.models.detectors import build_faster_rcnn
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances

# ---------------------------------------------------------------------------
# Lifecycle / hooks
# ---------------------------------------------------------------------------


class _RecordingHook:
    trainer: TrainerBase | None = None
    events: list[str]

    def __init__(self) -> None:
        self.events = []

    def before_train(self) -> None:
        self.events.append("before_train")

    def after_train(self) -> None:
        self.events.append("after_train")

    def before_step(self) -> None:
        self.events.append("before_step")

    def after_step(self) -> None:
        self.events.append("after_step")


class _NoopTrainer(TrainerBase):
    def run_step(self) -> None:
        return None


def test_trainer_base_calls_hooks_in_lifecycle_order() -> None:
    trainer = _NoopTrainer()
    hook = _RecordingHook()
    trainer.register_hooks([hook])  # type: ignore[list-item]
    trainer.train(start_iter=0, max_iter=2)
    assert hook.events == [
        "before_train",
        "before_step",
        "after_step",
        "before_step",
        "after_step",
        "after_train",
    ]


def test_trainer_iter_advances_after_loop() -> None:
    trainer = _NoopTrainer()
    trainer.train(start_iter=5, max_iter=8)
    # Detectron2 convention: post-loop iter = max_iter (one past the last).
    assert trainer.iter == 8


def test_trainer_validates_max_iter_geq_start() -> None:
    trainer = _NoopTrainer()
    try:
        trainer.train(start_iter=10, max_iter=5)
    except ValueError as exc:
        assert "max_iter" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Integration: SimpleTrainer over a real Faster R-CNN
# ---------------------------------------------------------------------------


def _tiny_cfg() -> MayakuConfig:
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
        solver=SolverConfig(
            base_lr=1e-4,
            momentum=0.0,
            max_iter=10,
            warmup_iters=2,
            warmup_factor=0.5,
            steps=(8,),
            checkpoint_period=5,
        ),
    )


def _toy_loader(device: torch.device) -> list[list[dict[str, Any]]]:
    image = torch.rand(3, 96, 96, device=device) * 255.0
    inst = Instances(image_size=(96, 96))
    inst.gt_boxes = Boxes(torch.tensor([[10.0, 10.0, 60.0, 60.0]], device=device))
    inst.gt_classes = torch.tensor([0], dtype=torch.long, device=device)
    batch = [{"image": image, "instances": inst, "height": 96, "width": 96}]
    return [batch]


def test_simple_trainer_runs_and_drops_loss(device: torch.device) -> None:
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device)
    optimizer = build_optimizer(model, cfg.solver)
    scheduler = build_lr_scheduler(optimizer, cfg.solver)
    loader = _toy_loader(device)

    trainer = SimpleTrainer(model, loader, optimizer, grad_clip_norm=10.0)
    trainer.register_hooks([IterationTimer(), LRScheduler(scheduler)])  # type: ignore[list-item]

    # Before training: capture the initial averaged loss.
    initial = _avg_loss(model, loader[0])
    trainer.train(start_iter=0, max_iter=cfg.solver.max_iter)
    final = _avg_loss(model, loader[0])

    assert torch.isfinite(torch.tensor(final)).item()
    assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"
    # Hook bookkeeping is wired correctly.
    timer = trainer.hooks[0]
    assert isinstance(timer, IterationTimer)
    assert timer.total_seconds > 0


def test_simple_trainer_checkpoint_hook_writes_files(device: torch.device, tmp_path) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device)
    optimizer = build_optimizer(model, cfg.solver)
    loader = _toy_loader(device)

    trainer = SimpleTrainer(model, loader, optimizer, grad_clip_norm=10.0)
    ckpt = PeriodicCheckpointer(model, output_dir=tmp_path, period=3, optimizer=optimizer)
    trainer.register_hooks([ckpt])  # type: ignore[list-item]
    trainer.train(start_iter=0, max_iter=6)

    # period=3 → saves at iters {3, 6} plus model_final.
    files = sorted(p.name for p in tmp_path.glob("*.pth"))
    assert "model_iter_0000003.pth" in files
    assert "model_iter_0000006.pth" in files
    assert "model_final.pth" in files


# ---------------------------------------------------------------------------
# AMPTrainer construction
# ---------------------------------------------------------------------------


def test_amp_trainer_rejects_unsupported_dtype() -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        AMPTrainer(model, [], opt, Device(kind="cuda"), amp_dtype="bogus")
    except ValueError as exc:
        assert "amp_dtype" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_amp_trainer_rejects_cpu_device() -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        AMPTrainer(model, [], opt, Device(kind="cpu"))
    except ValueError as exc:
        assert "AMP" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Gradient clipping (value vs norm)
# ---------------------------------------------------------------------------


class _HugeLossModel(nn.Module):
    """One vector of params; loss is `sum(weight) * 1000` so the gradient
    through every component is a uniform 1000 — easy to assert on.
    """

    def __init__(self, n: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n))

    def forward(self, _batch: Any) -> dict[str, Tensor]:
        return {"loss": (self.weight * 1000.0).sum()}


def test_simple_trainer_grad_clip_value_clamps_each_component() -> None:
    """`grad_clip_type='value'` calls `clip_grad_value_` → each component
    bounded by ±clip_value, no rescale."""
    torch.manual_seed(0)
    model = _HugeLossModel()
    # lr=0 keeps the weights at zero so we can read the post-clip grads
    # without any optimizer movement confusing the picture.
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = SimpleTrainer(
        model,
        [[{"dummy": True}]],
        opt,
        grad_clip_norm=0.5,
        grad_clip_type="value",
    )
    trainer.train(start_iter=0, max_iter=1)
    assert model.weight.grad is not None
    # Pre-clip grads were 1000 each; value-clip drops them to ±0.5.
    assert torch.allclose(model.weight.grad, torch.full_like(model.weight, 0.5))


def test_simple_trainer_grad_clip_norm_rescales_total_norm() -> None:
    """`grad_clip_type='norm'` (the default) calls `clip_grad_norm_` →
    total L2 norm rescaled to clip_value, components stay equal."""
    torch.manual_seed(0)
    model = _HugeLossModel(n=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = SimpleTrainer(
        model,
        [[{"dummy": True}]],
        opt,
        grad_clip_norm=1.0,
        grad_clip_type="norm",
    )
    trainer.train(start_iter=0, max_iter=1)
    assert model.weight.grad is not None
    # Pre-clip total norm was sqrt(8) * 1000; post-clip it should be exactly 1.0.
    assert abs(float(model.weight.grad.norm().item()) - 1.0) < 1e-5
    # Components are uniform so each one is 1 / sqrt(8).
    expected = torch.full_like(model.weight, 1.0 / (8**0.5))
    assert torch.allclose(model.weight.grad, expected, atol=1e-6)


class _TwoParamModel(nn.Module):
    """Two independent parameter vectors with controllable per-vector
    gradient norms — exercises the difference between *global* clip
    (concat both, scale jointly) and *per-parameter* clip (each tensor
    handled independently)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(8))
        self.b = nn.Parameter(torch.zeros(8))

    def forward(self, _batch: Any) -> dict[str, Tensor]:
        # Each component gets gradient = coefficient. Different per param.
        # `a`'s grad will be 1.06 / sqrt(8) per component → tensor norm ≈ 1.06.
        # `b`'s grad will be 0.5  / sqrt(8) per component → tensor norm ≈ 0.5.
        a_coef = 1.06 / (8**0.5)
        b_coef = 0.5 / (8**0.5)
        return {"loss": (self.a * a_coef).sum() + (self.b * b_coef).sum()}


def test_simple_trainer_grad_clip_norm_is_per_parameter() -> None:
    """`grad_clip_type='norm'` must clip each parameter tensor's gradient
    norm independently (matches detectron2/solver/build.py:36-37). A
    *global* clip would touch every parameter once the concatenated
    norm exceeds the threshold; a per-param clip leaves below-threshold
    tensors alone.

    Setup: two parameter vectors with gradient norms 1.06 and 0.5.
    Concatenated norm is sqrt(1.06^2 + 0.5^2) ≈ 1.17 > 1.0.

    Per-param semantics (what we want):
      - `a` (norm 1.06 > 1.0): scaled to norm 1.0
      - `b` (norm 0.5  < 1.0): untouched

    Global semantics (the bug we're locking out):
      - Both scaled by 1.0 / 1.17 ≈ 0.853 → `a` norm ~0.905, `b` norm ~0.427.
      - That's wrong — `b` would have been touched even though its own
        norm was already below 1.0.
    """
    torch.manual_seed(0)
    model = _TwoParamModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = SimpleTrainer(
        model,
        [[{"dummy": True}]],
        opt,
        grad_clip_norm=1.0,
        grad_clip_type="norm",
    )
    trainer.train(start_iter=0, max_iter=1)
    assert model.a.grad is not None and model.b.grad is not None
    a_norm = float(model.a.grad.norm().item())
    b_norm = float(model.b.grad.norm().item())
    # `a` clipped to exactly 1.0 (per-param).
    assert abs(a_norm - 1.0) < 1e-5, f"expected a-norm 1.0, got {a_norm}"
    # `b` untouched at 0.5 (per-param). Global clip would have driven this
    # to ~0.427.
    assert abs(b_norm - 0.5) < 1e-5, f"expected b-norm 0.5, got {b_norm}"


def test_simple_trainer_rejects_invalid_grad_clip_type() -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        SimpleTrainer(model, [], opt, grad_clip_type="bogus")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "grad_clip_type" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _avg_loss(model: nn.Module, batch: Sequence[dict[str, Any]], n: int = 5) -> float:
    model.train()
    with torch.no_grad():
        vals: list[float] = []
        for _ in range(n):
            losses = model(batch)
            assert isinstance(losses, dict)
            total: Tensor = sum(losses.values())  # type: ignore[assignment]
            vals.append(float(total.item()))
    return sum(vals) / len(vals)


# Re-export hint so unused-import lint stays happy if these helpers
# move around in future refactors.
_ = time
