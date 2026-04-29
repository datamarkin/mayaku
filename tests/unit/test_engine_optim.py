"""Tests for :mod:`mayaku.engine.optim`."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from mayaku.config.schemas import SolverConfig
from mayaku.engine.optim import build_lr_scheduler, build_optimizer
from mayaku.models.backbones._frozen_bn import FrozenBatchNorm2d


def _toy_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 4, 1),
        FrozenBatchNorm2d(4),
        nn.Linear(4, 2),
    )


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


def _solver_cfg(**overrides: object) -> SolverConfig:
    """Tiny SolverConfig that satisfies the schema's cross-field validators."""
    base: dict[str, object] = {
        "max_iter": 1000,
        "warmup_iters": 10,
        "steps": (500, 800),
    }
    base.update(overrides)
    return SolverConfig(**base)  # type: ignore[arg-type]


def test_optimizer_splits_norm_and_other_groups() -> None:
    model = _toy_model()
    opt = build_optimizer(model, _solver_cfg(weight_decay=1e-4, weight_decay_norm=0.0))
    # Two parameter groups: non-norm (wd=1e-4) and norm (wd=0).
    assert len(opt.param_groups) == 2
    weights = sorted(g["weight_decay"] for g in opt.param_groups)
    assert weights == [0.0, 1e-4]


def test_optimizer_skips_frozen_parameters() -> None:
    model = _toy_model()
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(model, _solver_cfg())


def test_optimizer_uses_base_lr_and_momentum() -> None:
    model = _toy_model()
    opt = build_optimizer(model, _solver_cfg(base_lr=0.05, momentum=0.5))
    for g in opt.param_groups:
        assert g["lr"] == 0.05
        assert g["momentum"] == 0.5


# ---------------------------------------------------------------------------
# build_lr_scheduler — multistep
# ---------------------------------------------------------------------------


def _make_opt_and_cfg(**overrides: object) -> tuple[torch.optim.SGD, SolverConfig]:
    model = nn.Linear(2, 2)
    cfg_kwargs: dict[str, object] = {
        "max_iter": 1000,
        "warmup_iters": 100,
        "warmup_factor": 0.001,
        "base_lr": 1.0,
        "steps": (500, 800),
        "gamma": 0.1,
    }
    cfg_kwargs.update(overrides)
    cfg = SolverConfig(**cfg_kwargs)  # type: ignore[arg-type]
    opt = torch.optim.SGD(model.parameters(), lr=cfg.base_lr)
    return opt, cfg


def test_warmup_multistep_warmup_grows_linearly_from_factor_to_one() -> None:
    opt, cfg = _make_opt_and_cfg()
    sched = build_lr_scheduler(opt, cfg)
    # Iter 0 → factor; iter warmup_iters → 1.0.
    assert math.isclose(opt.param_groups[0]["lr"], 0.001, abs_tol=1e-6)
    for _ in range(50):
        sched.step()
    # Half-way through warmup → ~0.5005 (lerp from 0.001 to 1.0)
    expected_mid = 0.001 * (1 - 0.5) + 0.5
    assert math.isclose(opt.param_groups[0]["lr"], expected_mid, abs_tol=1e-3)
    for _ in range(50):
        sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, abs_tol=1e-6)


def test_warmup_multistep_decays_at_each_step() -> None:
    opt, cfg = _make_opt_and_cfg()
    sched = build_lr_scheduler(opt, cfg)
    for _ in range(500):
        sched.step()
    # After the first decay step (500): lr = 1.0 * 0.1
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, abs_tol=1e-6)
    for _ in range(300):
        sched.step()
    # After both decay steps (800): lr = 1.0 * 0.1 * 0.1
    assert math.isclose(opt.param_groups[0]["lr"], 0.01, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# build_lr_scheduler — cosine
# ---------------------------------------------------------------------------


def test_warmup_cosine_decays_from_one_to_zero() -> None:
    opt, cfg = _make_opt_and_cfg(
        lr_scheduler_name="WarmupCosineLR",
        warmup_iters=10,
        max_iter=110,
        # Cosine doesn't need step milestones, but SolverConfig still
        # validates that every entry is < max_iter, so override here.
        steps=(50,),
    )
    sched = build_lr_scheduler(opt, cfg)
    for _ in range(10):
        sched.step()
    # At end of warmup, multiplier = 1.0.
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, abs_tol=1e-6)
    # At max_iter, cosine = 0.
    for _ in range(100):
        sched.step()
    assert opt.param_groups[0]["lr"] < 1e-5


def test_warmup_constant_method_holds_factor_through_warmup() -> None:
    opt, cfg = _make_opt_and_cfg(warmup_method="constant", warmup_iters=10)
    sched = build_lr_scheduler(opt, cfg)
    for _ in range(5):
        sched.step()
    # With constant method, lr stays at base_lr * warmup_factor through warmup.
    assert math.isclose(opt.param_groups[0]["lr"], 0.001, abs_tol=1e-6)
