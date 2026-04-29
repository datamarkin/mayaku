"""Optimizer + LR scheduler builders.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §3 (`solver/build.py`,
`solver/lr_scheduler.py`) for the in-scope SGD + warmup-multistep /
warmup-cosine combination.

Key behaviours preserved from the upstream defaults:

* **Per-parameter weight-decay groups.** Norm parameters
  (`BatchNorm`/`LayerNorm`/`GroupNorm`/`FrozenBatchNorm`) get
  ``weight_decay_norm`` (0 by default), everything else gets
  ``weight_decay`` (1e-4 by default). This is the standard practice
  for ResNet-FPN training and is what the spec §6.1 defaults imply.
* **`WarmupMultiStepLR`** linearly warms up for ``warmup_iters``
  iterations from ``warmup_factor * base_lr`` to ``base_lr``, then
  multiplies by ``gamma`` at every step in ``steps``.
* **`WarmupCosineLR`** uses the same warmup, then a half-cosine decay
  from ``base_lr`` to ``0`` over ``[warmup_iters, max_iter)``.

Both schedulers are returned as ``LambdaLR`` instances so the same
``lr_scheduler.step()`` plumbing in :class:`LRScheduler` (the engine
hook) works for either.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import nn

from mayaku.config.schemas import SolverConfig
from mayaku.models.backbones._frozen_bn import FrozenBatchNorm2d

__all__ = ["build_lr_scheduler", "build_optimizer"]


def build_optimizer(model: nn.Module, cfg: SolverConfig) -> torch.optim.SGD:
    """SGD with two parameter groups: norm params vs everything else.

    ``cfg.weight_decay_norm`` is applied to the norm group (default 0),
    ``cfg.weight_decay`` to everything else.
    """
    norm_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        is_norm = isinstance(
            module,
            nn.modules.batchnorm._BatchNorm | nn.LayerNorm | nn.GroupNorm | FrozenBatchNorm2d,
        )
        for p in module.parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            (norm_params if is_norm else other_params).append(p)

    param_groups: list[dict[str, object]] = []
    if other_params:
        param_groups.append({"params": other_params, "weight_decay": cfg.weight_decay})
    if norm_params:
        param_groups.append({"params": norm_params, "weight_decay": cfg.weight_decay_norm})
    if not param_groups:
        # Defensive: torch.optim.SGD raises on an empty params list. The
        # caller almost certainly has a bug (frozen everything?) but
        # we'd rather surface it as a clearer message.
        raise ValueError("build_optimizer: no trainable parameters found on the model")
    return torch.optim.SGD(
        param_groups,
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: SolverConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Construct ``WarmupMultiStepLR`` or ``WarmupCosineLR`` as a LambdaLR."""
    if cfg.lr_scheduler_name == "WarmupMultiStepLR":
        lr_fn = _warmup_multistep_lambda(
            steps=cfg.steps,
            gamma=cfg.gamma,
            warmup_iters=cfg.warmup_iters,
            warmup_factor=cfg.warmup_factor,
            warmup_method=cfg.warmup_method,
        )
    elif cfg.lr_scheduler_name == "WarmupCosineLR":
        lr_fn = _warmup_cosine_lambda(
            max_iter=cfg.max_iter,
            warmup_iters=cfg.warmup_iters,
            warmup_factor=cfg.warmup_factor,
            warmup_method=cfg.warmup_method,
        )
    else:  # pragma: no cover — defended by the schema's Literal
        raise ValueError(f"unknown lr_scheduler_name: {cfg.lr_scheduler_name!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)


# ---------------------------------------------------------------------------
# Lambda factories
# ---------------------------------------------------------------------------


def _warmup_factor_at_iter(method: str, it: int, warmup_iters: int, warmup_factor: float) -> float:
    if it >= warmup_iters:
        return 1.0
    if method == "constant":
        return warmup_factor
    # Linear (default): grow from `warmup_factor` to 1 over warmup_iters.
    alpha = it / max(warmup_iters, 1)
    return warmup_factor * (1.0 - alpha) + alpha


def _warmup_multistep_lambda(
    steps: Iterable[int],
    gamma: float,
    warmup_iters: int,
    warmup_factor: float,
    warmup_method: str,
) -> Callable[[int], float]:
    sorted_steps = sorted(steps)

    def lr_lambda(it: int) -> float:
        warm = _warmup_factor_at_iter(warmup_method, it, warmup_iters, warmup_factor)
        decays = sum(1 for s in sorted_steps if it >= s)
        return warm * (gamma**decays)

    return lr_lambda


def _warmup_cosine_lambda(
    max_iter: int,
    warmup_iters: int,
    warmup_factor: float,
    warmup_method: str,
) -> Callable[[int], float]:
    def lr_lambda(it: int) -> float:
        warm = _warmup_factor_at_iter(warmup_method, it, warmup_iters, warmup_factor)
        if it < warmup_iters:
            return warm
        progress = (it - warmup_iters) / max(max_iter - warmup_iters, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda
