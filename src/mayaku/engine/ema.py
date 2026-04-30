"""Exponential Moving Average of model weights.

Standard practice in modern detectors.
The EMA shadow is a separate :class:`nn.Module` deep-copied
from the live model; its parameters are updated each iteration as

    shadow ← decay * shadow + (1 - decay) * live

with a warmup-aware decay that ramps from 0 to the asymptotic value
during the first ~``tau`` iterations:

    decay_t = decay * (1 - exp(-t / tau))

This warmup avoids the "EMA lags init for thousands of steps" problem
that hurts early-training stability — at iteration 0 the EMA is
literally the live weights; by iteration ~``tau`` the EMA is tracking
the long-tail moving average.

The EMA shadow is **not** the training model. The trainer keeps optimising
the live model; the EMA hook just maintains a parallel copy. At eval /
save time, swap the EMA's ``state_dict()`` in (typically what gets
shipped as final checkpoint weights — they're consistently 0.3-0.5 AP
better than the raw live weights on COCO).

"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch import nn

from mayaku.engine.callbacks import _BaseHook

if TYPE_CHECKING:
    from mayaku.engine.trainer import TrainerBase

__all__ = ["EMAHook", "ModelEMA"]


class ModelEMA:
    """Exponential moving average of a model's parameters and buffers.

    Args:
        model: The live training model. EMA is initialised by deep-copying
            this model's state and is updated against it on every
            :meth:`update` call.
        decay: Asymptotic EMA decay. Default 0.9999. Values closer to 1
            track slower (more averaging); closer to 0 track faster
            (more like the live model).
        tau: Warmup time-constant for the decay ramp (in update steps).
            Default 2000. The effective decay at update ``t`` is
            ``decay * (1 - exp(-t / tau))``, so for the first ~tau
            iterations the EMA closely follows the live model and only
            gradually settles to long-window averaging.
        updates: Initial update counter. Set when resuming from a
            checkpoint so the warmup curve continues from the right
            point.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9999,
        tau: float = 2000.0,
        updates: int = 0,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1]; got {decay}")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0; got {tau}")
        if updates < 0:
            raise ValueError(f"updates must be >= 0; got {updates}")
        # Deep-copy in eval mode and freeze gradients — the shadow
        # never participates in autograd. Match the live model's device.
        self.shadow: nn.Module = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.decay_asymptote = float(decay)
        self.tau = float(tau)
        self.updates = int(updates)

    def decay_at(self, step: int) -> float:
        """Decay value at update step ``step`` (with warmup ramp)."""
        return self.decay_asymptote * (1.0 - math.exp(-step / self.tau))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the EMA shadow against the live ``model``'s state.

        Updates **all** floating-point tensors in the state dict — both
        learnable parameters and registered buffers (e.g. BatchNorm
        running stats). Integer / bool buffers (e.g. ``num_batches_tracked``)
        are passed through unchanged.
        """
        self.updates += 1
        d = self.decay_at(self.updates)
        live_state = model.state_dict()
        ema_state = self.shadow.state_dict()
        for k, ema_v in ema_state.items():
            live_v = live_state[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(live_v.detach(), alpha=1.0 - d)
            else:
                ema_v.copy_(live_v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the EMA shadow's ``state_dict``.

        Use this in place of ``model.state_dict()`` when saving the
        final checkpoint — that's where the AP gain materialises.
        """
        return self.shadow.state_dict()

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.shadow.parameters()


class EMAHook(_BaseHook):
    """Trainer hook that updates a :class:`ModelEMA` after every step.

    Register **after** ``LRScheduler`` and **before** ``EvalHook`` /
    ``PeriodicCheckpointer`` so:

    1. Live weights are updated by the optimiser (in ``run_step``).
    2. EMA shadow is updated against the new live weights (this hook).
    3. Eval / checkpoint reads either the live or EMA weights depending
       on configuration.

    Args:
        ema: The :class:`ModelEMA` instance to update.
        model: The live training model whose state drives the EMA.
            Typically the same module the trainer is updating.
    """

    def __init__(self, ema: ModelEMA, model: nn.Module) -> None:
        super().__init__()
        self.ema = ema
        self.model = model
        self.trainer: TrainerBase | None = None

    def after_step(self) -> None:
        self.ema.update(self.model)
