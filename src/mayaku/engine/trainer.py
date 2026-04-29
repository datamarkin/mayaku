"""Training loop drivers.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §3.1 (`engine/train_loop.py`):

* :class:`TrainerBase` — the lifecycle: ``train(start_iter, max_iter)``
  calls ``before_train`` → repeated ``before_step`` / ``run_step`` /
  ``after_step`` → ``after_train``. Subclass and override
  ``run_step``.
* :class:`SimpleTrainer` — synchronous SGD step:
  ``zero_grad → forward → backward → optimizer.step``.
* :class:`AMPTrainer` — same loop, wrapped in
  :func:`mayaku.backends.amp.autocast` and
  :func:`make_grad_scaler`. The dtype (``fp16``/``bf16``) is read off
  the :class:`Device` plus a config override (`spec §6.1`,
  ``SolverConfig.amp_dtype``).

The hook protocol comes from :mod:`mayaku.engine.callbacks`.
``register_hooks`` binds ``self`` onto each hook so hooks can read
``trainer.iter`` / ``trainer.last_loss`` without an explicit argument.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from mayaku.backends.amp import NullGradScaler, autocast, make_grad_scaler
from mayaku.backends.device import Device
from mayaku.engine.callbacks import HookBase

__all__ = ["AMPTrainer", "GradClipType", "SimpleTrainer", "TrainerBase"]

# Each loader yield is a list[dict] (DatasetMapper output, Step 6).
DataIterable = Iterable[Sequence[Mapping[str, Any]]]

# Matches `SolverConfig.clip_gradients_type` so the CLI can pass the
# config field through verbatim.
GradClipType = Literal["value", "norm"]


class TrainerBase:
    """Hook lifecycle wrapper. Subclasses implement :meth:`run_step`."""

    iter: int = 0
    start_iter: int = 0
    max_iter: int = 0
    storage: dict[str, float]
    hooks: list[HookBase]
    last_loss: float = 0.0

    def __init__(self) -> None:
        self.hooks = []
        self.storage = {}

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------

    def register_hooks(self, hooks: Sequence[HookBase]) -> None:
        for h in hooks:
            h.trainer = self
            self.hooks.append(h)

    def before_train(self) -> None:
        for h in self.hooks:
            h.before_train()

    def after_train(self) -> None:
        for h in self.hooks:
            h.after_train()

    def before_step(self) -> None:
        for h in self.hooks:
            h.before_step()

    def after_step(self) -> None:
        for h in self.hooks:
            h.after_step()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def train(self, start_iter: int, max_iter: int) -> None:
        if max_iter < start_iter:
            raise ValueError(f"max_iter={max_iter} < start_iter={start_iter}")
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        try:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            # Match Detectron2's post-loop convention: `iter` reflects
            # the count of *completed* iterations after the loop ends.
            self.iter += 1
        finally:
            self.after_train()

    def run_step(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SimpleTrainer
# ---------------------------------------------------------------------------


class SimpleTrainer(TrainerBase):
    """Synchronous SGD trainer.

    ``run_step``: ``optimizer.zero_grad → loss = sum(model(batch).values())
    → loss.backward → optimizer.step``. The model is expected to return
    a dict of named loss tensors (the contract used by every
    :class:`mayaku.models.detectors.FasterRCNN`).
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataIterable,
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip_norm: float | None = None,
        grad_clip_type: GradClipType = "norm",
    ) -> None:
        super().__init__()
        if grad_clip_type not in ("value", "norm"):
            raise ValueError(f"grad_clip_type must be 'value' or 'norm'; got {grad_clip_type!r}")
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_type: GradClipType = grad_clip_type
        self._data_iter: Any = None

    def _next_batch(self) -> Sequence[Mapping[str, Any]]:
        if self._data_iter is None:
            self._data_iter = iter(self.data_loader)
        try:
            return next(self._data_iter)
        except StopIteration:
            # Loader exhausted: re-iterate. Detectron2's TrainingSampler
            # is infinite (Step 6) so this branch is mainly for finite
            # toy loaders used by tests.
            self._data_iter = iter(self.data_loader)
            return next(self._data_iter)

    def _clip_grads(self) -> None:
        """Apply value or norm clipping to trainable params, per
        :attr:`grad_clip_type`. Caller is responsible for unscaling
        gradients first when AMP is in use.

        For ``grad_clip_type="norm"`` we clip *per-parameter*, not
        globally. PyTorch's ``clip_grad_norm_(parameters, max_norm)``
        flattens every parameter's gradient into one vector and rescales
        all of them uniformly if the concatenated norm exceeds
        ``max_norm`` — for a 50M-parameter detector at ``max_norm=1.0``
        that's roughly 10–100× more aggressive than the per-parameter
        check most parameters would individually pass. Detectron2's
        ``solver/build.py`` `OptimizerWithGradientClip.step` calls
        ``clip_grad_norm_`` once per parameter; we mirror that here so
        the configured ``base_lr`` translates to the same effective
        gradient magnitude.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        assert self.grad_clip_norm is not None
        if self.grad_clip_type == "value":
            # clip_grad_value_ is element-wise — list vs single is equivalent.
            torch.nn.utils.clip_grad_value_(params, clip_value=self.grad_clip_norm)
        else:
            for p in params:
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_(p, max_norm=self.grad_clip_norm)

    def run_step(self) -> None:
        batch = self._next_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_dict = self.model(batch)
        total = _sum_loss(loss_dict)
        total.backward()  # type: ignore[no-untyped-call]
        if self.grad_clip_norm is not None:
            self._clip_grads()
        self.optimizer.step()
        self._record(loss_dict, total)

    def _record(self, loss_dict: Mapping[str, Tensor], total: Tensor) -> None:
        self.last_loss = float(total.detach().item())
        self.storage = {
            "total_loss": self.last_loss,
            **{k: float(v.detach().item()) for k, v in loss_dict.items()},
        }


# ---------------------------------------------------------------------------
# AMPTrainer
# ---------------------------------------------------------------------------


class AMPTrainer(SimpleTrainer):
    """SGD with autocast + ``GradScaler`` for fp16 / bf16 training.

    Args (in addition to :class:`SimpleTrainer`):
        device: The :class:`mayaku.backends.device.Device` whose AMP
            kind / dtype the trainer will use.
        amp_dtype: ``"fp16"`` or ``"bf16"``. Overrides
            :attr:`Device.amp_dtype` so callers can opt into bf16 on
            CUDA Ampere+ via config (`spec §6.1`,
            ``SolverConfig.amp_dtype``).
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataIterable,
        optimizer: torch.optim.Optimizer,
        device: Device,
        *,
        amp_dtype: str = "fp16",
        grad_clip_norm: float | None = None,
        grad_clip_type: GradClipType = "norm",
    ) -> None:
        super().__init__(
            model,
            data_loader,
            optimizer,
            grad_clip_norm=grad_clip_norm,
            grad_clip_type=grad_clip_type,
        )
        if amp_dtype not in ("fp16", "bf16"):
            raise ValueError(f"amp_dtype must be 'fp16' or 'bf16'; got {amp_dtype!r}")
        if not device.supports_amp:
            raise ValueError(
                f"AMPTrainer requires a backend with AMP support; {device.kind!r} does not."
            )
        self.device = device
        self.amp_dtype = amp_dtype
        # GradScaler is only useful for fp16; bf16's range covers the
        # same gradient magnitudes that fp32 does, so the scaler is a no-op.
        self.scaler: torch.amp.GradScaler | NullGradScaler
        if amp_dtype == "fp16":
            self.scaler = make_grad_scaler(device)
        else:
            self.scaler = NullGradScaler()

    def run_step(self) -> None:
        batch = self._next_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, dtype=_torch_dtype(self.amp_dtype)):
            loss_dict = self.model(batch)
            total = _sum_loss(loss_dict)
        self.scaler.scale(total).backward()  # type: ignore[no-untyped-call]
        if self.grad_clip_norm is not None:
            # Unscale before clipping so the clip threshold is in
            # real-gradient units regardless of the scaler's loss scale.
            self.scaler.unscale_(self.optimizer)
            self._clip_grads()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._record(loss_dict, total)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sum_loss(loss_dict: Any) -> Tensor:
    if isinstance(loss_dict, Tensor):
        return loss_dict
    if isinstance(loss_dict, Mapping):
        values: list[Tensor] = list(loss_dict.values())
        if not values:
            raise ValueError("model returned an empty loss dict")
        total: Tensor = values[0]
        for v in values[1:]:
            total = total + v
        return total
    raise TypeError(
        f"model must return a Tensor or Mapping[str, Tensor]; got {type(loss_dict).__name__}"
    )


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16
