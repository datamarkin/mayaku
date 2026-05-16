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
from mayaku.engine.distributed import get_world_size

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
        grad_accum_steps: int = 1,
        grad_norm_log_enabled: bool = False,
    ) -> None:
        super().__init__()
        if grad_clip_type not in ("value", "norm"):
            raise ValueError(f"grad_clip_type must be 'value' or 'norm'; got {grad_clip_type!r}")
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1; got {grad_accum_steps}")
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_type: GradClipType = grad_clip_type
        self.grad_accum_steps = int(grad_accum_steps)
        self.grad_norm_log_enabled = bool(grad_norm_log_enabled)
        self._data_iter: Any = None
        # Resolved once on first call; param-name → group-name map.
        self._grad_norm_group_map: dict[str, str] | None = None

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

        ``grad_clip_type="norm"`` is global-norm clipping: PyTorch's
        ``clip_grad_norm_(parameters, max_norm)`` flattens every
        parameter's gradient into one vector and rescales them
        uniformly if the concatenated norm exceeds ``max_norm``.
        Detectron2's ``solver/build.py`` ``OptimizerWithGradientClip``
        does the same. An earlier version of this method clipped
        *per-parameter* at ``norm=1.0``, which is ~2–3 orders of
        magnitude more aggressive than typical "aggressive" clipping
        and starved RPN/FPN gradients enough that training stalled
        on Faster R-CNN — see commit history for the post-mortem.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        assert self.grad_clip_norm is not None
        if self.grad_clip_type == "value":
            # clip_grad_value_ is element-wise — list vs single is equivalent.
            torch.nn.utils.clip_grad_value_(params, clip_value=self.grad_clip_norm)
        else:
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)

    # ------------------------------------------------------------------
    # Gradient norm diagnostics
    # ------------------------------------------------------------------

    # Order matters: first-match wins, so more-specific prefixes go first.
    # Keys become metric column names (logged as ``grad_<key>``).
    _GRAD_NORM_GROUPS: tuple[tuple[str, str], ...] = (
        ("rpn_cls", "rpn.head.objectness_logits"),
        ("rpn_loc", "rpn.head.anchor_deltas"),
        ("rpn_conv", "rpn.head.conv"),
        ("roi_cls", "roi_heads.box_predictor.cls_score"),
        ("roi_loc", "roi_heads.box_predictor.bbox_pred"),
        ("roi_box_head", "roi_heads.box_head"),
        ("backbone_resnet", "backbone.bottom_up"),
        ("backbone_fpn", "backbone."),
    )

    def _build_grad_norm_group_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            matched = "other"
            for grp_key, grp_prefix in self._GRAD_NORM_GROUPS:
                if name.startswith(grp_prefix):
                    matched = grp_key
                    break
            mapping[name] = matched
        return mapping

    def _compute_grad_norms(self) -> dict[str, float]:
        """Per-group L2 grad norms (sqrt of sum-of-squares) plus ``total``.

        Cheap: one ``.pow(2).sum()`` per parameter, summed by group, then
        ``sqrt``. Numbers match what ``torch.nn.utils.clip_grad_norm_``
        would report as the pre-clip norm if you concatenated only the
        listed group's grads.
        """
        if self._grad_norm_group_map is None:
            self._grad_norm_group_map = self._build_grad_norm_group_map()
        sums: dict[str, float] = {"total": 0.0}
        for grp_key, _ in self._GRAD_NORM_GROUPS:
            sums[grp_key] = 0.0
        sums["other"] = 0.0
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            sq = float(p.grad.detach().pow(2).sum().item())
            sums["total"] += sq
            grp = self._grad_norm_group_map.get(name, "other")
            sums[grp] = sums.get(grp, 0.0) + sq
        return {k: v**0.5 for k, v in sums.items() if v > 0.0 or k == "total"}

    def run_step(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Gradient accumulation: forward+backward over ``grad_accum_steps``
        # micro-batches, scaling each loss by 1/N so the SUMMED gradients
        # equal the gradient of the mean loss across the full effective
        # batch. ``optimizer.step()`` and grad clipping fire once per
        # ``run_step`` (i.e. once per effective iteration).
        accum_total = 0.0
        accum_dict: dict[str, float] = {}
        for _ in range(self.grad_accum_steps):
            batch = self._next_batch()
            loss_dict = self.model(batch)
            micro_total = _sum_loss(loss_dict) / self.grad_accum_steps
            micro_total.backward()  # type: ignore[no-untyped-call]
            # Track raw (un-scaled) loss values for the metrics record.
            accum_total += float(micro_total.detach().item()) * self.grad_accum_steps
            for k, v in loss_dict.items():
                accum_dict[k] = accum_dict.get(k, 0.0) + float(v.detach().item())
        # Diagnostic: capture pre-clip per-group grad norms so a divergent
        # run can be localised to a specific head (RPN cls vs ROI cls vs ...).
        grad_norms = self._compute_grad_norms() if self.grad_norm_log_enabled else None
        if self.grad_clip_norm is not None:
            self._clip_grads()
        self.optimizer.step()
        # Mean across micro-batches for the metrics record.
        n = self.grad_accum_steps
        self._record_floats(
            {k: v / n for k, v in accum_dict.items()},
            accum_total / n,
            grad_norms=grad_norms,
        )

    def _record(self, loss_dict: Mapping[str, Tensor], total: Tensor) -> None:
        self.last_loss = float(total.detach().item())
        self.storage = {
            "total_loss": self.last_loss,
            **{k: float(v.detach().item()) for k, v in loss_dict.items()},
        }

    def _record_floats(
        self,
        loss_dict: Mapping[str, float],
        total: float,
        *,
        grad_norms: Mapping[str, float] | None = None,
    ) -> None:
        """Same shape as :meth:`_record` but with already-evaluated floats.

        Used by gradient-accumulation paths that have already reduced the
        per-micro-batch tensors to scalars across the accumulation loop.
        ``grad_norms`` (when provided) is recorded under ``grad_<key>``
        names so :class:`MetricsPrinter` can include them in the line.
        """
        world = get_world_size()
        if world > 1:
            # Average across ranks so logged numbers are comparable to
            # a single-GPU baseline rather than rank-0's local view. One
            # all_reduce per iter — negligible vs the forward/backward
            # work. Grad norms stay per-rank: they're diagnostic and
            # the per-rank value is what would actually be clipped.
            keys = sorted(loss_dict)
            tensor = torch.tensor(
                [loss_dict[k] for k in keys] + [total], dtype=torch.float32
            )
            if torch.distributed.get_backend() == "nccl":
                tensor = tensor.cuda()
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            reduced: list[float] = (tensor / world).cpu().tolist()
            loss_dict = dict(zip(keys, reduced[:-1], strict=True))
            total = reduced[-1]
        self.last_loss = total
        record: dict[str, float] = {"total_loss": total, **dict(loss_dict)}
        if grad_norms is not None:
            for k, v in grad_norms.items():
                record[f"grad_{k}"] = v
        self.storage = record


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
        grad_accum_steps: int = 1,
        grad_norm_log_enabled: bool = False,
    ) -> None:
        super().__init__(
            model,
            data_loader,
            optimizer,
            grad_clip_norm=grad_clip_norm,
            grad_clip_type=grad_clip_type,
            grad_accum_steps=grad_accum_steps,
            grad_norm_log_enabled=grad_norm_log_enabled,
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
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Same accumulation pattern as :meth:`SimpleTrainer.run_step`,
        # with ``scaler.scale(...).backward()`` per micro-batch but
        # ``scaler.unscale_`` / ``scaler.step`` / ``scaler.update``
        # only once per effective iteration. Unscale fires before grad
        # clip so the clip threshold is in real-gradient units regardless
        # of the scaler's loss scale.
        accum_total = 0.0
        accum_dict: dict[str, float] = {}
        for _ in range(self.grad_accum_steps):
            batch = self._next_batch()
            with autocast(self.device, dtype=_torch_dtype(self.amp_dtype)):
                loss_dict = self.model(batch)
                micro_total = _sum_loss(loss_dict) / self.grad_accum_steps
            self.scaler.scale(micro_total).backward()  # type: ignore[no-untyped-call]
            accum_total += float(micro_total.detach().item()) * self.grad_accum_steps
            for k, v in loss_dict.items():
                accum_dict[k] = accum_dict.get(k, 0.0) + float(v.detach().item())
        # Unscale before computing diagnostic norms so the values are in
        # real-gradient units (matching what the user would see at FP32).
        if self.grad_clip_norm is not None or self.grad_norm_log_enabled:
            self.scaler.unscale_(self.optimizer)
        grad_norms = self._compute_grad_norms() if self.grad_norm_log_enabled else None
        if self.grad_clip_norm is not None:
            self._clip_grads()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        n = self.grad_accum_steps
        self._record_floats(
            {k: v / n for k, v in accum_dict.items()},
            accum_total / n,
            grad_norms=grad_norms,
        )


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
