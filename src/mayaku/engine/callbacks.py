"""Hook protocol + a few built-in hooks.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §3.1 (`engine/hooks.py`):

* :class:`HookBase` — protocol with optional ``before_train``,
  ``after_train``, ``before_step``, ``after_step`` methods. Default
  implementations are no-ops so subclasses override only what they
  need.
* :class:`IterationTimer` — measures per-iter wall time and exposes a
  ``last_iter_seconds`` attribute the trainer can log.
* :class:`LRScheduler` — wraps a ``torch.optim.lr_scheduler``; calls
  ``.step()`` at the end of every iteration.
* :class:`PeriodicCheckpointer` — saves a single ``state_dict`` to
  ``output_dir/model_iter_<N>.pth`` every ``period`` iterations.

The hook list is registered on a :class:`TrainerBase` (Step 13's
trainer) and runs in registration order.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from mayaku.engine.evaluator import DatasetEvaluator
    from mayaku.engine.trainer import TrainerBase

__all__ = [
    "EvalHook",
    "HookBase",
    "IterationTimer",
    "LRScheduler",
    "MemoryTrim",
    "MetricsPrinter",
    "PeriodicCheckpointer",
]


class HookBase(Protocol):
    """Protocol with four optional hook points.

    Concrete hooks subclass this and override one or more methods.
    The trainer calls each method on every registered hook in
    registration order. ``self.trainer`` is set by
    :meth:`TrainerBase.register_hooks` so a hook can read iteration
    state without an explicit argument.
    """

    trainer: TrainerBase | None

    def before_train(self) -> None: ...
    def after_train(self) -> None: ...
    def before_step(self) -> None: ...
    def after_step(self) -> None: ...


class _BaseHook:
    """Default-no-op base class. Concrete hooks subclass this and
    override only the methods they care about."""

    trainer: TrainerBase | None = None

    def before_train(self) -> None:
        return None

    def after_train(self) -> None:
        return None

    def before_step(self) -> None:
        return None

    def after_step(self) -> None:
        return None


# ---------------------------------------------------------------------------
# IterationTimer
# ---------------------------------------------------------------------------


class IterationTimer(_BaseHook):
    """Records the wall-clock time of every training iteration.

    Exposes ``last_iter_seconds`` (a float, the most recent iteration's
    duration) and ``total_seconds`` (the sum across all iterations
    since ``before_train``). Trainers / external loggers can read
    these to emit ``data_time`` / ``iter_time`` metrics.
    """

    def __init__(self) -> None:
        self.last_iter_seconds: float = 0.0
        self.total_seconds: float = 0.0
        self._step_start: float = 0.0

    def before_step(self) -> None:
        self._step_start = time.perf_counter()

    def after_step(self) -> None:
        elapsed = time.perf_counter() - self._step_start
        self.last_iter_seconds = elapsed
        self.total_seconds += elapsed


# ---------------------------------------------------------------------------
# LRScheduler
# ---------------------------------------------------------------------------


class LRScheduler(_BaseHook):
    """Step a ``torch.optim.lr_scheduler`` every iteration."""

    def __init__(self, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        self.scheduler = scheduler

    def after_step(self) -> None:
        self.scheduler.step()


# ---------------------------------------------------------------------------
# MemoryTrim
# ---------------------------------------------------------------------------


class MemoryTrim(_BaseHook):
    """Periodically force the Python GC and return free heap to the OS.

    Long training runs accumulate two kinds of host-RAM pressure that
    don't show up on the GPU:

    1. **Python GC cycles.** Reference cycles (e.g. mutual references
       inside augmentation pipelines) sit in the cyclic GC's young
       generation and are reclaimed only when CPython's heuristic
       triggers. A periodic explicit ``gc.collect()`` keeps the cycle
       backlog bounded.

    2. **glibc malloc fragmentation.** glibc's default allocator rarely
       returns small freed regions to the OS — the resident-set drifts
       upward over a long run even when the live working-set stays
       bounded. ``malloc_trim(0)`` forces a release. No-op on
       non-glibc platforms (the ctypes load fails silently).

    Args:
        period: Trim every Nth iteration (matched on ``trainer.iter``).
            Iteration 0 is skipped. Default 1000 — cheap relative to a
            ~0.3 s/iter training step but frequent enough to keep the
            heap from drifting tens of GB.

    Has no effect on GPU memory; that's PyTorch's caching allocator and
    has its own knobs.
    """

    def __init__(self, *, period: int = 1000) -> None:
        if period <= 0:
            raise ValueError(f"period must be > 0; got {period}")
        self.period = period
        self._libc: Any = None
        try:
            import ctypes

            self._libc = ctypes.CDLL("libc.so.6")
        except OSError:
            # Non-glibc platform (macOS / Windows / musl) — gc.collect
            # alone still helps with Python-side cycles.
            self._libc = None

    def after_step(self) -> None:
        assert self.trainer is not None, "trainer reference not bound"
        it = self.trainer.iter + 1  # 1-indexed for the modulo check
        if it % self.period != 0:
            return
        import gc as _gc

        _gc.collect()
        if self._libc is not None:
            self._libc.malloc_trim(0)


# ---------------------------------------------------------------------------
# PeriodicCheckpointer
# ---------------------------------------------------------------------------


class PeriodicCheckpointer(_BaseHook):
    """Save the model + optimizer state every ``period`` iterations.

    Args:
        model: The :class:`nn.Module` whose ``state_dict`` is saved.
        optimizer: Optional optimizer; saved alongside the model when
            present so resumption picks up the same momentum buffers.
        output_dir: Directory under which checkpoints are written.
        period: Save every Nth iteration (matched on ``trainer.iter``).
            Iteration 0 is never saved (an empty model is rarely
            useful).
        save_final: Also save once on ``after_train``, naming the file
            ``model_final.pth``.
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str | Path,
        period: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        save_final: bool = True,
    ) -> None:
        if period <= 0:
            raise ValueError(f"PeriodicCheckpointer period must be > 0; got {period}")
        self.model = model
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.period = period
        self.save_final = save_final

    def before_train(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def after_step(self) -> None:
        assert self.trainer is not None, "trainer reference not bound"
        # 1-indexed: skip iter 0 (nothing to save), match every multiple.
        it = self.trainer.iter + 1
        if it % self.period == 0:
            self._save(self.output_dir / f"model_iter_{it:07d}.pth")

    def after_train(self) -> None:
        if self.save_final:
            self._save(self.output_dir / "model_final.pth")

    def _save(self, path: Path) -> None:
        state: dict[str, object] = {"model": self.model.state_dict()}
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        torch.save(state, path)


# ---------------------------------------------------------------------------
# MetricsPrinter
# ---------------------------------------------------------------------------


class MetricsPrinter(_BaseHook):
    """Print one line of training metrics every ``period`` iterations.

    Without this hook the trainer pushes losses into ``trainer.storage``
    but nothing surfaces them — the user sees a silent terminal and
    can't tell whether training is converging. Default period is small
    (20) so the feedback loop on tiny datasets is fast; bump it to 100+
    for full-scale training.

    Args:
        optimizer: Read the current LR from ``optimizer.param_groups[0]
            ["lr"]``. Optional — omit to drop ``lr`` from the line.
        period: Print every ``period`` iterations. The first iteration
            (``trainer.iter == start_iter``) always prints so users see
            something within seconds of launch.
        timer: Optional :class:`IterationTimer`; if provided, the line
            includes ``iter_t`` (seconds per recent iteration).
    """

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        period: int = 20,
        timer: IterationTimer | None = None,
    ) -> None:
        if period <= 0:
            raise ValueError(f"MetricsPrinter period must be > 0; got {period}")
        self.optimizer = optimizer
        self.period = period
        self.timer = timer
        self._first_printed = False

    def after_step(self) -> None:
        assert self.trainer is not None, "trainer reference not bound"
        it = self.trainer.iter
        # Always print the first iteration so the user gets feedback
        # within seconds. After that, throttle to every `period` steps.
        if self._first_printed and (it - self.trainer.start_iter) % self.period != 0:
            return
        self._first_printed = True

        storage = self.trainer.storage
        parts = [f"iter {it:>7d}/{self.trainer.max_iter}"]
        if "total_loss" in storage:
            parts.append(f"total_loss={storage['total_loss']:.4f}")
        # Print individual losses in a stable order so successive lines
        # are diff-friendly.
        for k in sorted(storage):
            if k == "total_loss":
                continue
            parts.append(f"{k}={storage[k]:.4f}")
        if self.optimizer is not None:
            parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
        if self.timer is not None and self.timer.last_iter_seconds > 0:
            parts.append(f"iter_t={self.timer.last_iter_seconds * 1000:.0f}ms")
        print("  ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# EvalHook
# ---------------------------------------------------------------------------


class EvalHook(_BaseHook):
    """Run a :class:`DatasetEvaluator` every ``period`` iterations.

    Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §3.1 ``EvalHook``: drives
    :func:`mayaku.engine.inference_on_dataset` against a held-out
    loader on a schedule, prints the resulting metrics dict so the user
    can see mid-training AP without stopping the run.

    The model is forced into ``eval()`` for the duration of the eval
    pass and restored to its prior mode by ``inference_on_dataset``.
    The hook itself does not gather across DDP ranks — that's
    :class:`COCOEvaluator`'s job.

    Args:
        period: Run eval every ``period`` iterations (matched on
            ``trainer.iter + 1``, the same 1-indexed convention
            :class:`PeriodicCheckpointer` uses). ``0`` disables periodic
            firing; the final eval still runs from ``after_train`` if
            ``eval_after_train=True``.
        evaluator: A :class:`DatasetEvaluator` (typically
            :class:`COCOEvaluator`).
        model: The model to evaluate. Usually the same module the
            trainer is updating.
        data_loader: A loader yielding val batches, contract identical
            to :func:`inference_on_dataset`'s ``data_loader`` argument.
        eval_after_train: Run one final eval in ``after_train``,
            regardless of period. Defaults to ``True`` so callers
            always see a closing AP.
    """

    def __init__(
        self,
        period: int,
        evaluator: DatasetEvaluator,
        model: nn.Module,
        data_loader: Iterable[Sequence[dict[str, Any]]],
        *,
        eval_after_train: bool = True,
    ) -> None:
        if period < 0:
            raise ValueError(f"EvalHook period must be >= 0; got {period}")
        self.period = period
        self.evaluator = evaluator
        self.model = model
        self.data_loader = data_loader
        self.eval_after_train = eval_after_train

    def after_step(self) -> None:
        assert self.trainer is not None, "trainer reference not bound"
        if self.period <= 0:
            return
        next_iter = self.trainer.iter + 1
        if next_iter % self.period != 0:
            return
        # Don't double-fire on the very last iteration when after_train
        # is also going to run an eval — that would be redundant.
        if self.eval_after_train and next_iter == self.trainer.max_iter:
            return
        self._run_eval(next_iter)

    def after_train(self) -> None:
        if not self.eval_after_train:
            return
        assert self.trainer is not None, "trainer reference not bound"
        self._run_eval(self.trainer.max_iter)

    def _run_eval(self, iter_: int) -> None:
        # Lazy import to avoid the engine.callbacks → engine.evaluator
        # import cycle at module import time.
        from mayaku.engine.evaluator import inference_on_dataset

        metrics = inference_on_dataset(self.model, self.data_loader, self.evaluator)
        print(f"[eval @ iter {iter_}] {metrics}", flush=True)
