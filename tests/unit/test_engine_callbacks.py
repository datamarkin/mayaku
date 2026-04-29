"""Tests for :mod:`mayaku.engine.callbacks`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from mayaku.engine.callbacks import (
    EvalHook,
    IterationTimer,
    LRScheduler,
    MetricsPrinter,
    PeriodicCheckpointer,
)
from mayaku.engine.evaluator import DatasetEvaluator


class _ToyTrainer:
    """Minimal stand-in so we can exercise hooks without a full TrainerBase."""

    def __init__(self) -> None:
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int = 100
        self.storage: dict[str, float] = {}


# ---------------------------------------------------------------------------
# IterationTimer
# ---------------------------------------------------------------------------


def test_iteration_timer_records_step_duration() -> None:
    t = IterationTimer()
    t.before_step()
    time.sleep(0.005)
    t.after_step()
    assert t.last_iter_seconds >= 0.005
    assert t.total_seconds == pytest.approx(t.last_iter_seconds, abs=1e-6)


def test_iteration_timer_accumulates_total() -> None:
    t = IterationTimer()
    for _ in range(3):
        t.before_step()
        time.sleep(0.001)
        t.after_step()
    assert t.total_seconds >= 3 * 0.001


# ---------------------------------------------------------------------------
# LRScheduler hook
# ---------------------------------------------------------------------------


def test_lr_scheduler_hook_steps_after_each_iter() -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda it: 0.5**it)
    hook = LRScheduler(sched)
    hook.after_step()
    assert opt.param_groups[0]["lr"] == 0.5
    hook.after_step()
    assert opt.param_groups[0]["lr"] == 0.25


# ---------------------------------------------------------------------------
# PeriodicCheckpointer
# ---------------------------------------------------------------------------


def test_periodic_checkpointer_writes_files(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    hook = PeriodicCheckpointer(
        model, output_dir=tmp_path, period=3, optimizer=opt, save_final=True
    )
    trainer = _ToyTrainer()
    hook.trainer = trainer  # type: ignore[assignment]
    hook.before_train()
    for i in range(7):
        trainer.iter = i
        hook.after_step()
    hook.after_train()
    files = sorted(p.name for p in tmp_path.glob("*.pth"))
    # Saves at iter 3, 6 (1-indexed: i+1 % 3 == 0 → i in {2, 5}, file names
    # use the post-iter count, so iter=2 → "model_iter_0000003.pth").
    assert "model_iter_0000003.pth" in files
    assert "model_iter_0000006.pth" in files
    assert "model_final.pth" in files


def test_periodic_checkpointer_state_dict_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    hook = PeriodicCheckpointer(
        model, output_dir=tmp_path, period=1, optimizer=opt, save_final=False
    )
    trainer = _ToyTrainer()
    hook.trainer = trainer  # type: ignore[assignment]
    hook.before_train()
    trainer.iter = 0
    hook.after_step()  # writes model_iter_0000001.pth

    state = torch.load(tmp_path / "model_iter_0000001.pth", weights_only=True)
    assert "model" in state and "optimizer" in state
    fresh = nn.Linear(2, 2)
    fresh.load_state_dict(state["model"])
    for p1, p2 in zip(fresh.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(p1, p2)


def test_periodic_checkpointer_rejects_zero_period(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="period"):
        PeriodicCheckpointer(nn.Linear(2, 2), output_dir=tmp_path, period=0)


# ---------------------------------------------------------------------------
# MetricsPrinter
# ---------------------------------------------------------------------------


def test_metrics_printer_prints_first_iteration(capsys: pytest.CaptureFixture[str]) -> None:
    trainer = _ToyTrainer()
    trainer.storage = {"total_loss": 1.5, "loss_cls": 0.7, "loss_box_reg": 0.8}
    hook = MetricsPrinter(period=10)
    hook.trainer = trainer  # type: ignore[assignment]
    trainer.iter = 0
    hook.after_step()
    out = capsys.readouterr().out
    assert "iter       0/100" in out
    assert "total_loss=1.5000" in out
    assert "loss_cls=0.7000" in out
    assert "loss_box_reg=0.8000" in out


def test_metrics_printer_throttles_to_period(capsys: pytest.CaptureFixture[str]) -> None:
    trainer = _ToyTrainer()
    trainer.storage = {"total_loss": 1.0}
    hook = MetricsPrinter(period=5)
    hook.trainer = trainer  # type: ignore[assignment]
    for i in range(12):
        trainer.iter = i
        trainer.storage = {"total_loss": 1.0 - 0.05 * i}
        hook.after_step()
    out = capsys.readouterr().out
    # First (i=0) always prints; then every 5th: i=5, i=10.
    lines = [line for line in out.splitlines() if line.startswith("iter")]
    assert len(lines) == 3
    iters = [int(line.split()[1].split("/")[0]) for line in lines]
    assert iters == [0, 5, 10]


def test_metrics_printer_includes_lr_when_optimizer_passed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = _ToyTrainer()
    trainer.storage = {"total_loss": 0.5}
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.01)
    hook = MetricsPrinter(optimizer=opt, period=1)
    hook.trainer = trainer  # type: ignore[assignment]
    trainer.iter = 0
    hook.after_step()
    out = capsys.readouterr().out
    assert "lr=1.00e-02" in out


def test_metrics_printer_includes_iter_time_when_timer_passed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = _ToyTrainer()
    trainer.storage = {"total_loss": 0.1}
    timer = IterationTimer()
    timer.last_iter_seconds = 0.123
    hook = MetricsPrinter(period=1, timer=timer)
    hook.trainer = trainer  # type: ignore[assignment]
    trainer.iter = 0
    hook.after_step()
    out = capsys.readouterr().out
    assert "iter_t=123ms" in out


def test_metrics_printer_rejects_zero_period() -> None:
    with pytest.raises(ValueError, match="period"):
        MetricsPrinter(period=0)


# ---------------------------------------------------------------------------
# EvalHook
# ---------------------------------------------------------------------------


class _CountingEvaluator(DatasetEvaluator):
    """Counts how many times `evaluate()` was called."""

    def __init__(self) -> None:
        self.eval_calls = 0

    def reset(self) -> None:
        return None

    def process(
        self,
        inputs: Sequence[dict[str, Any]],
        outputs: Sequence[dict[str, Any]],
    ) -> None:
        return None

    def evaluate(self) -> dict[str, Any]:
        self.eval_calls += 1
        return {"calls": self.eval_calls}


class _StubModel(nn.Module):
    def forward(self, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{} for _ in batch]


def _drive_hook(
    hook: EvalHook,
    trainer: _ToyTrainer,
    *,
    n_iters: int,
) -> None:
    """Walk a hook through `n_iters` step boundaries, mirroring the real
    trainer's iter increment after each `after_step` call."""
    hook.trainer = trainer  # type: ignore[assignment]
    for i in range(n_iters):
        trainer.iter = i
        hook.after_step()
    hook.after_train()


def test_eval_hook_fires_at_period_and_final(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = _ToyTrainer()
    trainer.max_iter = 10
    evaluator = _CountingEvaluator()
    loader = [[{"image_id": 1}]]
    hook = EvalHook(period=3, evaluator=evaluator, model=_StubModel(), data_loader=loader)
    _drive_hook(hook, trainer, n_iters=10)
    # Period=3 → fires at next_iter ∈ {3, 6, 9}; iter 10 is the final
    # eval handled by after_train (not double-counted at next_iter==10
    # because that condition gates against trainer.max_iter).
    assert evaluator.eval_calls == 4
    out = capsys.readouterr().out
    assert "[eval @ iter 3]" in out
    assert "[eval @ iter 6]" in out
    assert "[eval @ iter 9]" in out
    assert "[eval @ iter 10]" in out


def test_eval_hook_period_zero_only_runs_final(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = _ToyTrainer()
    trainer.max_iter = 5
    evaluator = _CountingEvaluator()
    hook = EvalHook(
        period=0,
        evaluator=evaluator,
        model=_StubModel(),
        data_loader=[[{"image_id": 1}]],
    )
    _drive_hook(hook, trainer, n_iters=5)
    assert evaluator.eval_calls == 1  # after_train only
    assert "[eval @ iter 5]" in capsys.readouterr().out


def test_eval_hook_period_zero_with_after_train_disabled_runs_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = _ToyTrainer()
    trainer.max_iter = 5
    evaluator = _CountingEvaluator()
    hook = EvalHook(
        period=0,
        evaluator=evaluator,
        model=_StubModel(),
        data_loader=[[{"image_id": 1}]],
        eval_after_train=False,
    )
    _drive_hook(hook, trainer, n_iters=5)
    assert evaluator.eval_calls == 0
    assert "[eval @" not in capsys.readouterr().out


def test_eval_hook_rejects_negative_period() -> None:
    with pytest.raises(ValueError, match="period"):
        EvalHook(
            period=-1,
            evaluator=_CountingEvaluator(),
            model=_StubModel(),
            data_loader=[],
        )


def test_eval_hook_does_not_double_eval_at_max_iter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When period divides max_iter, the periodic firing at max_iter
    is suppressed in favour of the after_train firing — otherwise we'd
    log the same eval twice."""
    trainer = _ToyTrainer()
    trainer.max_iter = 6
    evaluator = _CountingEvaluator()
    hook = EvalHook(period=3, evaluator=evaluator, model=_StubModel(), data_loader=[[{}]])
    _drive_hook(hook, trainer, n_iters=6)
    # Periodic fires at next_iter=3 (iter 2). next_iter=6 is suppressed.
    # after_train fires once at iter 6. Total: 2.
    assert evaluator.eval_calls == 2
    out = capsys.readouterr().out
    assert out.count("[eval @ iter 6]") == 1
