"""Tests for gradient accumulation in :mod:`mayaku.engine.trainer`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mayaku.engine.trainer import SimpleTrainer

# ---------------------------------------------------------------------------
# Test fixtures: a tiny loss-emitting model + a deterministic batch loader
# ---------------------------------------------------------------------------


class _TinyLossModel(nn.Module):
    """Linear regression head emitting a loss-dict (matches mayaku trainer contract)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 1)

    def forward(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        x = torch.cat([b["x"] for b in batch], dim=0)
        y = torch.cat([b["y"] for b in batch], dim=0)
        pred = self.linear(x).squeeze(-1)
        loss = (pred - y).pow(2).mean()
        return {"mse": loss}


def _samples(rng: np.random.Generator, n: int) -> list[dict[str, Tensor]]:
    """Build a list of n single-sample dicts with deterministic data."""
    return [
        {
            "x": torch.from_numpy(rng.standard_normal((1, 8)).astype(np.float32)),
            "y": torch.from_numpy(rng.standard_normal((1,)).astype(np.float32)),
        }
        for _ in range(n)
    ]


class _ListLoader:
    """Cycles over a fixed list of pre-built batches.

    The trainer expects ``data_loader`` to be ``Iterable[Sequence[Mapping]]``
    — i.e. each yield is a list[dict] (one batch). We pre-stage the batches
    to make the test deterministic.
    """

    def __init__(self, batches: list[list[dict[str, Tensor]]]) -> None:
        self.batches = batches

    def __iter__(self) -> Iterator[list[dict[str, Tensor]]]:
        return iter(self.batches)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_grad_accum_raises() -> None:
    model = _TinyLossModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.0)
    loader = _ListLoader([_samples(np.random.default_rng(0), 2)])
    with pytest.raises(ValueError, match="grad_accum_steps"):
        SimpleTrainer(model, loader, optim, grad_accum_steps=0)
    with pytest.raises(ValueError, match="grad_accum_steps"):
        SimpleTrainer(model, loader, optim, grad_accum_steps=-1)


# ---------------------------------------------------------------------------
# Numerical equivalence — the core correctness claim
# ---------------------------------------------------------------------------


def test_grad_accum_matches_single_batch() -> None:
    """``grad_accum_steps=k`` over micro-batches of N yields the same gradient
    as ``grad_accum_steps=1`` over a single batch of ``k * N``.

    This is THE correctness test for gradient accumulation. If it fails,
    the loss/N scaling is wrong, the zero_grad is in the wrong place, or
    something else fundamental is broken.
    """
    rng = np.random.default_rng(0)
    micro_size = 2
    accum = 4
    full_size = micro_size * accum

    # Pre-compute the SAME data for both runs.
    full_micro_batches = [_samples(rng, micro_size) for _ in range(accum)]
    full_single_batch = [s for batch in full_micro_batches for s in batch]
    assert len(full_single_batch) == full_size

    # Both models start from the same parameters.
    torch.manual_seed(7)
    model_accum = _TinyLossModel()
    model_single = _TinyLossModel()
    model_single.load_state_dict(model_accum.state_dict())

    # --- Path A: gradient accumulation through the trainer
    optim_a = torch.optim.SGD(model_accum.parameters(), lr=0.0)  # lr=0 → step is a no-op
    loader_a = _ListLoader(full_micro_batches)
    trainer_a = SimpleTrainer(model_accum, loader_a, optim_a, grad_accum_steps=accum)
    trainer_a.run_step()
    grads_a = {n: p.grad.detach().clone() for n, p in model_accum.named_parameters()}

    # --- Path B: single big batch, grad_accum_steps=1
    optim_b = torch.optim.SGD(model_single.parameters(), lr=0.0)
    loader_b = _ListLoader([full_single_batch])
    trainer_b = SimpleTrainer(model_single, loader_b, optim_b, grad_accum_steps=1)
    trainer_b.run_step()
    grads_b = {n: p.grad.detach().clone() for n, p in model_single.named_parameters()}

    # Gradients must match within fp tolerance.
    assert set(grads_a.keys()) == set(grads_b.keys())
    for k in grads_a:
        torch.testing.assert_close(grads_a[k], grads_b[k], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Step count / hook semantics
# ---------------------------------------------------------------------------


def test_optimizer_step_called_once_per_run_step() -> None:
    """``optimizer.step()`` fires once per ``run_step``, regardless of accum_steps."""
    rng = np.random.default_rng(1)
    model = _TinyLossModel()
    micro_batches = [_samples(rng, 2) for _ in range(5)]
    loader = _ListLoader(micro_batches)

    base = torch.optim.SGD(model.parameters(), lr=0.0)
    step_calls = 0
    real_step = base.step

    def counting_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_calls
        step_calls += 1
        return real_step(*args, **kwargs)

    base.step = counting_step  # type: ignore[method-assign]

    trainer = SimpleTrainer(model, loader, base, grad_accum_steps=4)
    trainer.run_step()  # one effective iter, should consume 4 micro-batches
    assert step_calls == 1


def test_loss_record_is_mean_across_micro_batches() -> None:
    """``trainer.last_loss`` and ``trainer.storage`` reflect the mean micro-batch loss."""
    rng = np.random.default_rng(2)
    model = _TinyLossModel()
    micro_batches = [_samples(rng, 2) for _ in range(3)]
    loader = _ListLoader(micro_batches)
    optim = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = SimpleTrainer(model, loader, optim, grad_accum_steps=3)

    # Pre-compute the expected per-micro-batch losses with no_grad so we
    # don't perturb autograd state, then take the mean.
    with torch.no_grad():
        per_batch_losses = []
        for batch in micro_batches:
            out = model(batch)
            per_batch_losses.append(float(out["mse"].item()))
    expected_mean = sum(per_batch_losses) / len(per_batch_losses)

    trainer.run_step()
    assert trainer.last_loss == pytest.approx(expected_mean, rel=1e-5, abs=1e-6)
    assert trainer.storage["mse"] == pytest.approx(expected_mean, rel=1e-5, abs=1e-6)
    assert trainer.storage["total_loss"] == pytest.approx(expected_mean, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# Default behaviour (no accumulation) — backwards compatibility
# ---------------------------------------------------------------------------


def test_default_grad_accum_steps_is_one() -> None:
    """Without specifying grad_accum_steps, the trainer behaves identically to before."""
    rng = np.random.default_rng(3)
    model = _TinyLossModel()
    loader = _ListLoader([_samples(rng, 4)])
    optim = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = SimpleTrainer(model, loader, optim)
    assert trainer.grad_accum_steps == 1
    trainer.run_step()  # should consume exactly 1 batch
