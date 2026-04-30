"""Tests for :mod:`mayaku.engine.ema`."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from mayaku.engine.ema import EMAHook, ModelEMA


def _tiny_model() -> nn.Module:
    """A 2-layer MLP — small enough for fast tests, still has BN to exercise buffers."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_ema_initialised_to_live_state() -> None:
    """At construction, EMA shadow is a deep copy of the live model."""
    torch.manual_seed(0)
    model = _tiny_model()
    ema = ModelEMA(model)

    live_sd = model.state_dict()
    ema_sd = ema.state_dict()
    assert set(live_sd) == set(ema_sd)
    for k in live_sd:
        torch.testing.assert_close(live_sd[k], ema_sd[k])


def test_ema_shadow_is_independent_of_live() -> None:
    """Mutating the live model should NOT mutate the EMA shadow."""
    torch.manual_seed(0)
    model = _tiny_model()
    # Set every parameter to a clearly-non-default value before constructing
    # the EMA so the post-construction divergence is detectable.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    ema = ModelEMA(model)

    # Mutate live params without calling update().
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)

    # Compare the parameter-only subset (buffers like BN's running_mean
    # weren't touched, so they'd match by coincidence and aren't a fair
    # check of "EMA shadow is independent").
    for live_p, ema_p in zip(list(model.parameters()), list(ema.shadow.parameters()), strict=True):
        assert not torch.allclose(live_p, ema_p)


def test_ema_parameters_have_no_grad() -> None:
    model = _tiny_model()
    ema = ModelEMA(model)
    for p in ema.parameters():
        assert p.requires_grad is False


# ---------------------------------------------------------------------------
# Decay schedule
# ---------------------------------------------------------------------------


def test_decay_at_zero_is_zero() -> None:
    ema = ModelEMA(_tiny_model())
    assert ema.decay_at(0) == 0.0


def test_decay_asymptotes_to_decay_value() -> None:
    """As step → ∞ the warmup-aware decay → asymptotic decay."""
    ema = ModelEMA(_tiny_model(), decay=0.9999, tau=2000.0)
    big_step = 50_000
    assert math.isclose(ema.decay_at(big_step), 0.9999, abs_tol=1e-9)


def test_decay_warms_up() -> None:
    """At t=tau the decay should be ≈ decay * (1 - 1/e) ≈ 63.2% of asymptote."""
    decay = 0.9999
    tau = 2000.0
    ema = ModelEMA(_tiny_model(), decay=decay, tau=tau)
    expected = decay * (1.0 - math.exp(-1.0))
    assert math.isclose(ema.decay_at(int(tau)), expected, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Update behaviour
# ---------------------------------------------------------------------------


def test_first_update_with_warmup_copies_live() -> None:
    """After 1 update the EMA decay is ~tiny (warmup), so EMA tracks live closely."""
    torch.manual_seed(0)
    model = _tiny_model()
    ema = ModelEMA(model, decay=0.9999, tau=2000.0)

    # Mutate live weights, then update.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    ema.update(model)

    # Decay at update 1 with tau=2000 ≈ 0.9999 * (1 - exp(-1/2000)) ≈ 5e-4.
    # So EMA ≈ (1 - 5e-4) * live + 5e-4 * old_init ≈ live.
    live_sd = model.state_dict()
    ema_sd = ema.state_dict()
    for k, live_v in live_sd.items():
        if live_v.dtype.is_floating_point:
            torch.testing.assert_close(ema_sd[k], live_v, atol=2e-3, rtol=0)


def test_many_updates_converge_toward_live_when_live_is_steady() -> None:
    """If live weights stay constant, EMA converges to them over time."""
    torch.manual_seed(0)
    model = _tiny_model()
    ema = ModelEMA(model, decay=0.99, tau=10.0)
    target = 7.0
    # Set BOTH parameters and float buffers to `target` so every entry of
    # the state_dict has the same expected limit.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(target)
        for b in model.buffers():
            if b.dtype.is_floating_point:
                b.fill_(target)
    for _ in range(5000):
        ema.update(model)

    for v in ema.state_dict().values():
        if v.dtype.is_floating_point:
            torch.testing.assert_close(v, torch.full_like(v, target), atol=1e-3, rtol=0)


def test_buffers_with_int_dtype_are_copied_not_blended() -> None:
    """``num_batches_tracked`` is int — it should mirror live exactly, not blend."""
    torch.manual_seed(0)
    model = _tiny_model()
    ema = ModelEMA(model)

    # Force the BN's num_batches_tracked to a non-zero value on the live model.
    bn = model[1]
    assert isinstance(bn, nn.BatchNorm1d)
    bn.num_batches_tracked.fill_(42)

    ema.update(model)
    ema_bn = ema.shadow[1]
    assert isinstance(ema_bn, nn.BatchNorm1d)
    assert int(ema_bn.num_batches_tracked.item()) == 42


# ---------------------------------------------------------------------------
# EMAHook
# ---------------------------------------------------------------------------


def test_ema_hook_increments_updates_per_step() -> None:
    model = _tiny_model()
    ema = ModelEMA(model)
    hook = EMAHook(ema, model)

    assert ema.updates == 0
    hook.after_step()
    assert ema.updates == 1
    hook.after_step()
    assert ema.updates == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_decay_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(_tiny_model(), decay=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_tau_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        ModelEMA(_tiny_model(), tau=bad)


def test_invalid_updates_raises() -> None:
    with pytest.raises(ValueError, match="updates"):
        ModelEMA(_tiny_model(), updates=-5)
