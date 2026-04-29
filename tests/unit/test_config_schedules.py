"""Tests for :mod:`mayaku.config.schedules`."""

from __future__ import annotations

from mayaku.config import schedule_1x, schedule_2x, schedule_3x


def test_schedule_3x_matches_spec_section_6_3() -> None:
    s = schedule_3x()
    assert s.ims_per_batch == 16
    assert s.base_lr == 0.02
    assert s.max_iter == 270_000
    assert s.steps == (210_000, 250_000)
    assert s.warmup_iters == 1000
    assert s.warmup_factor == 1.0 / 1000.0
    assert s.warmup_method == "linear"


def test_schedule_2x_lr_decays_at_two_thirds_and_eight_ninths() -> None:
    s = schedule_2x()
    assert s.max_iter == 180_000
    assert s.steps == (120_000, 160_000)


def test_schedule_1x_baseline_iters() -> None:
    s = schedule_1x()
    assert s.max_iter == 90_000
    assert s.steps == (60_000, 80_000)


def test_schedule_overrides_forwarded() -> None:
    s = schedule_3x(base_lr=0.001, amp_enabled=True)
    assert s.base_lr == 0.001
    assert s.amp_enabled is True
    # Untouched fields remain at the 3x defaults.
    assert s.max_iter == 270_000
