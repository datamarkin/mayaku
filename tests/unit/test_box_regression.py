"""Tests for :mod:`mayaku.models.proposals.box_regression`."""

from __future__ import annotations

import math

import torch

from mayaku.models.proposals.box_regression import Box2BoxTransform


def test_get_apply_round_trip_identity_with_unit_weights() -> None:
    bt = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
    src = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 25.0, 15.0]])
    target = torch.tensor([[1.0, 2.0, 11.0, 12.0], [4.0, 5.0, 28.0, 18.0]])
    deltas = bt.get_deltas(src, target)
    decoded = bt.apply_deltas(deltas, src)
    torch.testing.assert_close(decoded, target, atol=1e-5, rtol=1e-5)


def test_round_trip_with_box_head_weights() -> None:
    # The box head uses (10, 10, 5, 5) — round-trip must still recover the target.
    bt = Box2BoxTransform(weights=(10.0, 10.0, 5.0, 5.0))
    src = torch.tensor([[100.0, 100.0, 200.0, 250.0]])
    target = torch.tensor([[105.0, 110.0, 195.0, 240.0]])
    deltas = bt.get_deltas(src, target)
    decoded = bt.apply_deltas(deltas, src)
    torch.testing.assert_close(decoded, target, atol=1e-5, rtol=1e-5)


def test_apply_clamps_extreme_dw_dh() -> None:
    bt = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
    src = torch.tensor([[0.0, 0.0, 16.0, 16.0]])
    # Wildly large deltas: dw, dh = 50 → exp(50) overflow without clamp.
    deltas = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    decoded = bt.apply_deltas(deltas, src)
    width = decoded[0, 2] - decoded[0, 0]
    # Clamp = ln(1000/16); decoded width = 16 * 1000/16 = 1000.
    expected = math.exp(math.log(1000.0 / 16.0)) * 16.0
    torch.testing.assert_close(width, torch.tensor(expected), atol=1e-3, rtol=1e-3)


def test_apply_per_class_form() -> None:
    # Pass (N, K*4) deltas — output has the same per-class layout.
    bt = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
    src = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    # Two classes, both deltas zero → both decoded boxes equal src.
    deltas = torch.zeros(1, 8)
    decoded = bt.apply_deltas(deltas, src)
    assert decoded.shape == (1, 8)
    torch.testing.assert_close(decoded[0, :4], src[0])
    torch.testing.assert_close(decoded[0, 4:], src[0])


def test_apply_casts_to_float32_under_fp16_input() -> None:
    bt = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
    src = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    deltas = torch.zeros(1, 4, dtype=torch.float16)
    decoded = bt.apply_deltas(deltas, src)
    assert decoded.dtype == torch.float32
