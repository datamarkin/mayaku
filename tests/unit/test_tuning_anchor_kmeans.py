"""Tests for :mod:`mayaku.tuning.anchor_kmeans`."""

from __future__ import annotations

import itertools

import pytest

from mayaku.tuning.anchor_kmeans import cluster_aspect_ratios, cluster_sizes


def test_cluster_sizes_returns_k_sorted_ints() -> None:
    values = [10.0, 11.0, 12.0, 50.0, 51.0, 100.0, 101.0, 200.0, 500.0, 510.0]
    out = cluster_sizes(values, k=5)
    assert len(out) == 5
    assert list(out) == sorted(out)
    assert all(isinstance(s, int) for s in out)


def test_cluster_sizes_is_deterministic() -> None:
    # Quantile init makes this stable run-to-run.
    values = [float(v) for v in range(1, 200)]
    a = cluster_sizes(values, k=5)
    b = cluster_sizes(values, k=5)
    assert a == b


def test_cluster_sizes_enforces_strict_ascending_after_rounding() -> None:
    # Heavy-tail input where rounding could collapse two centers into
    # the same int. The function bumps duplicates up by 1 to keep
    # anchor levels distinct.
    values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    out = cluster_sizes(values, k=5)
    assert len(out) == 5
    assert all(b > a for a, b in itertools.pairwise(out))


def test_cluster_sizes_raises_below_k_samples() -> None:
    with pytest.raises(ValueError, match="k-means"):
        cluster_sizes([1.0, 2.0], k=5)


def test_cluster_sizes_finds_known_clusters() -> None:
    # Three tight clusters at 32, 128, 512; k=3 should recover them.
    values = [30, 31, 32, 33, 34, 126, 127, 128, 129, 130, 510, 511, 512, 513, 514]
    out = cluster_sizes([float(v) for v in values], k=3)
    # Round-trip stability check: each output cluster should be within
    # 5 px of one of the seed centers.
    seeds = [32, 128, 512]
    for c, seed in zip(out, seeds, strict=True):
        assert abs(c - seed) < 5


def test_cluster_aspect_ratios_dedupes_and_orders() -> None:
    values = [0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    out = cluster_aspect_ratios(values, k=3)
    assert len(out) == 3
    assert list(out) == sorted(out)
    # Values rounded to 2 decimal places.
    assert all(round(v, 2) == v for v in out)


def test_cluster_aspect_ratios_clamps_extreme_values() -> None:
    # Pathological labels (extreme AR) get clamped into [0.1, 10] so
    # the resulting anchors stay in a sane physical range.
    values = [0.001, 0.001, 1.0, 1.0, 1000.0, 1000.0]
    out = cluster_aspect_ratios(values, k=3)
    assert min(out) >= 0.1
    assert max(out) <= 10.0


def test_cluster_aspect_ratios_pads_when_dedupe_collapses() -> None:
    # All inputs identical; the function still returns k distinct centers.
    values = [1.0] * 20
    out = cluster_aspect_ratios(values, k=3)
    assert len(out) == 3
    assert len(set(out)) == 3
