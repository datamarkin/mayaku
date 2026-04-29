"""Tests for :mod:`mayaku.data.samplers` and :mod:`mayaku.data.collate`."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
import torch

from mayaku.data import (
    AspectRatioGroupedDataset,
    InferenceSampler,
    TrainingSampler,
    trivial_batch_collator,
)


def _take(it, n: int) -> list[int]:  # type: ignore[no-untyped-def]
    return list(itertools.islice(it, n))


# ---------------------------------------------------------------------------
# TrainingSampler
# ---------------------------------------------------------------------------


def test_training_sampler_yields_every_index_within_one_epoch() -> None:
    s = TrainingSampler(size=10, shuffle=True, seed=0)
    seen = set(_take(iter(s), 10))
    assert seen == set(range(10))


def test_training_sampler_per_rank_disjoint_streams() -> None:
    a = TrainingSampler(size=8, shuffle=True, seed=0, num_replicas=2, rank=0)
    b = TrainingSampler(size=8, shuffle=True, seed=0, num_replicas=2, rank=1)
    # Two different ranks see different first-128 element subsets.
    seg_a = _take(iter(a), 128)
    seg_b = _take(iter(b), 128)
    # Within one shared permutation each rank samples every-other index;
    # so seg_a and seg_b should not be element-wise equal.
    assert seg_a != seg_b


def test_training_sampler_seed_reproducible() -> None:
    a = TrainingSampler(size=5, shuffle=True, seed=42)
    b = TrainingSampler(size=5, shuffle=True, seed=42)
    assert _take(iter(a), 25) == _take(iter(b), 25)


def test_training_sampler_no_shuffle_yields_in_order() -> None:
    s = TrainingSampler(size=4, shuffle=False, seed=0)
    assert _take(iter(s), 8) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_training_sampler_validates_inputs() -> None:
    with pytest.raises(ValueError, match="size > 0"):
        TrainingSampler(size=0)
    with pytest.raises(ValueError, match="rank"):
        TrainingSampler(size=4, rank=2, num_replicas=2)


def test_training_sampler_len_unsupported() -> None:
    s = TrainingSampler(size=4)
    with pytest.raises(TypeError, match="infinite"):
        len(s)


# ---------------------------------------------------------------------------
# InferenceSampler
# ---------------------------------------------------------------------------


def test_inference_sampler_covers_every_index_exactly_once_across_ranks() -> None:
    size = 10
    rank0 = list(InferenceSampler(size, num_replicas=3, rank=0))
    rank1 = list(InferenceSampler(size, num_replicas=3, rank=1))
    rank2 = list(InferenceSampler(size, num_replicas=3, rank=2))
    union = rank0 + rank1 + rank2
    assert sorted(union) == list(range(size))
    assert len(union) == size  # no duplicates


def test_inference_sampler_len_matches_iter() -> None:
    s = InferenceSampler(size=10, num_replicas=3, rank=1)
    assert len(s) == sum(1 for _ in s)


def test_inference_sampler_size_zero_is_empty() -> None:
    s = InferenceSampler(size=0)
    assert list(s) == []
    assert len(s) == 0


def test_inference_sampler_validates_inputs() -> None:
    with pytest.raises(ValueError, match="size >= 0"):
        InferenceSampler(size=-1)
    with pytest.raises(ValueError, match="rank"):
        InferenceSampler(size=4, rank=3, num_replicas=3)


# ---------------------------------------------------------------------------
# AspectRatioGroupedDataset
# ---------------------------------------------------------------------------


def _make_sample(h: int, w: int, idx: int) -> dict[str, Any]:
    return {"image": torch.zeros(3, h, w), "image_id": idx}


def test_aspect_ratio_grouped_buckets_by_orientation() -> None:
    samples = [
        _make_sample(20, 30, 0),  # landscape  → bucket 0
        _make_sample(40, 20, 1),  # portrait   → bucket 1
        _make_sample(20, 30, 2),  # landscape  → bucket 0 (full → emit)
        _make_sample(50, 25, 3),  # portrait   → bucket 1 (full → emit)
    ]
    out = list(AspectRatioGroupedDataset(samples, batch_size=2))
    assert len(out) == 2
    landscapes = [b for b in out if b[0]["image_id"] in (0, 2)]
    portraits = [b for b in out if b[0]["image_id"] in (1, 3)]
    assert len(landscapes) == 1 and len(portraits) == 1
    assert {s["image_id"] for s in landscapes[0]} == {0, 2}
    assert {s["image_id"] for s in portraits[0]} == {1, 3}


def test_aspect_ratio_partial_bucket_not_emitted() -> None:
    # Three landscape samples with batch=2 → one batch yielded, one
    # sample left in the bucket.
    samples = [_make_sample(20, 30, i) for i in range(3)]
    batches = list(AspectRatioGroupedDataset(samples, batch_size=2))
    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_aspect_ratio_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        AspectRatioGroupedDataset([], batch_size=0)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def test_trivial_batch_collator_returns_list() -> None:
    batch = [{"a": 1}, {"a": 2}]
    out = trivial_batch_collator(batch)
    assert isinstance(out, list)
    assert out == batch
