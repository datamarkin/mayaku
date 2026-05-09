"""Tests for :mod:`mayaku.data.samplers` and :mod:`mayaku.data.collate`."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest
import torch

from mayaku.data import (
    AspectRatioGroupedDataset,
    InferenceSampler,
    RepeatFactorTrainingSampler,
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
# RepeatFactorTrainingSampler
# ---------------------------------------------------------------------------


def _dd(cat_ids: list[int]) -> dict[str, Any]:
    """Compact dataset-dict factory: one annotation per cat_id."""
    return {"annotations": [{"category_id": c} for c in cat_ids]}


def test_rfs_repeat_factors_match_lvis_formula() -> None:
    # 10 images: 5 contain class 0, 1 contains class 1 (rare).
    # f_0 = 0.5, f_1 = 0.1, t = 0.1.
    # r_0 = max(1, sqrt(0.1/0.5)) = 1.0; r_1 = max(1, sqrt(0.1/0.1)) = 1.0.
    # Bump t to 0.4: r_0 = sqrt(0.4/0.5) ≈ 0.894 → 1.0; r_1 = sqrt(0.4/0.1) = 2.0.
    dicts: list[dict[str, Any]] = []
    for _ in range(5):
        dicts.append(_dd([0]))
    dicts.append(_dd([1]))  # rare image
    for _ in range(4):
        dicts.append(_dd([0]))
    rep = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dicts, repeat_thresh=0.4
    )
    assert rep.shape == (10,)
    # Image 5 contains rare class → r_i = 2.0; everyone else → r_i = 1.0.
    assert rep[5].item() == pytest.approx(2.0, rel=1e-5)
    for i in (0, 1, 2, 3, 4, 6, 7, 8, 9):
        assert rep[i].item() == pytest.approx(1.0, rel=1e-5)


def test_rfs_per_image_factor_is_max_over_classes() -> None:
    # 4 images:
    #   [0]      common
    #   [1]      rare
    #   [0, 1]   common + rare → should take rare's r
    #   [0]      common
    dicts = [_dd([0]), _dd([1]), _dd([0, 1]), _dd([0])]
    rep = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dicts, repeat_thresh=0.5
    )
    # f_0 = 3/4 = 0.75 → r_0 = max(1, sqrt(0.5/0.75)) = 1.0
    # f_1 = 2/4 = 0.5  → r_1 = max(1, sqrt(0.5/0.5)) = 1.0
    # Both saturate to 1.0; bump t to make rare actually rare.
    rep = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dicts, repeat_thresh=2.0
    )
    # f_0 = 0.75 → r_0 = sqrt(2/0.75) ≈ 1.633
    # f_1 = 0.5  → r_1 = sqrt(2/0.5) = 2.0
    expected_r0 = math.sqrt(2.0 / 0.75)
    expected_r1 = 2.0
    assert rep[0].item() == pytest.approx(expected_r0, rel=1e-5)
    assert rep[1].item() == pytest.approx(expected_r1, rel=1e-5)
    assert rep[2].item() == pytest.approx(max(expected_r0, expected_r1), rel=1e-5)
    assert rep[3].item() == pytest.approx(expected_r0, rel=1e-5)


def test_rfs_empty_annotations_get_unit_factor() -> None:
    dicts = [_dd([0]), {"annotations": []}, _dd([1])]
    rep = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dicts, repeat_thresh=1.0
    )
    assert rep[1].item() == pytest.approx(1.0, rel=1e-5)


def test_rfs_iterator_oversamples_rare_images() -> None:
    # 9 common + 1 rare; with t=0.5 the rare image r_i = sqrt(0.5/0.1) ≈ 2.236.
    dicts = [_dd([0]) for _ in range(9)] + [_dd([1])]
    rep = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dicts, repeat_thresh=0.5
    )
    s = RepeatFactorTrainingSampler(rep, seed=0)
    # Pull a large window so per-epoch frac noise averages out.
    n_samples = 100_000
    counts = [0] * 10
    for i, idx in enumerate(s):
        if i >= n_samples:
            break
        counts[idx] += 1
    # Expected probability of rare image ≈ r_rare / (9 * r_common + r_rare)
    # = 2.236 / (9 + 2.236) ≈ 0.199.
    rare_frac = counts[9] / n_samples
    assert 0.16 < rare_frac < 0.24


def test_rfs_seed_reproducible() -> None:
    rep = torch.tensor([1.0, 2.0, 1.5, 3.0], dtype=torch.float32)
    a = RepeatFactorTrainingSampler(rep, seed=7)
    b = RepeatFactorTrainingSampler(rep, seed=7)
    assert _take(iter(a), 200) == _take(iter(b), 200)


def test_rfs_per_rank_disjoint_streams() -> None:
    rep = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
    a = RepeatFactorTrainingSampler(rep, seed=0, num_replicas=2, rank=0)
    b = RepeatFactorTrainingSampler(rep, seed=0, num_replicas=2, rank=1)
    seg_a = _take(iter(a), 64)
    seg_b = _take(iter(b), 64)
    # Two ranks slice the same shuffled epoch at offsets 0 and 1 — the
    # streams are deterministic but element-wise distinct.
    assert seg_a != seg_b
    # Every yielded index is in range.
    assert all(0 <= i < 6 for i in seg_a + seg_b)


def test_rfs_validates_inputs() -> None:
    with pytest.raises(ValueError, match="1-D"):
        RepeatFactorTrainingSampler(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="empty"):
        RepeatFactorTrainingSampler(torch.zeros(0))
    with pytest.raises(ValueError, match="non-negative"):
        RepeatFactorTrainingSampler(torch.tensor([1.0, -0.5]))
    with pytest.raises(ValueError, match="rank"):
        RepeatFactorTrainingSampler(torch.tensor([1.0]), rank=2, num_replicas=2)
    with pytest.raises(ValueError, match="repeat_thresh"):
        RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            [_dd([0])], repeat_thresh=0.0
        )
    with pytest.raises(ValueError, match="empty"):
        RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            [], repeat_thresh=0.001
        )


def test_rfs_len_unsupported() -> None:
    s = RepeatFactorTrainingSampler(torch.tensor([1.0, 2.0]))
    with pytest.raises(TypeError, match="infinite"):
        len(s)


def test_rfs_all_unit_factors_yields_every_index_per_epoch() -> None:
    # When every image has r=1, RFS reduces to a shuffled epoch sampler.
    rep = torch.ones(8, dtype=torch.float32)
    s = RepeatFactorTrainingSampler(rep, seed=0)
    seen = set(_take(iter(s), 8))
    assert seen == set(range(8))


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
