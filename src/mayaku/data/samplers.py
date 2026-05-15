"""Index samplers + aspect-ratio bucketing.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §5.2 / §5.6 for the in-scope
loaders. :class:`RepeatFactorTrainingSampler` implements Gupta et al.
(LVIS, 2019) frequency-based oversampling — useful for any imbalanced
detection dataset, not just LVIS. End users with skewed custom
datasets (e.g. 1000 common-class images, 30 rare-class images) get
the largest gain.

All samplers are picklable and support multi-worker DataLoader use; the
infinite ``TrainingSampler`` re-seeds per ``__iter__`` call so two
workers reading from the same sampler instance see different streams.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler

__all__ = [
    "AspectRatioGroupedDataset",
    "InferenceSampler",
    "RepeatFactorTrainingSampler",
    "TrainingSampler",
]


class TrainingSampler(Sampler[int]):
    """Infinite shuffled stream, sliced per rank (`spec §5.6`).

    Each call to ``__iter__`` produces an endless iterator yielding
    ``num_replicas``-spaced indices from a freshly shuffled
    permutation; when the permutation is exhausted, a new one is drawn.
    The ``rank``-th sample of every batch comes from rank ``rank``.

    The reference Detectron2 sampler reseeds via
    ``comm.shared_random_seed()`` so all DDP ranks generate the same
    permutation; we accept the seed explicitly for testability and
    leave DDP coordination to Step 14.
    """

    def __init__(
        self,
        size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if size <= 0:
            raise ValueError(f"TrainingSampler requires size > 0; got {size}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} must be in [0, num_replicas={num_replicas})")
        self.size = size
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        rank = self.rank
        stride = self.num_replicas
        size = self.size
        while True:
            perm = (
                torch.randperm(size, generator=gen).tolist() if self.shuffle else list(range(size))
            )
            for i in range(rank, len(perm), stride):
                yield perm[i]

    def __len__(self) -> int:  # pragma: no cover - infinite
        # PyTorch's Sampler doesn't strictly require __len__ for infinite
        # samplers, and DataLoader handles it via batch_sampler. Raise so
        # any caller depending on a finite length sees a clear error.
        raise TypeError("TrainingSampler is infinite; len() is not defined")


class InferenceSampler(Sampler[int]):
    """Deterministic ordered slice — every sample evaluated exactly once.

    Rank ``r`` of ``num_replicas`` reads ``indices[r::num_replicas]``,
    matching `spec §5.6`. The order is the input order so evaluation
    output ordering is reproducible.
    """

    def __init__(self, size: int, *, num_replicas: int = 1, rank: int = 0) -> None:
        if size < 0:
            raise ValueError(f"InferenceSampler requires size >= 0; got {size}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} must be in [0, num_replicas={num_replicas})")
        self.size = size
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        yield from range(self.rank, self.size, self.num_replicas)

    def __len__(self) -> int:
        # Number of indices this rank will yield.
        return (self.size - self.rank + self.num_replicas - 1) // self.num_replicas


class RepeatFactorTrainingSampler(Sampler[int]):
    """Frequency-balanced infinite sampler (Gupta et al. LVIS 2019, `spec §5.6`).

    Per epoch each image is repeated roughly ``r_i`` times, where
    ``r_i = max_{c in image_i} max(1, sqrt(t / f_c))`` and ``f_c`` is
    the fraction of training images that contain class ``c``. Images
    containing rare classes get oversampled; common-class images stay
    at ~1×. The fractional part of ``r_i`` is realised stochastically
    so the expected number of samples per epoch matches ``r_i``.

    Use :meth:`repeat_factors_from_category_frequency` to compute
    ``repeat_factors`` from a list of dataset dicts; pass the resulting
    tensor in. The two-step API lets callers cache or persist the
    factor tensor between runs and inspect the per-image distribution.

    DDP-aware via ``num_replicas``/``rank`` — each rank slices the
    same shuffled epoch ``rank::num_replicas``.
    """

    def __init__(
        self,
        repeat_factors: torch.Tensor,
        *,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if repeat_factors.ndim != 1:
            raise ValueError(f"repeat_factors must be 1-D; got shape {tuple(repeat_factors.shape)}")
        if repeat_factors.numel() == 0:
            raise ValueError("repeat_factors is empty; nothing to sample")
        if (repeat_factors < 0).any():
            raise ValueError("repeat_factors must be non-negative")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank={rank} must be in [0, num_replicas={num_replicas})")
        # Cache integer + fractional parts for the per-epoch realisation.
        self._int_part = repeat_factors.floor()
        self._frac_part = repeat_factors - self._int_part
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank

    @staticmethod
    def repeat_factors_from_category_frequency(
        dataset_dicts: Sequence[Mapping[str, Any]],
        repeat_thresh: float,
    ) -> torch.Tensor:
        """Compute per-image repeat factors from category frequencies.

        The threshold ``t`` controls how aggressive the oversampling is:
        classes appearing in a fraction of images at or above ``t``
        get ``r_c = 1`` (no oversampling); rarer classes get ``r_c =
        sqrt(t / f_c)``. The LVIS default ``t = 0.001`` is a reasonable
        starting point; raise it for more aggressive balancing on
        extremely imbalanced data.
        """
        if repeat_thresh <= 0.0:
            raise ValueError(f"repeat_thresh must be > 0; got {repeat_thresh}")
        if not dataset_dicts:
            raise ValueError("dataset_dicts is empty")

        # Step 1: per-category image-level frequency.
        category_count: dict[Any, int] = defaultdict(int)
        for d in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in d.get("annotations", ())}
            for cat_id in cat_ids:
                category_count[cat_id] += 1
        num_images = len(dataset_dicts)
        category_freq = {c: n / num_images for c, n in category_count.items()}

        # Step 2: per-category repeat factor.
        category_rep = {c: max(1.0, math.sqrt(repeat_thresh / f)) for c, f in category_freq.items()}

        # Step 3: per-image repeat factor — max over its labels.
        # Empty-annotation images get r=1.0 (sample at base rate); they
        # are normally filtered upstream but the safe default keeps the
        # sampler working when filter_empty_annotations=False.
        rep_factors: list[float] = []
        for d in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in d.get("annotations", ())}
            if not cat_ids:
                rep_factors.append(1.0)
            else:
                rep_factors.append(max(category_rep[c] for c in cat_ids))
        return torch.tensor(rep_factors, dtype=torch.float32)

    def _epoch_indices(self, generator: torch.Generator) -> list[int]:
        # Realise fractional parts: r_i = floor(r_i) + 1[u < frac(r_i)].
        rands = torch.rand(self._frac_part.shape, generator=generator)
        rep = self._int_part + (rands < self._frac_part).to(self._int_part.dtype)
        # Build the multiplied index list. ``repeat_interleave`` is the
        # vectorised equivalent of ``[idx]*int(rep) for idx, rep in ...``
        # and avoids a Python-level loop over potentially 10^6 images.
        idx = torch.arange(rep.numel(), dtype=torch.int64)
        indices = idx.repeat_interleave(rep.to(torch.int64))
        if indices.numel() == 0:
            # Pathological all-zero realisation; fall back to one pass
            # so the sampler doesn't infinite-loop on an empty epoch.
            indices = idx
        perm = torch.randperm(indices.numel(), generator=generator)
        return indices[perm].tolist()

    def __iter__(self) -> Iterator[int]:
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        rank = self.rank
        stride = self.num_replicas
        while True:
            indices = self._epoch_indices(gen)
            for i in range(rank, len(indices), stride):
                yield indices[i]

    def __len__(self) -> int:  # pragma: no cover - infinite
        raise TypeError("RepeatFactorTrainingSampler is infinite; len() is not defined")


class AspectRatioGroupedDataset:
    """Bucket samples by ``w >= h`` vs ``w < h`` and emit fixed-size batches.

    Wraps an iterable of dataset dicts (post-mapper, so each carries a
    ``"image"`` tensor with ``shape=(C, H, W)`` and an ``"image_size"``-
    equivalent we can read from the tensor). Two buckets of size
    ``batch_size`` are kept; whichever fills first is yielded.

    This avoids the 2x memory blow-up from padding mixed-orientation
    samples to a common square (`spec §5.2`). Detectron2's reference
    keys on ``dd["width"] > dd["height"]``; we use the same trigger but
    read the width/height off the rendered image tensor for robustness
    (the dict's ``"width"``/``"height"`` reflect the *original* image,
    pre-resize).
    """

    def __init__(self, dataset: Iterable[dict[str, Any]], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        buckets: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
        for sample in self.dataset:
            tensor = sample["image"]
            # tensor is (C, H, W) post-mapper.
            h, w = int(tensor.shape[1]), int(tensor.shape[2])
            bucket = buckets[0 if w >= h else 1]
            bucket.append(sample)
            if len(bucket) == self.batch_size:
                yield list(bucket)
                bucket.clear()
