"""Index samplers + aspect-ratio bucketing.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §5.2 / §5.6 for the in-scope
loaders. ``RepeatFactorTrainingSampler`` is omitted — it's an LVIS-only
helper and we ship COCO only at v1.

All samplers are picklable and support multi-worker DataLoader use; the
infinite ``TrainingSampler`` re-seeds per ``__iter__`` call so two
workers reading from the same sampler instance see different streams.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.utils.data import Sampler

__all__ = [
    "AspectRatioGroupedDataset",
    "InferenceSampler",
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
