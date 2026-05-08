"""Memory-efficient list of pickleable items.

For long training runs, holding the full ``dataset_dicts`` as a Python
``list[dict]`` is expensive in two ways:

1. **Resident memory.** COCO 2017 train carries 860k annotations; even
   bbox-only fields cost ~3 GB of small Python objects (each float is
   28 bytes, each list / dict adds header overhead). Mask polygons push
   that to ~5 GB.

2. **Heap fragmentation.** :class:`mayaku.data.mapper.DatasetMapper`
   deepcopies one dict per ``__getitem__`` (mapper.py:110) for tens of
   thousands of training steps. glibc malloc almost never returns
   fragmented memory back to the OS, so a working set of ~6 GB drifts
   into a resident-set of 60 GB+ over a 90k-iter run.

This module mirrors Detectron2's
``DatasetFromList(serialize=True)`` trick: pickle each dict once, store
all bytes in a single contiguous ``bytes`` buffer, and decode lazily on
``__getitem__``. The result:

* ~10× smaller resident-set than the equivalent Python list of dicts
  (one ``bytes`` allocation instead of millions of small objects).
* No drift from malloc fragmentation — the buffer is one allocation
  that lives for the whole run.
* Decoded dicts are fresh on every access (``pickle.loads`` allocates
  a new object), so callers no longer strictly need ``copy.deepcopy``
  before mutating. The mapper still deepcopies for safety; the cost is
  trivial on the small post-decode dict.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt

__all__ = ["SerializedList"]

T = TypeVar("T")


class SerializedList(Sequence[T], Generic[T]):
    """List-style accessor backed by a single pickled-bytes buffer.

    Implements :class:`collections.abc.Sequence` so it drops in anywhere
    the data layer asks for a ``Sequence[dict[str, Any]]`` (see
    :class:`mayaku.cli.train._MappedList` and
    :class:`mayaku.data.MultiSampleMappedDataset`).

    Args:
        items: Any sequence of pickleable items. Pickled eagerly during
            construction; the input sequence is no longer referenced
            after ``__init__`` returns, so callers can ``del`` their
            list and free the original Python objects.
    """

    def __init__(self, items: Sequence[T]) -> None:
        encoded: list[bytes] = [pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL) for x in items]
        sizes: npt.NDArray[np.int64] = np.asarray([len(b) for b in encoded], dtype=np.int64)
        # Offsets table: start byte of item ``i`` in the buffer.
        # ``np.cumsum([0] + sizes[:-1])`` would underflow on the empty
        # case; the concat-then-cumsum form below handles len==0 cleanly.
        if sizes.size > 0:
            offsets = np.cumsum(np.concatenate(([0], sizes)))[:-1]
        else:
            offsets = np.zeros(0, dtype=np.int64)
        self._sizes = sizes
        self._offsets = offsets
        # One contiguous bytes object — survives for the lifetime of
        # ``self``. ``b"".join`` allocates once; the encoded list goes
        # out of scope at the end of __init__ and the per-item bytes
        # objects are freed.
        self._buffer: bytes = b"".join(encoded)
        self._mv = memoryview(self._buffer)

    def __len__(self) -> int:
        return int(self._sizes.shape[0])

    @overload
    def __getitem__(self, idx: int) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...

    def __getitem__(self, idx: int | slice) -> T | Sequence[T]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"SerializedList index {idx} out of range")
        start = int(self._offsets[idx])
        size = int(self._sizes[idx])
        item: T = pickle.loads(self._mv[start : start + size])
        return item
