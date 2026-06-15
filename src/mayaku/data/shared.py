"""Per-node shared dataset loading for DDP.

Under DDP every process runs the training entrypoint and would otherwise parse
the full annotation JSON independently — N parses + N copies of the dataset
dicts *per node*, which OOMs host RAM for large datasets (e.g. Objects365's
14 GB / 25 M-annotation JSON × 4 GPUs).

This loads the dataset **once per node** (on the node's local rank 0), writes
the serialized buffer to a temp file, and broadcasts the small handle (path +
index arrays + metadata) to the node's other local ranks over a node-local
process subgroup. The other ranks read the buffer back instead of re-parsing.

Properties:
* **1× parse + 1× peak per node**, independent of GPUs-per-node.
* **Multi-node safe**: each node loads its own copy; nothing crosses node
  boundaries (only the tiny handle is broadcast, within the node). 4 servers ×
  1 GPU → each node's lone rank parses once (no sharing needed); 1 server × 4
  GPU → one parse shared to 3 ranks; M × K → one parse per node.
* **Single-process / single-GPU unchanged**: falls straight through to a plain
  ``SerializedList`` with no temp file.
"""

from __future__ import annotations

import gc
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from mayaku.data.serialize import SerializedList
from mayaku.engine.distributed import (
    get_local_process_group,
    get_local_rank,
    get_local_world_size,
    local_broadcast_object,
    synchronize,
)

__all__ = ["load_shared_dataset"]

E = TypeVar("E")


def load_shared_dataset(
    parse_fn: Callable[[], tuple[Any, list[dict[str, Any]]]],
    derive_fn: Callable[[Any, list[dict[str, Any]]], E] | None = None,
) -> tuple[Any, SerializedList[dict[str, Any]], E | None]:
    """Load a dataset once per node and share it across the node's ranks.

    Args:
        parse_fn: ``() -> (metadata, dataset_dicts)``. Runs only on the node's
            local rank 0 (the expensive JSON parse).
        derive_fn: optional ``(metadata, dataset_dicts) -> extra``, also computed
            on local rank 0 from the raw dicts and broadcast (e.g. auto-config
            overrides and RFS repeat factors, which both need the un-serialized
            dicts). ``None`` -> returns ``None``.

    Returns ``(metadata, SerializedList, extra)`` on every rank.
    """
    # Single process, or one rank per node (incl. N servers × 1 GPU): no
    # sharing to do — parse locally. This keeps the single-GPU path identical.
    if get_local_world_size() <= 1:
        metadata, dicts = parse_fn()
        extra = derive_fn(metadata, dicts) if derive_fn is not None else None
        return metadata, SerializedList(dicts), extra

    is_local_main = get_local_rank() == 0
    serialized: SerializedList[dict[str, Any]] | None = None
    payload: dict[str, Any] | None = None
    tmp_path: str | None = None

    if is_local_main:
        metadata, dicts = parse_fn()
        extra = derive_fn(metadata, dicts) if derive_fn is not None else None
        serialized = SerializedList(dicts)
        del dicts
        gc.collect()
        buffer, sizes, offsets = serialized.to_parts()
        fd, tmp_path = tempfile.mkstemp(prefix="mayaku-ds-", suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            f.write(buffer)
        payload = {
            "path": tmp_path,
            "sizes": sizes,
            "offsets": offsets,
            "metadata": metadata,
            "extra": extra,
        }

    payload = local_broadcast_object(payload)
    assert payload is not None

    if not is_local_main:
        buffer = Path(payload["path"]).read_bytes()
        serialized = SerializedList.from_buffer(buffer, payload["sizes"], payload["offsets"])

    # All node-local ranks have read the buffer; safe for rank 0 to clean up.
    synchronize(get_local_process_group())
    if is_local_main and tmp_path is not None:
        os.unlink(tmp_path)

    assert serialized is not None
    return payload["metadata"], serialized, payload["extra"]
