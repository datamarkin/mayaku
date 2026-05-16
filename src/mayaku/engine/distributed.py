"""DDP launch + comm primitives.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §3.5 / §3.6 (`engine/launch.py`,
`utils/comm.py`) for the in-scope multi-GPU / single-GPU paths and the
gloo CPU/MPS fallback documented in
`BACKEND_PORTABILITY_REPORT.md` §8:

* :func:`get_world_size`, :func:`get_rank`, :func:`is_main_process` —
  single source of truth for rank-0 gating. Return sane defaults
  (``1``, ``0``, ``True``) when the process group hasn't been
  initialised, so library code doesn't need a ``dist.is_initialized()``
  branch on every call site.
* :func:`synchronize` — ``dist.barrier()``; passes ``device_ids`` only
  on NCCL where it's required to avoid spurious stream syncs.
* :func:`all_reduce_dict` — sum-and-(optionally-)average a dict of
  scalar loss tensors across ranks. Used by the trainer to log a
  single coherent loss number across DDP workers.
* :func:`all_gather_object` — pickle-based, gloo-only, useful for
  evaluator outputs and metric aggregation.
* :func:`create_ddp_model` — wrap a model in
  :class:`~torch.nn.parallel.DistributedDataParallel` when
  ``world_size > 1``; pass-through otherwise.
* :func:`launch` — single entry point that either runs ``main_func``
  in-process (``world_size == 1``) or spawns ``world_size`` workers
  via ``torch.multiprocessing.spawn``.

MPS is intentionally documented as ``world_size == 1`` only — PyTorch
does not support MPS multi-device, and gloo over MPS would require an
extra cpu→mps→cpu hop per collective for no real-world benefit
(`BPR §8`).
"""

from __future__ import annotations

import datetime as dt
import os
import socket
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from mayaku.backends.device import Device

__all__ = [
    "all_gather_object",
    "all_reduce_dict",
    "create_ddp_model",
    "get_rank",
    "get_world_size",
    "init_from_env_if_needed",
    "is_main_process",
    "launch",
    "resolve_ddp_device",
    "synchronize",
]

T = TypeVar("T")

DEFAULT_TIMEOUT = dt.timedelta(minutes=30)


# ---------------------------------------------------------------------------
# Rank / world-size accessors
# ---------------------------------------------------------------------------


def get_world_size() -> int:
    """Total number of ranks; 1 when distributed isn't initialised."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def get_rank() -> int:
    """This rank's global index; 0 when distributed isn't initialised."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank())


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    """``dist.barrier()`` — no-op outside a process group."""
    if get_world_size() == 1:
        return
    if dist.get_backend() == "nccl":
        # NCCL needs the device id to avoid pulling streams from the
        # default device (`spec §3.6`).
        local = int(os.environ.get("LOCAL_RANK", get_rank()))
        dist.barrier(device_ids=[local])
    else:
        dist.barrier()


# ---------------------------------------------------------------------------
# Reductions / gathers
# ---------------------------------------------------------------------------


def all_reduce_dict(losses: Mapping[str, Tensor], *, average: bool = True) -> dict[str, Tensor]:
    """All-reduce every value (scalar tensor) across ranks.

    Returns a *new* dict (never mutates the input). When ``average=True``
    (the upstream default) the sum is divided by ``world_size`` so the
    returned numbers are directly comparable to a single-GPU run.
    """
    world = get_world_size()
    if world == 1:
        # Detach so the caller can `item()` on the result without
        # accidentally building a longer autograd graph than they expected.
        return {k: v.detach().clone() for k, v in losses.items()}
    # Stable iteration order so every rank reduces the same key sequence.
    keys = sorted(losses)
    stacked = torch.stack([losses[k].detach() for k in keys])
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
    if average:
        stacked = stacked / float(world)
    return {k: stacked[i] for i, k in enumerate(keys)}


def all_gather_object(obj: T) -> list[T]:
    """Pickle-based gather — every rank returns ``[obj_rank0, obj_rank1, …]``.

    Routed through gloo because NCCL doesn't carry Python objects
    (`spec §3.6`). Works with any picklable type; not for big tensors.
    """
    world = get_world_size()
    if world == 1:
        return [obj]
    out: list[T] = [None] * world  # type: ignore[list-item]
    dist.all_gather_object(out, obj)
    return out


# ---------------------------------------------------------------------------
# DDP wrap
# ---------------------------------------------------------------------------


def create_ddp_model(
    model: nn.Module,
    device: Device,
    *,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = True,
) -> nn.Module:
    """Wrap ``model`` in :class:`DistributedDataParallel` when world > 1.

    ``broadcast_buffers=False`` matches Detectron2 (`spec §3.2`'s
    `create_ddp_model`): backbone BNs are ``FrozenBatchNorm2d`` so the
    running stats don't change per-iteration, and broadcasting them
    every step is wasted bandwidth.

    ``find_unused_parameters=True`` is the safe default for detection
    models — Faster R-CNN heads can skip parameters on iters where no
    matching proposals/targets flow through a given branch (e.g. the
    bbox-regression rows for classes absent from the current batch).
    Without it, DDP's bucket reducer waits forever for gradients that
    never arrive and ``loss.backward()`` deadlocks. The ~3-5% per-step
    overhead is acceptable for correctness; pass ``False`` to opt out
    when the model is known-fully-used (e.g. classification-only).

    ``device_ids`` is passed only on CUDA — gloo's DDP path has no use
    for it and a non-empty list trips an assertion in PT 2.4+.
    """
    if get_world_size() == 1:
        return model
    if device.kind == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
        )
    # gloo / cpu / mps multi-machine: no device_ids.
    return DistributedDataParallel(
        model,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
    )


# ---------------------------------------------------------------------------
# Multi-GPU launch validation
# ---------------------------------------------------------------------------


def resolve_ddp_device(device_setting: str | None, num_gpus: int) -> Device:
    """Resolve a Device from a user-facing setting and validate it for DDP.

    ``device_setting`` accepts the same shapes as the CLI / API
    surfaces — ``None`` or ``"auto"`` triggers :meth:`Device.auto`,
    a concrete kind (``"cuda"``, ``"cpu"``, ``"mps"``) is honoured.

    Raises :class:`ValueError` when ``num_gpus > 1`` and the resolved
    device can't host that many ranks: MPS has no multi-process story,
    and ``num_gpus`` must not exceed visible CUDA devices.
    """
    dev = (
        Device.auto() if device_setting in (None, "auto") else Device(kind=device_setting)  # type: ignore[arg-type]
    )
    if num_gpus <= 1:
        return dev
    if dev.kind == "mps":
        raise ValueError("MPS does not support multi-GPU training; use num_gpus=1 on MPS.")
    if dev.kind == "cuda":
        visible = torch.cuda.device_count()
        if visible < num_gpus:
            raise ValueError(
                f"num_gpus={num_gpus} requested but only {visible} CUDA "
                "device(s) visible. On AMD hosts set MAYAKU_DEVICE=cuda."
            )
    return dev


# ---------------------------------------------------------------------------
# torchrun integration
# ---------------------------------------------------------------------------


def init_from_env_if_needed(device: Device) -> None:
    """Initialise distributed from torchrun-set env vars if not already initialised.

    ``torchrun`` (and ``torch.distributed.run``) sets ``WORLD_SIZE`` /
    ``RANK`` / ``LOCAL_RANK`` env vars and spawns the processes, but it
    does NOT call ``init_process_group`` on the user's behalf. When
    ``mayaku train --num-gpus 1`` runs under torchrun, this helper
    initialises the process group from those env vars so
    :func:`get_world_size` / :func:`get_rank` report the right values
    for the rest of the run.

    No-op when:
        - ``torch.distributed`` is unavailable
        - already initialised (e.g. inside our own :func:`launch`)
        - ``WORLD_SIZE`` is unset or ``<= 1``
    """
    if not dist.is_available() or dist.is_initialized():
        return
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    backend = "nccl" if device.kind == "cuda" else "gloo"
    if device.kind == "cuda":
        # NCCL ordering: set the active CUDA device BEFORE init_process_group
        # so the per-rank stream is bound to the right GPU.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=backend, timeout=DEFAULT_TIMEOUT)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch(
    main_func: Callable[..., None],
    world_size: int,
    *,
    device: Device | None = None,
    dist_url: str = "auto",
    args: Iterable[Any] = (),
    timeout: dt.timedelta = DEFAULT_TIMEOUT,
) -> None:
    """Run ``main_func`` either in-process or across ``world_size`` workers.

    Args:
        main_func: Entry point. Called as ``main_func(*args)``. Inside
            each worker, distributed has already been initialised so
            :func:`get_rank` / :func:`get_world_size` are usable.
        world_size: Total number of processes. ``1`` short-circuits to
            an in-process call (no spawn, no init).
        device: Optional :class:`Device`. Used to pick the
            ``dist_backend`` (``nccl`` for CUDA, ``gloo`` otherwise) and
            to set the active CUDA device per-rank. Defaults to
            :meth:`Device.auto`.
        dist_url: Init URL. ``"auto"`` finds a free TCP port on
            ``127.0.0.1`` (single-machine launch). Pass an explicit
            ``"tcp://host:port"`` for multi-machine.
        args: Positional args forwarded to ``main_func``.
        timeout: ``init_process_group`` timeout. Default 30 min.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be >= 1; got {world_size}")
    device = device or Device.auto()

    if world_size == 1:
        # Single-rank fast path: don't init dist, don't spawn.
        main_func(*args)
        return

    if device.kind == "mps":
        raise RuntimeError(
            "MPS does not support multi-process distributed training "
            "(BACKEND_PORTABILITY_REPORT §8). Run with world_size=1 on "
            "MPS hosts."
        )

    if dist_url == "auto":
        dist_url = f"tcp://127.0.0.1:{_pick_free_port()}"

    mp.spawn(  # type: ignore[no-untyped-call]
        _worker_entry,
        args=(world_size, dist_url, device.kind, main_func, tuple(args), timeout),
        nprocs=world_size,
        join=True,
    )


def _worker_entry(
    rank: int,
    world_size: int,
    dist_url: str,
    device_kind: str,
    main_func: Callable[..., None],
    args: tuple[Any, ...],
    timeout: dt.timedelta,
) -> None:
    backend = "nccl" if device_kind == "cuda" else "gloo"
    if device_kind == "cuda":
        # NCCL reads the current device at init time and registers this
        # rank against it. Setting the device AFTER init_process_group
        # leaves every rank registered against cuda:0 and the bootstrap
        # collective hangs (or every rank silently shares one GPU).
        # Restored from commit 67cb3c5 which was lost in the revert.
        torch.cuda.set_device(rank)
    os.environ.setdefault("LOCAL_RANK", str(rank))
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        timeout=timeout,
    )
    # ROCm hosts without an InfiniBand fabric stall in NCCL/RCCL's
    # IB-probe phase. Disabling IB by default is harmless on NVIDIA
    # single-node runs (Ethernet/NVLink are the actual transports).
    # Users running multi-node IB clusters can override these.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    try:
        synchronize()
        main_func(*args)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    with contextmanager_socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


@contextmanager
def contextmanager_socket():  # type: ignore[no-untyped-def]
    """Yield a transient TCP socket; close on exit."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        yield sock
    finally:
        sock.close()
