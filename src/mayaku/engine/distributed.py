"""Multi-GPU plumbing: starting ranks, and the few collectives training needs.

Two ways in, one process layout:

* `launch` spawns `world_size` processes on this machine (`mayaku.train(...,
  num_gpus=N)`), one per GPU;
* `init_from_env_if_needed` joins a group `torchrun` started, for multi-node
  runs (`torchrun --nproc-per-node N ... -m mayaku.cli train ...`).

Either way every rank runs the same training call and
`mayaku.engine.trainer.train` splits the work (see its docstring). NCCL on
CUDA, gloo elsewhere; gloo on the CPU exists so the distributed path can be
tested without a GPU. MPS has no multi-device support, so it is refused.

Outside a process group every helper degrades to the single-process answer
(world 1, rank 0, barriers and reductions no-ops), so library code calls them
unconditionally.
"""

from __future__ import annotations

import datetime as dt
import os
import socket
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from mayaku.backends.device import Device, DeviceKind

__all__ = [
    "all_reduce_dict",
    "broadcast_from_main",
    "get_rank",
    "get_world_size",
    "init_from_env_if_needed",
    "is_main_process",
    "launch",
    "local_device",
    "resolve_ddp_device",
    "synchronize",
]

T = TypeVar("T")

# Rank 0 evaluates and writes checkpoints while the other ranks wait at a
# barrier, so the timeout has to cover a full validation pass.
DEFAULT_TIMEOUT = dt.timedelta(hours=2)


def get_world_size() -> int:
    """Number of ranks; 1 outside a process group."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def get_rank() -> int:
    """This rank's index; 0 outside a process group."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank())


def is_main_process() -> bool:
    return get_rank() == 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", get_rank()))


def local_device(kind: DeviceKind) -> str:
    """This rank's device as torch spells it: its own GPU for "cuda"."""
    return str(Device(kind, _local_rank()).torch)


def synchronize() -> None:
    """A barrier across all ranks; a no-op outside a process group."""
    if get_world_size() == 1:
        return
    if dist.get_backend() == "nccl":
        # without the device id NCCL may pick the default device's stream
        dist.barrier(device_ids=[_local_rank()])
    else:
        dist.barrier()


def all_reduce_dict(values: Mapping[str, Tensor | float], *,
                    average: bool = True) -> dict[str, Tensor]:
    """Every value (a scalar tensor or number) summed, or averaged, over ranks,
    as a new dict of detached tensors."""
    device = next((v.device for v in values.values() if isinstance(v, Tensor)), None)
    as_tensor = {k: torch.as_tensor(v, dtype=torch.float32, device=device).detach()
                 for k, v in values.items()}
    world = get_world_size()
    if world == 1:
        return {k: v.clone() for k, v in as_tensor.items()}
    keys = sorted(values)                  # the same order on every rank
    stacked = torch.stack([as_tensor[k] for k in keys])
    dist.all_reduce(stacked)
    if average:
        stacked /= world
    return {k: stacked[i] for i, k in enumerate(keys)}


def broadcast_from_main(obj: T) -> T:
    """Rank 0's `obj` on every rank (the others' own `obj` is ignored);
    `obj` itself outside a process group."""
    if get_world_size() == 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def resolve_ddp_device(setting: str, num_gpus: int) -> Device:
    """The device for `num_gpus` ranks from a user setting ("auto", "cuda",
    "cpu", "mps"); raises when it cannot host them."""
    dev = Device(cast(DeviceKind, Device.resolve(setting)))
    if num_gpus <= 1:
        return dev
    if dev.kind == "mps":
        raise ValueError("MPS does not support multi-GPU training; use num_gpus=1")
    if dev.kind == "cuda" and torch.cuda.device_count() < num_gpus:
        raise ValueError(f"num_gpus={num_gpus} but only {torch.cuda.device_count()} CUDA "
                         "device(s) are visible")
    return dev


def init_from_env_if_needed(device: Device) -> None:
    """Join the process group `torchrun` describes in the environment
    (``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK``); a no-op when there is none
    or it is already joined."""
    if not dist.is_available() or dist.is_initialized():
        return
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return
    if device.kind == "cuda":
        # before init: NCCL binds each rank to the current device
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=device.dist_backend, timeout=DEFAULT_TIMEOUT)


def launch(main_func: Callable[..., Any], world_size: int, *, device: Device | None = None,
           args: Iterable[Any] = (), timeout: dt.timedelta = DEFAULT_TIMEOUT) -> None:
    """Run ``main_func(*args)`` on `world_size` ranks on this machine; with
    one rank, in this process. Inside `main_func` the process group is up.
    `main_func` and `args` must pickle (spawned processes import them)."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1; got {world_size}")
    device = device or Device.auto()
    if world_size == 1:
        main_func(*args)
        return
    if device.kind == "mps":
        raise RuntimeError("MPS does not support multi-process training; use one process")
    url = f"tcp://127.0.0.1:{_free_port()}"
    mp.spawn(_worker, args=(world_size, url, device.kind, main_func, tuple(args), timeout),
             nprocs=world_size, join=True)


def _worker(rank: int, world_size: int, url: str, kind: DeviceKind, main_func: Callable[..., Any],
            args: tuple[Any, ...], timeout: dt.timedelta) -> None:
    if kind == "cuda":
        # before init: NCCL registers each rank against the current device,
        # and all of them would land on cuda:0
        torch.cuda.set_device(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    # a single machine never talks InfiniBand; probing for it can stall
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    dist.init_process_group(backend=Device(kind).dist_backend, init_method=url,
                            world_size=world_size, rank=rank, timeout=timeout)
    try:
        synchronize()
        main_func(*args)
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
