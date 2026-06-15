"""Module-level worker entry points for ``mp.spawn``.

``torch.multiprocessing.spawn`` pickles the entry function before
sending it to the child process. Pickling closures or test-local
functions breaks under ``spawn`` start; module-level callables
(referenced by qualified name) round-trip cleanly.

These are intentionally tiny — the real assertions live back in the
parent test. Each worker writes a small file under ``out_dir/``
indexed by rank so the parent can read the result back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mayaku.backends.device import Device
from mayaku.engine.distributed import (
    all_reduce_dict,
    create_ddp_model,
    get_rank,
    get_world_size,
    is_main_process,
)


def all_reduce_sum_one(out_dir: str) -> None:
    """Each rank contributes 1.0; the all-reduced sum should equal world."""
    val = torch.tensor([1.0])
    reduced = all_reduce_dict({"x": val}, average=False)
    assert reduced["x"].item() == float(get_world_size()), (
        f"rank={get_rank()} got {reduced['x'].item()}, expected {get_world_size()}"
    )
    Path(out_dir, f"rank_{get_rank()}.ok").touch()


def assert_main_only_writes(out_dir: str) -> None:
    """Only rank 0 writes — exercises is_main_process under DDP."""
    if is_main_process():
        Path(out_dir, "main.txt").write_text("hello")
    Path(out_dir, f"rank_{get_rank()}.ok").touch()


def shared_dataset_one_parse(out_dir: str) -> None:
    """Two ranks share one parse — exercises ``load_shared_dataset``.

    The node's local rank 0 is the only one that should run ``parse_fn``
    (it drops a per-rank flag so the parent can confirm a single parse);
    both ranks must come away with byte-identical dataset dicts, the same
    broadcast metadata, and the same ``derive_fn`` result.
    """
    from mayaku.data.shared import load_shared_dataset

    rank = get_rank()
    expected = [{"image_id": i, "file_name": f"img_{i}.jpg"} for i in range(8)]

    def parse() -> tuple[Any, list[dict[str, Any]]]:
        Path(out_dir, f"parsed_rank_{rank}.flag").touch()
        return {"name": "toy", "n": len(expected)}, list(expected)

    def derive(meta: Any, dicts: list[dict[str, Any]]) -> int:
        return sum(d["image_id"] for d in dicts)

    metadata, serialized, extra = load_shared_dataset(parse_fn=parse, derive_fn=derive)

    assert metadata == {"name": "toy", "n": 8}, (rank, metadata)
    assert extra == 28, (rank, extra)
    decoded = [serialized[i] for i in range(len(serialized))]
    assert decoded == expected, (rank, decoded)
    Path(out_dir, f"rank_{rank}.ok").touch()


# ---------------------------------------------------------------------------
# Multi-GPU NCCL workers
# ---------------------------------------------------------------------------


def cuda_all_reduce_assert_per_gpu(out_dir: str) -> None:
    """Every rank does work on its own GPU; sum check + GPU-pin receipt.

    Each rank allocates a tensor on ``cuda:rank`` (verifies that
    ``launch`` actually called ``torch.cuda.set_device(rank)``), runs
    an NCCL all-reduce, and writes a receipt naming the GPU it used so
    the parent test can confirm both GPUs were touched.
    """
    rank = get_rank()
    world = get_world_size()
    device = torch.device(f"cuda:{rank}")
    val = torch.tensor([float(rank + 1)], device=device)
    reduced = all_reduce_dict({"x": val}, average=False)
    expected = float(world * (world + 1) // 2)  # 1+2+…+world
    assert reduced["x"].item() == expected, (
        f"rank={rank} got {reduced['x'].item()}, expected {expected}"
    )
    # Also confirm the tensor stayed on this rank's GPU through the reduce.
    assert reduced["x"].device == device, (
        f"rank={rank} all_reduce moved tensor off {device} to {reduced['x'].device}"
    )
    Path(out_dir, f"rank_{rank}_used_cuda_{rank}.ok").touch()


def ddp_grad_parity(out_dir: str) -> None:
    """Tiny DDP forward/backward; rank 0 dumps the averaged grad.

    Each rank sees a different synthetic batch
    (``x = full((2, 4), rank+1)``); after backward DDP all-reduces and
    averages the grads. We snapshot rank 0's grad to disk so the
    parent can compare against a single-process reference computed
    over the union of both ranks' batches.
    """
    rank = get_rank()
    device = Device(kind="cuda", index=rank)
    torch.manual_seed(0)  # identical init on every rank
    model = nn.Linear(4, 1, bias=False).to(device.torch)
    ddp = create_ddp_model(model, device)

    x = torch.full((2, 4), float(rank + 1), device=device.torch)
    y = torch.zeros((2, 1), device=device.torch)
    loss = ((ddp(x) - y) ** 2).mean()
    loss.backward()

    if is_main_process():
        # The wrapped grad lives on `model.weight.grad` (DDP forwards
        # gradients onto the underlying module's parameters).
        grad = model.weight.grad
        assert grad is not None
        torch.save(grad.detach().cpu(), Path(out_dir) / "rank0_grad.pt")
    Path(out_dir, f"rank_{rank}_ddp.ok").touch()
