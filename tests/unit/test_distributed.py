"""Tests for :mod:`mayaku.engine.distributed`.

Single-process tests cover every helper. The multi-process gloo
all-reduce test (the spec gate "loss parity vs. single-GPU on toy
data" at its smallest scale) spawns two workers and checks that the
sum-reduce produces the expected world-size value on every rank. We
mark it ``slow`` so users on slow Apple Silicon CPUs can skip with
``-m 'not slow'`` if needed.
"""

from __future__ import annotations

import multiprocessing as py_mp
import platform
from pathlib import Path

import pytest
import torch
from torch import nn

from mayaku.backends.device import Device
from mayaku.engine.distributed import (
    all_gather_object,
    all_reduce_dict,
    create_ddp_model,
    get_rank,
    get_world_size,
    is_main_process,
    launch,
    synchronize,
)

# ---------------------------------------------------------------------------
# Single-process semantics
# ---------------------------------------------------------------------------


def test_world_size_defaults_to_one_outside_dist() -> None:
    assert get_world_size() == 1


def test_rank_defaults_to_zero_outside_dist() -> None:
    assert get_rank() == 0


def test_is_main_process_outside_dist() -> None:
    assert is_main_process() is True


def test_synchronize_is_noop_outside_dist() -> None:
    # Must not raise even with no process group initialised.
    synchronize()


def test_all_reduce_dict_passthrough_when_world_size_one() -> None:
    losses = {"a": torch.tensor(1.5), "b": torch.tensor(2.5)}
    out = all_reduce_dict(losses, average=False)
    assert out.keys() == losses.keys()
    torch.testing.assert_close(out["a"], torch.tensor(1.5))
    torch.testing.assert_close(out["b"], torch.tensor(2.5))


def test_all_reduce_dict_returns_a_copy_not_an_alias() -> None:
    src = torch.tensor(1.0, requires_grad=True)
    out = all_reduce_dict({"a": src}, average=False)
    # Must be detached so callers can `item()` without graph plumbing.
    assert not out["a"].requires_grad
    # And not the same storage as the input.
    assert out["a"].data_ptr() != src.data_ptr()


def test_all_gather_object_returns_singleton_outside_dist() -> None:
    assert all_gather_object("hello") == ["hello"]


def test_create_ddp_model_passthrough_when_world_size_one() -> None:
    model = nn.Linear(2, 2)
    wrapped = create_ddp_model(model, Device(kind="cpu"))
    assert wrapped is model


# ---------------------------------------------------------------------------
# launch()
# ---------------------------------------------------------------------------


def test_launch_world_size_one_calls_main_inline(tmp_path: Path) -> None:
    out = tmp_path / "single.txt"

    def _main(path: Path) -> None:
        path.write_text("hi")

    launch(_main, world_size=1, args=(out,))
    assert out.read_text() == "hi"


def test_launch_validates_world_size() -> None:
    with pytest.raises(ValueError, match="world_size"):
        launch(lambda: None, world_size=0)


def test_launch_rejects_mps_multi_process() -> None:
    with pytest.raises(RuntimeError, match="MPS"):
        launch(lambda: None, world_size=2, device=Device(kind="mps"))


# ---------------------------------------------------------------------------
# Multi-process gloo all-reduce — the "loss parity" gate (smallest-scale)
# ---------------------------------------------------------------------------


_SPAWN_OK = (
    py_mp.get_all_start_methods()  # type: ignore[no-untyped-call]
) and platform.system() in ("Darwin", "Linux")


@pytest.mark.slow
@pytest.mark.skipif(not _SPAWN_OK, reason="multiprocessing spawn unavailable")
def test_launch_two_gloo_ranks_perform_an_all_reduce(tmp_path: Path) -> None:
    # The worker function lives in tests/unit/_distributed_workers.py so
    # mp.spawn (which pickles by qualified name) can find it.
    from tests.unit._distributed_workers import all_reduce_sum_one

    launch(
        all_reduce_sum_one,
        world_size=2,
        device=Device(kind="cpu"),
        args=(str(tmp_path),),
    )
    # Every rank wrote its receipt; the all-reduced sum was correct.
    assert (tmp_path / "rank_0.ok").exists()
    assert (tmp_path / "rank_1.ok").exists()


@pytest.mark.slow
@pytest.mark.skipif(not _SPAWN_OK, reason="multiprocessing spawn unavailable")
def test_is_main_process_is_only_true_on_rank_zero(tmp_path: Path) -> None:
    from tests.unit._distributed_workers import assert_main_only_writes

    launch(
        assert_main_only_writes,
        world_size=2,
        device=Device(kind="cpu"),
        args=(str(tmp_path),),
    )
    assert (tmp_path / "main.txt").read_text() == "hello"
    # Both ranks dropped a receipt regardless of who's main.
    assert (tmp_path / "rank_0.ok").exists()
    assert (tmp_path / "rank_1.ok").exists()


# ---------------------------------------------------------------------------
# Multi-GPU NCCL — closes the "GPU 1 idle" gap on multi-GPU CUDA hosts
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_launch_two_nccl_ranks_all_reduce_across_gpus(tmp_path: Path) -> None:
    """Spawn 2 NCCL ranks pinned to ``cuda:0`` / ``cuda:1``; each does
    real work on its own GPU and contributes to a sum-reduce.

    Validates the CUDA-specific code paths of :func:`launch` /
    :func:`all_reduce_dict` that the gloo CPU test never reaches:
    NCCL backend selection, ``torch.cuda.set_device(rank)``,
    ``device_ids=[rank]`` on the NCCL barrier.
    """
    from tests.unit._distributed_workers import cuda_all_reduce_assert_per_gpu

    launch(
        cuda_all_reduce_assert_per_gpu,
        world_size=2,
        device=Device(kind="cuda"),
        args=(str(tmp_path),),
    )
    # Both ranks executed and used their assigned GPU.
    assert (tmp_path / "rank_0_used_cuda_0.ok").exists()
    assert (tmp_path / "rank_1_used_cuda_1.ok").exists()


@pytest.mark.cuda
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_ddp_grad_parity_across_two_gpus(tmp_path: Path) -> None:
    """Spec gate (Step 14): 'loss parity vs. single-GPU on toy data'.

    DDP averages grads across ranks, so a 2-rank run with batches
    ``x_rank = full((2, 4), rank+1)`` should produce the same parameter
    gradient as a single-process run that sees the *concatenation*
    ``[1, 1, 2, 2]`` repeated. We compute the single-process reference
    on CPU and compare to rank 0's snapshot under a generous tolerance.
    """
    from tests.unit._distributed_workers import ddp_grad_parity

    launch(
        ddp_grad_parity,
        world_size=2,
        device=Device(kind="cuda"),
        args=(str(tmp_path),),
    )
    assert (tmp_path / "rank_0_ddp.ok").exists()
    assert (tmp_path / "rank_1_ddp.ok").exists()
    ddp_grad = torch.load(tmp_path / "rank0_grad.pt", weights_only=True)

    # Reference: identical init (manual_seed(0)) + nn.Linear(4, 1, bias=False);
    # batch is the union of rank 0's and rank 1's per-rank batches.
    torch.manual_seed(0)
    ref = nn.Linear(4, 1, bias=False)
    x = torch.cat([torch.full((2, 4), 1.0), torch.full((2, 4), 2.0)], dim=0)
    y = torch.zeros((4, 1))
    # DDP averages the per-rank-mean gradients across ranks.
    # Per-rank loss is mean over 2 elements; mean across 2 ranks is
    # equivalent to the mean over the concatenated 4-element batch.
    loss = ((ref(x) - y) ** 2).mean()
    loss.backward()
    ref_grad = ref.weight.grad
    assert ref_grad is not None
    torch.testing.assert_close(ddp_grad, ref_grad, atol=1e-5, rtol=1e-5)
