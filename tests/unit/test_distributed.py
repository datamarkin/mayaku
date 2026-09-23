"""Multi-GPU training: the helpers' single-process behaviour, and real
two-rank runs over gloo on the CPU (the NCCL variants need a 2-GPU host).

The one number that has to be right is the gradient: DDP averages the ranks'
gradients and the loss is a per-image sum over a batch-wide normaliser, so
two ranks on half a batch each must reproduce one process on the whole batch.
"""

from __future__ import annotations

import multiprocessing as py_mp
from pathlib import Path

import pytest
import torch

import mayaku
from mayaku.backends.device import Device
from mayaku.engine.distributed import (
    all_reduce_dict,
    get_rank,
    get_world_size,
    is_main_process,
    launch,
    local_device,
    resolve_ddp_device,
    synchronize,
)

from . import _distributed_workers as workers
from ._coco_fixture import synthetic_coco
from .test_api import FAST

_SPAWN = "spawn" in py_mp.get_all_start_methods()
multi = pytest.mark.skipif(not _SPAWN, reason="multiprocessing spawn unavailable")


def test_helpers_outside_a_process_group() -> None:
    assert get_world_size() == 1 and get_rank() == 0 and is_main_process()
    synchronize()
    x = torch.tensor(2.0)
    out = all_reduce_dict({"x": x})
    assert out["x"].item() == 2.0 and out["x"] is not x
    assert local_device("cpu") == "cpu" and local_device("cuda") == "cuda:0"


def test_launch_one_rank_runs_inline(tmp_path: Path) -> None:
    launch(workers.main_only_writes, 1, device=Device("cpu"), args=(str(tmp_path),))
    assert (tmp_path / "main.txt").exists()


def test_launch_refuses_what_it_cannot_run() -> None:
    with pytest.raises(ValueError):
        launch(print, 0)
    with pytest.raises(RuntimeError, match="MPS"):
        launch(print, 2, device=Device("mps"))
    with pytest.raises(ValueError, match="MPS"):
        resolve_ddp_device("mps", 2)


@pytest.mark.slow
@multi
def test_two_gloo_ranks_reduce_and_only_rank_zero_writes(tmp_path: Path) -> None:
    launch(workers.all_reduce_sum_one, 2, device=Device("cpu"), args=(str(tmp_path),))
    assert (tmp_path / "rank_0.ok").exists() and (tmp_path / "rank_1.ok").exists()
    out = tmp_path / "w"
    out.mkdir()
    launch(workers.main_only_writes, 2, device=Device("cpu"), args=(str(out),))
    assert (out / "main.txt").exists() and (out / "rank_1.ok").exists()


def _reference_grads() -> dict[str, torch.Tensor]:
    net = workers.parity_model()
    workers.loss_grads(net, *workers.parity_batch())
    return {k: p.grad for k, p in net.named_parameters()}


def _assert_same_grads(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> None:
    assert got.keys() == want.keys()
    for k in want:
        torch.testing.assert_close(got[k], want[k], rtol=1e-4, atol=1e-6, msg=k)


@pytest.mark.slow
@multi
def test_two_ranks_on_half_a_batch_give_the_single_process_gradient(tmp_path: Path) -> None:
    launch(workers.ddp_grad_parity, 2, device=Device("cpu"), args=(str(tmp_path), "cpu"))
    _assert_same_grads(torch.load(tmp_path / "grads.pt", weights_only=True), _reference_grads())


@pytest.mark.slow
@multi
@pytest.mark.parametrize("aux", [False, True], ids=["det", "seg-kpt"])
def test_two_rank_training_keeps_the_ranks_identical(tmp_path: Path, aux: bool) -> None:
    ann = synthetic_coco(str(tmp_path / "data"), n=16, seed=3)
    launch(workers.tiny_ddp_run, 2, device=Device("cpu"),
           args=(str(tmp_path), str(tmp_path / "data"), ann, aux))
    p0, p1 = (torch.load(tmp_path / f"params_{r}.pt", weights_only=True) for r in (0, 1))
    assert all(torch.equal(p0[k], p1[k]) for k in p0)
    run = tmp_path / "run"
    assert (run / "last.pt").exists() and (run / "state.pt").exists()
    assert len((run / "log.jsonl").read_text().splitlines()) == 2    # rank 0 only


@pytest.mark.slow
@multi
def test_train_on_two_ranks_through_the_api(tmp_path: Path) -> None:
    tr, va = tmp_path / "train", tmp_path / "val"
    result = mayaku.train(
        train_annotations=synthetic_coco(str(tr), n=16, seed=1), train_images=tr,
        val_annotations=synthetic_coco(str(va), n=6, seed=2), val_images=va,
        output_dir=tmp_path / "run", size_budget=96, num_epochs=2, num_gpus=2, device="cpu",
        overrides=FAST,
        log=lambda *_: None)
    assert result["metadata"]["world_size"] == 2 and result["metrics"] is not None
    assert result["best"] is not None and result["final_weights"].name == "best.pt"
    with pytest.raises(ValueError, match="multiple of 2"):
        mayaku.train(train_annotations=tr / "instances.json", train_images=tr, num_gpus=2,
                     device="cpu", overrides={"train": {"batch": 5}}, log=lambda *_: None)


@pytest.mark.cuda
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_two_nccl_ranks_each_use_their_gpu(tmp_path: Path) -> None:
    launch(workers.cuda_all_reduce_per_gpu, 2, device=Device("cuda"), args=(str(tmp_path),))
    assert (tmp_path / "rank_0_used_cuda:0.ok").exists()
    assert (tmp_path / "rank_1_used_cuda:1.ok").exists()


@pytest.mark.cuda
@pytest.mark.multi_gpu
@pytest.mark.slow
def test_nccl_gradient_matches_the_single_process_one(tmp_path: Path) -> None:
    launch(workers.ddp_grad_parity, 2, device=Device("cuda"), args=(str(tmp_path), "cuda"))
    _assert_same_grads(torch.load(tmp_path / "grads.pt", weights_only=True), _reference_grads())
