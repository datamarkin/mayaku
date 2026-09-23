"""Rank entry points for the distributed tests.

`torch.multiprocessing.spawn` pickles the entry function, so these live at
module level. Each writes its result under `out_dir`, by rank, for the parent
test to read back.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from mayaku.data import DEFAULT_AUG, CocoDetection
from mayaku.engine.distributed import (
    all_reduce_dict,
    get_rank,
    get_world_size,
    is_main_process,
    local_device,
)
from mayaku.engine.loss import DetectionLoss
from mayaku.engine.trainer import BASE, train
from mayaku.model import TINY, Detector


def all_reduce_sum_one(out_dir: str) -> None:
    """Each rank contributes 1; the sum is the world size on every rank."""
    reduced = all_reduce_dict({"x": torch.tensor(1.0)}, average=False)
    assert reduced["x"].item() == get_world_size()
    Path(out_dir, f"rank_{get_rank()}.ok").touch()


def main_only_writes(out_dir: str) -> None:
    if is_main_process():
        Path(out_dir, "main.txt").write_text("hello")
    Path(out_dir, f"rank_{get_rank()}.ok").touch()


def cuda_all_reduce_per_gpu(out_dir: str) -> None:
    """Every rank works on its own GPU; the receipt names it."""
    rank, world = get_rank(), get_world_size()
    device = torch.device(local_device("cuda"))
    reduced = all_reduce_dict({"x": torch.tensor(float(rank + 1), device=device)},
                              average=False)
    assert reduced["x"].item() == world * (world + 1) // 2
    assert reduced["x"].device == device
    Path(out_dir, f"rank_{rank}_used_{device}.ok").touch()


def parity_batch(n=4, size=64, nc=4):
    """`n` fixed images with two boxes each, as (images, target rows)."""
    g = torch.Generator().manual_seed(1)
    imgs = torch.rand(n, 3, size, size, generator=g)
    rows = []
    for i in range(n):
        for k in range(2):
            x1, y1 = 4 + 20 * k + i, 6 + 10 * k
            rows.append([i, (i + k) % nc, x1, y1, x1 + 24, y1 + 30])
    return imgs, torch.tensor(rows, dtype=torch.float32)


def parity_model(device="cpu"):
    """The same TINY detector on every rank, BatchNorm on its running
    statistics so each image's forward does not depend on its batch-mates."""
    torch.manual_seed(0)
    return Detector(TINY, 4, 64).to(device).eval()


def loss_grads(model, imgs, targets):
    """Backward the detection loss the way the trainer does: DDP averages the
    ranks' gradients, so each rank's share is scaled back to a sum."""
    crit = DetectionLoss(nc=4, reg_max=TINY.reg_max)
    loss, _ = crit(model(imgs), targets)
    (loss * get_world_size()).backward()


def ddp_grad_parity(out_dir: str, kind: str) -> None:
    """Each rank takes its share of `parity_batch` through DDP; rank 0 saves
    the synchronised gradients for the parent to compare against one process
    on the whole batch."""
    rank, world = get_rank(), get_world_size()
    device = local_device(kind)
    net = parity_model(device)
    model = DistributedDataParallel(net, device_ids=[torch.device(device).index]
                                    if kind == "cuda" else None)
    imgs, targets = parity_batch()
    per = len(imgs) // world
    mine = (targets[:, 0] >= rank * per) & (targets[:, 0] < (rank + 1) * per)
    t = targets[mine].clone()
    t[:, 0] -= rank * per
    loss_grads(model, imgs[rank * per:(rank + 1) * per].to(device), t)
    if rank == 0:
        torch.save({k: p.grad.cpu() for k, p in net.named_parameters()},
                   Path(out_dir) / "grads.pt")


def tiny_ddp_run(out_dir: str, root: str, ann: str, aux: bool) -> None:
    """Two epochs of the real trainer on every rank; each rank saves its final
    parameters, which DDP must have kept identical. With `aux`, masks and
    keypoints too: steps without their positives must still train."""
    k = 3 if aux else 0
    ds = CocoDetection(root, ann, 96, aug=DEFAULT_AUG, seed=3, masks=aux, kpt=k)
    val = CocoDetection(root, ann, 96, masks=aux, kpt=k) if is_main_process() else None
    torch.manual_seed(0)
    model = Detector(dataclasses.replace(TINY, seg=aux, kpt=k), ds.nc, 96)
    r = dataclasses.replace(BASE, epochs=2, batch=4, lr_ref_batch=4, warmup_iters_min=2,
                            recalibrate_images=4, amp=False)
    best, ema, _ = train(model, ds, val, r, out=str(Path(out_dir) / "run"), log_every=0,
                         log=lambda *_: None)
    assert (best is None) == (ema is None) == (get_rank() != 0)
    torch.save(dict(model.named_parameters()), Path(out_dir) / f"params_{get_rank()}.pt")
