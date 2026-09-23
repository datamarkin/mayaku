"""The training loop, the schedule, and the recipe that defines a run.

Recipe matters as much as architecture for a detector this size, so every
choice that moves AP is a field of `Recipe` rather than a line of code, and
two runs compare as a diff of two dataclasses. Where the run happens -- device,
worker count, how often it evaluates, where it writes -- are arguments to
`train`, not recipe fields.

Choices with a published number behind them:

  no decay on norm and bias   +0.9 AP (RTMDet, arXiv 2212.07784)
  cosine over step decay      +0.46 to +1.82 AP
  clean final stage           +2.3 AP (RTMDet)
  ATSS warmup into TAL        the alignment metric is noise while the
                              predictions are, which is worst from scratch

EMA is universally adopted and has no published isolated ablation: it is here
because the whole field uses it.
"""

import copy
import dataclasses
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader, Dataset

from mayaku.data.augment import CLEAN_AUG, DEFAULT_AUG, Augment
from mayaku.data.batch import (
    batch_to,
    collate,
    multiscale_sizes,
    rescale_batch,
    seed_worker,
    to_tensor,
)
from mayaku.data.geometry import as_canvas
from mayaku.engine.evaluation import DEPLOY, STATS, Decode, evaluate, summary
from mayaku.engine.loss import DetectionLoss
from mayaku.model.quant import enable_qat, ranges_frozen, recalibrate_ranges


@dataclasses.dataclass(frozen=True)
class Recipe:
    """One training run, and everything that defines it.

    Frozen so a run has a stable identity; vary one with
    `dataclasses.replace(BASE, optimizer="adamw")`. The augmentation of each
    stage and the host decode are whole objects rather than names, so the
    record this serializes into reproduces the run.
    """

    epochs: int = 125
    batch: int = 16
    imgsz: int = 640
    optimizer: str = "sgd"          # sgd | adamw | musgd
    lr: float = 0.01
    lr_final_frac: float = 0.01     # final LR as a fraction of `lr`
    momentum: float = 0.937         # SGD only
    beta1: float = 0.9              # AdamW only; not the same number
    # musgd only: share of each hidden conv-weight step taken along the
    # orthogonalised direction (see `mayaku.engine.muon`); 0 is plain SGD
    muon_mix: float = 0.5
    weight_decay: float = 5e-4
    lr_ref_batch: int = 64          # batch size `lr` is quoted against
    warmup_epochs: float = 3.0
    warmup_iters_min: int = 100     # a floor, so a tiny set still warms up
    warmup_momentum_start: float = 0.8  # momentum ramps up from this
    warmup_bias_lr_start: float = 0.1   # bias LR ramps down from this
    final_epochs: int = 20          # final epochs run `final_aug`
    assigner_warmup: int = 5        # epochs of ATSS before TAL
    atss_soft: bool = False         # scale the ATSS class target by best IoU
    cls_loss: str = "bce"           # bce | vfl
    ema_decay: float = 0.9999
    ema_tau: float = 2000.0
    # Unaugmented training images the EMA's BatchNorm statistics (and, with
    # QAT, its int8 activation ranges) are recomputed over before every
    # evaluation and every checkpoint. 0 disables it. See `recalibrate`.
    recalibrate_images: int = 640
    # TaskAligned alignment exponent beta in s^alpha * u^beta (TOOD's 6.0).
    # Lower values let lower-IoU small positives rank.
    tal_beta: float = 6.0
    loc_weight_floor: float = 0.0   # floor the box/DFL loss weight (0 = off)
    # Multi-scale training: the minimum input size, 0 = off (fixed `imgsz`).
    # When set, every batch is rendered at `imgsz` (the operating point, the
    # maximum) and downscaled to a random multiple of 32 in [multiscale,
    # imgsz]. Eval always runs at `imgsz`.
    multiscale: int = 0
    # auxiliary heads: the Dice loss weight and the positive cap for masks.
    # The tier's `seg`/`kpt` flags turn the heads on; these tune them.
    seg_gain: float = 2.0
    seg_cap: int = 250
    # Quantization-aware training: fake-quantize (per-channel symmetric
    # weights, per-tensor affine activations) so training and the scored AP
    # reflect the int8 deploy. Near-lossless by design (QARepVGG blocks).
    # Set False for an fp32 run.
    qat: bool = True
    seed: int = 0
    amp: bool = True
    aug: Augment = DEFAULT_AUG
    final_aug: Augment = CLEAN_AUG
    decode: Decode = DEPLOY

    def __post_init__(self):
        assert self.optimizer in ("sgd", "adamw", "musgd"), self.optimizer
        assert self.final_epochs < self.epochs


BASE = Recipe()


def param_groups(model, decay):
    """Weights decay; norms and biases do not.

    Decaying a BatchNorm scale pulls the layer's gain toward zero, which the
    following layer then has to undo.

    Three groups, not two. `warmup_bias_lr_start` starts the bias group above
    `lr` and brings it down, which is right for a detection bias and wrong
    for a BatchNorm gain, so norms get a group of their own.
    """
    decayed, norms, biases = [], [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name == "bias":
                biases.append(p)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                norms.append(p)
            else:
                decayed.append(p)
    return [{"params": decayed, "weight_decay": decay, "bias": False},
            {"params": norms, "weight_decay": 0.0, "bias": False},
            {"params": biases, "weight_decay": 0.0, "bias": True}]


def build_optimizer(model, r, accumulate):
    """`lr` is quoted against `lr_ref_batch` images, not against `batch`, and
    weight decay is scaled to the real gradient batch, so a run at batch 8
    and a run at batch 64 are the same recipe.

    SGD's `momentum` and AdamW's `beta1` are separate fields: they are
    different quantities that happen to occupy the same slot.
    """
    decay = r.weight_decay * r.batch * accumulate / r.lr_ref_batch
    groups = param_groups(model, decay)
    if r.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=r.lr, momentum=r.momentum, nesterov=True)
    if r.optimizer == "musgd":
        # Hidden conv weights get the orthogonalised direction. The RGB input
        # convs and the output projections stay plain SGD: Muon is defined
        # for hidden layers, not for the first layer or the readout.
        from mayaku.engine.muon import MuonSGD
        readout = {id(c.weight) for c in model.readouts()}
        hidden, plain = [], []
        for w in groups[0]["params"]:
            is_hidden = w.ndim == 4 and w.shape[1] != 3 and id(w) not in readout
            (hidden if is_hidden else plain).append(w)
        groups = ([{**groups[0], "params": hidden, "muon": True},
                   {**groups[0], "params": plain, "muon": False}]
                  + groups[1:])
        return MuonSGD(groups, lr=r.lr, momentum=r.momentum, mix=r.muon_mix)
    return torch.optim.AdamW(groups, lr=r.lr, betas=(r.beta1, 0.999))


def cosine(r, epoch):
    """LR multiplier at an epoch: a half cosine from 1 at the start down to
    `lr_final_frac` at the end (Loshchilov & Hutter 2017, no restarts)."""
    end = r.lr_final_frac
    return end + (1 - end) * (1 + math.cos(math.pi * epoch / r.epochs)) / 2


def set_lr(opt, r, epoch, it, warmup_iters):
    """The only writer of the learning rate, called every iteration.

    Warmup and cosine are one expression rather than an iteration-level ramp
    racing an epoch-level scheduler. Biases start high and come down,
    everything else starts at zero and comes up, momentum ramps with them,
    and past `warmup_iters` the whole thing collapses to the cosine.
    """
    cos = cosine(r, epoch)
    x = min(it / warmup_iters, 1.0) if warmup_iters else 1.0
    for g in opt.param_groups:
        start = r.warmup_bias_lr_start if g["bias"] else 0.0
        g["lr"] = start + x * (r.lr * cos - start)
        if "momentum" in g:
            m0 = r.warmup_momentum_start
            g["momentum"] = m0 + x * (r.momentum - m0)
    return cos


class EMA:
    """A shadow copy of the weights, averaged with a ramping decay.

    The ramp, decay * (1 - exp(-updates / tau)), makes it usable from step
    one: far below `tau` updates the shadow tracks the model, and it only
    becomes a long average once there is something worth averaging.

    Parameters are averaged; buffers are copied, not averaged. A running
    variance is a second moment and does not commute with a weight average
    through a deep stack: averaged statistics stop describing what the
    averaged weights produce, silently, with nothing in the loss to show it.

    The tensor lists are paired once and updated with foreach kernels, and
    hold references to the live tensors, so an optimizer that replaced
    parameters rather than updating them in place would go unnoticed. None do.
    """

    def __init__(self, model, decay=0.9999, tau=2000.0):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0
        shadow, live = self.model.state_dict(), model.state_dict()
        names = {k for k, _ in self.model.named_parameters()}
        param_keys = [k for k in shadow if k in names]
        param_set = set(param_keys)
        self.shadow = [shadow[k] for k in param_keys]
        self.live = [live[k] for k in param_keys]
        buffers = [k for k in shadow if k not in param_set]
        self.shadow_buffers = [shadow[k] for k in buffers]
        self.live_buffers = [live[k] for k in buffers]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        torch._foreach_mul_(self.shadow, d)
        torch._foreach_add_(self.shadow, self.live, alpha=1 - d)
        torch._foreach_copy_(self.shadow_buffers, self.live_buffers)


def hms(seconds):
    """A short duration: 95 -> 1m35s, 7000 -> 1h56m."""
    s = int(seconds)
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


class _Unaugmented(Dataset):
    """The first `n` training images, letterboxed and unaugmented. Module
    level, not local, so worker processes can pickle it under spawn."""

    def __init__(self, ds, n):
        self.ds, self.n = ds, min(n, len(ds))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = self.ds.load(i)[0]
        while img is None:                       # skip a missing/corrupt file
            img = self.ds.load(torch.randint(len(self.ds), (1,)).item())[0]
        return to_tensor(img)


@torch.no_grad()
def recalibrate(model, train_ds, n, batch=16, device="cpu", workers=4):
    """Recompute BatchNorm running statistics, then int8 activation ranges,
    for the model's own weights.

    The EMA averages weights over the last ~tau steps, but its BatchNorm
    buffers describe the live model's current weights; at a high learning
    rate, where the two drift apart, its statistics stop matching what its
    weights produce. Nothing in the loss shows it; only the scored AP
    collapses, and it is worst where a BatchNorm sees the least data.

    So the statistics are recomputed as a true population estimate (momentum
    None, cumulative; `torch.optim.swa_utils.update_bn`) over `n` unaugmented
    training images drawn through `train_ds.load`, which letterboxes and never
    touches augmentation state. Runs before every evaluation and every
    checkpoint, so the best checkpoint is chosen on honest numbers and
    `fuse()` folds honest statistics into the deployed conv bias.

    With QAT the activation ranges are statistics of the weights in the same
    way, so they follow: BatchNorm is recomputed with the ranges frozen, then
    the ranges are recomputed with BatchNorm in eval mode (`recalibrate_ranges`),
    which is exactly how evaluation and export will run. Without QAT the
    second pass does not happen.
    """
    if n <= 0:
        return
    loader = DataLoader(_Unaugmented(train_ds, n), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=device.startswith("cuda"))
    with ranges_frozen(model):
        update_bn((batch_to(imgs, device) for imgs in loader), model)
    recalibrate_ranges(model, (batch_to(imgs, device) for imgs in loader))


def train(model, train_ds, val_ds, r=BASE, device="cpu", out=None,
          workers=0, eval_every=1, log_every=100, log=print):
    """Run the recipe. Returns the best metrics, the EMA model, and the
    per-epoch records.

    Every `log_every` iterations the running loss is printed with a rate and
    an estimate; that line is the only place the host waits on the device
    inside an epoch.

    A record per epoch carries the loss parts, the assigner counters and the
    full COCO metrics. They are returned, and appended to `out/log.jsonl` when
    there is an `out`, next to `recipe.json` and `tier.json`, so a run
    directory can always reproduce and rebuild its own model.
    """
    assert model.nc == train_ds.nc == val_ds.nc, "head and labels disagree"
    assert train_ds.canvas == val_ds.canvas == as_canvas(r.imgsz), \
        "recipe imgsz disagrees with the data canvas"
    torch.manual_seed(r.seed)
    model = model.to(device)
    if r.qat:
        enable_qat(model)
    accumulate = max(1, round(r.lr_ref_batch / r.batch))
    opt = build_optimizer(model, r, accumulate)
    crit = DetectionLoss(nc=model.nc, reg_max=model.cfg.reg_max,
                         cls_loss=r.cls_loss, warmup=r.assigner_warmup,
                         atss_soft=r.atss_soft, tal_beta=r.tal_beta,
                         loc_weight_floor=r.loc_weight_floor,
                         seg=model.cfg.seg, seg_gain=r.seg_gain, seg_cap=r.seg_cap,
                         kpt=model.cfg.kpt).to(device)
    ema = EMA(model, r.ema_decay, r.ema_tau)
    amp = r.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def make_loader():
        return DataLoader(train_ds, batch_size=r.batch, shuffle=True,
                          num_workers=workers, collate_fn=collate,
                          drop_last=True, pin_memory=device.startswith("cuda"),
                          worker_init_fn=seed_worker,
                          persistent_workers=workers > 0)

    loader = make_loader()
    # multi-scale training downscales each batch from the rendered `imgsz`
    # (the maximum) to a random size in [multiscale, imgsz]; empty = fixed
    ms_sizes = multiscale_sizes(r.multiscale, r.imgsz) if r.multiscale else []
    if ms_sizes and out:
        log("multi-scale training over %s" % ms_sizes)
    warmup_iters = max(round(r.warmup_epochs * len(loader)), r.warmup_iters_min)
    log_path = os.path.join(out, "log.jsonl") if out else None
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "recipe.json"), "w") as f:
            json.dump(dataclasses.asdict(r), f, indent=2)
        with open(os.path.join(out, "tier.json"), "w") as f:
            json.dump(dataclasses.asdict(model.cfg), f, indent=2)
        open(log_path, "w").close()
    records, best = [], {"AP": -1.0}

    for epoch in range(r.epochs):
        if epoch == r.epochs - r.final_epochs and train_ds.aug is not None:
            log("epoch %d: mosaic off, final stage %s" % (epoch, r.final_aug))
            train_ds.aug = r.final_aug
            # persistent workers hold their own copy of the dataset, so the
            # switch only reaches them through a fresh loader; without this
            # the final stage silently never happens. The old loader is
            # dropped first so its workers exit before new ones fork.
            loader = None
            loader = make_loader()
        model.train()
        totals, t0 = {}, time.perf_counter()
        for i, (imgs, targets, _, masks) in enumerate(loader):
            it = epoch * len(loader) + i
            lr = set_lr(opt, r, epoch, it, warmup_iters)
            imgs = batch_to(imgs, device)
            if ms_sizes:                        # multi-scale: one size per batch
                s = ms_sizes[torch.randint(len(ms_sizes), (1,)).item()]
                imgs, targets, masks = rescale_batch(
                    imgs, targets, masks, s, model.cfg.seg, model.cfg.kpt)
            with torch.amp.autocast("cuda", enabled=amp):
                loss, parts = crit(model(imgs), targets, epoch=epoch, masks=masks)
            scaler.scale(loss / accumulate).backward()
            if (i + 1) % accumulate == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update()
            # kept as device tensors and reduced once below; reading them
            # every iteration would sync the host against the GPU
            for k, v in parts.items():
                totals[k] = totals.get(k, 0) + v
            if log_every and (i + 1) % log_every == 0:
                done, spent = i + 1, time.perf_counter() - t0
                log("  %5d/%-5d  box %.3f cls %.3f dfl %.3f  lr %.5f  "
                    "%.0f img/s  epoch eta %s"
                    % (done, len(loader), float(totals["box"]) / done,
                       float(totals["cls"]) / done, float(totals["dfl"]) / done,
                       opt.param_groups[0]["lr"], done * r.batch / spent,
                       hms(spent / done * (len(loader) - done))))

        left = (r.epochs - epoch - 1) * (time.perf_counter() - t0)
        rec = {"epoch": epoch, "lr": r.lr * lr,
               "secs": time.perf_counter() - t0,
               **{k: float(v) / len(loader) for k, v in totals.items()}}
        # honest statistics for everything scored or saved below
        recalibrate(ema.model, train_ds, r.recalibrate_images,
                    device=device, workers=min(workers, 4))
        if (epoch + 1) % eval_every == 0 or epoch == r.epochs - 1:
            rec.update(evaluate(ema.model, val_ds, device=device,
                                batch=r.batch, workers=workers, d=r.decode))
            if rec["AP"] > best["AP"]:
                best = {"epoch": epoch, **{k: rec[k] for k in STATS}}
                if out:
                    torch.save(ema.model.state_dict(), os.path.join(out, "best.pt"))
            log("epoch %3d  box %.3f cls %.3f dfl %.3f  %s"
                % (epoch, rec["box"], rec["cls"], rec["dfl"], summary(rec)))
        else:
            log("epoch %3d  box %.3f cls %.3f dfl %.3f  %s, %s left"
                % (epoch, rec["box"], rec["cls"], rec["dfl"],
                   hms(rec["secs"]), hms(left)))
        records.append(rec)
        if out:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            torch.save(ema.model.state_dict(), os.path.join(out, "last.pt"))
    return best, ema.model, records
