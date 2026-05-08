"""Train a Mayaku model. Edit the constants below, then run:

    python tools/train_mayaku.py

Writes checkpoints, mid-training eval results, and metadata.json to OUTPUT_DIR/train/.
EMA checkpoint (if enabled) has num_batches_tracked stripped so it loads with strict=True.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from mayaku.cli.eval import run_eval
from mayaku.cli.train import run_train
from mayaku.config import load_yaml, merge_overrides

# ---------------------------------------------------------------------------
# Edit these before running.
# ---------------------------------------------------------------------------

CONFIG = Path("configs/detection/faster_rcnn_R_50_FPN_1x.yaml")

TRAIN_JSON   = Path("/path/coco/annotations/instances_train2017.json")
TRAIN_IMAGES = Path("/path/coco/train2017")
VAL_JSON     = Path("/path/coco/annotations/instances_val2017.json")
VAL_IMAGES   = Path("/path/coco/val2017")

OUTPUT_DIR = Path("./runs/r50_1x")

# Gradient batch = IMS_PER_BATCH × GRAD_ACCUM_STEPS (target 16).
# With NORM=BN: BN statistics are computed on IMS_PER_BATCH images only — grad_accum
#   does NOT improve BN quality, only gradient quality. Prefer larger IMS_PER_BATCH.
# With NORM=FrozenBN: BN is frozen, so BN quality is unaffected by batch size and
#   any IMS_PER_BATCH / GRAD_ACCUM_STEPS split is equivalent.
IMS_PER_BATCH    = 8
GRAD_ACCUM_STEPS = 2

# Mid-training eval cadence. Set to 0 to skip and only run final eval.
EVAL_PERIOD = 5000

# Backbone norm and frozen-layer policy.
# BN / freeze_at=0: best AP for from-scratch training — BN stats co-evolve with
#   weights and EMA averages them. Maximise IMS_PER_BATCH, not just effective batch.
# FrozenBN / freeze_at=2: BN is fixed, safe to use small IMS_PER_BATCH with large
#   GRAD_ACCUM_STEPS when VRAM is tight — BN quality is unaffected.
NORM      = "BN"
FREEZE_AT = 0

# ---------------------------------------------------------------------------

def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _strip_num_batches_tracked(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["model"] = {k: v for k, v in state["model"].items()
                      if not k.endswith("num_batches_tracked")}
    torch.save(state, path)


def main() -> int:
    if not torch.cuda.is_available():
        print("[train_mayaku] FAIL — CUDA not available.", file=sys.stderr)
        return 2

    cfg = load_yaml(CONFIG)
    cfg = merge_overrides(cfg, {
        "model":  {"backbone": {"norm": NORM, "freeze_at": FREEZE_AT}},
        "solver": {"ims_per_batch": IMS_PER_BATCH, "grad_accum_steps": GRAD_ACCUM_STEPS},
        "test":   {"eval_period": EVAL_PERIOD},
    })

    train_dir = OUTPUT_DIR / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = IMS_PER_BATCH * GRAD_ACCUM_STEPS
    print(f"[train_mayaku] config:  {CONFIG}")
    print(f"[train_mayaku] backbone: {cfg.model.backbone.name}, norm: {NORM}, freeze_at: {FREEZE_AT}")
    print(f"[train_mayaku] {cfg.solver.max_iter} iters  batch {IMS_PER_BATCH}×{GRAD_ACCUM_STEPS}={effective_batch}")
    print(f"[train_mayaku] output:  {train_dir}")

    t0 = time.time()
    run_train(
        cfg,
        coco_gt_json=TRAIN_JSON,
        image_root=TRAIN_IMAGES,
        output_dir=train_dir,
        pretrained_backbone=True,
        device="cuda",
        val_json=VAL_JSON if EVAL_PERIOD > 0 else None,
        val_image_root=VAL_IMAGES if EVAL_PERIOD > 0 else None,
    )
    train_secs = time.time() - t0
    print(f"[train_mayaku] training done in {train_secs:.0f}s ({train_secs / 3600:.2f}h)")

    ema_final  = train_dir / "ema" / "model_final.pth"
    live_final = train_dir / "model_final.pth"
    final_weights: Path

    if ema_final.exists():
        print("[train_mayaku] stripping num_batches_tracked from EMA checkpoint")
        _strip_num_batches_tracked(ema_final)
        final_weights = ema_final
    elif live_final.exists():
        final_weights = live_final
    else:
        candidates: list[Path] = sorted(train_dir.glob("model_iter_*.pth"))
        if not candidates:
            print(f"[train_mayaku] FAIL — no checkpoint under {train_dir}", file=sys.stderr)
            return 1
        final_weights = candidates[-1]

    print(f"[train_mayaku] evaluating {final_weights.relative_to(OUTPUT_DIR)}")
    t1 = time.time()
    metrics = run_eval(cfg, weights=final_weights, coco_gt_json=VAL_JSON,
                       image_root=VAL_IMAGES, output_dir=OUTPUT_DIR / "eval", device="cuda")
    eval_secs = time.time() - t1

    bbox = metrics.get("bbox", {})
    final_box_ap = float(bbox.get("AP", 0.0)) if bbox else None
    if final_box_ap is not None:
        print(f"[train_mayaku] box AP = {final_box_ap * 100:.2f}")

    (train_dir / "metadata.json").write_text(json.dumps({
        "config":               str(CONFIG),
        "backbone":             cfg.model.backbone.name,
        "num_classes":          cfg.model.roi_heads.num_classes,
        "max_iter":             cfg.solver.max_iter,
        "ims_per_batch":        IMS_PER_BATCH,
        "grad_accum_steps":     GRAD_ACCUM_STEPS,
        "effective_batch_size": effective_batch,
        "ema_enabled":          cfg.solver.ema_enabled,
        "final_weights":        str(final_weights),
        "final_box_ap":         final_box_ap,
        "train_seconds":        train_secs,
        "eval_seconds":         eval_secs,
        "git_hash":             _git_hash(),
        "torch_version":        torch.__version__,
        "cuda_version":         torch.version.cuda,
        "device_name":          torch.cuda.get_device_name(0),
    }, indent=2))
    print(f"[train_mayaku] metadata written to {train_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())