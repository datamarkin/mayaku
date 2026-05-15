"""Train a Mayaku model. Edit the constants below, then run:

    python tools/train_mayaku.py

Writes checkpoints, mid-training eval results, and metadata.json to OUTPUT_DIR/train/.
EMA checkpoint (if enabled) has num_batches_tracked stripped so it loads with strict=True.

Supports ResNet/ResNeXt and ConvNeXt backbones — pick a config from
configs/detection/ and the script handles the family-specific quirks
(``norm`` doesn't apply to ConvNeXt; ``weights_path`` in the YAML
auto-disables the torchvision-ImageNet fallback). For ConvNeXt, the
YAML's ``weights_path`` field selects the pretrained checkpoint (e.g.,
the DINOv3 LVD-1689M release, the original Liu et al. release, or a
user fine-tune).
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
from mayaku.models.backbones import is_convnext_variant

# ---------------------------------------------------------------------------
# Edit these before running.
# ---------------------------------------------------------------------------

# Pick a config from configs/detection/. Examples:
#   ResNet-50     : configs/detection/faster_rcnn_R_50_FPN_1x.yaml
#   ConvNeXt-Tiny : configs/detection/faster_rcnn_convnext_tiny_FPN_1x.yaml
#   ConvNeXt-Small: configs/detection/faster_rcnn_convnext_small_FPN_1x.yaml
#   ConvNeXt-Base : configs/detection/faster_rcnn_convnext_base_FPN_1x.yaml
#   ConvNeXt-Large: configs/detection/faster_rcnn_convnext_large_FPN_1x.yaml
# The four `faster_rcnn_convnext_*` configs ship with `weights_path`
# pre-filled for the DINOv3 LVD-1689M checkpoints in `models/dinov3/`;
# edit that field in the YAML to use any other compatible checkpoint,
# or drop the field and pass `pretrained_backbone=True` here for
# torchvision ImageNet-1k init.
CONFIG = Path("configs/detection/faster_rcnn_convnext_small_FPN_1x.yaml")

TRAIN_JSON   = Path("/path/coco/annotations/instances_train2017.json")
TRAIN_IMAGES = Path("/path/coco/train2017")
VAL_JSON     = Path("/path/coco/annotations/instances_val2017.json")
VAL_IMAGES   = Path("/path/coco/val2017")

OUTPUT_DIR = Path("./runs/convnext_small_1x")

# Gradient batch = IMS_PER_BATCH × GRAD_ACCUM_STEPS (target 16).
# The YAML already carries variant-tuned defaults (see header in each
# ConvNeXt config for VRAM guidance). Override here only when your GPU
# doesn't match the YAML's assumed budget.
#
# With NORM=BN (ResNet only): BN statistics are computed on IMS_PER_BATCH
#   images per step — grad_accum does NOT improve BN quality, only gradient
#   quality. Prefer larger IMS_PER_BATCH.
# With NORM=FrozenBN (ResNet) or ConvNeXt (LayerNorm): per-step batch size
#   does not affect normalisation, so any IMS_PER_BATCH × GRAD_ACCUM_STEPS
#   split is gradient-equivalent.
IMS_PER_BATCH    = 4
GRAD_ACCUM_STEPS = 4

# Mid-training eval cadence. Set to 0 to skip and only run final eval.
EVAL_PERIOD = 5000

# Backbone norm + freeze policy. **ResNet/ResNeXt only** — these knobs
# are silently ignored when CONFIG points at a ConvNeXt variant (the
# schema validator rejects non-default ``norm``/``stride_in_1x1`` on
# ConvNeXt, so the script branches and skips that part of the override).
# ConvNeXt's freeze_at comes from the YAML; edit the YAML directly to
# change it.
#
# BN / freeze_at=0: best AP for from-scratch training — BN stats co-evolve with
#   weights and EMA averages them. Maximise IMS_PER_BATCH.
# FrozenBN / freeze_at=2: BN is fixed, safe to use small IMS_PER_BATCH with
#   large GRAD_ACCUM_STEPS when VRAM is tight — BN quality is unaffected.
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

    # Branch the override on backbone family. ConvNeXt's schema validator
    # rejects ``norm`` / ``stride_in_1x1`` / ``res5_dilation`` overrides
    # (LayerNorm is intrinsic; there's no bottleneck stride to relocate).
    is_cnx = is_convnext_variant(cfg.model.backbone.name)
    backbone_override: dict[str, object] = {"freeze_at": FREEZE_AT}
    if not is_cnx:
        backbone_override["norm"] = NORM
    cfg = merge_overrides(cfg, {
        "model":  {"backbone": backbone_override},
        "solver": {"ims_per_batch": IMS_PER_BATCH, "grad_accum_steps": GRAD_ACCUM_STEPS},
        "test":   {"eval_period": EVAL_PERIOD},
    })

    # If the YAML pins a backbone-local checkpoint via ``weights_path``,
    # don't also request torchvision ImageNet weights — they would download
    # and be silently overwritten by the local file, wasting time and disk.
    use_torchvision_pretrained = cfg.model.backbone.weights_path is None

    train_dir = OUTPUT_DIR / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = IMS_PER_BATCH * GRAD_ACCUM_STEPS
    bb_name = cfg.model.backbone.name
    bb_freeze = cfg.model.backbone.freeze_at
    print(f"[train_mayaku] config:  {CONFIG}")
    if is_cnx:
        wp = cfg.model.backbone.weights_path or "(random init / torchvision ImageNet)"
        print(f"[train_mayaku] backbone: {bb_name}, freeze_at: {bb_freeze}, weights: {wp}")
    else:
        src = "torchvision IMAGENET1K_V2" if use_torchvision_pretrained else "from config / --weights"
        print(f"[train_mayaku] backbone: {bb_name}, norm: {NORM}, freeze_at: {bb_freeze}, init: {src}")
    print(f"[train_mayaku] {cfg.solver.max_iter} iters  batch {IMS_PER_BATCH}x{GRAD_ACCUM_STEPS}={effective_batch}")
    print(f"[train_mayaku] output:  {train_dir}")

    t0 = time.time()
    run_train(
        cfg,
        coco_gt_json=TRAIN_JSON,
        image_root=TRAIN_IMAGES,
        output_dir=train_dir,
        pretrained_backbone=use_torchvision_pretrained,
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
