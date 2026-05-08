"""Full COCO 2017 training validator — Mayaku from-scratch on a CUDA host.

The expensive one. ~24-48 GPU-hours on a single A100/H100. Pinned to
``faster_rcnn_R_50_FPN_1x`` (cheapest D2 schedule) and compared against
D2's published 37.9 box AP for the same config.

**Configure by editing the constants below**, then run:

    python benchmarks/training_validation/tier3.py

Pass criterion: final box AP in [0.374, 0.384] (= 37.4-38.4 in
0-100 percent units; D2 published 37.9 ± 0.5).
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from mayaku.cli.eval import run_eval
from mayaku.cli.train import run_train
from mayaku.config import load_yaml, merge_overrides

# ---------------------------------------------------------------------------
# Run configuration — edit these constants to retarget.
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/detection/faster_rcnn_R_50_FPN_1x_modern.yaml")

COCO_TRAIN_JSON = Path("/home/dataserver/Downloads/coco/annotations/instances_train2017.json")
COCO_TRAIN_IMAGES = Path("/home/dataserver/Downloads/coco/train2017")
COCO_VAL_JSON = Path("/home/dataserver/Downloads/coco/annotations/instances_val2017.json")
COCO_VAL_IMAGES = Path("/home/dataserver/Downloads/coco/val2017")

OUTPUT_DIR = Path("./tv_tier3_1b")

# 1x schedule defaults (90k iters @ batch 16). Increase EMS_PER_BATCH /
# decrease GRAD_ACCUM_STEPS for a larger GPU; do the inverse for smaller.
# Effective batch size = IMS_PER_BATCH * GRAD_ACCUM_STEPS.
MAX_ITER = 90_000
BASE_LR = 0.02
IMS_PER_BATCH = 8
GRAD_ACCUM_STEPS = 2  # set to 2 for 12 GB GPU at IMS_PER_BATCH=8 → effective batch=16

# Mid-training eval cadence. Costs ~5-10 min per firing on val2017; in
# exchange you get an early warning if the run silently collapses to AP=0.
# Set to 0 to skip mid-training eval and only run final eval.
EVAL_PERIOD = 5000

# ---------------------------------------------------------------------------
# Pass band — D2 published 37.9 box AP at 1x; 0.5 AP either side is the
# accepted variance for full-COCO from-scratch training.
# ---------------------------------------------------------------------------

PASS_BAND_LOW = 0.374
PASS_BAND_HIGH = 0.384
D2_PUBLISHED_BOX_AP = 0.379


def _gpu_info() -> dict[str, str | int | bool | None]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_name": None, "device_count": 0}
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print(
            "[tier3] FAIL — CUDA not available. tier3 is CUDA-only in practice; "
            "edit the script if you want to force CPU.",
            file=sys.stderr,
        )
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(CONFIG_PATH)
    cfg = merge_overrides(
        cfg,
        {
            "solver": {
                "ims_per_batch": IMS_PER_BATCH,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "base_lr": BASE_LR,
                "max_iter": MAX_ITER,
                # D2's 1x schedule decays at 60k and 80k of 90k iters.
                "steps": (
                    max(1, int(MAX_ITER * 60_000 / 90_000)),
                    max(1, int(MAX_ITER * 80_000 / 90_000)),
                ),
                "warmup_iters": min(1000, MAX_ITER // 90),
                "checkpoint_period": max(MAX_ITER // 10, 1),
                # Safety net for gradient spikes — global L2 norm across
                # all parameters (the standard idiom). Threshold 5.0
                # catches genuine blow-ups (a single bad batch whose
                # combined gradient norm exceeds 5.0) without throttling
                # normal training (a healthy step is in the 0.5–2.0 range).
                # D2's COCO 1x has clipping off entirely; mayaku has
                # shown enough spike-driven divergence in early training
                # that we ship a wide safety net rather than discover it
                # by NaN-ing out at iter 100.
                "clip_gradients_enabled": True,
                "clip_gradients_value": 5.0,
                "clip_gradients_type": "norm"
            },
            "test": {"eval_period": EVAL_PERIOD}
        },
    )

    effective_batch = IMS_PER_BATCH * GRAD_ACCUM_STEPS
    print(
        f"[tier3] from-scratch COCO 2017 — {MAX_ITER} iters at base_lr={BASE_LR}, "
        f"ims_per_batch={IMS_PER_BATCH} × grad_accum={GRAD_ACCUM_STEPS} = effective batch={effective_batch}"
    )
    print(f"[tier3] config: {CONFIG_PATH}")
    print(
        f"[tier3] target band: [{PASS_BAND_LOW}, {PASS_BAND_HIGH}] "
        f"(= [{PASS_BAND_LOW * 100:.1f}, {PASS_BAND_HIGH * 100:.1f}] on 0-100 scale; "
        f"D2 published {D2_PUBLISHED_BOX_AP * 100:.1f})"
    )
    if EVAL_PERIOD > 0:
        print(
            f"[tier3] mid-training eval every {EVAL_PERIOD} iters — watch for "
            f"`[eval @ iter ...]` lines in stdout."
        )
    else:
        print("[tier3] mid-training eval disabled (EVAL_PERIOD=0); only final eval will run.")

    train_dir = OUTPUT_DIR / "train"
    t_train_start = time.time()
    run_train(
        cfg,
        coco_gt_json=COCO_TRAIN_JSON,
        image_root=COCO_TRAIN_IMAGES,
        output_dir=train_dir,
        pretrained_backbone=True,
        device="cuda",
        val_json=COCO_VAL_JSON if EVAL_PERIOD > 0 else None,
        val_image_root=COCO_VAL_IMAGES if EVAL_PERIOD > 0 else None,
    )
    train_secs = time.time() - t_train_start
    print(f"[tier3] train wall-clock = {train_secs:.0f}s ({train_secs / 3600:.2f}h)")

    ema_final = train_dir / "ema" / "model_final.pth"
    live_final = train_dir / "model_final.pth"
    if ema_final.exists():
        final_weights = ema_final
    elif live_final.exists():
        final_weights = live_final
    else:
        candidates = sorted(train_dir.glob("model_iter_*.pth"))
        if not candidates:
            print(f"[tier3] FAIL — no checkpoint produced under {train_dir}", file=sys.stderr)
            return 1
        final_weights = candidates[-1]

    print(f"[tier3] evaluating {final_weights.name} on val2017")
    t_eval_start = time.time()
    metrics = run_eval(
        cfg,
        weights=final_weights,
        coco_gt_json=COCO_VAL_JSON,
        image_root=COCO_VAL_IMAGES,
        output_dir=OUTPUT_DIR / "eval",
        device="cuda",
    )
    eval_secs = time.time() - t_eval_start

    bbox = metrics.get("bbox", {})
    final_box_ap = float(bbox.get("AP", 0.0))

    in_band = PASS_BAND_LOW <= final_box_ap <= PASS_BAND_HIGH
    verdict = "PASS" if in_band else "INVESTIGATE"

    result = {
        "tier": 3,
        "config": CONFIG_PATH.stem,
        "max_iter": MAX_ITER,
        "base_lr": BASE_LR,
        "ims_per_batch": IMS_PER_BATCH,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": effective_batch,
        "device": "cuda",
        "train_seconds": train_secs,
        "eval_seconds": eval_secs,
        "wall_clock_seconds": train_secs + eval_secs,
        "final_box_ap": final_box_ap,
        "final_box_ap50": float(bbox.get("AP50", 0.0)),
        "final_box_ap75": float(bbox.get("AP75", 0.0)),
        "d2_published_box_ap": D2_PUBLISHED_BOX_AP,
        "delta_vs_d2": final_box_ap - D2_PUBLISHED_BOX_AP,
        "pass_band": [PASS_BAND_LOW, PASS_BAND_HIGH],
        "verdict": verdict,
        "metrics": metrics,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **_gpu_info(),
        },
    }
    out_path = OUTPUT_DIR / "result.json"
    out_path.write_text(json.dumps(result, indent=2))

    print()
    print(f"[tier3] final box AP   = {final_box_ap:.4f}  ({final_box_ap * 100:.2f} on 0-100 scale)")
    print(
        f"[tier3] D2 published   = {D2_PUBLISHED_BOX_AP}  "
        f"({D2_PUBLISHED_BOX_AP * 100:.1f} on 0-100 scale)"
    )
    delta = result["delta_vs_d2"]
    assert isinstance(delta, float)
    print(f"[tier3] Δ vs D2        = {delta:+.4f}  ({delta * 100:+.2f} AP points)")
    print(f"[tier3] verdict        = {verdict}")
    print(f"[tier3] result written to {out_path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
