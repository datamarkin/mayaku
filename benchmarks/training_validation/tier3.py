"""Full COCO 2017 training validator — Mayaku from-scratch on a CUDA host.

The expensive one. ~24-48 GPU-hours on a single A100/H100. Pinned to
``faster_rcnn_R_50_FPN_1x`` (cheapest D2 schedule) and compared against
D2's published 37.9 box AP for the same config.

**Configure by editing the constants below**, then run:

    python benchmarks/training_validation/tier3.py

Pass criterion: final box AP in [0.374, 0.384] (= 37.4-38.4 in
0-100 percent units; D2 published 37.9 ± 0.5).

Orchestration lives in :func:`mayaku.train`; this file only carries
tier-3-specific knobs (the from-scratch 1x recipe overrides + the
pass-band check).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from mayaku import train

# ---------------------------------------------------------------------------
# Run configuration — edit these constants to retarget.
# ---------------------------------------------------------------------------

CONFIG_PATH       = Path("configs/detection/faster_rcnn_R_50_FPN_1x.yaml")

COCO_TRAIN_JSON   = Path("/home/dataserver/Downloads/coco/annotations/instances_train2017.json")
COCO_TRAIN_IMAGES = Path("/home/dataserver/Downloads/coco/train2017")
COCO_VAL_JSON     = Path("/home/dataserver/Downloads/coco/annotations/instances_val2017.json")
COCO_VAL_IMAGES   = Path("/home/dataserver/Downloads/coco/val2017")

OUTPUT_DIR        = Path("./tv_tier3_1b")

# 1x schedule: 90k iters at batch 16 (8 × 2 grad-accum). D2 published
# 37.9 box AP at 1x; ±0.5 AP either side is the accepted variance for
# full-COCO from-scratch training.
SOLVER_OVERRIDES = {
    "base_lr":               0.02,
    "max_iter":              90_000,
    "ims_per_batch":         8,
    "grad_accum_steps":      2,
    "steps":                 (60_000, 80_000),
    "warmup_iters":          1_000,
    # Safety net for gradient spikes — D2's COCO 1x runs without
    # clipping; mayaku has shown enough early-training spike-driven
    # divergence that we ship a wide global-L2 safety net rather than
    # discover it by NaN-ing out at iter 100. Threshold 5.0 catches
    # genuine blow-ups while leaving healthy steps (norm 0.5-2.0)
    # untouched.
    "clip_gradients_enabled": True,
    "clip_gradients_value":   5.0,
    "clip_gradients_type":    "norm",
}

EVAL_PERIOD = 5000

PASS_BAND_LOW       = 0.374
PASS_BAND_HIGH      = 0.384
D2_PUBLISHED_BOX_AP = 0.379


def main() -> int:
    result = train(
        config=CONFIG_PATH,
        train_json=COCO_TRAIN_JSON,
        train_images=COCO_TRAIN_IMAGES,
        val_json=COCO_VAL_JSON,
        val_images=COCO_VAL_IMAGES,
        output_dir=OUTPUT_DIR,
        overrides={
            "solver": SOLVER_OVERRIDES,
            "test": {"eval_period": EVAL_PERIOD},
        },
        device="cuda",
    )

    ap = result["final_box_ap"]
    if ap is None:
        print("[tier3] FAIL — eval did not produce a box AP", file=sys.stderr)
        return 1

    in_band = PASS_BAND_LOW <= ap <= PASS_BAND_HIGH
    verdict = "PASS" if in_band else "INVESTIGATE"
    delta = ap - D2_PUBLISHED_BOX_AP
    print(
        f"[tier3] box AP = {ap:.4f} ({ap * 100:.2f})  "
        f"Δ vs D2 = {delta:+.4f} ({delta * 100:+.2f} pts)  "
        f"verdict = {verdict}"
    )
    return 0 if in_band else 1


if __name__ == "__main__":
    sys.exit(main())
