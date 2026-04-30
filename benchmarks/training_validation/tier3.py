"""Full COCO 2017 training validator — Mayaku from-scratch on a CUDA host.

The expensive one. ~24-48 GPU-hours on a single A100/H100, ~12-15h on a
2-GPU DDP setup. Pinned to ``faster_rcnn_R_50_FPN_1x`` (cheapest D2
schedule) and compared against D2's published 37.9 box AP.

**Configure by editing the constants below**, then run:

    python benchmarks/training_validation/tier3.py

For multi-GPU (DDP) the same script does the spawn — no torchrun needed.
Just set WORLD_SIZE=2 (or higher) and the launcher splits work across
``WORLD_SIZE`` ranks. Effective batch = ``IMS_PER_BATCH * WORLD_SIZE *
GRAD_ACCUM_STEPS``.

Pass criterion: final box AP in [0.374, 0.384] (= 37.4-38.4 in
0-100 percent units; D2 published 37.9 ± 0.5).
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from mayaku.backends.device import Device
from mayaku.cli.eval import run_eval
from mayaku.cli.train import run_train
from mayaku.config import load_yaml, merge_overrides
from mayaku.engine import launch

# ---------------------------------------------------------------------------
# Run configuration — edit these constants to retarget.
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/detection/faster_rcnn_R_50_FPN_1x.yaml")

COCO_TRAIN_JSON = Path("/data/coco/annotations/instances_train2017.json")
COCO_TRAIN_IMAGES = Path("/data/coco/train2017")
COCO_VAL_JSON = Path("/data/coco/annotations/instances_val2017.json")
COCO_VAL_IMAGES = Path("/data/coco/val2017")

OUTPUT_DIR = Path("./tv_tier3")

# Multi-GPU: number of ranks. Set to 1 for single-GPU; 2+ for DDP. Each
# rank runs an independent process pinned to one CUDA device, with
# gradient all-reduce per iteration. Heterogeneous GPUs (e.g. 24 GB +
# 12 GB) work — both ranks use IMS_PER_BATCH that fits the SMALLER GPU,
# so the bigger one runs with spare capacity.
WORLD_SIZE = 1

# Per-rank batch size. EFFECTIVE batch (what base_lr is tuned to via the
# linear-scaling rule) = IMS_PER_BATCH * WORLD_SIZE * GRAD_ACCUM_STEPS.
# Examples:
#   - Single 24 GB GPU: WORLD_SIZE=1, IMS_PER_BATCH=16, GRAD_ACCUM=1 → eff 16
#   - Single 12 GB GPU: WORLD_SIZE=1, IMS_PER_BATCH=8,  GRAD_ACCUM=2 → eff 16 (slow)
#   - Dual 24 GB+12 GB: WORLD_SIZE=2, IMS_PER_BATCH=8,  GRAD_ACCUM=1 → eff 16 (~2× faster)
MAX_ITER = 90_000
BASE_LR = 0.02
IMS_PER_BATCH = 16
GRAD_ACCUM_STEPS = 1

# Mid-training eval cadence. Costs ~5-10 min per firing on val2017; in
# exchange you get an early warning if the run silently collapses to AP=0.
# Set to 0 to skip mid-training eval and only run final eval.
EVAL_PERIOD = 10_000

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


def _build_cfg() -> tuple[Any, int]:
    """Construct the resolved config + effective batch size.

    Effective batch = ``IMS_PER_BATCH * WORLD_SIZE * GRAD_ACCUM_STEPS`` —
    this is the number ``base_lr`` is tuned against under the linear
    scaling rule. The schema's ``ims_per_batch`` is the PER-RANK batch.
    """
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
                "clip_gradients_enabled": True,
                "clip_gradients_value": 1.0,
                "clip_gradients_type": "norm",
            },
            "test": {"eval_period": EVAL_PERIOD},
        },
    )
    effective_batch = IMS_PER_BATCH * WORLD_SIZE * GRAD_ACCUM_STEPS
    return cfg, effective_batch


def _train_only(cfg: Any) -> None:
    """The per-rank training body. Wrapped by ``launch()`` when WORLD_SIZE > 1.

    Each rank loads the same config, runs ``run_train`` (which is rank-aware:
    sampler, model, checkpointer, metrics-printer all gate on the active
    process group). Side-effects (config dump, checkpoints, stdout) only
    happen on rank 0.
    """
    train_dir = OUTPUT_DIR / "train"
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


def main() -> int:
    if not torch.cuda.is_available():
        print(
            "[tier3] FAIL — CUDA not available. tier3 is CUDA-only in practice; "
            "edit the script if you want to force CPU.",
            file=sys.stderr,
        )
        return 2
    if WORLD_SIZE > 1 and torch.cuda.device_count() < WORLD_SIZE:
        print(
            f"[tier3] FAIL — WORLD_SIZE={WORLD_SIZE} requested but only "
            f"{torch.cuda.device_count()} CUDA device(s) visible.",
            file=sys.stderr,
        )
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg, effective_batch = _build_cfg()

    print(
        f"[tier3] from-scratch COCO 2017 — {MAX_ITER} iters at base_lr={BASE_LR}, "
        f"world_size={WORLD_SIZE} × ims_per_batch={IMS_PER_BATCH} × "
        f"grad_accum={GRAD_ACCUM_STEPS} = effective batch={effective_batch}"
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
    if WORLD_SIZE == 1:
        _train_only(cfg)
    else:
        # ``launch()`` spawns ``WORLD_SIZE`` worker processes, each pinned
        # to one CUDA device, then runs ``_train_only`` inside each. The
        # NCCL process group is initialised before ``_train_only`` fires
        # so all the rank-aware machinery in ``run_train`` (sampler,
        # DDP wrap, checkpoint gating) sees the right world size.
        launch(_train_only, world_size=WORLD_SIZE, device=Device("cuda", 0), args=(cfg,))
    train_secs = time.time() - t_train_start
    print(f"[tier3] train wall-clock = {train_secs:.0f}s ({train_secs / 3600:.2f}h)")

    final_weights = train_dir / "model_final.pth"
    if not final_weights.exists():
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
