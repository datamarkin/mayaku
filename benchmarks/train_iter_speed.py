"""Train for a fixed number of iterations and print per-iter duration.

This is a benchmark, not a real training run — we point at val2017 as
the "training" set just so something feeds the loader. The reported
number includes one-time startup (model build, first batch load), so
keep ITERS large enough to dwarf it.
"""

from pathlib import Path

from mayaku import configs
from mayaku.api import train

CONFIG       = configs.path("faster_rcnn_R_50_FPN_3x")
TRAIN_JSON   = Path("instances_val2017.json")
TRAIN_IMAGES = Path("val2017")
ITERS = 100

result = train(
    CONFIG,
    train_json=TRAIN_JSON,
    train_images=TRAIN_IMAGES,
    overrides={
        "auto_config": {"enabled": False},
        "solver": {
            "max_iter": ITERS,
            "steps": [],
            "warmup_iters": 0,
            "ema_enabled": False,
            "ims_per_batch": 32,
        },
    },
)

elapsed = result["train_seconds"]
print("\nResults")
print("=" * 60)
print(f"  Iterations            {ITERS}")
print(f"  Elapsed (s)           {elapsed:.2f}")
print(f"  Sec / iter            {elapsed / ITERS:.3f}")
print(f"  Iter / sec            {ITERS / elapsed:.2f}")
