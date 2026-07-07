"""Train a Mayaku model. Edit the constants below, then run:

    python tools/train_mayaku.py

The actual orchestration (config load, train, pick best checkpoint,
eval, metadata.json) lives in :func:`mayaku.train`. For per-run knobs
beyond the YAML's defaults, pass ``overrides=`` (a dict in the schema
shape, e.g. ``{"solver": {"base_lr": 1e-3}}``).

Backbones are architecture-only (random init); warm-start a full mayaku
checkpoint by passing ``weights=`` to :func:`mayaku.train`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mayaku import train

# ---------------------------------------------------------------------------
# Edit these before running.
# ---------------------------------------------------------------------------

# Pick a config from configs/detection/. Examples:
#   ResNet-50      : configs/detection/faster_rcnn_R_50_FPN.yaml
#   ConvNeXt-Tiny  : configs/detection/faster_rcnn_convnext_tiny_FPN.yaml
#   ConvNeXt-Small : configs/detection/faster_rcnn_convnext_small_FPN.yaml
#   ConvNeXt-Base  : configs/detection/faster_rcnn_convnext_base_FPN.yaml
#   ConvNeXt-Large : configs/detection/faster_rcnn_convnext_large_FPN.yaml
CONFIG       = Path("configs/detection/faster_rcnn_convnext_small_FPN.yaml")

TRAIN_JSON   = Path("/path/coco/annotations/instances_train2017.json")
TRAIN_IMAGES = Path("/path/coco/train2017")
VAL_JSON     = Path("/path/coco/annotations/instances_val2017.json")
VAL_IMAGES   = Path("/path/coco/val2017")

OUTPUT_DIR   = Path("./runs/convnext_small_1x")
EVAL_PERIOD  = 5000


def main() -> int:
    result = train(
        config=CONFIG,
        train_annotations=TRAIN_JSON,
        train_images=TRAIN_IMAGES,
        val_annotations=VAL_JSON,
        val_images=VAL_IMAGES,
        output_dir=OUTPUT_DIR,
        overrides={"test": {"eval_period": EVAL_PERIOD}},
        device="cuda",
    )
    ap = result["final_box_ap"]
    if ap is not None:
        print(f"[train_mayaku] final box AP = {ap * 100:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
