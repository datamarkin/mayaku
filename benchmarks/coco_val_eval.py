"""Evaluate Faster R-CNN on COCO val2017 (5000 images) and print results."""

from pathlib import Path

from mayaku import configs
from mayaku.cli.eval import run_eval

CONFIG       = configs.path("faster_rcnn_R_50_FPN_3x")
WEIGHTS    = Path("ema-model_iter_0090000.pth")
COCO_GT_JSON = Path("instances_val2017.json")
COCO_IMAGES  = Path("val2017")

metrics = run_eval(
    CONFIG,
    weights=WEIGHTS,
    coco_gt_json=COCO_GT_JSON,
    image_root=COCO_IMAGES,
    device="cuda",
)

print("\nResults")
print("=" * 60)
for task, task_metrics in metrics.items():
    print(f"\n[{task}]")
    for k, v in task_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<24s} {v:.4f}")
        else:
            print(f"  {k:<24s} {v}")
