"""Fine-tune any Mayaku detector on a COCO-format dataset.

Works with detection, segmentation, and keypoint configs — pass the right
config and weights for the architecture you want:

    # Detection
    python examples/finetune.py \\
        --config configs/detection/faster_rcnn_R_50_FPN_3x.yaml \\
        --train  /data/myproject/train \\
        --val    /data/myproject/valid \\
        --weights faster_rcnn_R_50_FPN_3x

    # Segmentation
    python examples/finetune.py \\
        --config configs/segmentation/mask_rcnn_R_50_FPN_3x.yaml \\
        --train  /data/myproject/train \\
        --val    /data/myproject/valid \\
        --weights mask_rcnn_R_50_FPN_3x

    # Keypoints
    python examples/finetune.py \\
        --config configs/keypoints/keypoint_rcnn_R_50_FPN_3x.yaml \\
        --train  /data/myproject/train \\
        --val    /data/myproject/valid \\
        --weights keypoint_rcnn_R_50_FPN_3x

Dataset layout (Roboflow export or any COCO-format split):

    train/
        image1.jpg
        image2.jpg
        _annotations.coco.json   ← or annotations.json / instances.json

The number of classes is auto-discovered from the train annotations.
Backbone and FPN stay pretrained; only the head classifier layers are
re-initialised when num_classes differs from the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO

from mayaku.backends.device import Device
from mayaku.cli._weights import resolve_weights
from mayaku.cli.eval import run_eval
from mayaku.cli.train import run_train
from mayaku.config import load_yaml, merge_overrides


def _find_json(directory: Path) -> Path:
    for name in ("_annotations.coco.json", "annotations.json", "instances.json"):
        p = directory / name
        if p.exists():
            return p
    candidates = sorted(directory.glob("*.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"No annotation JSON found in {directory}.\n"
        "Expected one of: _annotations.coco.json, annotations.json, instances.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune any Mayaku detector on a COCO-format dataset.")
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config (detection / segmentation / keypoints)")
    parser.add_argument("--train", type=Path, required=True,
                        help="Train directory containing images and annotation JSON")
    parser.add_argument("--val", type=Path, required=True,
                        help="Val directory containing images and annotation JSON")
    parser.add_argument("--weights", default=None,
                        help="Pretrained checkpoint: model name (auto-downloaded) or .pth path")
    parser.add_argument("--output", type=Path, default=Path("runs/finetune"))
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Base learning rate (D2's 0.02 / 16 ≈ 1.25e-3 for batch 1)")
    parser.add_argument("--eval-period", type=int, default=500,
                        help="Run COCO eval on val split every N iterations")
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    args = parser.parse_args()

    train_json = _find_json(args.train)
    val_json   = _find_json(args.val)

    coco = COCO(str(train_json))
    cat_ids     = sorted(coco.cats.keys())
    class_names = [coco.cats[cid]["name"] for cid in cat_ids]
    num_classes = len(class_names)
    print(f"Dataset : {num_classes} classes — {class_names}")

    device = Device.auto().kind if args.device == "auto" else args.device
    print(f"Device  : {device}")

    cfg = merge_overrides(
        load_yaml(args.config),
        {
            "model": {"roi_heads": {"num_classes": num_classes}},
            "solver": {
                "ims_per_batch": args.batch,
                "base_lr": args.lr,
                "max_iter": args.iters,
                "steps": (int(args.iters * 0.67), int(args.iters * 0.89)),
                "warmup_iters": min(200, args.iters // 10),
                "clip_gradients_enabled": True,
                "clip_gradients_value": 1.0,
                "clip_gradients_type": "norm",
            },
            "test": {"eval_period": args.eval_period},
        },
    )

    weights = resolve_weights(args.weights) if args.weights else None

    print(f"\nTraining for {cfg.solver.max_iter} iters → {args.output}")
    run_train(
        cfg,
        coco_gt_json=train_json,
        image_root=args.train,
        output_dir=args.output,
        weights=weights,
        val_json=val_json,
        val_image_root=args.val,
        device=device,
    )

    final_weights = args.output / "model_final.pth"
    print(f"\nFinal eval on {val_json.name}…")
    metrics = run_eval(
        cfg,
        weights=final_weights,
        coco_gt_json=val_json,
        image_root=args.val,
        output_dir=args.output / "eval",
        device=device,
    )
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
