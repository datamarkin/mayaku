"""Evaluate a Mayaku checkpoint on a COCO-format validation set.

Computes COCO mAP (bbox / segm / keypoints — auto-detected from the
config) using ``pycocotools``. Mirrors the ``mayaku eval`` CLI verb;
this script exists so a first-time user has a discoverable example for
the "did my fine-tune actually improve mAP?" question.

Quickstart on a Roboflow-style val split:

    python examples/evaluate.py \\
        --weights runs/finetune/model_final.pth \\
        --config  configs/detection/faster_rcnn_R_50_FPN_3x.yaml \\
        --val     /data/myproject/valid

Hybrid eval — measure the deployment artifact's accuracy, not just the
eager model — by pointing at the exported backbone:

    # ONNX (cross-platform, default CPU provider)
    python examples/evaluate.py --weights ... --val ... \\
        --backbone-onnx examples/outputs/model.onnx

    # CoreML (macOS only)
    python examples/evaluate.py --weights ... --val ... \\
        --backbone-mlpackage examples/outputs/model.mlpackage

The exported backbone is swapped into ``model.backbone`` while RPN /
NMS / heads stay in PyTorch — same hybrid pattern as ``run_onnx.py`` /
``run_coreml.py``, just measured against ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mayaku.backends.device import Device
from mayaku.cli.eval import run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _print_metrics(metrics: dict) -> None:
    """Pretty-print the per-task metrics dict from `run_eval`."""
    if not metrics:
        print("(no metrics returned — empty val set?)")
        return
    print("\nResults")
    print("=" * 60)
    for task, task_metrics in metrics.items():
        print(f"\n[{task}]")
        if isinstance(task_metrics, dict):
            for k, v in task_metrics.items():
                if isinstance(v, float):
                    print(f"  {k:<24s} {v:.4f}")
                else:
                    print(f"  {k:<24s} {v}")
        else:
            print(f"  {task_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO mAP evaluation for a Mayaku checkpoint.")
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config (must match the trained checkpoint architecture)")
    parser.add_argument("--weights", required=True,
                        help="Path to .pth checkpoint OR a model name to auto-download")
    parser.add_argument("--val", type=Path, default=None,
                        help="Val directory (with annotations + images); shortcut for "
                             "--coco-gt-json + --image-root")
    parser.add_argument("--coco-gt-json", type=Path, default=None,
                        help="Explicit ground-truth annotations JSON (overrides --val)")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Explicit image directory (overrides --val)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write metrics.json + COCO eval intermediates")
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--backbone-onnx", type=Path, default=None,
                        help="Use this ONNX artifact for backbone+FPN (hybrid eval)")
    parser.add_argument("--onnx-providers", default=None,
                        help="Comma-separated ORT providers, e.g. "
                             "CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--backbone-mlpackage", type=Path, default=None,
                        help="Use this CoreML .mlpackage for backbone+FPN (macOS only)")
    parser.add_argument("--coreml-compute-units", default="CPU_AND_GPU",
                        choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"])
    args = parser.parse_args()

    if args.val is None and (args.coco_gt_json is None or args.image_root is None):
        raise SystemExit(
            "Provide --val DIR (auto-finds annotations + uses DIR as image root), "
            "or both --coco-gt-json and --image-root explicitly."
        )

    coco_gt_json = args.coco_gt_json or _find_json(args.val)
    image_root = args.image_root or args.val

    device = Device.auto().kind if args.device == "auto" else args.device
    print(f"Config       : {args.config}")
    print(f"Weights      : {args.weights}")
    print(f"GT JSON      : {coco_gt_json}")
    print(f"Image root   : {image_root}")
    print(f"Device       : {device}")
    if args.backbone_onnx:
        print(f"ONNX backbone: {args.backbone_onnx}")
    if args.backbone_mlpackage:
        print(f"CoreML bb    : {args.backbone_mlpackage} ({args.coreml_compute_units})")

    metrics = run_eval(
        args.config,
        weights=args.weights,
        coco_gt_json=coco_gt_json,
        image_root=image_root,
        output_dir=args.output_dir,
        device=device,
        backbone_mlpackage=args.backbone_mlpackage,
        coreml_compute_units=args.coreml_compute_units,
        backbone_onnx=args.backbone_onnx,
        onnx_providers=args.onnx_providers,
    )

    _print_metrics(metrics)
    if args.output_dir:
        print(f"\nSaved → {args.output_dir / 'metrics.json'}")
    else:
        # Always print the raw dict for downstream scripting.
        print("\n" + json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
