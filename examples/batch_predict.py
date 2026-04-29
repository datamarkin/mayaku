"""Run detection on every image in a directory.

    python examples/batch_predict.py --images /path/to/images/

Writes one JSON file per image to --output (default: alongside --images)
and a combined results.json summary.

Switch model or architecture:

    python examples/batch_predict.py \\
        --images /path/to/images/ \\
        --model  mask_rcnn_R_50_FPN_3x \\
        --config configs/segmentation/mask_rcnn_R_50_FPN_3x.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference import Predictor
from mayaku.utils.image import read_image

REPO_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch object detection on a directory of images.")
    parser.add_argument("--images", type=Path, required=True, help="Directory of images")
    parser.add_argument("--output", type=Path, help="Output directory (default: --images/detections/)")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x",
                        help="Model name (auto-downloaded) or path to .pth")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    output_dir = args.output or (args.images / "detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = Device.auto().kind if args.device == "auto" else args.device
    print(f"Found {len(image_paths)} images  |  model={args.model}  |  device={device}")

    cfg = load_yaml(args.config)
    weights_path = resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    predictor = Predictor.from_config(cfg, model)

    summary: list[dict] = []
    t0 = time.time()

    for i, image_path in enumerate(image_paths, 1):
        t_img = time.time()
        instances = predictor(read_image(image_path))

        boxes   = instances.pred_boxes.tensor.tolist()
        scores  = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        detections = [
            {
                "class": COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls),
                "score": round(score, 4),
                "box":   [round(v, 1) for v in box],
            }
            for box, score, cls in zip(boxes, scores, classes, strict=True)
            if score >= args.threshold
        ]

        out_path = output_dir / (image_path.stem + ".json")
        out_path.write_text(json.dumps(detections, indent=2))

        elapsed = time.time() - t_img
        print(f"  [{i:>{len(str(len(image_paths)))}}/{len(image_paths)}]  "
              f"{image_path.name:<40s}  {len(detections):3d} detections  {elapsed:.2f}s")

        summary.append({"image": image_path.name, "detections": detections})

    total = time.time() - t0
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone — {len(image_paths)} images in {total:.1f}s "
          f"({total / len(image_paths):.2f}s avg)  →  {output_dir}")


if __name__ == "__main__":
    main()
