"""Detect objects in a single image with Faster R-CNN.

Zero-configuration quickstart — downloads model and sample image automatically:

    python examples/predict.py

Run on your own image or switch model:

    python examples/predict.py --image photo.jpg
    python examples/predict.py --image photo.jpg --model faster_rcnn_R_101_FPN_3x
    python examples/predict.py --image photo.jpg --device cpu
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference import Predictor
from mayaku.utils.image import read_image

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / "assets"
SAMPLE_URL = "https://dtmfiles.com/assets/dog.jpg"

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


def _ensure_sample_image() -> Path:
    path = ASSETS / "sample.jpg"
    if path.exists():
        return path
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("Downloading sample image…")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, path)
    except Exception:
        from PIL import Image
        Image.new("RGB", (640, 480), color=(100, 149, 237)).save(path)
        print("  (network unavailable — using placeholder; pass --image for real detections)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Faster R-CNN detection on a single image.")
    parser.add_argument("--image", type=Path, help="Path to image (default: COCO sample)")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x",
                        help="Model name (auto-downloaded) or path to .pth")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--output", type=Path, help="Save detections JSON here")
    args = parser.parse_args()

    image_path = args.image or _ensure_sample_image()
    device = Device.auto().kind if args.device == "auto" else args.device

    print(f"Config : {args.config}")
    print(f"Model  : {args.model}")
    print(f"Device : {device}")

    cfg = load_yaml(args.config)
    weights_path = resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    predictor = Predictor.from_config(cfg, model)
    instances = predictor(read_image(image_path))

    boxes   = instances.pred_boxes.tensor.tolist()
    scores  = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    print(f"\nDetections in '{image_path.name}' (threshold ≥ {args.threshold}):")
    for box, score, cls in zip(boxes, scores, classes, strict=True):
        if score < args.threshold:
            continue
        label = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
        print(f"  {label:<20s}  {score:.3f}  {[round(v, 1) for v in box]}")
        results.append({"class": label, "score": round(score, 4), "box": [round(v, 1) for v in box]})

    if not results:
        print("  (no detections above threshold)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
