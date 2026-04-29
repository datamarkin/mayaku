"""Run inference using a previously-exported ONNX backbone artifact.

Closes the export → deploy loop. The ``examples/export_onnx.py`` script
produces a ``model.onnx`` covering backbone + FPN; this script swaps that
artifact into the eager detector's ``model.backbone`` slot and runs the
rest of the pipeline (RPN, NMS, ROI heads) in PyTorch as usual.

Zero-configuration quickstart — auto-exports the model on first run:

    python examples/export_onnx.py        # produces examples/outputs/model.onnx
    python examples/run_onnx.py           # uses it for inference

Switch ONNX Runtime providers (CPU is the safe default; GPU/TensorRT
require the matching ``onnxruntime-gpu`` install):

    python examples/run_onnx.py --providers CUDAExecutionProvider,CPUExecutionProvider
    python examples/run_onnx.py --providers CoreMLExecutionProvider,CPUExecutionProvider

This is the deployment pattern most teams use: ONNX for portability,
PyTorch for the dynamic post-processing that doesn't trace cleanly.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference import Predictor
from mayaku.inference.export import ONNXBackbone
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
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection with the ONNX-exported backbone + eager heads.")
    parser.add_argument("--onnx", type=Path,
                        default=Path("examples/outputs/model.onnx"),
                        help="Exported .onnx file (run examples/export_onnx.py first)")
    parser.add_argument("--image", type=Path, help="Path to image (default: COCO sample)")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated ORT providers; default CPUExecutionProvider")
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1344)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto "
                        "(controls eager heads; backbone runs via ORT providers)")
    args = parser.parse_args()

    if not args.onnx.exists():
        raise SystemExit(
            f"ONNX artifact not found at {args.onnx}. "
            "Run `python examples/export_onnx.py` first."
        )

    image_path = args.image or _ensure_sample_image()
    device = Device.auto().kind if args.device == "auto" else args.device
    providers = (
        tuple(p.strip() for p in args.providers.split(",") if p.strip())
        if args.providers else None
    )

    print(f"Config    : {args.config}")
    print(f"ONNX      : {args.onnx}")
    print(f"Heads on  : {device}")

    cfg = load_yaml(args.config)
    weights_path = resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    prev_size_div = getattr(model.backbone, "size_divisibility", 32)
    onnx_backbone = ONNXBackbone(
        args.onnx,
        input_height=args.input_height,
        input_width=args.input_width,
        size_divisibility=prev_size_div,
        providers=providers,
    )
    model.backbone = onnx_backbone
    print(f"Providers : {onnx_backbone.active_providers}")

    predictor = Predictor.from_config(cfg, model)
    instances = predictor(read_image(image_path))

    boxes = instances.pred_boxes.tensor.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    print(f"\nDetections in '{image_path.name}' (threshold ≥ {args.threshold}):")
    n = 0
    for box, score, cls in zip(boxes, scores, classes, strict=True):
        if score < args.threshold:
            continue
        label = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
        print(f"  {label:<20s}  {score:.3f}  {[round(v, 1) for v in box]}")
        n += 1
    if not n:
        print("  (no detections above threshold)")


if __name__ == "__main__":
    main()
