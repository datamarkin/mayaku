"""Run inference using a previously-exported CoreML backbone artifact.

macOS counterpart to ``run_onnx.py``. Closes the export → deploy loop
for Apple Silicon: the ``.mlpackage`` produced by ``export_coreml.py``
runs the backbone + FPN on ANE / GPU via Core ML, while RPN, NMS, and
ROI heads stay in PyTorch.

Zero-configuration quickstart:

    python examples/export_coreml.py     # produces examples/outputs/model.mlpackage
    python examples/run_coreml.py        # uses it for inference

Switch compute units to compare runtime targets:

    python examples/run_coreml.py --compute-units ALL          # ANE + GPU
    python examples/run_coreml.py --compute-units CPU_ONLY     # debugging baseline

Performance note: on M1/M2/M3 the eager MPS path is usually within
10-15% of CoreML CPU_AND_GPU on R-CNN — the CoreML artifact's value is
deployment portability (.mlpackage for iOS/macOS apps), not raw speed.
"""

from __future__ import annotations

import argparse
import platform
import urllib.request
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference import Predictor
from mayaku.inference.export import CoreMLBackbone
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
    if platform.system() != "Darwin":
        raise SystemExit("CoreML inference requires macOS. On Linux/Windows use run_onnx.py.")

    parser = argparse.ArgumentParser(
        description="Run detection with the CoreML-exported backbone + eager heads.")
    parser.add_argument("--mlpackage", type=Path,
                        default=Path("examples/outputs/model.mlpackage"),
                        help="Exported .mlpackage (run examples/export_coreml.py first)")
    parser.add_argument("--image", type=Path, help="Path to image (default: COCO sample)")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--compute-units", default="CPU_AND_GPU",
                        choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"])
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1344)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto",
                        help="cpu | cuda | mps | auto (eager heads; backbone runs on ANE/GPU)")
    args = parser.parse_args()

    if not args.mlpackage.exists():
        raise SystemExit(
            f"CoreML artifact not found at {args.mlpackage}. "
            "Run `python examples/export_coreml.py` first."
        )

    image_path = args.image or _ensure_sample_image()
    device = Device.auto().kind if args.device == "auto" else args.device

    print(f"Config        : {args.config}")
    print(f"mlpackage     : {args.mlpackage}")
    print(f"Compute units : {args.compute_units}")
    print(f"Heads on      : {device}")

    cfg = load_yaml(args.config)
    weights_path = resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    prev_size_div = getattr(model.backbone, "size_divisibility", 32)
    coreml_backbone = CoreMLBackbone(
        args.mlpackage,
        input_height=args.input_height,
        input_width=args.input_width,
        size_divisibility=prev_size_div,
        compute_units=args.compute_units,
    )
    model.backbone = coreml_backbone

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
