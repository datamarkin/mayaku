"""Detect people and estimate their pose with Keypoint R-CNN.

Zero-configuration quickstart — downloads model and sample image automatically:

    python examples/predict_keypoint.py

Run on your own image or switch model:

    python examples/predict_keypoint.py --image photo.jpg
    python examples/predict_keypoint.py --image photo.jpg --model keypoint_rcnn_R_101_FPN_3x
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

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
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
    parser = argparse.ArgumentParser(description="Keypoint R-CNN pose estimation on a single image.")
    parser.add_argument("--image", type=Path, help="Path to image (default: COCO sample)")
    parser.add_argument("--model", default="keypoint_rcnn_R_50_FPN_3x",
                        help="Model name (auto-downloaded) or path to .pth")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config (default: bundled config matching --model)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Person detection score threshold")
    parser.add_argument("--vis-threshold", type=float, default=0.5,
                        help="Keypoint visibility threshold")
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--output", type=Path, help="Save results JSON here")
    args = parser.parse_args()

    if args.config is not None:
        config_path: Path = args.config
    else:
        from mayaku import configs
        config_path = configs.path(args.model)
    image_path = args.image or _ensure_sample_image()
    device = Device.auto().kind if args.device == "auto" else args.device

    print(f"Config : {config_path}")
    print(f"Model  : {args.model}")
    print(f"Device : {device}")

    cfg = load_yaml(config_path)
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
    has_kpts = hasattr(instances, "pred_keypoints") and instances.pred_keypoints is not None

    results = []
    print(f"\nPose results in '{image_path.name}' (threshold ≥ {args.threshold}):")
    for i, (box, score) in enumerate(zip(boxes, scores, strict=True)):
        if score < args.threshold:
            continue
        print(f"  Person {i + 1}  score={score:.3f}  box={[round(v, 1) for v in box]}")

        keypoints = {}
        if has_kpts:
            kpts = instances.pred_keypoints[i]  # (17, 3): x, y, visibility
            for j, name in enumerate(KEYPOINT_NAMES):
                x, y, vis = kpts[j].tolist()
                if vis >= args.vis_threshold:
                    print(f"    {name:<16s}  ({x:.1f}, {y:.1f})  vis={vis:.2f}")
                    keypoints[name] = {"x": round(x, 1), "y": round(y, 1), "visibility": round(vis, 3)}

        results.append({"score": round(score, 4), "box": [round(v, 1) for v in box], "keypoints": keypoints})

    if not results:
        print("  (no persons detected above threshold)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
