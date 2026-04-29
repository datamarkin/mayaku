"""Detect objects in a video, frame-by-frame, with boxes drawn over each frame.

Zero-configuration quickstart — downloads model and a sample highway clip:

    python examples/predict_video.py

Run on your own video, switch model, or limit frames for a quick smoke test:

    python examples/predict_video.py --video myclip.mp4
    python examples/predict_video.py --video 0                  # webcam (cv2 device id)
    python examples/predict_video.py --max-frames 60 --threshold 0.6

Requires OpenCV (``pip install opencv-python``); not a hard dependency of
mayaku itself.

The exported video is encoded with mp4v — most players handle it. Each
frame runs through the eager Predictor; on Apple Silicon you should see
~5-10 FPS at 720p depending on model size.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference import Predictor

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / "assets"
SAMPLE_URL = "https://dtmfiles.com/assets/highway.mp4"

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


def _ensure_sample_video() -> Path:
    path = ASSETS / "highway.mp4"
    if path.exists():
        return path
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample video from {SAMPLE_URL} …")
    urllib.request.urlretrieve(SAMPLE_URL, path)
    return path


def _open_capture(spec: str):
    import cv2
    if spec.isdigit():
        cap = cv2.VideoCapture(int(spec))
    else:
        cap = cv2.VideoCapture(spec)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {spec}")
    return cap


def _draw(frame_bgr, boxes, scores, classes, threshold: float) -> int:
    import cv2
    drawn = 0
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes, strict=True):
        if score < threshold:
            continue
        label = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame_bgr, p1, p2, (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (p1[0], p1[1] - th - 4), (p1[0] + tw + 4, p1[1]), (0, 255, 0), -1)
        cv2.putText(frame_bgr, text, (p1[0] + 2, p1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        drawn += 1
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser(description="Faster R-CNN detection on a video.")
    parser.add_argument("--video", type=str, default=None,
                        help="Path or webcam index (e.g. '0'); default: download highway.mp4")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config (default: bundled config matching --model)")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--output", type=Path,
                        default=Path("examples/outputs/highway_detected.mp4"),
                        help="Output mp4 path; pass '' to skip writing")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = process whole video)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as e:
        raise SystemExit("predict_video.py requires OpenCV: pip install opencv-python") from e

    if args.config is not None:
        config_path: Path = args.config
    else:
        from mayaku import configs
        config_path = configs.path(args.model)
    video_spec = args.video if args.video else str(_ensure_sample_video())
    device = Device.auto().kind if args.device == "auto" else args.device

    print(f"Config : {config_path}")
    print(f"Model  : {args.model}")
    print(f"Device : {device}")
    print(f"Source : {video_spec}")

    cfg = load_yaml(config_path)
    weights_path = resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)
    predictor = Predictor.from_config(cfg, model)

    cap = _open_capture(video_spec)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output and str(args.output):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    print(f"\nProcessing {width}×{height} @ {fps:.1f} FPS")

    n_frames = 0
    n_dets = 0
    t0 = time.perf_counter()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        instances = predictor(frame_rgb)
        boxes = instances.pred_boxes.tensor.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        n_dets += _draw(frame_bgr, boxes, scores, classes, args.threshold)
        if writer is not None:
            writer.write(frame_bgr)
        n_frames += 1
        if n_frames % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  frame {n_frames:>5d}  {n_frames / elapsed:5.1f} FPS  "
                  f"({n_dets} detections drawn so far)")
        if args.max_frames and n_frames >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - t0
    fps_proc = n_frames / elapsed if elapsed > 0 else 0.0
    print(f"\nDone — {n_frames} frames in {elapsed:.1f}s ({fps_proc:.1f} FPS), "
          f"{n_dets} detections ≥ {args.threshold:.2f}")
    if writer is not None:
        print(f"Wrote → {args.output}")


if __name__ == "__main__":
    main()
