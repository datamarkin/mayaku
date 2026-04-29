"""Detect frame-by-frame on a video; write an annotated mp4.

    python examples/predict_video.py [video.mp4]

If no path is given, downloads a sample highway clip. Requires opencv-python.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import cv2

from mayaku.inference import Predictor

ASSETS = Path(__file__).parent / "assets"
OUT = Path(__file__).parent / "outputs" / "annotated.mp4"
SAMPLE_URL = "https://dtmfiles.com/assets/highway.mp4"


def _sample_video() -> Path:
    p = ASSETS / "highway.mp4"
    if not p.exists():
        ASSETS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, p)
    return p


video = sys.argv[1] if len(sys.argv) > 1 else str(_sample_video())
predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
OUT.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

n, t0 = 0, time.perf_counter()
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    instances = predictor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    for box, score in zip(
        instances.pred_boxes.tensor.tolist(), instances.scores.tolist(), strict=True
    ):
        if score >= 0.6:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    writer.write(frame_bgr)
    n += 1

cap.release()
writer.release()
print(f"{n} frames in {time.perf_counter() - t0:.1f}s → {OUT}")
