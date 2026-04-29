"""Detect objects in an image with Faster R-CNN.

    python examples/predict.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from mayaku.inference import Predictor
from mayaku.utils.image import read_image

ASSETS = Path(__file__).parent / "assets"
SAMPLE_URL = "https://dtmfiles.com/assets/dog.jpg"


def _sample_image() -> Path:
    p = ASSETS / "sample.jpg"
    if not p.exists():
        ASSETS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, p)
    return p


predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
instances = predictor(read_image(_sample_image()))

for box, score, cls in zip(
    instances.pred_boxes.tensor.tolist(),
    instances.scores.tolist(),
    instances.pred_classes.tolist(),
    strict=True,
):
    if score >= 0.5:
        print(f"  class={cls:<3d}  score={score:.3f}  bbox={[round(v, 1) for v in box]}")
