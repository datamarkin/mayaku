"""Instance segmentation with Mask R-CNN.

    python examples/predict_segmentation.py
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


predictor = Predictor.from_pretrained("mask_rcnn_R_50_FPN_3x")
instances = predictor(read_image(_sample_image()))

masks = instances.pred_masks  # (N, H, W) bool
for i, (score, cls) in enumerate(
    zip(instances.scores.tolist(), instances.pred_classes.tolist(), strict=True)
):
    if score >= 0.5:
        n_pixels = int(masks[i].sum().item())
        print(f"  class={cls:<3d}  score={score:.3f}  mask_pixels={n_pixels}")
