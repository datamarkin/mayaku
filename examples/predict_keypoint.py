"""Human pose estimation with Keypoint R-CNN.

    python examples/predict_keypoint.py
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


predictor = Predictor.from_pretrained("keypoint_rcnn_R_50_FPN_3x")
instances = predictor(read_image(_sample_image()))

# (N, 17, 3) — last dim is (x, y, score) per keypoint.
kpts = instances.pred_keypoints
for i, score in enumerate(instances.scores.tolist()):
    if score < 0.5:
        continue
    visible = int((kpts[i, :, 2] > 0.5).sum().item())
    print(f"  person {i}  score={score:.3f}  visible_keypoints={visible}/17")
