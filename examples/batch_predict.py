"""Detect on every image in a directory; write one JSON per input.

    python examples/batch_predict.py /path/to/images
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mayaku.inference import Predictor
from mayaku.utils.image import read_image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if len(sys.argv) != 2:
    raise SystemExit("usage: python examples/batch_predict.py <image-dir>")
images_dir = Path(sys.argv[1])
out_dir = images_dir / "detections"
out_dir.mkdir(exist_ok=True)

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
print(f"Processing {len(image_paths)} images …")

for img_path in image_paths:
    instances = predictor(read_image(img_path))
    payload = [
        {"class": int(c), "score": round(float(s), 4), "box": [round(float(v), 1) for v in b]}
        for b, s, c in zip(
            instances.pred_boxes.tensor.tolist(),
            instances.scores.tolist(),
            instances.pred_classes.tolist(),
            strict=True,
        )
        if s >= 0.5
    ]
    (out_dir / (img_path.stem + ".json")).write_text(json.dumps(payload, indent=2))
    print(f"  {img_path.name}: {len(payload)} detections")
