"""Run inference using a previously-exported ONNX backbone.

    python examples/export_onnx.py     # produces examples/outputs/model.onnx
    python examples/run_onnx.py        # uses it for inference

Closes the export → deploy loop: the ONNX artifact runs the backbone+FPN
via ONNX Runtime; PyTorch handles RPN / NMS / heads as usual.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from mayaku.inference import Predictor
from mayaku.inference.export import ONNXBackbone
from mayaku.utils.image import read_image

ONNX = Path("examples/outputs/model.onnx")
ASSETS = Path(__file__).parent / "assets"
SAMPLE_URL = "https://dtmfiles.com/assets/dog.jpg"

if not ONNX.exists():
    raise SystemExit(f"{ONNX} not found — run examples/export_onnx.py first.")


def _sample_image() -> Path:
    p = ASSETS / "sample.jpg"
    if not p.exists():
        ASSETS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, p)
    return p


predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
size_div = getattr(predictor.model.backbone, "size_divisibility", 32)
predictor.model.backbone = ONNXBackbone(ONNX, size_divisibility=size_div)
print(f"Providers: {predictor.model.backbone.active_providers}")

instances = predictor(read_image(_sample_image()))
for box, score, cls in zip(
    instances.pred_boxes.tensor.tolist(),
    instances.scores.tolist(),
    instances.pred_classes.tolist(),
    strict=True,
):
    if score >= 0.5:
        print(f"  class={cls:<3d}  score={score:.3f}  bbox={[round(v, 1) for v in box]}")
