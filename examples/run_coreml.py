"""Run inference using a previously-exported CoreML backbone (macOS only).

    python examples/export_coreml.py    # produces examples/outputs/model.mlpackage
    python examples/run_coreml.py       # uses it for inference

Hybrid: CoreML runs backbone+FPN on ANE/GPU; PyTorch handles RPN/NMS/heads.
"""

from __future__ import annotations

import platform
import urllib.request
from pathlib import Path

from mayaku.inference import Predictor
from mayaku.inference.export import CoreMLBackbone
from mayaku.utils.image import read_image

if platform.system() != "Darwin":
    raise SystemExit("CoreML inference requires macOS — try run_onnx.py instead.")

MLPACKAGE = Path("examples/outputs/model.mlpackage")
ASSETS = Path(__file__).parent / "assets"
SAMPLE_URL = "https://dtmfiles.com/assets/dog.jpg"

if not MLPACKAGE.exists():
    raise SystemExit(f"{MLPACKAGE} not found — run examples/export_coreml.py first.")


def _sample_image() -> Path:
    p = ASSETS / "sample.jpg"
    if not p.exists():
        ASSETS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, p)
    return p


predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
size_div = getattr(predictor.model.backbone, "size_divisibility", 32)
predictor.model.backbone = CoreMLBackbone(
    MLPACKAGE, size_divisibility=size_div, compute_units="CPU_AND_GPU"
)

instances = predictor(read_image(_sample_image()))
for box, score, cls in zip(
    instances.pred_boxes.tensor.tolist(),
    instances.scores.tolist(),
    instances.pred_classes.tolist(),
    strict=True,
):
    if score >= 0.5:
        print(f"  class={cls:<3d}  score={score:.3f}  bbox={[round(v, 1) for v in box]}")
