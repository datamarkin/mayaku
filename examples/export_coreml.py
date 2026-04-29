"""Export Faster R-CNN backbone+FPN to CoreML (.mlpackage).

    pip install mayaku[coreml]   # macOS only
    python examples/export_coreml.py

The exported .mlpackage is a deployment artifact for iOS / macOS apps;
RPN / NMS / heads stay in Python on the inference side.
"""

from __future__ import annotations

import platform
from pathlib import Path

import torch

from mayaku.inference import Predictor
from mayaku.inference.export import CoreMLExporter

if platform.system() != "Darwin":
    raise SystemExit("CoreML export requires macOS.")

OUT = Path(__file__).parent / "outputs" / "model.mlpackage"
OUT.parent.mkdir(parents=True, exist_ok=True)

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
sample = torch.zeros(1, 3, 800, 1344, device=predictor.device)

exporter = CoreMLExporter(compute_precision="fp16", compute_units="CPU_AND_GPU")
result = exporter.export(predictor.model, sample, OUT)
print(f"✓ {result.path}")

# fp16 drift sits ~2e-2 on FPN; widen tolerance to match the realistic envelope.
parity = exporter.parity_check(predictor.model, OUT, sample, atol=5e-2)
for name, (abs_err, _) in parity.per_output.items():
    mark = "✓" if abs_err <= parity.atol else "✗"
    print(f"  {mark} {name:<3s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")
print("Parity:", "PASS" if parity.passed else "FAIL")
