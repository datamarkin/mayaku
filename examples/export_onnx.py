"""Export Faster R-CNN backbone+FPN to ONNX and verify parity.

    pip install mayaku[onnx]
    python examples/export_onnx.py

Backbone + FPN go to ONNX; RPN, NMS, mask paste, keypoint decode stay in
Python — same split for the CoreML / OpenVINO / TensorRT exporters.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mayaku.inference import Predictor
from mayaku.inference.export import ONNXExporter

OUT = Path(__file__).parent / "outputs" / "model.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
sample = torch.zeros(1, 3, 800, 1344, device=predictor.device)

exporter = ONNXExporter(dynamic_input_shape=True)
result = exporter.export(predictor.model, sample, OUT)
print(f"✓ {result.path}  ({result.path.stat().st_size / 1e6:.1f} MB)")

parity = exporter.parity_check(predictor.model, OUT, sample)
for name, (abs_err, _) in parity.per_output.items():
    mark = "✓" if abs_err <= parity.atol else "✗"
    print(f"  {mark} {name:<3s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")
print("Parity:", "PASS" if parity.passed else "FAIL")
