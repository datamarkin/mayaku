"""Export Faster R-CNN backbone+FPN to a TensorRT engine.

    pip install mayaku[tensorrt]   # CUDA Linux/Windows only
    python examples/export_tensorrt.py

Fails fast on macOS or hosts without a CUDA GPU.
"""

from __future__ import annotations

import platform
from pathlib import Path

import torch

from mayaku.inference import Predictor
from mayaku.inference.export import TensorRTExporter

if platform.system() == "Darwin":
    raise SystemExit("TensorRT is not supported on macOS — try export_coreml.py instead.")
if not torch.cuda.is_available():
    raise SystemExit("TensorRT requires a CUDA GPU — try export_onnx.py for CPU.")

OUT = Path(__file__).parent / "outputs" / "model.engine"
OUT.parent.mkdir(parents=True, exist_ok=True)

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x", device="cuda")
sample = torch.zeros(1, 3, 800, 1344, device=predictor.device)

exporter = TensorRTExporter(fp16=False)
print("Building TRT engine (can take several minutes the first time)…")
result = exporter.export(predictor.model, sample, OUT)
print(f"✓ {result.path}  ({result.path.stat().st_size / 1e6:.1f} MB)")

parity = exporter.parity_check(predictor.model, OUT, sample)
for name, (abs_err, _) in parity.per_output.items():
    mark = "✓" if abs_err <= parity.atol else "✗"
    print(f"  {mark} {name:<3s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")
print("Parity:", "PASS" if parity.passed else "FAIL")
