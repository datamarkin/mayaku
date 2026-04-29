"""Export Faster R-CNN backbone+FPN to OpenVINO IR (.xml + .bin).

    pip install mayaku[openvino]
    python examples/export_openvino.py

OpenVINO is best on Intel CPUs (especially AVX-512) and iGPUs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mayaku.inference import Predictor
from mayaku.inference.export import OpenVINOExporter

OUT = Path(__file__).parent / "outputs" / "model.xml"
OUT.parent.mkdir(parents=True, exist_ok=True)

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
sample = torch.zeros(1, 3, 800, 1344, device=predictor.device)

exporter = OpenVINOExporter(compress_to_fp16=False)
result = exporter.export(predictor.model, sample, OUT)
bin_path = OUT.with_suffix(".bin")
print(f"✓ {result.path}  ({result.path.stat().st_size / 1e6:.1f} MB)")
print(f"✓ {bin_path}  ({bin_path.stat().st_size / 1e6:.1f} MB)")

parity = exporter.parity_check(predictor.model, OUT, sample)
for name, (abs_err, _) in parity.per_output.items():
    mark = "✓" if abs_err <= parity.atol else "✗"
    print(f"  {mark} {name:<3s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")
print("Parity:", "PASS" if parity.passed else "FAIL")
