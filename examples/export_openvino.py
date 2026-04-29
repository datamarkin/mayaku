"""Export a Mayaku model to OpenVINO IR (``.xml`` + ``.bin``) and verify parity.

Zero-configuration quickstart — downloads model automatically:

    python examples/export_openvino.py

Requires: pip install mayaku[openvino]

OpenVINO is Intel's deployment runtime — best on x86 CPUs (especially
Intel Core / Xeon with AVX-512), iGPUs, and VPUs. The exported IR
covers backbone + FPN; RPN top-k, per-class NMS, mask paste, and
keypoint decode stay in Python (same split as the ONNX / CoreML
exporters). The ``compress_to_fp16`` flag halves disk size at the cost
of ~1e-3 drift — leave it off for tight parity, turn it on for
deployment artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference.export import OpenVINOExporter

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Mayaku model to OpenVINO IR.")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x",
                        help="Model name (auto-downloaded) or leave empty when using --weights")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to .pth checkpoint (overrides --model auto-download)")
    parser.add_argument("--output", type=Path, default=Path("examples/outputs/model.xml"),
                        help="Output .xml path; the .bin is written next to it")
    parser.add_argument("--compress-fp16", action="store_true",
                        help="Compress weights to fp16 (smaller .bin, ~1e-3 drift)")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    args = parser.parse_args()

    device = Device.auto().kind if args.device == "auto" else args.device
    print(f"Config       : {args.config}")
    print(f"Device       : {device}")
    print(f"Compress fp16: {args.compress_fp16}")

    cfg = load_yaml(args.config)
    weights_path = args.weights or resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    exporter = OpenVINOExporter(compress_to_fp16=args.compress_fp16)
    sample = torch.zeros(1, 3, args.height, args.width, device=device)

    print(f"\nExporting {args.height}×{args.width} → {args.output}")
    result = exporter.export(model, sample, args.output)
    bin_path = args.output.with_suffix(".bin")
    xml_size = result.path.stat().st_size / 1e6
    bin_size = bin_path.stat().st_size / 1e6 if bin_path.exists() else 0.0
    print(f"  ✓ Written  {result.path}  ({xml_size:.1f} MB)")
    print(f"  ✓ Written  {bin_path}  ({bin_size:.1f} MB)")

    # fp16 weights drift ~1e-3; loosen tolerance to match if requested.
    atol = 5e-3 if args.compress_fp16 else 1e-3
    print("\nVerifying parity (eager vs OpenVINO CPU)…")
    parity = exporter.parity_check(model, args.output, sample, atol=atol)
    for name, (abs_err, _rel_err) in parity.per_output.items():
        status = "✓" if abs_err <= parity.atol else "✗"
        print(f"  {status} {name:<6s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")

    if parity.passed:
        print("\nParity PASS — OpenVINO output matches PyTorch eager.")
    else:
        print("\nParity FAIL — check compress_to_fp16 / quantisation settings.")


if __name__ == "__main__":
    main()
