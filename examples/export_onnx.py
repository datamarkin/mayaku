"""Export a Mayaku model to ONNX and verify numerical parity.

Zero-configuration quickstart — downloads model automatically:

    python examples/export_onnx.py

Export a specific model or use your own checkpoint:

    python examples/export_onnx.py --model faster_rcnn_R_101_FPN_3x
    python examples/export_onnx.py --weights runs/myproject/model_final.pth \\
        --config configs/detection/faster_rcnn_R_50_FPN_3x.yaml

Requires: pip install mayaku[onnx]

The exported graph covers backbone + FPN (the compute-intensive, static part).
RPN top-k, per-class NMS, mask paste, and keypoint decoding stay in Python —
this is what makes the ONNX artifact portable across runtimes without custom ops.

For CoreML, OpenVINO, or TensorRT export run the equivalent CLI verbs:
    mayaku export coreml   ...
    mayaku export openvino ...
    mayaku export tensorrt ...   # CUDA Linux host required
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference.export import ONNXExporter

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Mayaku model to ONNX.")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x",
                        help="Model name (auto-downloaded) or leave empty when using --weights")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs/detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to .pth checkpoint (overrides --model auto-download)")
    parser.add_argument("--output", type=Path, default=Path("examples/outputs/model.onnx"))
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--fixed-shape", action="store_true",
                        help="Export with fixed H×W instead of dynamic axes (TensorRT-friendly)")
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    args = parser.parse_args()

    device = Device.auto().kind if args.device == "auto" else args.device
    print(f"Config : {args.config}")
    print(f"Device : {device}")

    cfg = load_yaml(args.config)
    weights_path = args.weights or resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    exporter = ONNXExporter(dynamic_input_shape=not args.fixed_shape)
    sample = torch.zeros(1, 3, args.height, args.width, device=device)

    print(f"\nExporting {args.height}×{args.width} "
          f"({'fixed' if args.fixed_shape else 'dynamic'} shape) → {args.output}")
    result = exporter.export(model, sample, args.output)
    print(f"  ✓ Written  {result.path}  ({result.path.stat().st_size / 1e6:.1f} MB)")

    print("\nVerifying parity (eager vs ONNX Runtime)…")
    parity = exporter.parity_check(model, args.output, sample)
    for name, (abs_err, _rel_err) in parity.per_output.items():
        status = "✓" if abs_err <= parity.atol else "✗"
        print(f"  {status} {name:<6s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")

    if parity.passed:
        print("\nParity PASS — ONNX output matches PyTorch eager within 1e-3.")
    else:
        print("\nParity FAIL — check your model and opset version.")


if __name__ == "__main__":
    main()
