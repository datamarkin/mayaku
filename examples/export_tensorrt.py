"""Export a Mayaku model to a TensorRT serialised engine (``.engine``).

Requires CUDA Linux/Windows + ``pip install mayaku[tensorrt]``. On macOS
or hosts without a CUDA GPU this script will fail fast with a clear hint
— TensorRT is the deployment target for NVIDIA GPUs only; use ONNX
(`export_onnx.py`) for portable inference or CoreML (`export_coreml.py`)
for Apple Silicon.

Quickstart on a CUDA host:

    python examples/export_tensorrt.py
    python examples/export_tensorrt.py --fp16            # ~2x throughput
    python examples/export_tensorrt.py --workspace-gb 4  # bigger builder workspace

The exported engine covers backbone + FPN (the dense compute). RPN
top-k, per-class NMS, mask paste, and keypoint decode stay in Python.
TensorRT typically yields 3-10× over PyTorch eager on the same GPU,
with looser parity (~1e-2) due to optimised kernel selection. Set
``--fp16`` to halve memory and ~double throughput; expect parity to
loosen further.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import torch

from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference.export import TensorRTExporter

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if platform.system() == "Darwin":
        raise SystemExit(
            "TensorRT is not supported on macOS. Use export_coreml.py for "
            "Apple Silicon or export_onnx.py for portable inference."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "TensorRT requires a CUDA-enabled GPU. None detected. "
            "Use export_onnx.py for CPU inference."
        )

    parser = argparse.ArgumentParser(description="Export a Mayaku model to TensorRT.")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config (default: bundled config matching --model)")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("examples/outputs/model.engine"))
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 mode (~2x throughput, looser parity)")
    parser.add_argument("--workspace-gb", type=float, default=1.0,
                        help="TRT builder workspace cap in GiB (default 1.0)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1344)
    args = parser.parse_args()

    if args.config is not None:
        config_path: Path = args.config
    else:
        from mayaku import configs
        config_path = configs.path(args.model)
    device = "cuda"
    print(f"Config        : {config_path}")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0)})")
    print(f"FP16          : {args.fp16}")
    print(f"Workspace     : {args.workspace_gb:.1f} GiB")

    cfg = load_yaml(config_path)
    weights_path = args.weights or resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    exporter = TensorRTExporter(
        fp16=args.fp16,
        workspace_bytes=int(args.workspace_gb * (1 << 30)),
        opset=args.opset,
    )
    sample = torch.zeros(1, 3, args.height, args.width, device=device)

    print(f"\nBuilding TRT engine {args.height}×{args.width} → {args.output}")
    print("(this can take several minutes the first time)")
    try:
        result = exporter.export(model, sample, args.output)
    except ModuleNotFoundError as e:
        raise SystemExit(f"TensorRT export failed: {e}\n"
                         "Install with: pip install mayaku[tensorrt]") from e
    size_mb = result.path.stat().st_size / 1e6
    print(f"  ✓ Written  {result.path}  ({size_mb:.1f} MB)")

    # FP16 drift is real on TRT; loosen tolerance accordingly.
    atol = 5e-2 if args.fp16 else 1e-2
    print("\nVerifying parity (eager vs TensorRT)…")
    parity = exporter.parity_check(model, args.output, sample, atol=atol)
    for name, (abs_err, _rel_err) in parity.per_output.items():
        status = "✓" if abs_err <= parity.atol else "✗"
        print(f"  {status} {name:<6s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")

    if parity.passed:
        print(f"\nParity PASS — TRT output matches PyTorch eager within "
              f"{'fp16' if args.fp16 else 'fp32'} tolerance.")
    else:
        print("\nParity FAIL — kernel selection differences exceed tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
