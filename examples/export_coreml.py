"""Export a Mayaku model to CoreML for Apple Silicon / iOS / macOS deployment.

Zero-configuration quickstart — downloads model automatically:

    python examples/export_coreml.py

Requires: pip install mayaku[coreml]   (macOS only)

The exported .mlpackage covers backbone + FPN. Inference still uses the
Python predictor for RPN, NMS, and head postprocessing — the CoreML artifact
is the feature extractor that runs on ANE or GPU.

Performance note (measured on M1 Max, R-50 FPN 736×1344):
  - MPS eager ≈ CoreML CPU_AND_GPU within ~5-15% — neither is a clear winner
  - CoreML shines on pure classifier-shaped graphs (6× over PyTorch on 224×224)
  - For R-CNN, CoreML's value is the deployment artifact format (.mlpackage
    for iOS/macOS apps), not raw throughput on a developer machine.

See docs/decisions/004-coreml-export-positioning.md for the full analysis.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import torch

from mayaku.backends.device import Device
from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import load_yaml
from mayaku.inference.export import CoreMLExporter

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if platform.system() != "Darwin":
        raise SystemExit("CoreML export and inference require macOS.")

    parser = argparse.ArgumentParser(description="Export a Mayaku model to CoreML (.mlpackage).")
    parser.add_argument("--model", default="faster_rcnn_R_50_FPN_3x",
                        help="Model name (auto-downloaded) or leave empty when using --weights")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config (default: bundled config matching --model)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to .pth checkpoint (overrides --model auto-download)")
    parser.add_argument("--output", type=Path, default=Path("examples/outputs/model.mlpackage"))
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                        help="fp16 targets ANE/GPU; fp32 is CPU-only (slower but always available)")
    parser.add_argument("--compute-units", default="CPU_AND_GPU",
                        choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"],
                        help="Apple compute units (CPU_AND_GPU is fastest for R-CNN on most Macs)")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1344)
    args = parser.parse_args()

    if args.config is not None:
        config_path: Path = args.config
    else:
        from mayaku import configs
        config_path = configs.path(args.model)
    device = Device.auto().kind
    print(f"Config        : {config_path}")
    print(f"Device        : {device}")
    print(f"Precision     : {args.precision}")
    print(f"Compute units : {args.compute_units}")

    cfg = load_yaml(config_path)
    weights_path = args.weights or resolve_weights(args.model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = build_detector(cfg)
    model.load_state_dict(state)
    model = model.eval().to(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    exporter = CoreMLExporter(
        compute_precision=args.precision,
        compute_units=args.compute_units,
    )
    sample = torch.zeros(1, 3, args.height, args.width, device=device)

    print(f"\nExporting {args.height}×{args.width} → {args.output}")
    result = exporter.export(model, sample, args.output)
    print(f"  ✓ Written  {result.path}")

    # fp16 weights drift ~2e-2 on FPN outputs — well within deployment
    # tolerance but over the parity_check default (1e-2, tuned for fp32).
    atol = 5e-2 if args.precision == "fp16" else 1e-3
    print("\nVerifying parity (eager vs CoreML)…")
    parity = exporter.parity_check(model, args.output, sample, atol=atol)
    for name, (abs_err, _rel_err) in parity.per_output.items():
        status = "✓" if abs_err <= parity.atol else "✗"
        print(f"  {status} {name:<6s}  max_abs={abs_err:.2e}  (atol {parity.atol:.0e})")

    if parity.passed:
        print(f"\nParity PASS — CoreML output matches PyTorch eager within {args.precision} tolerance.")
    else:
        print("\nParity FAIL — check precision settings.")


if __name__ == "__main__":
    main()
