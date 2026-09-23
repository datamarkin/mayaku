"""Check multi-GPU training on a CUDA host: one command, a pass/fail table.

    python tools/check_multi_gpu.py --annotations A --images I \
        --val-annotations VA --val-images VI [--gpus 2] [--epochs 3] [--tier n]

1. Runs the NCCL tests (`pytest -m multi_gpu` with MAYAKU_DEVICE=cuda): every
   rank on its own GPU, and the DDP gradient equal to the single-process one.
2. Trains the same recipe on 1 GPU and on `--gpus` GPUs (with SyncBatchNorm,
   and, with --no-sync-bn-too, without) and compares the per-epoch losses,
   the throughput and the final AP. The recipe's batch is global, so the runs
   train the same thing; they differ only by float noise and data order.

Use a subset of a few thousand images: the point is agreement, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_tests() -> bool:
    env = {**os.environ, "MAYAKU_DEVICE": "cuda"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "multi_gpu", "tests/unit/test_distributed.py"]
    return subprocess.run(cmd, env=env).returncode == 0


def train(args, n_images: int, gpus: int, sync_bn: bool, out: Path) -> dict:
    import mayaku

    model = {} if args.weights else {"model": {"tier": args.tier}}
    result = mayaku.train(
        weights=args.weights,
        train_annotations=args.annotations, train_images=args.images,
        val_annotations=args.val_annotations, val_images=args.val_images,
        output_dir=out, num_epochs=args.epochs, num_gpus=gpus, device="cuda",
        overrides={**model, "train": {"batch": args.batch, "sync_bn": sync_bn}})
    log = [json.loads(line) for line in (out / "train" / "log.jsonl").open()]
    return {"loss": [round(r["box"] + r["cls"] + r["dfl"], 4) for r in log],
            "img_s": round(n_images / min(r["secs"] for r in log), 1),
            "AP": round(result["metrics"]["AP"], 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--annotations", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--val-annotations", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--weights", help="warm-start from a checkpoint (else from scratch)")
    p.add_argument("--tier", default="n")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32, help="global batch")
    p.add_argument("--no-sync-bn-too", action="store_true",
                   help="also run the multi-GPU case without SyncBatchNorm")
    p.add_argument("--out", default="runs/multi_gpu_check")
    args = p.parse_args()

    ok = run_tests()
    print("NCCL tests:", "PASS" if ok else "FAIL")
    out = Path(args.out)
    n = len(json.loads(Path(args.annotations).read_text())["images"])
    runs = {"1 GPU": train(args, n, 1, True, out / "1gpu"),
            f"{args.gpus} GPU sync-bn": train(args, n, args.gpus, True, out / "ngpu_syncbn")}
    if args.no_sync_bn_too:
        runs[f"{args.gpus} GPU plain-bn"] = train(args, n, args.gpus, False,
                                                  out / "ngpu_plainbn")

    ref = runs["1 GPU"]
    print(f"\n{'run':<20}{'img/s':>9}{'speedup':>9}{'AP':>9}{'dAP':>8}  loss per epoch")
    for name, r in runs.items():
        print(f"{name:<20}{r['img_s']:>9}{r['img_s'] / ref['img_s']:>9.2f}{r['AP']:>9}"
              f"{r['AP'] - ref['AP']:>+8.4f}  {r['loss']}")
    # same recipe: the last epoch's loss within 5 %, the AP within 1 point
    for name, r in list(runs.items())[1:]:
        close = abs(r["loss"][-1] / ref["loss"][-1] - 1) < 0.05 and abs(r["AP"] - ref["AP"]) < 0.01
        ok &= close
        print(f"{name}: {'PASS' if close else 'CHECK'}")
    (out / "summary.json").write_text(json.dumps(runs, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
