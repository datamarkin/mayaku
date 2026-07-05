"""Train YOLO on every RF100-VL dataset — library default, nothing tuned.

    python train_yolo.py --datasets <coco_root> [--out results/yolo] [--device 0]

The train call below is Ultralytics' stock recipe (its own default epochs, LR,
augmentation, everything). The only added argument is checkpoint cadence, so the
run leaves a per-epoch trail to score afterwards. YOLO needs YOLO-format labels,
so common.to_yolo prepares them (data-format prep, not a recipe change). Scoring
lives in eval_yolo.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Ultralytics is not installed. Run: pip install ultralytics") from exc

MODEL = "yolo26n.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", required=True, help="COCO dataset root (one subdir per dataset)")
    p.add_argument("--out", default="results/yolo")
    p.add_argument("--device", default="")
    args = p.parse_args()

    out_root = Path(args.out)
    for name, dataset_dir in common.iter_datasets(Path(args.datasets)):
        run = out_root / name
        if (run / "meta.json").exists():
            print(f"[yolo] skip {name} (already trained)")
            continue
        print(f"[yolo] train {name}")
        t0 = time.time()
        kw = {"device": args.device} if args.device else {}
        YOLO(MODEL).train(
            data=str(common.to_yolo(dataset_dir)),
            save_period=1,
            project=str(out_root),
            name=name,
            exist_ok=True,
            **kw,
        )
        common.write_meta(run, lib="yolo", dataset=name, t0=t0)


if __name__ == "__main__":
    main()
