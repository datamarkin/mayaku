"""Train RF-DETR on every RF100-VL dataset — library default, nothing tuned.

    python train_rfdetr.py --datasets <coco_root> [--out results/rfdetr]

The train call below is RF-DETR's stock recipe (its own default epochs, LR,
augmentation, everything). The only added argument is checkpoint cadence, so the
run leaves a per-epoch trail to score afterwards. Scoring lives in eval_rfdetr.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common

try:
    from rfdetr import RFDETRNano
except ImportError as exc:
    raise SystemExit("RF-DETR is not installed. Run: pip install rfdetr") from exc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", required=True, help="COCO dataset root (one subdir per dataset)")
    p.add_argument("--out", default="results/rfdetr")
    args = p.parse_args()

    out_root = Path(args.out)
    for name, dataset_dir in common.iter_datasets(Path(args.datasets)):
        run = out_root / name
        if (run / "meta.json").exists():
            print(f"[rfdetr] skip {name} (already trained)")
            continue
        print(f"[rfdetr] train {name}")
        t0 = time.time()
        RFDETRNano().train(
            dataset_dir=str(dataset_dir),
            checkpoint_interval=1,
            output_dir=str(run),
        )
        common.write_meta(run, lib="rfdetr", dataset=name, t0=t0)


if __name__ == "__main__":
    main()
