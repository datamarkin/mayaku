"""Train Mayaku on every RF100-VL dataset — library default, nothing tuned.

    python train_mayaku.py --datasets <coco_root> [--out results/mayaku]

The train call below is Mayaku's stock fine-tune: point at the pretrained nano
weights (a self-describing checkpoint — architecture comes from it) and let
auto-config derive the recipe from the dataset. Same shape as the other legs'
`yolo26n.pt` / `RFDETRNano()` — pretrained model + library defaults. Trains on
the dataset's `train/` split and validates on `valid/` every epoch, like the
others. The added arguments are per-epoch checkpoints and per-epoch eval; the
final (uniform pycocotools) scoring is offline in eval_mayaku.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common
from mayaku import train

# Pretrained nano — a bundled model name (downloads) or a local .pth checkpoint path.
WEIGHTS = "mayaku-n-det"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", required=True, help="COCO dataset root (one subdir per dataset)")
    p.add_argument("--out", default="results/mayaku")
    args = p.parse_args()

    out_root = Path(args.out)
    for name, dataset_dir in common.iter_datasets(Path(args.datasets)):
        run = out_root / name
        if (run / "meta.json").exists():
            print(f"[mayaku] skip {name} (already trained)")
            continue
        print(f"[mayaku] train {name}")
        split = common.train_split(dataset_dir)  # train/ split, same as the other legs
        val = common.val_split(dataset_dir)  # valid/ split — per-epoch eval, like the other legs
        t0 = time.time()
        train(
            weights=WEIGHTS,
            train_annotations=split / common.COCO_ANN,
            train_images=split,
            val_annotations=val / common.COCO_ANN,
            val_images=val,
            output_dir=run,
            # per-epoch checkpoints; the only added argument (a curve needs them).
            overrides={"solver": {"checkpoint_period": 1}},
        )
        common.write_meta(run, lib="mayaku", dataset=name, t0=t0)


if __name__ == "__main__":
    main()
