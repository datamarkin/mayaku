"""YOLO leg of the RF100-VL AP-vs-wall-clock benchmark — one self-contained script.

    python run_yolo.py --datasets <coco_root> [--out results/yolo]
                       [--device 0] [--variants nano,small,medium,large]

For every variant and every dataset it does the same three steps in the same
order — nothing library-specific, nothing tuned:

  1. TRAIN with Ultralytics' stock recipe (its own default epochs, LR, aug — we
     don't even pass `epochs`). The only added argument is `save_period=1`, so the
     run leaves a per-epoch checkpoint trail; a learning curve needs it.
  2. SCORE every checkpoint with the shared pycocotools evaluator — the *same*
     metric as the RF-DETR and Mayaku legs, never YOLO's internal mAP. Loading a
     checkpoint with `YOLO(ckpt)` gives the EMA weights (Ultralytics loads
     `ckpt["ema"]` for inference), i.e. YOLO's default deploy model.
  3. PURGE that dataset's checkpoints once its curve.csv is written, so 100
     datasets never need 100 datasets of checkpoints on disk at once.

Wall-clock per checkpoint is the checkpoint file's mtime minus the train-start
`t0` in meta.json — no timing callbacks. Resumable: a (variant, dataset) with a
curve.csv is skipped; one trained but not yet scored is scored without retraining.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import common

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Ultralytics is not installed. Run: pip install ultralytics") from exc

# The four size variants — nano / small / medium / large — as their stock weights.
VARIANTS = {
    "nano": "yolo26n.pt",
    "small": "yolo26s.pt",
    "medium": "yolo26m.pt",
    "large": "yolo26l.pt",
}


def train_one(model_id: str, variant: str, name: str, dataset_dir: Path, out_root: Path, device: str = "") -> None:
    """Default YOLO fine-tune on one dataset; per-epoch checkpoints the only extra."""
    out_dir = out_root / variant  # Ultralytics writes out_dir/name/weights/epoch*.pt
    run = out_dir / name
    t0 = time.time()
    kw = {"device": device} if device else {}
    YOLO(model_id).train(
        data=str(common.to_yolo(dataset_dir)),
        save_period=1,
        project=str(out_dir),
        name=name,
        exist_ok=True,
        **kw,
    )
    common.write_meta(run, lib="yolo", variant=variant, dataset=name, t0=t0)


def score(run: Path, dataset_dir: Path, device: str) -> None:
    """Score every checkpoint against the COCO val split with shared pycocotools."""
    t0 = common.read_meta(run)["t0"]
    val = common.val_split(dataset_dir)
    gt = val / common.COCO_ANN
    cat_ids = common.category_ids(gt)
    val_images = common.images(gt, val)
    ckpts = sorted((run / "weights").glob("epoch*.pt"), key=lambda p: p.stat().st_mtime)
    print(f"[yolo] eval {run.name}: {len(ckpts)} checkpoints")
    pred_kw = {"device": device} if device else {}
    rows = []
    for ck in ckpts:
        model = YOLO(str(ck))
        dets = []
        for image_id, path in val_images:
            # conf low so COCOeval sees the full PR curve (default conf=0.25 truncates recall).
            res = model.predict(str(path), conf=0.001, verbose=False, **pred_kw)[0]
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(
                    {
                        "image_id": image_id,
                        "category_id": cat_ids[int(box.cls)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(box.conf),
                    }
                )
        rows.append({"checkpoint": ck.name, "wall_clock_s": common.walltime(ck, t0), **common.coco_ap(gt, dets)})
    common.write_curve(run, rows)
    print(f"[yolo]   wrote {run / 'curve.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", required=True, help="COCO dataset root (one subdir per dataset)")
    p.add_argument("--out", default="results/yolo")
    p.add_argument("--device", default="")
    p.add_argument("--variants", default="nano,small,medium,large", help="comma-separated subset of VARIANTS")
    args = p.parse_args()

    out_root = Path(args.out)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")

    datasets = list(common.iter_datasets(Path(args.datasets)))  # scan the tree once, not per variant
    for variant in variants:
        model_id = VARIANTS[variant]
        for name, dataset_dir in datasets:
            run = out_root / variant / name
            if (run / "curve.csv").exists():
                print(f"[yolo] skip {variant}/{name} (already done)")
                continue
            if not (run / "meta.json").exists():  # 1. TRAIN (unless a prior run already did)
                print(f"[yolo] train {variant}/{name}")
                train_one(model_id, variant, name, dataset_dir, out_root, args.device)
            score(run, dataset_dir, args.device)  # 2. SCORE
            shutil.rmtree(run / "weights", ignore_errors=True)  # 3. PURGE — keep curve.csv
            print(f"[yolo] done {variant}/{name} (checkpoints deleted)")


if __name__ == "__main__":
    main()
