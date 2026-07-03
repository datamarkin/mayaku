"""Score YOLO checkpoints offline against the COCO val split (pycocotools).

    python eval_yolo.py --datasets <coco_root> [--out results/yolo] [--force]

Runs after train_yolo.py. For each dataset it loads every ``epoch*.pt``, predicts
on the val images, and scores with the shared pycocotools evaluator — the same
metric as the RF-DETR leg, never YOLO's internal mAP. Wall-clock per checkpoint is
its file mtime minus the train-start ``t0`` stamped in meta.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Ultralytics is not installed. Run: pip install ultralytics") from exc


def score(run: Path, dataset_dir: Path, device: str) -> None:
    t0 = common.read_meta(run)["t0"]
    val = common.val_split(dataset_dir)
    gt = val / common.COCO_ANN
    cat_ids = common.category_ids(gt)
    val_images = common.images(gt, val)
    ckpts = sorted((run / "weights").glob("epoch*.pt"), key=lambda p: p.stat().st_mtime)
    print(f"[yolo] eval {run.name}: {len(ckpts)} checkpoints")
    rows = []
    for ck in ckpts:
        model = YOLO(str(ck))
        dets = []
        for image_id, path in val_images:
            # conf low so COCOeval sees the full PR curve (default conf=0.25 truncates recall).
            res = model.predict(str(path), conf=0.001, verbose=False, **({"device": device} if device else {}))[0]
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
    p.add_argument("--datasets", required=True)
    p.add_argument("--out", default="results/yolo")
    p.add_argument("--device", default="")
    p.add_argument("--force", action="store_true", help="re-score even if curve.csv exists")
    args = p.parse_args()

    out_root = Path(args.out)
    for name, dataset_dir in common.iter_datasets(Path(args.datasets)):
        run = out_root / name
        if not (run / "meta.json").exists():
            continue
        if (run / "curve.csv").exists() and not args.force:
            print(f"[yolo] skip {name} (already scored)")
            continue
        score(run, dataset_dir, args.device)


if __name__ == "__main__":
    main()
