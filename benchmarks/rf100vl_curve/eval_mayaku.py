"""Score Mayaku checkpoints offline against the COCO val split (pycocotools).

    python eval_mayaku.py --datasets <coco_root> [--out results/mayaku] [--device auto] [--force]

Runs after train_mayaku.py. For each dataset it loads every per-epoch checkpoint,
predicts on the val images, and scores with the shared pycocotools evaluator — the
same metric as the YOLO and RF-DETR legs. Wall-clock per checkpoint is its file
mtime minus the train-start ``t0`` stamped in meta.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common
from mayaku import from_pretrained


def score(run: Path, dataset_dir: Path, device: str) -> None:
    import torch

    t0 = common.read_meta(run)["t0"]
    val = common.val_split(dataset_dir)
    gt = val / common.COCO_ANN
    cat_ids = common.category_ids(gt)
    val_images = common.images(gt, val)
    ckpts = sorted((run / "train").glob("model_iter_*.pth"), key=lambda p: p.stat().st_mtime)
    print(f"[mayaku] eval {run.name}: {len(ckpts)} checkpoints")
    rows = []
    for ck in ckpts:
        predictor = from_pretrained(str(ck), device=device)
        # Match the 0.001 the other legs eval at so COCOeval sees the full PR curve
        # (the checkpoint bakes score_thresh_test=0.05, which would truncate recall).
        if hasattr(predictor.model, "score_thresh"):
            predictor.model.score_thresh = 0.001
        dets = []
        for image_id, path in val_images:
            inst = predictor(str(path))
            boxes = inst.pred_boxes.tensor.tolist()
            scores = inst.scores.tolist()
            classes = inst.pred_classes.tolist()
            for (x1, y1, x2, y2), s, cls in zip(boxes, scores, classes):
                dets.append(
                    {
                        "image_id": image_id,
                        "category_id": cat_ids[int(cls)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(s),
                    }
                )
        rows.append({"checkpoint": ck.name, "wall_clock_s": common.walltime(ck, t0), **common.coco_ap(gt, dets)})
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    common.write_curve(run, rows)
    print(f"[mayaku]   wrote {run / 'curve.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", required=True)
    p.add_argument("--out", default="results/mayaku")
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="re-score even if curve.csv exists")
    args = p.parse_args()

    out_root = Path(args.out)
    for name, dataset_dir in common.iter_datasets(Path(args.datasets)):
        run = out_root / name
        if not (run / "meta.json").exists():
            continue
        if (run / "curve.csv").exists() and not args.force:
            print(f"[mayaku] skip {name} (already scored)")
            continue
        score(run, dataset_dir, args.device)


if __name__ == "__main__":
    main()
