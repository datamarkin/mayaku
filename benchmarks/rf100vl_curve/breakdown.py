"""Full COCO metric breakdown for the BEST Mayaku vs YOLO checkpoint on one dataset.

Re-scores the single best checkpoint (per curve.csv) and dumps the complete
pycocotools stats — AP@50 vs AP@75 (classification/detection vs localization),
AP small/medium/large (object-size failure points), AR, and per-class AP — so we
can see exactly where Mayaku wins and loses vs YOLO.

    python breakdown.py --dataset-dir <coco_dataset> --name floating-waste [--device cpu]
"""
from __future__ import annotations
import argparse, csv, io, contextlib, json
from pathlib import Path
import numpy as np
import common


def best_ckpt(curve: Path) -> str:
    r = list(csv.DictReader(open(curve)))
    return max(r, key=lambda x: float(x["ap"]))["checkpoint"]


def full_stats(gt_json: Path, dets: list[dict]):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json)); dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iouType="bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    stats = [float(x) for x in ev.stats]  # 12 COCO metrics
    # per-class AP@[.5:.95]: precision[T, R, K, area=0(all), maxdet=2(100)]
    prec = ev.eval["precision"]
    cats = sorted(gt.dataset["categories"], key=lambda c: c["id"])
    per_class = {}
    for k, c in enumerate(cats):
        p = prec[:, :, k, 0, 2]
        per_class[c["name"]] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return stats, per_class


def mayaku_dets(ckpt: Path, gt: Path, val_dir: Path, device: str):
    from mayaku import from_pretrained
    cat_ids = common.category_ids(gt)
    pred = from_pretrained(str(ckpt), device=device)
    if hasattr(pred.model, "score_thresh"):
        pred.model.score_thresh = 0.001
    dets = []
    for image_id, path in common.images(gt, val_dir):
        inst = pred(str(path))
        for (x1, y1, x2, y2), s, cls in zip(inst.pred_boxes.tensor.tolist(), inst.scores.tolist(), inst.pred_classes.tolist()):
            ci = int(cls)
            if ci >= len(cat_ids):
                continue
            dets.append({"image_id": image_id, "category_id": cat_ids[ci], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return dets


def yolo_dets(ckpt: Path, gt: Path, val_dir: Path, device: str):
    from ultralytics import YOLO
    cat_ids = common.category_ids(gt)
    model = YOLO(str(ckpt))
    dets = []
    for image_id, path in common.images(gt, val_dir):
        res = model.predict(str(path), conf=0.001, verbose=False, device=device)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append({"image_id": image_id, "category_id": cat_ids[int(box.cls)], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(box.conf)})
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results_medium")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dsd = Path(a.dataset_dir)
    val = common.val_split(dsd); gt = val / common.COCO_ANN
    out = Path(a.out)
    labels = ["AP@[.50:.95]", "AP@.50", "AP@.75", "AP small", "AP medium", "AP large",
              "AR@1", "AR@10", "AR@100", "AR small", "AR medium", "AR large"]

    results = {}
    for lib, dets_fn, sub in [("mayaku", mayaku_dets, "train/ema"), ("yolo", yolo_dets, "weights")]:
        curve = out / lib / a.name / "curve.csv"
        if not curve.exists():
            print(f"[skip] {lib}: no curve.csv"); continue
        ck_name = best_ckpt(curve)
        ck = out / lib / a.name / sub / ck_name
        print(f"[{lib}] scoring best ckpt {ck_name} on {a.name} ({a.device}) ...", flush=True)
        dets = dets_fn(ck, gt, val, a.device)
        stats, per_class = full_stats(gt, dets)
        results[lib] = (stats, per_class, ck_name, len(dets))
        print(f"[{lib}] done ({len(dets)} dets)", flush=True)

    if len(results) == 2:
        ms, mpc, mck, mn = results["mayaku"]; ys, ypc, yck, yn = results["yolo"]
        print("\n===== floating COCO breakdown: Mayaku vs YOLO =====")
        print(f"{'metric':16} {'Mayaku':>8} {'YOLO':>8} {'Δ(M−Y)':>8}")
        for i, lab in enumerate(labels):
            print(f"{lab:16} {ms[i]:8.3f} {ys[i]:8.3f} {ms[i]-ys[i]:+8.3f}")
        print(f"\n{'class':16} {'Mayaku':>8} {'YOLO':>8} {'Δ(M−Y)':>8}")
        for c in mpc:
            print(f"{c:16} {mpc[c]:8.3f} {ypc.get(c,float('nan')):8.3f} {mpc[c]-ypc.get(c,0):+8.3f}")
        json.dump({"mayaku": {"stats": ms, "per_class": mpc, "ckpt": mck},
                   "yolo": {"stats": ys, "per_class": ypc, "ckpt": yck}, "labels": labels},
                  open(out / f"breakdown_{a.name}.json", "w"), indent=2)
        print(f"\nwrote {out}/breakdown_{a.name}.json")


if __name__ == "__main__":
    main()
