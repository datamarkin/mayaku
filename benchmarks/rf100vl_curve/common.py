"""Shared helpers for the RF100-VL AP-vs-wall-clock benchmark.

Every dataset is COCO format (Roboflow export): ``<dataset>/{train,valid,test}/``
each holding images + ``_annotations.coco.json``. One COCO root is the single
source of truth — it feeds RF-DETR training directly, is converted to YOLO
format for YOLO training, and is the ground truth for the offline pycocotools
scoring of both.

The x-axis of the curve is wall-clock, read from each checkpoint file's mtime
relative to the ``t0`` stamped in the run's ``meta.json`` at train start. The
per-epoch label is irrelevant; a checkpoint is just a ``(wall_clock, AP)`` point.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import io
import json
from pathlib import Path

VAL_SPLIT_NAMES = ("valid", "val", "test")
COCO_ANN = "_annotations.coco.json"

# The 12 COCO summary stats, in pycocotools ``COCOeval.stats`` order: 6 AP figures
# then 6 AR figures. Same names, same order, for every leg's curve.csv.
COCO_STATS = (
    "ap",  # AP @ IoU=0.50:0.95, all areas, maxDets=100  (the headline metric)
    "ap50",  # AP @ IoU=0.50
    "ap75",  # AP @ IoU=0.75
    "ap_small",  # AP for small objects
    "ap_medium",  # AP for medium objects
    "ap_large",  # AP for large objects
    "ar1",  # AR @ maxDets=1
    "ar10",  # AR @ maxDets=10
    "ar100",  # AR @ maxDets=100
    "ar_small",  # AR for small objects
    "ar_medium",  # AR for medium objects
    "ar_large",  # AR for large objects
)


def _split_dir(dataset_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        sub = dataset_dir / name
        if sub.is_dir() and (sub / COCO_ANN).exists():
            return sub
    return None


def iter_datasets(root: Path):
    """Yield ``(name, dataset_dir)`` for every dataset subdir that has a train split."""
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if _split_dir(d, ("train",)) is not None:
            yield d.name, d


def train_split(dataset_dir: Path) -> Path:
    sub = _split_dir(dataset_dir, ("train",))
    if sub is None:
        raise FileNotFoundError(f"no train/{COCO_ANN} in {dataset_dir}")
    return sub


def val_split(dataset_dir: Path) -> Path:
    """The split scored against — first of valid/val/test that exists."""
    sub = _split_dir(dataset_dir, VAL_SPLIT_NAMES)
    if sub is None:
        raise FileNotFoundError(f"no {VAL_SPLIT_NAMES} split with {COCO_ANN} in {dataset_dir}")
    return sub


def category_ids(gt_json: Path) -> list[int]:
    """COCO category ids ascending. A model's class index ``i`` maps to ``[i]``.

    Both trainers see categories in this ascending-id order (YOLO via the
    generated data.yaml, RF-DETR via its own label remap), so predicted class
    ``i`` is category ``category_ids(gt)[i]``.
    """
    data = json.loads(Path(gt_json).read_text())
    return [c["id"] for c in sorted(data["categories"], key=lambda c: c["id"])]


def images(gt_json: Path, images_dir: Path) -> list[tuple[int, Path]]:
    """``(image_id, image_path)`` pairs from a COCO json."""
    data = json.loads(Path(gt_json).read_text())
    return [(im["id"], images_dir / im["file_name"]) for im in data["images"]]


@functools.lru_cache(maxsize=4)
def _coco_gt(gt_json: str):
    """Parse + index a COCO ground-truth file once. Cached so a dataset's many
    per-checkpoint ``coco_ap`` calls reuse one immutable GT object instead of
    re-reading and re-indexing the same file every time (``loadRes``/``COCOeval``
    only read it). maxsize=4 bounds memory to a few datasets' GT at a time.
    """
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(gt_json)


def coco_ap(gt_json: Path, detections: list[dict]) -> dict:
    """pycocotools bbox metrics for ``detections`` in COCO result format.

    Returns all 12 ``COCOeval.stats`` (see ``COCO_STATS``) plus ``n_dets``. Each
    detection: ``{image_id, category_id, bbox:[x,y,w,h], score}``.
    """
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {**dict.fromkeys(COCO_STATS, 0.0), "n_dets": 0}
    coco_gt = _coco_gt(str(gt_json))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {**{k: float(v) for k, v in zip(COCO_STATS, ev.stats)}, "n_dets": len(detections)}


def to_yolo(dataset_dir: Path) -> Path:
    """Convert a COCO dataset into a YOLO-format tree and return its data.yaml.

    Data-format prep only — it does not touch the training recipe. Written once
    into a sibling ``_yolo/`` cache and reused thereafter.
    """
    dst = dataset_dir / "_yolo"
    data_yaml = dst / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    import yaml

    train_src = train_split(dataset_dir)
    val_src = val_split(dataset_dir)
    cats = sorted(json.loads((train_src / COCO_ANN).read_text())["categories"], key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    for src, split in ((train_src, "train"), (val_src, "val")):
        img_out = dst / "images" / split
        lbl_out = dst / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        coco = json.loads((src / COCO_ANN).read_text())
        anns: dict[int, list] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            name = im["file_name"]
            link = img_out / name
            if not link.exists():
                target = (src / name).resolve()
                try:
                    link.symlink_to(target)
                except OSError:
                    import shutil

                    shutil.copy2(target, link)
            w, h = im["width"], im["height"]
            lines = [
                f"{id_to_idx[a['category_id']]} "
                f"{(a['bbox'][0] + a['bbox'][2] / 2) / w:.6f} {(a['bbox'][1] + a['bbox'][3] / 2) / h:.6f} "
                f"{a['bbox'][2] / w:.6f} {a['bbox'][3] / h:.6f}"
                for a in anns.get(im["id"], [])
            ]
            (lbl_out / f"{Path(name).stem}.txt").write_text("\n".join(lines))

    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dst.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {i: c["name"] for i, c in enumerate(cats)},
            }
        )
    )
    return data_yaml


def write_meta(run_dir: Path, **fields) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(fields, indent=2))


def read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def walltime(checkpoint: Path, t0: float) -> float:
    """Wall-clock seconds from train start to when this checkpoint was written."""
    return checkpoint.stat().st_mtime - t0


def write_curve(run_dir: Path, rows: list[dict]) -> None:
    """Write ``curve.csv`` (one row per checkpoint), sorted by wall-clock."""
    rows = sorted(rows, key=lambda r: r["wall_clock_s"])
    with (Path(run_dir) / "curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["checkpoint", "wall_clock_s", *COCO_STATS, "n_dets"])
        w.writeheader()
        w.writerows(rows)
