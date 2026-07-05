"""Score RF-DETR checkpoints offline against the COCO val split (pycocotools).

    python eval_rfdetr.py --datasets <coco_root> [--out results/rfdetr] [--force]

Runs after train_rfdetr.py. For each dataset it loads every per-epoch checkpoint,
predicts on the val images, and scores with the shared pycocotools evaluator — the
same metric as the YOLO leg. Wall-clock per checkpoint is its file mtime minus the
train-start ``t0`` stamped in meta.json.

Two things to validate on the first real checkpoint:
  * ``_to_rfdetr_weights`` converts the plain PyTorch-Lightning ``.ckpt`` (keys
    prefixed ``model.``, no ``model`` entry) into the ``{"model": ...}`` file
    RF-DETR loads; it fails loud if RF-DETR's module layout ever changes.
  * These are the *base* weights. RF-DETR deploys the *EMA* weights as its headline
    model (kept separately in the checkpoint's callback state), so this slightly
    understates it — switch to EMA if that gap matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common

try:
    from rfdetr import RFDETRNano
except ImportError as exc:
    raise SystemExit("RF-DETR is not installed. Run: pip install rfdetr") from exc


def _to_rfdetr_weights(ckpt: Path, dst: Path) -> None:
    """Convert a per-epoch checkpoint into an RF-DETR-loadable ``{"model": ...}`` file."""
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" in ck:  # already RF-DETR native
        payload = {"model": ck["model"]}
        if "args" in ck:
            payload["args"] = ck["args"]
        torch.save(payload, dst)
        return

    sd = ck["state_dict"]  # plain PTL: strip the single "model." LightningModule prefix
    prefix = "model._orig_mod." if any(k.startswith("model._orig_mod.") for k in sd) else "model."
    raw = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if "class_embed.bias" not in raw:
        raise RuntimeError(
            f"{ckpt.name}: no 'class_embed.bias' after stripping '{prefix}'. RF-DETR's "
            f"LightningModule layout may have changed — inspect the state_dict keys."
        )
    torch.save({"model": raw}, dst)


def score(run: Path, dataset_dir: Path, device: str) -> None:
    import torch

    t0 = common.read_meta(run)["t0"]
    val = common.val_split(dataset_dir)
    gt = val / common.COCO_ANN
    cat_ids = common.category_ids(gt)
    val_images = common.images(gt, val)
    ckpts = sorted(run.glob("checkpoint_*.ckpt"), key=lambda p: p.stat().st_mtime)
    print(f"[rfdetr] eval {run.name}: {len(ckpts)} checkpoints")
    tmp = run / "_eval_weights.pth"
    rows = []
    for ck in ckpts:
        _to_rfdetr_weights(ck, tmp)
        model = RFDETRNano(pretrain_weights=str(tmp))
        dets = []
        for image_id, path in val_images:
            det = model.predict(str(path), threshold=0.001)
            for (x1, y1, x2, y2), s, cls in zip(det.xyxy, det.confidence, det.class_id):
                # RF-DETR (LW-DETR) builds an (N+1)-slot classification head: indices
                # 0..N-1 are the dataset's ascending categories, index N is the DETR
                # "no-object" class. At threshold 0.001 that slot leaks detections; it
                # maps to no real category, so drop anything past the last category.
                ci = int(cls)
                if ci >= len(cat_ids):
                    continue
                dets.append(
                    {
                        "image_id": image_id,
                        "category_id": cat_ids[ci],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    }
                )
        rows.append({"checkpoint": ck.name, "wall_clock_s": common.walltime(ck, t0), **common.coco_ap(gt, dets)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tmp.unlink(missing_ok=True)
    common.write_curve(run, rows)
    print(f"[rfdetr]   wrote {run / 'curve.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", required=True)
    p.add_argument("--out", default="results/rfdetr")
    p.add_argument("--device", default="")
    p.add_argument("--force", action="store_true", help="re-score even if curve.csv exists")
    args = p.parse_args()

    out_root = Path(args.out)
    for name, dataset_dir in common.iter_datasets(Path(args.datasets)):
        run = out_root / name
        if not (run / "meta.json").exists():
            continue
        if (run / "curve.csv").exists() and not args.force:
            print(f"[rfdetr] skip {name} (already scored)")
            continue
        score(run, dataset_dir, args.device)


if __name__ == "__main__":
    main()
