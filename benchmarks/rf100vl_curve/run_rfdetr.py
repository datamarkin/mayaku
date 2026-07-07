"""RF-DETR leg of the RF100-VL AP-vs-wall-clock benchmark — one self-contained script.

    python run_rfdetr.py --datasets <coco_root> [--out results/rfdetr]
                         [--device cuda] [--variants nano,small,medium,large]

For every variant and every dataset it does the same three steps in the same
order — nothing library-specific, nothing tuned:

  1. TRAIN with RF-DETR's stock recipe (its own default epochs, LR, aug — we don't
     even pass `epochs`). The only added argument is `checkpoint_interval=1`, so the
     run leaves a per-epoch checkpoint trail; a learning curve needs it.
  2. SCORE every checkpoint with the shared pycocotools evaluator — the *same*
     metric as the YOLO and Mayaku legs. We score the **EMA** weights, RF-DETR's
     default deploy model (`use_ema=True`; it ships the EMA-based
     `checkpoint_best_total`). YOLO is scored at its EMA too and Mayaku at its EMA
     shadow, so all three legs sit on each library's own default deploy weights.
  3. PURGE that dataset's checkpoints once its curve.csv is written, so 100
     datasets never need 100 datasets of checkpoints on disk at once.

Wall-clock per checkpoint is the checkpoint file's mtime minus the train-start
`t0` in meta.json — no timing callbacks. Resumable: a (variant, dataset) with a
curve.csv is skipped; one trained but not yet scored is scored without retraining.

RF-DETR is the slow leg — meant to run on its own server while the YOLO and
Mayaku legs share the other one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import common

try:
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
except ImportError as exc:
    raise SystemExit("RF-DETR is not installed. Run: pip install rfdetr") from exc

# The four size variants — nano / small / medium / large — as their model classes.
VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def _to_rfdetr_weights(ckpt: Path, dst: Path) -> None:
    """Convert a per-epoch checkpoint into an RF-DETR-loadable ``{"model": ...}`` file.

    Uses the **EMA** weights — RF-DETR's default deploy model. In a PyTorch-Lightning
    ``.ckpt`` those live under ``callbacks``, keyed by the EMA callback's state_key,
    as ``average_model_state_dict`` (a flattened ``AveragedModel`` state where the
    detection weights sit under the ``module.model.`` prefix). A native RF-DETR
    ``.pth`` (``{"model": ...}`` with no ``state_dict``) is passed through unchanged.
    """
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)

    # Native RF-DETR .pth — already deploy-shaped, pass through.
    if "model" in ck and "state_dict" not in ck:
        payload = {"model": ck["model"]}
        if "args" in ck:
            payload["args"] = ck["args"]
        torch.save(payload, dst)
        return

    # PTL .ckpt: pull the EMA snapshot out of the EMA callback's state. Locate it by
    # the payload it carries (average_model_state_dict) rather than a hard-coded key,
    # so it survives a state_key rename.
    ema_state = next(
        (
            cb["average_model_state_dict"]
            for cb in ck.get("callbacks", {}).values()
            if isinstance(cb, dict) and "average_model_state_dict" in cb
        ),
        None,
    )
    if ema_state is None:
        raise RuntimeError(
            f"{ckpt.name}: no EMA callback state ('average_model_state_dict') under "
            f"'callbacks'. RF-DETR trains with use_ema=True by default and the EMA "
            f"weights are the deploy model — inspect the checkpoint's callback keys."
        )
    # AveragedModel.state_dict() flattens as 'module.<pl_module keys>'; the detection
    # model is 'module.model.<...>' ('._orig_mod.' inserted under torch.compile).
    prefix = (
        "module.model._orig_mod."
        if any(k.startswith("module.model._orig_mod.") for k in ema_state)
        else "module.model."
    )
    raw = {k[len(prefix):]: v for k, v in ema_state.items() if k.startswith(prefix)}
    if "class_embed.bias" not in raw:
        raise RuntimeError(
            f"{ckpt.name}: no 'class_embed.bias' after stripping '{prefix}' from the EMA "
            f"state. RF-DETR's EMA/module layout may have changed — inspect the keys."
        )
    torch.save({"model": raw}, dst)


def train_one(model_cls, variant: str, name: str, dataset_dir: Path, out_root: Path) -> None:
    """Default RF-DETR fine-tune on one dataset; per-epoch checkpoints the only extra."""
    run = out_root / variant / name
    t0 = time.time()
    model_cls().train(
        dataset_dir=str(dataset_dir),
        checkpoint_interval=1,
        output_dir=str(run),
    )
    common.write_meta(run, lib="rfdetr", variant=variant, dataset=name, t0=t0)


def score(run: Path, model_cls, dataset_dir: Path, device: str) -> None:
    """Score every checkpoint against the COCO val split with shared pycocotools."""
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
        model = model_cls(pretrain_weights=str(tmp))
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
    p.add_argument("--datasets", required=True, help="COCO dataset root (one subdir per dataset)")
    p.add_argument("--out", default="results/rfdetr")
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
        model_cls = VARIANTS[variant]
        for name, dataset_dir in datasets:
            run = out_root / variant / name
            if (run / "curve.csv").exists():
                print(f"[rfdetr] skip {variant}/{name} (already done)")
                continue
            if not (run / "meta.json").exists():  # 1. TRAIN (unless a prior run already did)
                print(f"[rfdetr] train {variant}/{name}")
                train_one(model_cls, variant, name, dataset_dir, out_root)
            score(run, model_cls, dataset_dir, args.device)  # 2. SCORE
            for ckpt in run.glob("*.ckpt"):  # 3. PURGE — keep curve.csv
                ckpt.unlink()
            (run / "_eval_weights.pth").unlink(missing_ok=True)
            print(f"[rfdetr] done {variant}/{name} (checkpoints deleted)")


if __name__ == "__main__":
    main()
