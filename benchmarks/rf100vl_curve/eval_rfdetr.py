"""Score RF-DETR checkpoints offline against the COCO val split (pycocotools).

    python eval_rfdetr.py --datasets <coco_root> [--out results/rfdetr] [--force]

Runs after train_rfdetr.py. For each dataset it loads every per-epoch checkpoint,
predicts on the val images, and scores with the shared pycocotools evaluator — the
same metric as the YOLO leg. Wall-clock per checkpoint is its file mtime minus the
train-start ``t0`` stamped in meta.json.

``_to_rfdetr_weights`` converts each ``.ckpt`` into the ``{"model": ...}`` file
RF-DETR loads, preferring the **EMA** (deploy) weights from the checkpoint's callback
state and falling back to the base weights when EMA is off — so this matches the
EMA-grade weights the YOLO and Mayaku legs are scored on. It fails loud if RF-DETR's
layout ever changes. Worth a smoke-test on the first real checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common

try:
    from rfdetr import RFDETRNano
except ImportError as exc:
    raise SystemExit("RF-DETR is not installed. Run: pip install rfdetr") from exc


def _strip_to_model(state: dict, anchor_suffix: str = "class_embed.bias") -> dict | None:
    """Strip a state_dict's wrapper prefix down to RF-DETR's raw model keys.

    The prefix (``model.``, ``module.model.``, optionally with ``_orig_mod.``) is
    detected from whichever key ends in ``class_embed.bias``, so this handles the
    LightningModule wrap, the EMA ``AveragedModel`` wrap, and torch.compile alike.
    Returns None if no anchor key is present.
    """
    anchor = next((k for k in state if k.endswith(anchor_suffix)), None)
    if anchor is None:
        return None
    prefix = anchor[: -len(anchor_suffix)]
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def _to_rfdetr_weights(ckpt: Path, dst: Path) -> str:
    """Convert a per-epoch checkpoint into an RF-DETR-loadable ``{"model": ...}`` file.

    Prefers the EMA (deploy) weights — RF-DETR ships EMA as its headline model and
    they are far less noisy epoch-to-epoch — falling back to the base weights when
    EMA is off. RF-DETR auto-resizes its head from ``class_embed.bias`` on load.
    Returns the weight source used: ``"ema"``, ``"base"``, or ``"native"``.
    """
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)

    # EMA weights live in the RFDETREMACallback state (an AveragedModel state_dict).
    for cb_state in ck.get("callbacks", {}).values():
        if isinstance(cb_state, dict) and "average_model_state_dict" in cb_state:
            ema = _strip_to_model(cb_state["average_model_state_dict"])
            if ema is not None:
                torch.save({"model": ema}, dst)
                return "ema"

    if "model" in ck:  # already RF-DETR native (.pth)
        payload = {"model": ck["model"]}
        if "args" in ck:
            payload["args"] = ck["args"]
        torch.save(payload, dst)
        return "native"

    raw = _strip_to_model(ck["state_dict"])  # plain PTL base weights
    if raw is None:
        raise RuntimeError(
            f"{ckpt.name}: no 'class_embed.bias' in the checkpoint's EMA callback state "
            f"or base state_dict — RF-DETR's layout may have changed."
        )
    torch.save({"model": raw}, dst)
    return "base"


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
    src = "none"
    for ck in ckpts:
        src = _to_rfdetr_weights(ck, tmp)
        model = RFDETRNano(pretrain_weights=str(tmp))
        dets = []
        for image_id, path in val_images:
            det = model.predict(str(path), threshold=0.001)
            for (x1, y1, x2, y2), s, cls in zip(det.xyxy, det.confidence, det.class_id):
                dets.append(
                    {
                        "image_id": image_id,
                        "category_id": cat_ids[int(cls)],
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
    print(f"[rfdetr]   wrote {run / 'curve.csv'} ({src} weights)")


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
