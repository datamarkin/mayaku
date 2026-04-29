"""Per-stage numerical comparison: Mayaku vs Detectron2 on one image.

Throwaway diagnostic for ADR 003 attempt 5. Loads D2's
``faster_rcnn_R_50_FPN_3x`` from the original ``.pkl`` and Mayaku's
detector with the converted ``.pth``, feeds both the same preprocessed
input tensor (CPU, deterministic), and reports the max-abs-diff at:

  1. backbone.bottom_up outputs (res2..res5)
  2. FPN outputs (p2..p6)
  3. RPN per-level objectness logits + anchor deltas
  4. RPN top-1000 proposals (pre-NMS) per level
  5. ROI Align features for the kept proposals
  6. Box-head cls_logits + box_deltas
  7. final detections (per-class NMS output)

Run from a venv that has *both* mayaku (editable) and detectron2
installed, OR run as two separate invocations that each dump tensors
to disk and a third that diffs the two. We use the disk-handoff form
because installing detectron2 in the mayaku env is fragile.

Usage:
    # In d2 venv (with detectron2 installed):
    python tools/d2_parity_diff.py dump-d2 \\
        --pkl model_final_280758.pkl \\
        --image /path/val2017/000000000139.jpg \\
        --out /tmp/d2_dump.pt

    # In mayaku venv:
    python tools/d2_parity_diff.py dump-mayaku \\
        --weights model_final.mayaku.pth \\
        --config cfg_d2_parity.yaml \\
        --image /path/val2017/000000000139.jpg \\
        --out /tmp/mayaku_dump.pt

    # In either:
    python tools/d2_parity_diff.py diff \\
        --d2 /tmp/d2_dump.pt --mayaku /tmp/mayaku_dump.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# --------------------------------------------------------------------------
# Shared input preprocessor
# --------------------------------------------------------------------------


def load_image_as_d2_input(
    image_path: Path, *, min_size: int = 800, max_size: int = 1333
) -> tuple[torch.Tensor, int, int]:
    """Read image with PIL (Mayaku's convention per ADR 002) and resize
    using ResizeShortestEdge semantics. Returns (CHW uint8-as-float,
    original_h, original_w). Both engines see the same tensor."""
    from PIL import Image

    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size
    short, long = min(orig_h, orig_w), max(orig_h, orig_w)
    scale = min_size / short
    if scale * long > max_size:
        scale = max_size / long
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    resized = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    import numpy as np

    arr = np.asarray(resized)  # HWC RGB uint8
    chw = torch.as_tensor(arr.transpose(2, 0, 1).copy(), dtype=torch.float32)
    return chw, orig_h, orig_w


# --------------------------------------------------------------------------
# Mayaku dump
# --------------------------------------------------------------------------


def dump_mayaku(weights: Path, config: Path, image: Path, out: Path) -> None:
    from mayaku.cli._factory import build_detector
    from mayaku.config import load_yaml

    cfg = load_yaml(config)
    model = build_detector(cfg)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    chw, h, w = load_image_as_d2_input(image)
    batched = [{"image": chw, "height": h, "width": w}]

    captured: dict[str, torch.Tensor] = {}

    def cap(name: str):
        def hook(_m, _i, o):
            if isinstance(o, dict):
                for k, v in o.items():
                    captured[f"{name}.{k}"] = v.detach().cpu()
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    if isinstance(v, torch.Tensor):
                        captured[f"{name}[{i}]"] = v.detach().cpu()
            elif isinstance(o, torch.Tensor):
                captured[name] = o.detach().cpu()

        return hook

    # Hook the backbone (FPN wraps bottom_up; the FPN outputs are
    # what the rest of the pipeline consumes, but bottom_up gives us
    # the pre-FPN features for sanity).
    model.backbone.bottom_up.register_forward_hook(cap("backbone.bottom_up"))
    model.backbone.register_forward_hook(cap("backbone.fpn"))
    # RPN head (per-level)
    model.rpn.head.register_forward_hook(cap("rpn.head"))
    # ROI box head
    if hasattr(model, "roi_heads"):
        model.roi_heads.box_head.register_forward_hook(cap("roi_heads.box_head"))
        model.roi_heads.box_predictor.register_forward_hook(
            cap("roi_heads.box_predictor")
        )

    with torch.no_grad():
        outputs = model(batched)
    inst = outputs[0]["instances"]
    captured["final.pred_boxes"] = inst.pred_boxes.tensor.detach().cpu()
    captured["final.scores"] = inst.scores.detach().cpu()
    captured["final.pred_classes"] = inst.pred_classes.detach().cpu()
    captured["__input__"] = chw

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, out)
    print(f"Wrote {out} with {len(captured)} tensors:")
    for k, v in sorted(captured.items()):
        print(f"  {k:50s} {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")


# --------------------------------------------------------------------------
# D2 dump (runs in detectron2 venv)
# --------------------------------------------------------------------------


def dump_d2(pkl: Path, image: Path, out: Path) -> None:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(pkl)
    cfg.MODEL.DEVICE = "cpu"
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(pkl))
    model.eval()

    chw, h, w = load_image_as_d2_input(image)
    # D2 expects BGR uint8-as-float by default. Mayaku's load gives RGB.
    # Reverse so the model's pixel_mean (BGR) lines up.
    chw_bgr = chw.flip(0)
    batched = [{"image": chw_bgr, "height": h, "width": w}]

    captured: dict[str, torch.Tensor] = {}

    def cap(name: str):
        def hook(_m, _i, o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, torch.Tensor):
                        captured[f"{name}.{k}"] = v.detach().cpu()
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    if isinstance(v, torch.Tensor):
                        captured[f"{name}[{i}]"] = v.detach().cpu()
            elif isinstance(o, torch.Tensor):
                captured[name] = o.detach().cpu()

        return hook

    model.backbone.bottom_up.register_forward_hook(cap("backbone.bottom_up"))
    model.backbone.register_forward_hook(cap("backbone.fpn"))
    model.proposal_generator.rpn_head.register_forward_hook(cap("rpn.head"))
    model.roi_heads.box_head.register_forward_hook(cap("roi_heads.box_head"))
    model.roi_heads.box_predictor.register_forward_hook(cap("roi_heads.box_predictor"))

    with torch.no_grad():
        outputs = model(batched)
    inst = outputs[0]["instances"]
    captured["final.pred_boxes"] = inst.pred_boxes.tensor.detach().cpu()
    captured["final.scores"] = inst.scores.detach().cpu()
    captured["final.pred_classes"] = inst.pred_classes.detach().cpu()
    captured["__input__"] = chw  # log the RGB input (pre-flip) for parity

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, out)
    print(f"Wrote {out} with {len(captured)} tensors:")
    for k, v in sorted(captured.items()):
        print(f"  {k:50s} {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")


# --------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------


def diff(d2_path: Path, mayaku_path: Path) -> None:
    d2 = torch.load(d2_path, map_location="cpu", weights_only=True)
    my = torch.load(mayaku_path, map_location="cpu", weights_only=True)

    # Compare keys
    common = sorted(set(d2) & set(my))
    only_d2 = sorted(set(d2) - set(my))
    only_my = sorted(set(my) - set(d2))
    print(f"common keys: {len(common)}; only-d2: {len(only_d2)}; only-mayaku: {len(only_my)}")
    if only_d2:
        print("  only in D2:", only_d2)
    if only_my:
        print("  only in Mayaku:", only_my)

    print(f"\n{'key':45s} {'D2 shape':25s} {'My shape':25s} {'max_abs':>10s} {'match?'}")
    print("-" * 120)
    for k in common:
        a, b = d2[k], my[k]
        as_, bs_ = tuple(a.shape) if hasattr(a, "shape") else "-", tuple(b.shape) if hasattr(b, "shape") else "-"
        if as_ != bs_:
            print(f"{k:45s} {str(as_):25s} {str(bs_):25s} {'SHAPE MISMATCH'}")
            continue
        if not torch.is_floating_point(a):
            eq = bool(torch.equal(a, b))
            print(f"{k:45s} {str(as_):25s} {str(bs_):25s} {'-':>10s} {'==' if eq else '!='}")
            continue
        diff_ = (a - b).abs()
        mx = float(diff_.max())
        ok = "✓" if mx < 1e-3 else ("~" if mx < 1e-1 else "✗")
        print(f"{k:45s} {str(as_):25s} {str(bs_):25s} {mx:10.4e} {ok}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sd = sub.add_parser("dump-d2")
    sd.add_argument("--pkl", type=Path, required=True)
    sd.add_argument("--image", type=Path, required=True)
    sd.add_argument("--out", type=Path, required=True)
    sm = sub.add_parser("dump-mayaku")
    sm.add_argument("--weights", type=Path, required=True)
    sm.add_argument("--config", type=Path, required=True)
    sm.add_argument("--image", type=Path, required=True)
    sm.add_argument("--out", type=Path, required=True)
    sf = sub.add_parser("diff")
    sf.add_argument("--d2", type=Path, required=True)
    sf.add_argument("--mayaku", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "dump-d2":
        dump_d2(args.pkl, args.image, args.out)
    elif args.cmd == "dump-mayaku":
        dump_mayaku(args.weights, args.config, args.image, args.out)
    elif args.cmd == "diff":
        diff(args.d2, args.mayaku)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
