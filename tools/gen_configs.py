#!/usr/bin/env python3
"""Generate the ``mayaku-{n,s,m,l,xl,xxl}`` config family — the source of truth.

The 18 family configs (6 tiers × {detection, segmentation, keypoints}) are *derived* from
the :data:`TIERS` table below, not hand-maintained. A tier supplies five inputs
— backbone, ``hidden_dim``, ``num_stages``, ``infer_size``, and whether it is a
real-time tier — and every other family-varying field is computed from them so
the invariants can never drift:

    fpn.out_channels        = hidden_dim          (FPN must feed the head width)
    head.dim_feedforward    = 4 * hidden_dim       (Sparse R-CNN ratio)
    head.dim_dynamic        = hidden_dim / 4        (Sparse R-CNN ratio)
    head.pooler_sampling_ratio = 1 if real-time else 0   (TRT-fp16 vs adaptive)
    mask.conv_dim           = hidden_dim           (scale the mask head too)

Everything else (solver, dataloader, costs, num_proposals, num_heads, pooler 7,
pixel mean/std) is constant across the family and lives in the templates.

Usage::

    python tools/gen_configs.py            # (re)write all 18 configs
    python tools/gen_configs.py --check    # CI gate: exit 1 if any file drifted

The ``--check`` mode is the 0-drift guarantee: regenerate in memory and diff
against what's committed.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"


@dataclass(frozen=True)
class Tier:
    """The five per-tier inputs; all other family fields derive from these."""

    name: str  # n, s, m, l, xl, xxl
    backbone: str  # convnext_femto | _nano | _tiny | _base
    hidden_dim: int  # 128 (real-time) | 256 (accuracy)
    num_stages: int  # QGN cascade depth
    infer_size: int  # fixed square letterbox size for inference + export
    realtime: bool  # True -> pooler_sampling_ratio 1; False -> 0
    desc: str  # one-phrase positioning, used in the header comment
    params: str  # detection param estimate, used in the header comment


TIERS: tuple[Tier, ...] = (
    Tier("n", "convnext_femto", 128, 2, 640, True, "real-time edge", "10M"),
    Tier("s", "convnext_nano", 128, 2, 640, True, "real-time", "20M"),
    Tier("m", "convnext_tiny", 128, 2, 640, True, "real-time hero", "33M"),
    Tier("l", "convnext_tiny", 256, 2, 640, False, "best AP/speed", "57M"),
    Tier("xl", "convnext_base", 256, 2, 640, False, "max accuracy", "117M"),
    Tier("xxl", "convnext_base", 256, 6, 800, False, "accuracy ceiling", "167M"),
)


def _backbone_display(backbone: str) -> str:
    """``convnext_femto`` -> ``ConvNeXt-femto`` for the header comment."""
    _, _, variant = backbone.partition("_")
    return f"ConvNeXt-{variant}"


def _header(tier: Tier, task: str) -> str:
    """The 3-line header comment block (task-aware title; shared body)."""
    bb = _backbone_display(tier.backbone)
    return (
        f"# mayaku-{tier.name} ({task}) — {tier.desc}. "
        f"{bb} + FPN + UniQuery (QGN, {tier.num_stages} stages).\n"
        f"# ~{tier.params} params (detection). Proven head dims: dim_feedforward = 4*hidden_dim,\n"
        f"# dim_dynamic = hidden_dim/4 (Sparse R-CNN ratio), pooler 7.\n"
    )


def render(tier: Tier, task: str) -> str:
    """Return the full YAML text for one (tier, task) family member."""
    hd = tier.hidden_dim
    ff = 4 * hd
    dd = hd // 4
    # Real-time tier: 1 sample/ROI bin (skips the sub-bin average → faster, no
    # ReduceMean op). Accuracy tier: 2 samples/bin. The deploy one-pass pooler
    # can't do per-box adaptive sampling, so these are the literal fixed counts
    # both train and deploy use (see ROIPooler._eff_sampling_ratio).
    sr = 1 if tier.realtime else 2

    # Task = detection body + one optional head block. Segmentation adds a mask
    # head (conv_dim scales with the model); keypoints adds a keypoint head
    # (width auto-scales via fpn.out_channels, so no per-tier knob). COCO
    # keypoints is person-only → num_classes 1, vs 80 for detection/segmentation.
    head_block = ""
    extra_format = ""
    if task == "segmentation":
        head_block = (
            "  uniquery_mask:\n"
            "    pooler_resolution: 14\n"
            "    mask_resolution: 28\n"
            "    num_conv: 4\n"
            f"    conv_dim: {hd}     # scale the mask head with the model\n"
            "    loss_weight: 1.0       # real GT masks; lower to 0.5 only for noisy pseudo-masks\n"
        )
        extra_format = "  mask_format: polygon   # COCO instance masks; use bitmask for RLE\n"
    elif task == "keypoints":
        head_block = (
            "  uniquery_keypoint:\n"
            "    pooler_resolution: 14\n"
            "    num_keypoints: 17        # COCO person-pose keypoints\n"
            "    loss_weight: 1.0\n"
        )
    num_classes = 1 if task == "keypoints" else 80

    return (
        f"{_header(tier, task)}"
        "\n"
        "model:\n"
        "  meta_architecture: uniquery\n"
        "  pixel_mean: [123.675, 116.28, 103.53]\n"
        "  pixel_std:  [58.395, 57.12, 57.375]\n"
        "  backbone:\n"
        f"    name: {tier.backbone}\n"
        "\n"
        "    weights_path: null\n"
        "    freeze_at: 0\n"
        "    norm: FrozenBN\n"
        "  fpn:\n"
        "    in_features: [res2, res3, res4, res5]\n"
        f"    out_channels: {hd}   # MUST equal uniquery_head.hidden_dim\n"
        "    norm: ''\n"
        "    fuse_type: sum\n"
        "  roi_heads:\n"
        f"    num_classes: {num_classes}\n"
        "  uniquery_head:\n"
        "    num_proposals: 300\n"
        f"    hidden_dim: {hd}\n"
        "    num_heads: 8\n"
        f"    num_stages: {tier.num_stages}\n"
        "    uniquery_generator: true\n"
        f"    dim_feedforward: {ff}    # 4 x hidden_dim\n"
        f"    dim_dynamic: {dd}         # hidden_dim / 4\n"
        "    pooler_resolution: 7\n"
        f"    pooler_sampling_ratio: {sr}   # samples per ROI bin (fixed; same in train + deploy). 1 = real-time tier, 2 = accuracy tier\n"
        "    dropout: 0.0\n"
        "    cost_class: 2.0\n"
        "    cost_bbox: 5.0\n"
        "    cost_giou: 2.0\n"
        "    cascade_iou_thresholds: []\n"
        f"{head_block}"
        "\n"
        "input:\n"
        "  resize_mode: letterbox\n"
        f"  infer_size: {tier.infer_size}   # compute-budget dial (canvas = largest 128-aligned (H,W) under infer_size^2)\n"
        "  train_scale_min: 0.5   # multi-scale letterbox train: budget fraction floor → full deploy canvas\n"
        "  random_flip: horizontal\n"
        "  color_jitter_enabled: true\n"
        "  color_jitter_brightness: 0.4\n"
        "  color_jitter_contrast: 0.4\n"
        "  color_jitter_saturation: 0.7\n"
        "  color_jitter_hue: 0.015\n"
        "  color_jitter_prob: 0.5\n"
        f"{extra_format}"
        "\n"
        "solver:\n"
        "  optimizer_name: AdamW\n"
        "  base_lr: 2.5e-5\n"
        "  weight_decay: 1.0e-4\n"
        "  weight_decay_norm: 0.0\n"
        "  betas: [0.9, 0.999]\n"
        "  eps: 1.0e-8\n"
        "  llrd_enabled: true\n"
        "  llrd_decay: 0.8\n"
        "  lr_scheduler_name: WarmupCosineLR\n"
        "  max_iter: 90000\n"
        "  steps: [60000, 80000]\n"
        "  warmup_iters: 1500\n"
        "  warmup_factor: 0.001\n"
        "  warmup_method: linear\n"
        "  ims_per_batch: 2\n"
        "  grad_accum_steps: 8\n"
        "  clip_gradients_enabled: true\n"
        "  clip_gradients_value: 1.0\n"
        "  clip_gradients_type: norm\n"
        "  grad_norm_log_enabled: false\n"
        "  amp_enabled: true\n"
        "  amp_dtype: bf16\n"
        "  checkpoint_period: 5000\n"
        "  ema_enabled: true\n"
        "  ema_decay: 0.9999\n"
        "  ema_tau: 2000.0\n"
        "\n"
        "test:\n"
        "  detections_per_image: 100\n"
        "  eval_period: 5000\n"
        "  precise_bn_enabled: false\n"
        "\n"
        "dataloader:\n"
        "  num_workers: 4\n"
        "  aspect_ratio_grouping: true\n"
        "  sampler_train: TrainingSampler\n"
        "  filter_empty_annotations: true\n"
        "\n"
        "auto_config:\n"
        "  enabled: false\n"
    )


def _validate(text: str) -> None:
    """Fail loudly if a rendered config is not schema-valid.

    ``render`` derives ``dim_feedforward``/``dim_dynamic`` from ``hidden_dim``,
    and ``MayakuConfig`` enforces ``fpn.out_channels == uniquery_head.hidden_dim``,
    so a schema-valid render is an invariant-correct one — no extra asserts needed.
    """
    from mayaku.config import MayakuConfig

    MayakuConfig.model_validate(yaml.safe_load(text))  # raises ValidationError on any drift


def _targets() -> list[tuple[Path, str]]:
    """Every (output_path, rendered_text) the family comprises."""
    out: list[tuple[Path, str]] = []
    for task in ("detection", "segmentation", "keypoints"):
        for tier in TIERS:
            text = render(tier, task)
            _validate(text)
            out.append((CONFIGS / task / f"mayaku-{tier.name}.yaml", text))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify on-disk configs match the generator; exit 1 on any drift (CI gate)",
    )
    args = parser.parse_args()

    targets = _targets()

    if args.check:
        drifted = []
        for path, want in targets:
            have = path.read_text(encoding="utf-8") if path.exists() else ""
            if have != want:
                drifted.append(path)
                rel = path.relative_to(REPO)
                sys.stdout.writelines(
                    difflib.unified_diff(
                        have.splitlines(keepends=True),
                        want.splitlines(keepends=True),
                        fromfile=f"a/{rel}",
                        tofile=f"b/{rel}",
                    )
                )
        if drifted:
            print(
                f"\n{len(drifted)} config(s) out of date. Run: python tools/gen_configs.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK — all {len(targets)} family configs match the generator.")
        return 0

    for path, text in targets:
        path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(targets)} family configs to {CONFIGS.relative_to(REPO)}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
