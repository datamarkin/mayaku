"""Convert a Detectron2 model-zoo checkpoint to Mayaku's state_dict layout.

Throwaway validation tool. See tools/README.md for the deletion plan.

Architecture support: Faster / Mask / Keypoint R-CNN with R-50 / R-101 /
X-101_32x8d FPN backbones, against Mayaku's torchvision-backed ResNet
(and ResNeXt). Inert rules pass through cleanly: a Faster R-CNN .pkl has
no mask/keypoint head keys, so the head rules add no risk for it.

Usage:
    python tools/convert_d2_checkpoint.py INPUT.pkl -o OUTPUT.pth
    python tools/convert_d2_checkpoint.py INPUT.pkl -o OUT.pth --channel-order rgb

The output is a plain state_dict (.pth) loadable via Mayaku's existing
`mayaku train --weights` / `mayaku eval --weights` flag.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

_Replacement = str | Callable[[re.Match[str]], str]

# Detectron2-side keys that carry no learned state and should be dropped silently.
_METADATA_KEYS = {"__author__", "matching_heuristics"}

# D2 keys that have no Mayaku equivalent in the state_dict and are
# safe to drop. Each entry is a regex matched against the full key.
# Anchor cell-buffers and pixel mean/std fall here: they're either
# precomputed-from-config in Mayaku (cell_anchors) or registered with
# persistent=False (pixel_mean/pixel_std).
_DROP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^proposal_generator\.anchor_generator\.cell_anchors\.\d+$"),
    re.compile(r"^pixel_mean$"),
    re.compile(r"^pixel_std$"),
]

# Rename rules: (compiled-regex, replacement). Replacement may use a callable
# that receives the regex Match and returns the new key, for index arithmetic.
# Order matters: the first matching rule wins.
_RENAME_RULES: list[tuple[re.Pattern[str], _Replacement]] = [
    # Stem: D2 BasicStem.conv1 → torchvision Sequential index 0; its
    # FrozenBN ".norm" → Sequential index 1. FrozenBN buffer names
    # (weight/bias/running_mean/running_var) match Mayaku's FrozenBN.
    (re.compile(r"^backbone\.bottom_up\.stem\.conv1\.weight$"),
     "backbone.bottom_up.stem.0.weight"),
    (re.compile(r"^backbone\.bottom_up\.stem\.conv1\.norm\.(weight|bias|running_mean|running_var)$"),
     r"backbone.bottom_up.stem.1.\1"),

    # Bottleneck convs: convN unchanged; convN.norm → bnN.
    (re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.conv([1-3])\.weight$"),
     r"backbone.bottom_up.\1.\2.conv\3.weight"),
    (re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.conv([1-3])\.norm\.(weight|bias|running_mean|running_var)$"),
     r"backbone.bottom_up.\1.\2.bn\3.\4"),

    # Bottleneck shortcut → torchvision downsample (Sequential of conv,bn).
    (re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.shortcut\.weight$"),
     r"backbone.bottom_up.\1.\2.downsample.0.weight"),
    (re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.shortcut\.norm\.(weight|bias|running_mean|running_var)$"),
     r"backbone.bottom_up.\1.\2.downsample.1.\3"),

    # FPN: D2's flat fpn_lateral{N}/fpn_output{N} → Mayaku's ModuleList
    # indexed from 0 over in_features=(res2,res3,res4,res5).
    (re.compile(r"^backbone\.fpn_lateral([2-5])\.(weight|bias)$"),
     lambda m: f"backbone.lateral_convs.{int(m.group(1)) - 2}.{m.group(2)}"),
    (re.compile(r"^backbone\.fpn_output([2-5])\.(weight|bias)$"),
     lambda m: f"backbone.output_convs.{int(m.group(1)) - 2}.{m.group(2)}"),

    # RPN: D2 wraps its head in proposal_generator.rpn_head; Mayaku in rpn.head.
    (re.compile(r"^proposal_generator\.rpn_head\.(conv|objectness_logits|anchor_deltas)\.(weight|bias)$"),
     r"rpn.head.\1.\2"),

    # Box head FCs: D2 fc1/fc2 → Mayaku fcs.0/fcs.1.
    (re.compile(r"^roi_heads\.box_head\.fc(\d+)\.(weight|bias)$"),
     lambda m: f"roi_heads.box_head.fcs.{int(m.group(1)) - 1}.{m.group(2)}"),

    # Box predictor: identity (cls_score, bbox_pred); kept because COCO
    # has 80 classes which matches Mayaku's COCO config.
    (re.compile(r"^roi_heads\.box_predictor\.(cls_score|bbox_pred)\.(weight|bias)$"),
     r"roi_heads.box_predictor.\1.\2"),

    # Mask head: D2 stores convs as flat mask_fcn{1..4}; Mayaku puts them
    # in self.convs (ModuleList, 0-indexed). deconv and predictor names
    # match between the two.
    (re.compile(r"^roi_heads\.mask_head\.mask_fcn(\d+)\.(weight|bias)$"),
     lambda m: f"roi_heads.mask_head.convs.{int(m.group(1)) - 1}.{m.group(2)}"),
    (re.compile(r"^roi_heads\.mask_head\.(deconv|predictor)\.(weight|bias)$"),
     r"roi_heads.mask_head.\1.\2"),

    # Keypoint head: D2 conv_fcn{1..8} → Mayaku convs.0..7. D2's upsampling
    # ConvTranspose2d is named score_lowres; Mayaku calls it deconv.
    # Tensor shape (channels, num_keypoints, 4, 4) is identical.
    (re.compile(r"^roi_heads\.keypoint_head\.conv_fcn(\d+)\.(weight|bias)$"),
     lambda m: f"roi_heads.keypoint_head.convs.{int(m.group(1)) - 1}.{m.group(2)}"),
    (re.compile(r"^roi_heads\.keypoint_head\.score_lowres\.(weight|bias)$"),
     r"roi_heads.keypoint_head.deconv.\1"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load a Detectron2 .pkl (caffe2 numpy) or .pth (torch) checkpoint.

    Returns a dict[str, Tensor] of the inner model state, with metadata
    keys stripped. .pkl files are validated to have a {"model": dict-of-
    numpy-arrays, ...} top level — anything else is rejected to avoid
    executing arbitrary pickled objects.
    """
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        # Trust check: the only shape we accept from a .pkl is a dict
        # whose "model" entry is itself a dict mapping str → numpy.ndarray.
        # Anything else is rejected before any further work.
        if not isinstance(obj, dict) or "model" not in obj:
            raise ValueError(
                f"Refusing to load {path}: expected a Detectron2 .pkl with "
                "a top-level dict containing 'model'. Got: "
                f"{type(obj).__name__}"
            )
        inner = obj["model"]
        if not isinstance(inner, dict):
            raise ValueError(
                f"Refusing to load {path}: 'model' entry is not a dict "
                f"(got {type(inner).__name__})."
            )
        state: dict[str, torch.Tensor] = {}
        for k, v in inner.items():
            if k in _METADATA_KEYS:
                continue
            if not isinstance(v, np.ndarray):
                raise ValueError(
                    f"Refusing to load {path}: model[{k!r}] is not a "
                    f"numpy.ndarray (got {type(v).__name__})."
                )
            state[k] = torch.from_numpy(v).clone()
        return state
    if suffix == ".pth":
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], dict):
            loaded = loaded["model"]
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Refusing to load {path}: expected a state_dict mapping; "
                f"got {type(loaded).__name__}."
            )
        return {k: v for k, v in loaded.items() if k not in _METADATA_KEYS}
    raise ValueError(f"Unsupported file extension {suffix!r}; expected .pkl or .pth.")


# ---------------------------------------------------------------------------
# Renaming
# ---------------------------------------------------------------------------


def rename_key(d2_key: str) -> str | None:
    """Apply the rename table to a single D2 key.

    Returns the renamed key on a hit, or None if no rule matched.
    """
    for pattern, repl in _RENAME_RULES:
        m = pattern.match(d2_key)
        if m is None:
            continue
        if callable(repl):
            return repl(m)
        return pattern.sub(repl, d2_key)
    return None


def _is_dropped(d2_key: str) -> bool:
    return any(p.match(d2_key) is not None for p in _DROP_PATTERNS)


def rename_state(
    d2_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Apply the rename table to every key.

    Returns (mayaku_state, dropped_keys, unrecognised_keys). The caller
    decides whether unrecognised keys are fatal — the script-level
    main() treats any unrecognised key as fatal because that's the
    signal that the rename table needs an update. Dropped keys are
    expected and reported only as a count.
    """
    out: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    unrecognised: list[str] = []
    for k, v in d2_state.items():
        if _is_dropped(k):
            dropped.append(k)
            continue
        new = rename_key(k)
        if new is None:
            unrecognised.append(k)
            continue
        out[new] = v
    return out, dropped, unrecognised


# ---------------------------------------------------------------------------
# Channel-order fix
# ---------------------------------------------------------------------------


_STEM_CONV_KEY = "backbone.bottom_up.stem.0.weight"


def reverse_stem_input_channels(state: dict[str, torch.Tensor]) -> None:
    """Reverse the input channel dim of conv1 in-place.

    D2 model-zoo .pkl checkpoints are caffe2 BGR-trained. Mayaku is
    RGB-native (ADR 002). Reversing dim=1 of the stem's first conv
    is a weight-domain transformation equivalent to a runtime BGR↔RGB
    channel swap — once applied, the rest of the network sees the same
    activations regardless of the input convention.
    """
    if _STEM_CONV_KEY not in state:
        raise KeyError(
            f"Expected {_STEM_CONV_KEY!r} after rename; got "
            f"{sorted(k for k in state if k.startswith('backbone.bottom_up.stem.'))}"
        )
    w = state[_STEM_CONV_KEY]
    if w.ndim != 4 or w.shape[1] != 3:
        raise ValueError(
            f"{_STEM_CONV_KEY} has unexpected shape {tuple(w.shape)}; "
            "expected [out, 3, kH, kW]."
        )
    state[_STEM_CONV_KEY] = w[:, [2, 1, 0], :, :].clone().contiguous()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a Detectron2 model-zoo checkpoint to Mayaku's state_dict layout.",
    )
    p.add_argument("input", type=Path, help="Path to the D2 checkpoint (.pkl or .pth).")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output .pth path.")
    p.add_argument(
        "--channel-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help=(
            "Channel order of the source checkpoint's conv1. Default 'bgr' "
            "matches D2's caffe2 model-zoo convention; pass 'rgb' for a "
            "checkpoint trained with INPUT.FORMAT='RGB'."
        ),
    )
    return p.parse_args(argv)


def _summarise(state: dict[str, torch.Tensor]) -> str:
    n_keys = len(state)
    n_params = sum(int(t.numel()) for t in state.values())
    bytes_total = sum(int(t.numel() * t.element_size()) for t in state.values())
    return f"keys={n_keys}, tensors={n_params:,} elements ({bytes_total / 1e6:.1f} MB)"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.input.exists():
        print(f"error: input {args.input} does not exist", file=sys.stderr)
        return 2

    print(f"Loading {args.input} ...")
    d2_state = load_checkpoint(args.input)
    print(f"  loaded:    {_summarise(d2_state)}")

    print("Applying rename table ...")
    mayaku_state, dropped, unrecognised = rename_state(d2_state)

    if unrecognised:
        print(
            f"error: {len(unrecognised)} key(s) had no matching rename rule. "
            "The rename table needs an entry for each. Sample (up to 20):",
            file=sys.stderr,
        )
        for k in unrecognised[:20]:
            print(f"  {k}", file=sys.stderr)
        return 1

    if dropped:
        print(f"  dropped:   {len(dropped)} key(s) with no Mayaku equivalent (e.g. {dropped[0]})")
    print(f"  renamed:   {_summarise(mayaku_state)}")

    if args.channel_order == "bgr":
        print("Reversing stem.conv1 input channels (BGR → RGB) ...")
        before = mayaku_state[_STEM_CONV_KEY].clone()
        reverse_stem_input_channels(mayaku_state)
        after = mayaku_state[_STEM_CONV_KEY]
        print(
            f"  conv1 mean per-channel before: {before.mean(dim=(0, 2, 3)).tolist()}"
        )
        print(
            f"  conv1 mean per-channel after:  {after.mean(dim=(0, 2, 3)).tolist()}"
        )
    else:
        print("--channel-order rgb: skipping stem.conv1 channel reverse.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mayaku_state, args.output)
    out_size = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output} ({out_size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
