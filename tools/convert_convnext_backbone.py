"""Convert an external ConvNeXt backbone checkpoint to Mayaku's layout.

**Why this exists.** The `mayaku` library loads *mayaku* weights, period — it
has no knowledge of timm / facebookresearch / torchvision as *weight sources*
(no auto-download, no runtime format remapping). This standalone tool is the one
place that knows how to lift an external ConvNeXt backbone into mayaku's key
layout, so a maintainer can warm-start a from-scratch train from a public
ImageNet-pretrained backbone. It is **not** part of the installed package.

Supported source formats (auto-detected from the key naming):

* **timm** (`stem.0`, `stages.N.blocks.Y.{conv_dw,norm,mlp.fc1,mlp.fc2,gamma}`,
  `stages.N.downsample.{0,1}`) — e.g. `convnext_femto.d1_in1k`. timm packs the
  pointwise MLP as 1×1 Conv2d; we squeeze `(out,in,1,1) → (out,in)` to match
  torchvision's Linear.
* **facebookresearch / Liu et al.** (`downsample_layers.k`,
  `stages.k.j.{dwconv,norm,pwconv1,pwconv2,gamma}`) — the original ConvNeXt
  release, DINOv3 LVD-1689M, and most forks. `gamma (C,) → layer_scale (C,1,1)`.
* **torchvision** (`features.k.j.block.{0,2,3,5}`, `layer_scale` already
  `(C,1,1)`) — `torchvision.models`, torch.hub, lightly_train exports. Only the
  top-level `features.{i}` prefix is rewritten; the per-block sub-structure is
  identical to mayaku's.

The output is a bare backbone `state_dict` (`.pth`) whose keys match
`ConvNeXtBackbone(name).state_dict()` — `stem.*` / `_res_downs.resN.*` /
`_res_stages.resN.*`. The tool builds that backbone and does a strict load to
prove every parameter is filled and nothing is left over before it writes.

Usage:
    python tools/convert_convnext_backbone.py INPUT --variant convnext_femto -o OUT.pth

    INPUT may be .pth / .pt / .bin (PyTorch pickle) or .safetensors.
    Source-only classification-head keys (head.*, norm.*, classifier.*) are
    dropped automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

# Import from the installed mayaku package to get the exact target key set.
from mayaku.models.backbones.convnext import ConvNeXtBackbone

# k → carved-stage name, shared by every remapper. Index k is the source's
# 0-based stage/downsample index; k=0 is res2 (fed by the stem, no downsample).
_STAGE_NAMES = ("res2", "res3", "res4", "res5")


# ---------------------------------------------------------------------------
# Read + unwrap
# ---------------------------------------------------------------------------


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Read a ConvNeXt checkpoint file into a top-level dict.

    The return type is ``dict[str, Any]`` because wrapped checkpoints
    (``{"model": state_dict, "epoch": 42, ...}``) have non-tensor leaves at the
    top level; :func:`_unwrap_checkpoint` narrows to the state-dict afterwards.
    """
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors requires the 'safetensors' package; "
                "install with `pip install safetensors`, or download the "
                f"`.pth` variant of {path.name} instead."
            ) from exc
        return cast(dict[str, Any], load_file(str(path), device="cpu"))
    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    return cast(dict[str, Any], loaded)


def _unwrap_checkpoint(raw: dict[str, Any]) -> dict[str, Tensor]:
    # Common wrappers: {"model": state_dict}, {"state_dict": state_dict},
    # {"teacher": state_dict} (DINOv3 SSL checkpoints sometimes nest under
    # "teacher"/"student" for the EMA copy). Detect by absence of tensor
    # leaves at the top level.
    if not raw:
        return cast(dict[str, Tensor], raw)
    sample_value = next(iter(raw.values()))
    if isinstance(sample_value, torch.Tensor):
        return cast(dict[str, Tensor], raw)
    for key in ("model", "state_dict", "teacher", "student"):
        if key in raw and isinstance(raw[key], dict):
            return cast(dict[str, Tensor], raw[key])
    raise RuntimeError(
        "ConvNeXt checkpoint has a non-tensor top level but no known wrapper "
        f"key (model/state_dict/teacher/student). Got top-level keys: "
        f"{sorted(raw.keys())[:8]}"
    )


# ---------------------------------------------------------------------------
# Format detection + dispatch
# ---------------------------------------------------------------------------


def remap_convnext_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename external ConvNeXt keys → Mayaku's internal layout.

    Dispatches on a *positive, backbone-defining* signature unique to each
    source — a marker the format's ConvNeXt backbone is guaranteed to emit and
    the other two never do — rather than a substring scan that a bundled
    sub-module (SSL/detection head with its own ``.blocks.``) could trip. An
    input matching none of the three is rejected, not silently assumed-facebook.
    Source-only keys (``head.*``, ``norm.*``, ``classifier.*``, register tokens,
    …) pass through unchanged so the caller's strict-load filter can drop them.
    """
    keys = state.keys()
    if any("conv_dw" in k for k in keys):  # timm's depthwise conv name
        return _remap_timm_convnext_state_dict(state)
    if any(k.startswith("features.") for k in keys):  # torchvision Sequential
        return _remap_torchvision_convnext_state_dict(state)
    if any("dwconv" in k or k.startswith("downsample_layers.") for k in keys):
        return _remap_facebook_convnext_state_dict(state)  # facebookresearch / Liu et al.
    raise RuntimeError(
        "Unrecognized ConvNeXt source format: no timm (conv_dw), torchvision "
        "(features.*), or facebookresearch (dwconv / downsample_layers.*) key "
        f"signature matched. Top-level keys: {sorted(keys)[:8]}"
    )


def _remap_facebook_convnext_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """facebookresearch / Liu et al. ConvNeXt keys → Mayaku's layout.

    * ``downsample_layers.0.{0,1}.*``     → ``stem.{0,1}.*``
    * ``downsample_layers.k.{0,1}.*``     → ``_res_downs.res{k+1}.{0,1}.*``
    * ``stages.k.j.dwconv.*``  → ``_res_stages.resN.j.block.0.*``
    * ``stages.k.j.norm.*``    → ``_res_stages.resN.j.block.2.*``
    * ``stages.k.j.pwconv1.*`` → ``_res_stages.resN.j.block.3.*``
    * ``stages.k.j.pwconv2.*`` → ``_res_stages.resN.j.block.5.*``
    * ``stages.k.j.gamma``     → ``_res_stages.resN.j.layer_scale`` (C,)→(C,1,1)
    """
    out: dict[str, Tensor] = {}
    # torchvision CNBlock.block Sequential index: dwconv=0, Permute=1,
    # LayerNorm=2, pwconv1=3, GELU=4, pwconv2=5, Permute=6. Only the
    # parametric layers appear in a state_dict.
    _BLOCK_SUB = {"dwconv": 0, "norm": 2, "pwconv1": 3, "pwconv2": 5}

    for key, value in state.items():
        parts = key.split(".")
        # ---- downsample_layers.k.{0,1}.{weight,bias} ---------------------
        if parts[0] == "downsample_layers":
            k = int(parts[1])  # 0 = stem, 1..3 = pre-res{3,4,5} downsamples
            if k == 0:
                new_key = "stem." + ".".join(parts[2:])
            else:
                new_key = f"_res_downs.{_STAGE_NAMES[k]}." + ".".join(parts[2:])
            out[new_key] = value
            continue

        # ---- stages.k.j.{dwconv,norm,pwconv1,pwconv2,gamma}.{w,b} --------
        if parts[0] == "stages":
            k = int(parts[1])
            j = int(parts[2])
            sub = parts[3]
            stage = _STAGE_NAMES[k]
            if sub == "gamma":
                out[f"_res_stages.{stage}.{j}.layer_scale"] = value.view(-1, 1, 1)
                continue
            if sub not in _BLOCK_SUB:
                out[key] = value
                continue
            tail = ".".join(parts[4:])  # "weight" / "bias"
            out[f"_res_stages.{stage}.{j}.block.{_BLOCK_SUB[sub]}.{tail}"] = value
            continue

        # ---- everything else (norm, norms, mask_token, ...) --------------
        out[key] = value

    return out


def _remap_timm_convnext_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """timm ConvNeXt keys (dot naming) → Mayaku's layout.

    timm 1.0.x lays each block out as:
        stem.{0,1}                        → stem.{0,1}
        stages.k.downsample.{0,1}         → _res_downs.res{k+1}.{0,1}
        stages.k.blocks.j.conv_dw         → _res_stages.resN.j.block.0
        stages.k.blocks.j.norm            → _res_stages.resN.j.block.2
        stages.k.blocks.j.mlp.fc1         → _res_stages.resN.j.block.3
        stages.k.blocks.j.mlp.fc2         → _res_stages.resN.j.block.5
        stages.k.blocks.j.gamma           → _res_stages.resN.j.layer_scale (C→C,1,1)

    timm packs the pointwise MLP as 1×1 Conv2d; torchvision uses Linear, so the
    ``fc1``/``fc2`` weights are squeezed ``(out,in,1,1) → (out,in)``.
    """
    out: dict[str, Tensor] = {}
    _BLOCK_SUB = {"conv_dw": 0, "norm": 2}
    _MLP_SUB = {"fc1": 3, "fc2": 5}

    for key, value in state.items():
        parts = key.split(".")

        # ---- stem.{0,1}.{weight,bias} ----
        if parts[0] == "stem":
            out[key] = value  # already mayaku's stem.{0,1} naming
            continue

        # ---- stages.k.downsample.{0,1}.{weight,bias} ----
        # stages.k.downsample sits before stage k, i.e. between res{k}→res{k+1}.
        if parts[0] == "stages" and len(parts) >= 3 and parts[2] == "downsample":
            k = int(parts[1])
            tail = ".".join(parts[3:])  # "0.weight" / "1.bias"
            out[f"_res_downs.{_STAGE_NAMES[k]}.{tail}"] = value
            continue

        # ---- stages.k.blocks.j.{conv_dw,norm,mlp.fc1,mlp.fc2,gamma} ----
        if parts[0] == "stages" and len(parts) >= 4 and parts[2] == "blocks":
            k = int(parts[1])
            j = int(parts[3])
            stage = _STAGE_NAMES[k]
            sub = parts[4]

            if sub == "gamma":
                out[f"_res_stages.{stage}.{j}.layer_scale"] = value.view(-1, 1, 1)
                continue
            if sub == "mlp" and len(parts) >= 6:
                mlp_sub = parts[5]  # "fc1" or "fc2"
                if mlp_sub in _MLP_SUB:
                    tail = ".".join(parts[6:])  # "weight" / "bias"
                    # timm Conv2d(1×1) → torchvision Linear: drop spatial dims.
                    if tail == "weight" and value.dim() == 4:
                        value = value.squeeze(-1).squeeze(-1)
                    out[f"_res_stages.{stage}.{j}.block.{_MLP_SUB[mlp_sub]}.{tail}"] = value
                    continue
            if sub in _BLOCK_SUB:
                tail = ".".join(parts[5:])
                out[f"_res_stages.{stage}.{j}.block.{_BLOCK_SUB[sub]}.{tail}"] = value
                continue

        out[key] = value

    return out


def _remap_torchvision_convnext_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """torchvision-native ConvNeXt keys → Mayaku's layout.

    torchvision lays the whole network out as one ``features`` Sequential:
        features.0.{0,1}          stem (Conv 4×4 + LayerNorm2d)
        features.{1,3,5,7}.j...   stages res2..res5 (the CNBlock list)
        features.{2,4,6}.{0,1}    downsamples before res3 / res4 / res5

    The per-block sub-structure (``block.{0,2,3,5}`` with Linear pointwise,
    ``layer_scale`` already ``(C,1,1)``) is identical to mayaku's, so only the
    top-level ``features.{i}`` prefix is rewritten — no per-tensor reshape.
    ``classifier.*`` passes through for the strict-load filter to drop.
    """
    out: dict[str, Tensor] = {}
    _STAGE_IDX = {1: "res2", 3: "res3", 5: "res4", 7: "res5"}
    _DOWN_IDX = {2: "res3", 4: "res4", 6: "res5"}
    for key, value in state.items():
        parts = key.split(".")
        if parts[0] != "features" or len(parts) < 3:
            out[key] = value
            continue
        fi = int(parts[1])
        tail = ".".join(parts[2:])
        if fi == 0:
            out[f"stem.{tail}"] = value
        elif fi in _STAGE_IDX:
            out[f"_res_stages.{_STAGE_IDX[fi]}.{tail}"] = value
        elif fi in _DOWN_IDX:
            out[f"_res_downs.{_DOWN_IDX[fi]}.{tail}"] = value
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def convert(input_path: Path, variant: str) -> dict[str, Tensor]:
    """Read + remap an external checkpoint, strict-validated against ``variant``.

    Returns the mayaku-format backbone state_dict. Raises if any backbone
    parameter is left unfilled after remap (a wrong variant, or an unsupported
    source layout).
    """
    raw = _read_checkpoint(input_path)
    state = _unwrap_checkpoint(raw)
    remapped = remap_convnext_state_dict(state)

    # The architecture-only backbone gives the exact target key set.
    backbone = ConvNeXtBackbone(name=variant)  # type: ignore[arg-type]
    target = backbone.state_dict()
    target_keys = set(target)

    filtered = {k: v for k, v in remapped.items() if k in target_keys}
    missing = target_keys - set(filtered)
    dropped = sorted(set(remapped) - target_keys)

    if missing:
        raise RuntimeError(
            f"{input_path.name}: {len(missing)} backbone parameters unfilled "
            f"after remap into {variant!r} — is the --variant right?\n  "
            + "\n  ".join(sorted(missing)[:12])
            + (" ..." if len(missing) > 12 else "")
        )

    # Shape check: catch a same-key/wrong-shape mismatch (e.g. wrong variant
    # whose key set happens to coincide) before the load_state_dict error.
    shape_bad = [
        k for k, v in filtered.items() if tuple(v.shape) != tuple(target[k].shape)
    ]
    if shape_bad:
        raise RuntimeError(
            f"{input_path.name}: shape mismatch on {len(shape_bad)} keys for "
            f"variant {variant!r} (wrong variant?): e.g. {shape_bad[0]} "
            f"{tuple(filtered[shape_bad[0]].shape)} != {tuple(target[shape_bad[0]].shape)}"
        )

    # Prove it loads cleanly into the real architecture.
    backbone.load_state_dict(filtered, strict=True)

    if dropped:
        print(
            f"  dropped {len(dropped)} source-only key(s) "
            f"(classification head etc.): {dropped[:4]}"
            + (" ..." if len(dropped) > 4 else ""),
            file=sys.stderr,
        )
    return filtered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an external ConvNeXt backbone (timm / facebookresearch "
        "/ torchvision) to a Mayaku-format backbone state_dict.",
    )
    parser.add_argument("input", type=Path, help="external ConvNeXt checkpoint (.pth/.pt/.bin/.safetensors)")
    parser.add_argument(
        "--variant",
        required=True,
        help="target Mayaku ConvNeXt variant, e.g. convnext_femto / convnext_tiny",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="output mayaku-format backbone .pth")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input checkpoint not found: {args.input}")

    print(f"Converting {args.input.name} → {args.variant} …", file=sys.stderr)
    state = convert(args.input, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(args.output))
    print(f"  wrote {len(state)} tensors → {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
