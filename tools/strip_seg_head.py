"""Strip the segmentation head from a UniQuery checkpoint, in place.

Throwaway tool. The `models/detection/*` checkpoints were seeded as
byte-identical copies of the segmentation checkpoints, so each still carries
a dynamic mask head in two coupled places:

  * state_dict   — 16 tensors under the ``mask_head.*`` prefix.
  * sidecar cfg  — ``model.uniquery_mask`` is populated.

`build_uniquery` builds a mask head iff ``cfg.model.uniquery_mask is not None``
(src/mayaku/models/detectors/uniquery.py), and the deploy loader does a
*strict* ``load_state_dict`` (src/mayaku/cli/_factory.py). So the two must be
edited together — drop the tensors AND null the config — or the next
`from_pretrained` fails on missing/unexpected keys. This script does both and
re-saves over the same file (mutate in place; no backup — the originals are
the segmentation checkpoints, kept separately).

It also drops a third field when present: ``model.backbone.weights_path``, the
training-box pretrain-init path (e.g. ``/home/vision/mayaku/convnext_*.pth``)
recorded by checkpoints trained before the backbone went architecture-only. The
current ``BackboneConfig`` removed that field and forbids extras, so a checkpoint
carrying it fails ``model_validate`` (i.e. won't load) until the key is dropped.

The edit is idempotent: a checkpoint already stripped (no ``mask_head.*``
tensors, ``uniquery_mask`` null, and no ``backbone.weights_path`` key) is
reported and left untouched.

Usage:
    python tools/strip_seg_head.py models/detection/mayaku-n-det/**/mayaku-n-det.pth
    python tools/strip_seg_head.py models/detection            # recurse for *.pth
    python tools/strip_seg_head.py models/detection --no-verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_MASK_PREFIX = "mask_head."


def _find_checkpoints(paths: list[Path]) -> list[Path]:
    """Expand each argument: a .pth is taken as-is, a directory is globbed."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pth")))
        else:
            out.append(p)
    return out


def strip_checkpoint(path: Path, *, verify: bool) -> str:
    """Remove the mask head from one checkpoint in place.

    Returns a one-line human status. Raises on a malformed checkpoint (no
    ``mayaku`` sidecar, or a config the schema rejects after the edit) — the
    file is left untouched in that case because ``torch.save`` runs last.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and "model" in obj and isinstance(obj.get("mayaku"), dict)):
        raise ValueError(
            f"{path} is not a self-describing Mayaku checkpoint "
            "(need a top-level 'model' state_dict and a 'mayaku' sidecar)."
        )

    state: dict = obj["model"]
    sidecar: dict = obj["mayaku"]
    config = sidecar.get("config")
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(f"{path} sidecar has no embedded model config.")

    mask_keys = [k for k in state if k.startswith(_MASK_PREFIX)]
    model_cfg = config["model"]
    backbone_cfg = model_cfg.get("backbone") or {}
    had_uniquery_mask = model_cfg.get("uniquery_mask") is not None
    had_weights_path = "weights_path" in backbone_cfg

    if not mask_keys and not had_uniquery_mask and not had_weights_path:
        return f"{path.name}: already detection-only — skipped"

    # 1) Drop the mask-head tensors from the weights.
    for k in mask_keys:
        del state[k]

    # 2) Null the mask config and DELETE any ``backbone.weights_path`` — the
    #    latter is a training-box pretrain-init path (e.g.
    #    /home/vision/mayaku/convnext_*.pth) recorded by checkpoints trained
    #    before the backbone went architecture-only. The current BackboneConfig
    #    schema removed that field and forbids extras, so the key must be dropped
    #    (not nulled) or model_validate below rejects it — which is also why such
    #    a checkpoint fails to load until stripped. Re-validate so a malformed
    #    edit is caught before we overwrite the file; model_validate normalises
    #    the dump.
    from mayaku.config import MayakuConfig

    model_cfg["uniquery_mask"] = None
    if had_weights_path:
        del model_cfg["backbone"]["weights_path"]
    validated = MayakuConfig.model_validate(config)
    sidecar["config"] = validated.model_dump(mode="json")

    torch.save(obj, path)

    if verify:
        # The real proof the two edits agree: build from the sidecar and load
        # the weights strictly, exactly as `from_pretrained` will.
        from mayaku.cli._factory import load_detector

        load_detector(path)  # raises on any missing/unexpected key

    verified = " verified" if verify else ""
    wp = ", backbone.weights_path removed" if had_weights_path else ""
    return (
        f"{path.name}: dropped {len(mask_keys)} mask_head tensor(s), "
        f"uniquery_mask -> null{wp}{verified}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strip the segmentation (mask) head from UniQuery checkpoints, in place.",
    )
    p.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Checkpoint .pth files, or directories to search recursively for *.pth.",
    )
    p.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip the post-save load_detector() strict-load check.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    checkpoints = _find_checkpoints(args.paths)
    if not checkpoints:
        print("error: no .pth checkpoints found in the given paths", file=sys.stderr)
        return 2

    missing = [p for p in checkpoints if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: {p} does not exist", file=sys.stderr)
        return 2

    print(f"Stripping segmentation head from {len(checkpoints)} checkpoint(s):")
    failures = 0
    for path in checkpoints:
        try:
            print(f"  {strip_checkpoint(path, verify=args.verify)}")
        except Exception as exc:
            failures += 1
            print(f"  {path.name}: FAILED — {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
