"""Auto-config: derive the fields a user left unset from the training data.

Two kinds of override, both skipped for any field the user set explicitly:

* **Structural facts**, at any dataset size: the class count, the canvas (the
  data's aspect under the size budget, `mayaku.data.canvas.canvas_for_data`),
  and, for a keypoint model, the keypoint names and left/right flip pairs the
  annotations define.
* **The fine-tune recipe**, applied only when training starts from pretrained
  weights and the dataset has at least `MIN_IMAGES_FOR_AUTO_CONFIG` images:
  run length, the clean final stage, augmentation strength, learning rate and
  warmup, keyed on dataset size. A from-scratch run keeps its recipe.

With ``auto_config.enabled`` False nothing is tuned, but a model cannot be
built without a class count and a canvas: left unset, they are filled plainly
-- the data's class count, a square canvas at the size budget. A class count
that disagrees with the data raises either way.

Auto-config adapts the run to the data; it never changes the model or its
loss. The only fields it may set are listed in `AUTO_PATHS`, and a test holds
every override to that list.

The fine-tune numbers below are starting points carried over from mayaku 2.x
and are to be re-tuned on RF100-VL for this detector.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final

from mayaku.config import MayakuConfig, merge_overrides
from mayaku.data.canvas import canvas_for_data

if TYPE_CHECKING:
    from mayaku.data.coco import CocoLabels

__all__ = ["AUTO_PATHS", "MIN_IMAGES_FOR_AUTO_CONFIG", "apply_auto_config", "collect_set_paths"]

#: Every dotted config path auto-config may set. Anything else -- the tier,
#: the heads, QAT, the loss, the optimizer -- belongs to the user or the
#: checkpoint.
AUTO_PATHS: Final = frozenset({
    "model.num_classes",
    "model.keypoints.names",
    "model.keypoints.flip_pairs",
    "input.canvas_hw",
    "train.epochs",
    "train.final_frac",
    "train.lr",
    "train.warmup_epochs",
    "train.assigner_warmup",
    "train.aug.mosaic",
    "train.aug.mixup",
})

#: Below this image count there is too little signal to pick a fine-tune
#: schedule; only the structural facts are applied.
MIN_IMAGES_FOR_AUTO_CONFIG: Final = 10

# Fine-tune run length: a log taper from MAX epochs on small sets to MIN on
# large ones, so total training work still grows with the data.
MIN_FINETUNE_EPOCHS: Final = 16
MAX_FINETUNE_EPOCHS: Final = 30
_EPOCH_TAPER_IMG_LO: Final = 1_000
_EPOCH_TAPER_IMG_HI: Final = 20_000

# Fine-tune learning rate at the recipe's `lr_ref_batch`, and a one-epoch
# warmup. Starting from pretrained weights the predictions already mean
# something, so the task-aligned assigner runs from the first epoch.
FINETUNE_LR: Final = 2.0e-3
FINETUNE_WARMUP_EPOCHS: Final = 1.0

# The share of the run, at the end, trained on clean (mosaic-free) images.
FINETUNE_FINAL_FRAC: Final = 0.2

# (upper exclusive image count, mosaic probability, mixup probability)
_AUG_BY_SIZE: Final = (
    (500, 0.1, 0.0),
    (2_000, 0.2, 0.0),
    (5_000, 0.3, 0.0),
    (math.inf, 0.5, 0.1),
)


def _finetune_epochs(num_images: int) -> int:
    """MAX epochs up to 1,000 images, MIN from 20,000, log-linear between."""
    t = (math.log(max(1, num_images)) - math.log(_EPOCH_TAPER_IMG_LO)) / (
        math.log(_EPOCH_TAPER_IMG_HI) - math.log(_EPOCH_TAPER_IMG_LO))
    t = min(1.0, max(0.0, t))
    return round(MAX_FINETUNE_EPOCHS - t * (MAX_FINETUNE_EPOCHS - MIN_FINETUNE_EPOCHS))


def _structural(coco: CocoLabels, cfg: MayakuConfig) -> dict[str, Any]:
    """The dataset's structural facts as config overrides."""
    out: dict[str, Any] = {
        "model": {"num_classes": len(coco.cat_ids)},
        "input": {"canvas_hw": canvas_for_data(coco.shapes, cfg.input.size_budget)},
    }
    if cfg.model.keypoints is not None and coco.kpt_names:
        out["model"]["keypoints"] = {"names": tuple(coco.kpt_names),
                                     "flip_pairs": coco.kpt_flip_pairs}
    return out


def _required(coco: CocoLabels, cfg: MayakuConfig) -> dict[str, Any]:
    """The class count and canvas still unset after tuning, filled plainly;
    raises when a set class count disagrees with the data."""
    nc = len(coco.cat_ids)
    out: dict[str, Any] = {}
    if cfg.model.num_classes is None:
        out["model"] = {"num_classes": nc}
    elif cfg.model.num_classes != nc:
        raise ValueError(f"model.num_classes is {cfg.model.num_classes} and the training "
                         f"annotations have {nc} categories")
    if cfg.input.canvas_hw is None:
        out["input"] = {"canvas_hw": (cfg.input.size_budget,) * 2}
    return out


def _finetune(num_images: int) -> dict[str, Any]:
    """The fine-tune recipe for a dataset of this size; empty below
    `MIN_IMAGES_FOR_AUTO_CONFIG`."""
    if num_images < MIN_IMAGES_FOR_AUTO_CONFIG:
        return {}
    mosaic, mixup = next((m, x) for upper, m, x in _AUG_BY_SIZE if num_images < upper)
    return {"train": {
        "epochs": _finetune_epochs(num_images),
        "final_frac": FINETUNE_FINAL_FRAC,
        "lr": FINETUNE_LR,
        "warmup_epochs": FINETUNE_WARMUP_EPOCHS,
        "assigner_warmup": 0,
        "aug": {"mosaic": mosaic, "mixup": mixup},
    }}


def apply_auto_config(cfg: MayakuConfig, coco: CocoLabels, user_set_paths: Collection[str] = (),
                      finetune: bool = True) -> tuple[MayakuConfig, list[tuple[str, Any, Any]]]:
    """`cfg` with the auto-derived fields filled in from the training
    annotations `coco` (`mayaku.data.coco.CocoLabels`), and the list of
    ``(path, old, new)`` changes made. Paths in `user_set_paths` are never
    touched; `finetune` False (training from scratch) applies the structural
    facts only. With ``auto_config.enabled`` False only the class count and
    canvas are filled, when unset (see the module docstring).
    """
    overrides: dict[str, Any] = {}
    if cfg.auto_config.enabled:
        overrides = _structural(coco, cfg) | (_finetune(len(coco.shapes)) if finetune else {})
        overrides = _filter_unset(overrides, user_set_paths)
    new = merge_overrides(cfg, overrides)
    required = _required(coco, new)
    new = merge_overrides(new, required)
    before = dict(_walk_leaves(cfg.model_dump(mode="json")))
    after = dict(_walk_leaves(new.model_dump(mode="json")))
    paths = {p for ov in (overrides, required) for p, _ in _walk_leaves(ov)}
    changes = [(p, before.get(p), after[p]) for p in sorted(paths) if before.get(p) != after[p]]
    return new, changes


def collect_set_paths(raw: Any) -> set[str]:
    """Dotted paths of every leaf in a parsed user config, e.g.
    ``{"train": {"lr": 0.01}}`` -> ``{"train.lr"}``: the fields auto-config
    must leave alone."""
    return {p for p, _ in _walk_leaves(raw)}


def _walk_leaves(payload: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """``(dotted_path, value)`` for every leaf of a nested mapping."""
    if isinstance(payload, Mapping):
        for k, v in payload.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, Mapping):
                yield from _walk_leaves(v, path)
            else:
                yield path, v


def _filter_unset(overrides: Mapping[str, Any], user_set_paths: Collection[str],
                  prefix: str = "") -> dict[str, Any]:
    """`overrides` without the paths the user set; empty branches pruned."""
    result: dict[str, Any] = {}
    for k, v in overrides.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            sub = _filter_unset(v, user_set_paths, path)
            if sub:
                result[k] = sub
        elif path not in user_set_paths:
            result[k] = v
    return result
