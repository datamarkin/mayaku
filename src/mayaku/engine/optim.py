"""Optimizer + LR scheduler builders.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §3 (`solver/build.py`,
`solver/lr_scheduler.py`) for the in-scope SGD + warmup-multistep /
warmup-cosine combination.

Key behaviours preserved from the upstream defaults:

* **Per-parameter weight-decay groups.** Norm parameters
  (`BatchNorm`/`LayerNorm`/`GroupNorm`/`FrozenBatchNorm`) get
  ``weight_decay_norm`` (0 by default), everything else gets
  ``weight_decay`` (1e-4 by default). This is the standard practice
  for ResNet-FPN training and is what the spec §6.1 defaults imply.
* **`WarmupMultiStepLR`** linearly warms up for ``warmup_iters``
  iterations from ``warmup_factor * base_lr`` to ``base_lr``, then
  multiplies by ``gamma`` at every step in ``steps``.
* **`WarmupCosineLR`** uses the same warmup, then a half-cosine decay
  from ``base_lr`` to ``0`` over ``[warmup_iters, max_iter)``.

Both schedulers are returned as ``LambdaLR`` instances so the same
``lr_scheduler.step()`` plumbing in :class:`LRScheduler` (the engine
hook) works for either.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable

import torch
from torch import nn

from mayaku.config.schemas import SolverConfig
from mayaku.models.backbones._frozen_bn import FrozenBatchNorm2d

__all__ = ["build_lr_scheduler", "build_optimizer"]

logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, cfg: SolverConfig) -> torch.optim.Optimizer:
    """Build the optimizer chosen by ``cfg.optimizer_name``.

    Two parameter groups are always emitted: norm params (BatchNorm /
    LayerNorm / GroupNorm / FrozenBatchNorm) get ``cfg.weight_decay_norm``
    (default 0), everything else gets ``cfg.weight_decay``. This split
    is the standard D2 practice and is preserved for both SGD and AdamW.

    SGD uses ``momentum`` / ``nesterov``; AdamW uses ``betas`` / ``eps``.
    The unused pair is silently ignored.
    """
    param_groups = _split_norm_groups(model, cfg)
    if cfg.optimizer_name == "SGD":
        return torch.optim.SGD(
            param_groups,
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    elif cfg.optimizer_name == "AdamW":
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.base_lr,
            betas=cfg.betas,
            eps=cfg.eps,
        )
    else:
        # Defended by the schema's Literal — unreachable in normal flow.
        raise ValueError(f"unknown optimizer_name: {cfg.optimizer_name!r}")


def _split_norm_groups(model: nn.Module, cfg: SolverConfig) -> list[dict[str, object]]:
    if cfg.llrd_enabled:
        return _build_llrd_groups(model, cfg)

    norm_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        is_norm = isinstance(
            module,
            nn.modules.batchnorm._BatchNorm | nn.LayerNorm | nn.GroupNorm | FrozenBatchNorm2d,
        )
        for p in module.parameters(recurse=False):
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            (norm_params if is_norm else other_params).append(p)

    param_groups: list[dict[str, object]] = []
    if other_params:
        param_groups.append({"params": other_params, "weight_decay": cfg.weight_decay})
    if norm_params:
        param_groups.append({"params": norm_params, "weight_decay": cfg.weight_decay_norm})
    if not param_groups:
        # Defensive: torch.optim raises on an empty params list. The
        # caller almost certainly has a bug (frozen everything?) but
        # we'd rather surface it as a clearer message.
        raise ValueError("build_optimizer: no trainable parameters found on the model")
    return param_groups


# ---------------------------------------------------------------------------
# LLRD (layer-wise learning rate decay)
# ---------------------------------------------------------------------------
#
# Scheme: each backbone parameter is assigned a depth ``layer_id`` (0 at the
# input, increasing toward the output), then its LR is scaled by
# ``decay ** ((num_layers + 2) - layer_id - 1)``. The detector neck/heads
# (FPN, RPN, ROI) are treated as the top layer (``layer_id = num_layers + 1``)
# and train at full ``base_lr`` (scale ``decay^0 = 1.0``). Frozen parameters
# (``requires_grad=False``) are filtered out and contribute no group, so
# ``freeze_at`` composes orthogonally. LLRD also composes with the existing
# norm-vs-non-norm weight-decay split — each ``layer_id`` may emit up to two
# groups (norm wd vs non-norm wd) at the same scaled LR.
#
# ConvNeXt scheme: verbatim from MMDet's ``get_layer_id_for_convnext`` in
# ``mmdet/engine/optimizers/layer_decay_optimizer_constructor.py``, with our
# carved-name → MMDet-name adapter applied first. Stage 2 uses ``block_id //
# 3`` bucketing — this is what makes ``num_layers = 6`` (Tiny) and
# ``num_layers = 12`` (Small/Base/Large) fall out of the formula directly.
#
# ResNet scheme: per-stage (stem=0, res{2,3,4,5}={1,2,3,4}, head=5). MMDet
# has no reference for ResNet detection LLRD. KNOWN LIMITATION: all of
# ``res4``'s blocks collapse to a single ``layer_id``. For ResNet-101,
# ``res4`` has 23 blocks — the bulk of the network on one LR — and LLRD will
# be near-flat over its largest stage. If R101 transfer quality on
# distilled-backbone experiments matters, add ``block_id // K`` bucketing
# for ``res4`` (analogous to ConvNeXt stage-2) as v2.


_NORM_MODULE_TYPES: tuple[type, ...] = (
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    FrozenBatchNorm2d,
)


def _build_llrd_groups(model: nn.Module, cfg: SolverConfig) -> list[dict[str, object]]:
    backbone_prefix, family = _find_backbone(model)
    num_layers = _resolve_llrd_num_layers(model, backbone_prefix, family)
    head_layer_id = num_layers + 1
    n_internal = num_layers + 2

    norm_param_ids = _collect_norm_param_ids(model)

    # Bucket (layer_id, is_norm) → list[Parameter], plus parallel lists of
    # parameter names per bucket for the startup log.
    buckets: dict[tuple[int, bool], list[nn.Parameter]] = {}
    bucket_names: dict[tuple[int, bool], list[str]] = {}

    backbone_prefix_dot = backbone_prefix + "." if backbone_prefix else ""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(backbone_prefix_dot) and backbone_prefix_dot:
            local = name[len(backbone_prefix_dot):]
            if family == "convnext":
                layer_id = _layer_id_for_convnext_param(local, num_layers)
            else:  # family == "resnet"
                layer_id = _layer_id_for_resnet_param(local, num_layers)
        else:
            layer_id = head_layer_id

        is_norm = id(p) in norm_param_ids
        key = (layer_id, is_norm)
        buckets.setdefault(key, []).append(p)
        bucket_names.setdefault(key, []).append(name)

    if not buckets:
        raise ValueError("build_optimizer: no trainable parameters found on the model")

    groups: list[dict[str, object]] = []
    for (layer_id, is_norm), params in sorted(buckets.items()):
        scale = cfg.llrd_decay ** (n_internal - layer_id - 1)
        lr = cfg.base_lr * scale
        wd = cfg.weight_decay_norm if is_norm else cfg.weight_decay
        group_name = f"layer{layer_id}_{'norm' if is_norm else 'decay'}"
        groups.append(
            {
                "params": params,
                "weight_decay": wd,
                "lr": lr,
                # Bookkeeping fields ignored by torch but consumed by the
                # startup logger and the monotonicity assertion. Carry
                # them on the group dict so a debugger inspection still
                # reveals the depth mapping after optimizer construction.
                "name": group_name,
                "layer_id": layer_id,
                "is_norm_group": is_norm,
                "lr_scale": scale,
            }
        )

    _log_and_assert_llrd_groups(groups, cfg, num_layers, head_layer_id, bucket_names)
    return groups


def _find_backbone(model: nn.Module) -> tuple[str, str]:
    """Locate the LLRD-supported backbone and return ``(qualified_prefix, family)``.

    Walks ``model.named_modules()`` looking for the first instance of a
    ConvNeXt or ResNet backbone. Imports are local to avoid a circular
    import between ``engine.optim`` and the model package at module load.
    """
    from mayaku.models.backbones.convnext import ConvNeXtBackbone
    from mayaku.models.backbones.resnet import ResNetBackbone

    for name, module in model.named_modules():
        if isinstance(module, ConvNeXtBackbone):
            return name, "convnext"
        if isinstance(module, ResNetBackbone):
            return name, "resnet"
    raise ValueError(
        "llrd_enabled=True but no supported backbone (ConvNeXtBackbone / "
        "ResNetBackbone) found on the model. LLRD currently supports those "
        "two families only."
    )


def _resolve_llrd_num_layers(model: nn.Module, backbone_prefix: str, family: str) -> int:
    if family == "resnet":
        return 4  # stem(0) + res{2,3,4,5}(1..4); head at 5
    # ConvNeXt: ``num_layers`` is fully determined by stage-2's block count
    # via MMDet's ``block_id // 3`` bucketing. Tiny (depth 9) buckets to
    # ``{0, 1, 2}`` → layer_ids ``{3, 4, 5}`` → max_layer_id 6. Small/Base/
    # Large (depth 27) bucket to ``{0..8}`` → layer_ids ``{3..11}`` → 12.
    # This is mechanical, not a free choice.
    backbone = model.get_submodule(backbone_prefix)
    stage2 = backbone._res_stages["res4"]  # our "res4" carved-name == MMDet's stages.2
    n_blocks = len(stage2)
    return 6 if n_blocks <= 9 else 12


def _collect_norm_param_ids(model: nn.Module) -> set[int]:
    """Mirror the legacy splitter's norm detection by ``id(p)``.

    Matches ``_split_norm_groups``' rule exactly: a parameter owned
    *directly* by a norm-typed module is a norm parameter; everything
    else is not. Using ``id()`` keeps composition with the LLRD layer-id
    grouping cheap and unambiguous.
    """
    norm_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, _NORM_MODULE_TYPES):
            for p in module.parameters(recurse=False):
                norm_ids.add(id(p))
    return norm_ids


def _layer_id_for_convnext_param(local_name: str, num_layers: int) -> int:
    """Map a ConvNeXtBackbone-local parameter name to MMDet's ``layer_id``.

    ``local_name`` is the parameter name with the backbone prefix stripped
    (e.g. ``"stem.0.weight"``, ``"_res_stages.res4.6.block.0.weight"``).

    Adapter from our carved naming to MMDet's:

    * ``stem.*``                  → MMDet ``downsample_layers.0.*``
    * ``_res_downs.res2.*``       → (Identity in our model; should not appear)
    * ``_res_downs.res{3,4,5}.*`` → MMDet ``downsample_layers.{1,2,3}.*``
    * ``_res_stages.res{2,3,4,5}.{block}.*`` → MMDet ``stages.{0,1,2,3}.{block}.*``

    Then MMDet's verbatim rules apply (``max_layer_id = num_layers``):

    * ``downsample_layers.0``  → 0
    * ``downsample_layers.1``  → 2
    * ``downsample_layers.2``  → 3
    * ``downsample_layers.3``  → max_layer_id
    * ``stages.0.{block}``     → 1
    * ``stages.1.{block}``     → 2
    * ``stages.2.{block}``     → 3 + block_id // 3   (bucketed)
    * ``stages.3.{block}``     → max_layer_id
    """
    parts = local_name.split(".")
    head = parts[0]
    max_layer_id = num_layers

    if head == "stem":
        # MMDet downsample_layers.0
        return 0

    if head == "_res_downs":
        stage = parts[1]
        if stage == "res2":
            # Our _res_downs.res2 is Identity (parameterless). Reaching
            # this branch means a checkpoint or refactor put a parameter
            # under an Identity module — surface the contract violation
            # loudly rather than silently dropping it into layer 0.
            raise AssertionError(
                f"_res_downs.res2 must be parameterless (Identity); got parameter {local_name!r}. "
                "The LLRD adapter assumes our _res_downs.res2 ↔ no MMDet downsample."
            )
        # res3 → downsample_layers.1; res4 → .2; res5 → .3
        ds_idx = {"res3": 1, "res4": 2, "res5": 3}[stage]
        if ds_idx == 1:
            return 2
        if ds_idx == 2:
            return 3
        return max_layer_id  # ds_idx == 3

    if head == "_res_stages":
        stage = parts[1]
        block_id = int(parts[2])
        stage_idx = {"res2": 0, "res3": 1, "res4": 2, "res5": 3}[stage]
        if stage_idx == 0:
            return 1
        if stage_idx == 1:
            return 2
        if stage_idx == 2:
            return 3 + block_id // 3
        return max_layer_id  # stage_idx == 3

    # Unknown structure under the backbone — surface it. Silently
    # bucketing into head would let a model refactor break LLRD invisibly.
    raise AssertionError(
        f"unrecognised ConvNeXt backbone parameter name {local_name!r}; "
        "expected one of stem.*, _res_downs.res{2,3,4,5}.*, _res_stages.res{2,3,4,5}.*"
    )


def _layer_id_for_resnet_param(local_name: str, num_layers: int) -> int:
    """Per-stage ResNet layer-id assignment.

    ``num_layers`` is fixed at 4 for ResNet (see ``_resolve_llrd_num_layers``).
    Known coarseness: ``res4`` collapses all blocks to one layer_id (23
    blocks for R101).
    """
    head = local_name.split(".", 1)[0]
    if head == "stem":
        return 0
    if head == "res2":
        return 1
    if head == "res3":
        return 2
    if head == "res4":
        return 3
    if head == "res5":
        return num_layers  # == 4
    raise AssertionError(
        f"unrecognised ResNet backbone parameter name {local_name!r}; "
        "expected one of stem.*, res{2,3,4,5}.*"
    )


def _log_and_assert_llrd_groups(
    groups: list[dict[str, object]],
    cfg: SolverConfig,
    num_layers: int,
    head_layer_id: int,
    bucket_names: dict[tuple[int, bool], list[str]],
) -> None:
    """Emit a per-group log line (rank-0 only) and assert sanity invariants.

    Two invariants must hold for any correct LLRD schedule:

    * Backbone group LRs are monotonically non-decreasing with ``layer_id``
      (input layers get smaller LRs than output layers).
    * The head/neck group LR equals ``base_lr`` exactly (it sits at the
      top ``layer_id`` and gets ``decay^0 = 1.0``).
    """
    # Local import to avoid a heavy distributed dependency at module
    # load — engine.distributed pulls in ``torch.distributed`` machinery
    # that's not needed when this module is imported standalone (e.g.
    # by config-validation tooling).
    from mayaku.engine.distributed import is_main_process

    if is_main_process():
        for g in groups:
            layer_id = g["layer_id"]
            names = bucket_names.get((layer_id, g["is_norm_group"]), [])
            logger.info(
                "[llrd] group=%s layer_id=%d lr=%.6g wd=%.6g lr_scale=%.6g n_params=%d",
                g["name"],
                layer_id,
                g["lr"],
                g["weight_decay"],
                g["lr_scale"],
                len(names),
            )
        logger.info(
            "[llrd] base_lr=%.6g decay=%.6g num_layers=%d (head layer_id=%d) total_groups=%d",
            cfg.base_lr,
            cfg.llrd_decay,
            num_layers,
            head_layer_id,
            len(groups),
        )

    # Monotonicity check across distinct layer_ids (the same layer_id may
    # appear twice — norm vs non-norm — at the same LR, which is fine).
    seen_lrs: dict[int, float] = {}
    for g in groups:
        seen_lrs.setdefault(g["layer_id"], float(g["lr"]))
    ordered = sorted(seen_lrs.items())
    for (lid_a, lr_a), (lid_b, lr_b) in zip(ordered, ordered[1:], strict=False):
        if lr_b < lr_a - 1e-12:
            raise AssertionError(
                f"[llrd] LR monotonicity violated: layer_id {lid_a} lr={lr_a} > "
                f"layer_id {lid_b} lr={lr_b}. Indicates a bug in the layer-id "
                "adapter or scale formula."
            )

    # Head/neck must sit at exactly base_lr (scale 1.0).
    head_lr = seen_lrs.get(head_layer_id)
    if head_lr is None:
        # No head params — possible only on a backbone-only model. Allow
        # it but skip the head-LR assertion.
        return
    if not math.isclose(head_lr, cfg.base_lr, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"[llrd] head/neck LR {head_lr} != base_lr {cfg.base_lr}; expected "
            f"scale=decay^0=1.0 at layer_id {head_layer_id}."
        )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: SolverConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Construct ``WarmupMultiStepLR`` or ``WarmupCosineLR`` as a LambdaLR."""
    if cfg.lr_scheduler_name == "WarmupMultiStepLR":
        lr_fn = _warmup_multistep_lambda(
            steps=cfg.steps,
            gamma=cfg.gamma,
            warmup_iters=cfg.warmup_iters,
            warmup_factor=cfg.warmup_factor,
            warmup_method=cfg.warmup_method,
        )
    elif cfg.lr_scheduler_name == "WarmupCosineLR":
        lr_fn = _warmup_cosine_lambda(
            max_iter=cfg.max_iter,
            warmup_iters=cfg.warmup_iters,
            warmup_factor=cfg.warmup_factor,
            warmup_method=cfg.warmup_method,
        )
    else:  # pragma: no cover — defended by the schema's Literal
        raise ValueError(f"unknown lr_scheduler_name: {cfg.lr_scheduler_name!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)


# ---------------------------------------------------------------------------
# Lambda factories
# ---------------------------------------------------------------------------


def _warmup_factor_at_iter(method: str, it: int, warmup_iters: int, warmup_factor: float) -> float:
    if it >= warmup_iters:
        return 1.0
    if method == "constant":
        return warmup_factor
    # Linear (default): grow from `warmup_factor` to 1 over warmup_iters.
    alpha = it / max(warmup_iters, 1)
    return warmup_factor * (1.0 - alpha) + alpha


def _warmup_multistep_lambda(
    steps: Iterable[int],
    gamma: float,
    warmup_iters: int,
    warmup_factor: float,
    warmup_method: str,
) -> Callable[[int], float]:
    sorted_steps = sorted(steps)

    def lr_lambda(it: int) -> float:
        warm = _warmup_factor_at_iter(warmup_method, it, warmup_iters, warmup_factor)
        decays = sum(1 for s in sorted_steps if it >= s)
        return warm * (gamma**decays)

    return lr_lambda


def _warmup_cosine_lambda(
    max_iter: int,
    warmup_iters: int,
    warmup_factor: float,
    warmup_method: str,
) -> Callable[[int], float]:
    def lr_lambda(it: int) -> float:
        warm = _warmup_factor_at_iter(warmup_method, it, warmup_iters, warmup_factor)
        if it < warmup_iters:
            return warm
        progress = (it - warmup_iters) / max(max_iter - warmup_iters, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda
