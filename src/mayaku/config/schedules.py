"""Canonical solver schedules.

The 1x / 2x / 3x naming inherited from Detectron2 refers to multiples of
the COCO 1x baseline: 90k iters at batch 16. The 3x schedule is the
default for every in-scope detection / segmentation / keypoint model
(`DETECTRON2_TECHNICAL_SPEC.md` §6.2-§6.3); 1x is preserved as a quick
sanity / smoke target.

Numbers come from spec §6.3 and `Base-RCNN-FPN.yaml`. Anything not in
that table inherits the :class:`SolverConfig` defaults.
"""

from __future__ import annotations

from mayaku.config.schemas import SolverConfig

__all__ = ["schedule_1x", "schedule_2x", "schedule_3x"]


_BASE_KW: dict[str, object] = {
    "ims_per_batch": 16,
    "base_lr": 0.02,
    "warmup_iters": 1000,
    "warmup_factor": 1.0 / 1000.0,
    "warmup_method": "linear",
}


def _build(max_iter: int, steps: tuple[int, int], **overrides: object) -> SolverConfig:
    kw = dict(_BASE_KW)
    kw["max_iter"] = max_iter
    kw["steps"] = steps
    kw.update(overrides)  # explicit overrides win
    return SolverConfig(**kw)  # type: ignore[arg-type]


def schedule_3x(**overrides: object) -> SolverConfig:
    """Default Detectron2 3x schedule (`DETECTRON2_TECHNICAL_SPEC.md` §6.3).

    270k iterations at batch 16 (~36.5 epochs of COCO train2017), with
    LR decay by ``GAMMA=0.1`` at iters 210k and 250k, 1k-iter linear
    warm-up at ``warmup_factor=1/1000``.

    ``overrides`` are forwarded to :class:`SolverConfig`; useful for
    flipping ``amp_enabled`` or scaling ``base_lr`` for a smaller batch.
    """
    return _build(270_000, (210_000, 250_000), **overrides)


def schedule_2x(**overrides: object) -> SolverConfig:
    """Detectron2 2x schedule: 180k iters; LR decay at 120k and 160k."""
    return _build(180_000, (120_000, 160_000), **overrides)


def schedule_1x(**overrides: object) -> SolverConfig:
    """Detectron2 1x baseline: 90k iters; LR decay at 60k and 80k."""
    return _build(90_000, (60_000, 80_000), **overrides)
