"""1-D k-means for anchor sizes and aspect ratios.

Two clusterings drive the auto-config anchor generator:

* :func:`cluster_sizes` runs k-means on ``sqrt(box_area)`` to assign one
  anchor scale per FPN level (P3–P7). One-size-per-level matches the
  FPN convention from ``DETECTRON2_TECHNICAL_SPEC.md`` §2.3 and
  :class:`mayaku.config.AnchorGeneratorConfig`'s default ladder.
* :func:`cluster_aspect_ratios` runs k-means on ``w/h`` to produce a
  shared aspect-ratio set used at every level.

Deterministic init via quantiles makes outputs stable across runs without
needing torch / numpy seeding, and matches the way users naturally read
"anchor sizes": small → large.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

__all__ = ["cluster_aspect_ratios", "cluster_sizes"]

_MAX_ITERS: Final = 50
_CONVERGENCE_EPS: Final = 1e-6


def _kmeans_1d(values: Sequence[float], k: int) -> list[float]:
    """Lloyd's algorithm in 1-D with deterministic quantile init.

    Returns the ``k`` cluster centers sorted ascending. ``values`` is
    not modified. Raises if ``k <= 0`` or ``len(values) < k`` — callers
    are expected to gate on a minimum sample count.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")
    n = len(values)
    if n < k:
        raise ValueError(f"need at least k={k} samples for k-means; got {n}")

    sorted_vals = sorted(values)

    # Quantile init: center i sits at the (i+0.5)/k quantile. Avoids
    # the random-init non-determinism that would otherwise produce
    # slightly different anchors run-to-run.
    centers: list[float] = []
    for i in range(k):
        q = (i + 0.5) / k
        idx = min(n - 1, int(q * n))
        centers.append(sorted_vals[idx])

    # Make sure the init has k distinct centers — duplicate quantile
    # picks happen on tiny / heavily-skewed inputs. Falling back to
    # evenly-spaced indices preserves the "k distinct seeds" invariant
    # that Lloyd's algorithm needs to avoid empty clusters.
    if len(set(centers)) < k:
        step = max(1, n // k)
        centers = [sorted_vals[min(n - 1, i * step)] for i in range(k)]
        if len(set(centers)) < k:
            # Pathological tie-heavy input — perturb so all k centers
            # remain distinct without changing the rounded result.
            centers = [sorted_vals[min(n - 1, i)] + 1e-9 * i for i in range(k)]

    for _ in range(_MAX_ITERS):
        # Assignment step: assign each value to its nearest center.
        # 1-D + sorted centers means we can do this via boundary
        # midpoints, but a per-value loop is plenty fast for the sample
        # sizes we work on (≤ 10^6 boxes).
        clusters: list[list[float]] = [[] for _ in range(k)]
        for v in values:
            best_i = 0
            best_d = abs(v - centers[0])
            for i in range(1, k):
                d = abs(v - centers[i])
                if d < best_d:
                    best_d = d
                    best_i = i
            clusters[best_i].append(v)

        # Update step.
        new_centers: list[float] = []
        for i in range(k):
            if clusters[i]:
                new_centers.append(sum(clusters[i]) / len(clusters[i]))
            else:
                # Empty cluster: keep the previous center. Won't move
                # again unless something gets reassigned to it on a
                # later iteration.
                new_centers.append(centers[i])

        shift = max(abs(a - b) for a, b in zip(centers, new_centers, strict=True))
        centers = new_centers
        if shift < _CONVERGENCE_EPS:
            break

    return sorted(centers)


def cluster_sizes(sqrt_areas: Sequence[float], k: int = 5) -> tuple[int, ...]:
    """Cluster GT box √area into ``k`` anchor scales, sorted ascending.

    The Mayaku / D2 FPN convention is one anchor scale per FPN level
    (P3→P7), so the default ``k=5`` returns the ladder used by
    :class:`mayaku.config.AnchorGeneratorConfig`. Output is rounded to
    ints — the schema declares ``sizes: tuple[tuple[int, ...], ...]``.
    """
    centers = _kmeans_1d(sqrt_areas, k=k)
    # Round to int + dedupe-with-min-spacing so two adjacent FPN levels
    # never get an identical anchor scale (the anchor generator would
    # silently produce overlapping anchors). Spacing of 1 pixel is the
    # smallest distinguishable step at the schema's int resolution.
    out: list[int] = []
    for c in centers:
        rounded = max(1, round(c))
        if out and rounded <= out[-1]:
            rounded = out[-1] + 1
        out.append(rounded)
    return tuple(out)


def cluster_aspect_ratios(aspect_ratios: Sequence[float], k: int = 3) -> tuple[float, ...]:
    """Cluster GT box w/h into ``k`` anchor aspect ratios, sorted ascending.

    Output is rounded to 2 decimals — anchor ARs don't need more
    precision than that, and round numbers make the resolved YAML
    easier to read and audit.
    """
    centers = _kmeans_1d(aspect_ratios, k=k)
    # Clamp to a sane physical range: rotated-box behaviour out of scope
    # (ADR 001), so an AR < 0.1 or > 10 is almost always a labeling bug
    # rather than a real shape we want anchors for.
    clamped = [max(0.1, min(10.0, c)) for c in centers]
    rounded = sorted({round(c, 2) for c in clamped})
    # Re-pad if dedupe collapsed two centers — unlikely with k=3 but
    # cheap insurance. Spread the duplicates by 0.01 to keep AR ordering
    # strict.
    while len(rounded) < k:
        rounded.append(round(rounded[-1] + 0.01, 2))
    return tuple(rounded[:k])
