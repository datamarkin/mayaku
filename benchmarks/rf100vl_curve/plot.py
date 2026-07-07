"""Aggregate per-dataset curves into per-variant AP-vs-wall-clock comparisons.

    python plot.py [--results results] [--budgets 60,300,900,1800]

Reads ``<results>/<lib>/<variant>/<dataset>/curve.csv`` for every library and
variant, interpolates each dataset's curve onto a shared per-variant time grid,
averages across datasets (only datasets that have actually reached a given time
contribute to it), and writes — one comparison per size class:

  * ``<results>/curve_<variant>.png`` — mean AP vs wall-clock, one line per library
  * ``<results>/summary.csv``         — AP at each time budget, per (variant, library)

Each size class gets its own figure and its own time grid, because nano and large
live on very different wall-clock scales. A single seed per dataset is fine here:
the aggregate is a mean over ~100 datasets, where the variance averages out.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# Draw variants in size order, not alphabetical; unknown names sort after, by name.
VARIANT_ORDER = ("nano", "small", "medium", "large")


def _variant_key(name: str) -> tuple[int, str]:
    return (VARIANT_ORDER.index(name) if name in VARIANT_ORDER else len(VARIANT_ORDER), name)


def _load_datasets(variant_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(times, aps), ...] for each dataset curve under a <lib>/<variant> dir."""
    curves = []
    for curve in sorted(variant_dir.glob("*/curve.csv")):
        rows = list(csv.DictReader(curve.open()))
        if not rows:
            continue
        t = np.array([float(r["wall_clock_s"]) for r in rows])
        ap = np.array([float(r["ap"]) for r in rows])
        order = np.argsort(t)
        curves.append((t[order], ap[order]))
    return curves


def _aggregate(curves, grid: np.ndarray) -> np.ndarray:
    """Mean AP across datasets on ``grid``; NaN outside a dataset's own time range."""
    stacked = np.full((len(curves), grid.size), np.nan)
    for i, (t, ap) in enumerate(curves):
        vals = np.interp(grid, t, ap)
        vals[(grid < t[0]) | (grid > t[-1])] = np.nan  # no extrapolation
        stacked[i] = vals
    return np.nanmean(stacked, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--budgets", default="60,300,900,1800", help="comma-separated seconds")
    args = p.parse_args()

    results = Path(args.results)
    # series[variant][lib] = [(times, aps), ...]
    series: dict[str, dict[str, list]] = defaultdict(dict)
    for lib_dir in sorted(results.iterdir()):
        if not lib_dir.is_dir():
            continue
        for variant_dir in sorted(lib_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            cs = _load_datasets(variant_dir)
            if cs:
                series[variant_dir.name][lib_dir.name] = cs
    if not series:
        raise SystemExit(f"no <lib>/<variant>/<dataset>/curve.csv found under {results}")

    budgets = [float(b) for b in args.budgets.split(",")]
    variants = sorted(series, key=_variant_key)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
        print("matplotlib not installed — skipping figures (summary.csv still written)")

    summary_rows = []
    for variant in variants:
        libs = series[variant]
        # Per-variant shared log-spaced grid spanning all observed times in this class.
        all_t = np.concatenate([t for cs in libs.values() for t, _ in cs])
        grid = np.geomspace(max(all_t.min(), 1.0), all_t.max(), 200)
        agg = {lib: _aggregate(cs, grid) for lib, cs in libs.items()}

        for lib, cs in libs.items():
            aps = [f"{np.interp(b, grid, agg[lib]):.4f}" for b in budgets]
            summary_rows.append([variant, lib, len(cs), *aps])

        if plt is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for lib in sorted(libs):
            ax.plot(grid, agg[lib], label=f"{lib} (n={len(libs[lib])})", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("wall-clock (s, log)")
        ax.set_ylabel("mean COCO AP @[.50:.95]")
        ax.set_title(f"RF100-VL {variant}: AP vs training wall-clock")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = results / f"curve_{variant}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    with (results / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "library", "n_datasets", *[f"AP@{int(b)}s" for b in budgets]])
        w.writerows(summary_rows)
    print(f"wrote {results / 'summary.csv'}")


if __name__ == "__main__":
    main()
