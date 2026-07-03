"""Aggregate per-dataset curves into the AP-vs-wall-clock comparison.

    python plot.py [--results results] [--budgets 60,300,900,1800]

Reads ``<results>/<lib>/<dataset>/curve.csv`` for every library, interpolates
each dataset's curve onto a shared time grid, averages across datasets (only
datasets that have actually reached a given time contribute to it), and writes:

  * ``<results>/curve.png``     — aggregate mean AP vs wall-clock, one line per library
  * ``<results>/summary.csv``   — AP at each time budget, per library

A single seed per dataset is fine here: the aggregate is a mean over ~100
datasets, which is where the variance averages out.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_lib(lib_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(times, aps), ...] for each dataset curve under a library dir."""
    curves = []
    for curve in sorted(lib_dir.glob("*/curve.csv")):
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
    libs = {d.name: _load_lib(d) for d in sorted(results.iterdir()) if d.is_dir() and any(d.glob("*/curve.csv"))}
    if not libs:
        raise SystemExit(f"no <lib>/<dataset>/curve.csv found under {results}")

    # Shared log-spaced grid spanning all observed times.
    all_t = np.concatenate([t for cs in libs.values() for t, _ in cs])
    grid = np.geomspace(max(all_t.min(), 1.0), all_t.max(), 200)
    agg = {lib: _aggregate(cs, grid) for lib, cs in libs.items()}

    # summary.csv — AP at each budget.
    budgets = [float(b) for b in args.budgets.split(",")]
    with (results / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["library", "n_datasets", *[f"AP@{int(b)}s" for b in budgets]])
        for lib, cs in libs.items():
            aps = [f"{np.interp(b, grid, agg[lib]):.4f}" for b in budgets]
            w.writerow([lib, len(cs), *aps])
    print(f"wrote {results / 'summary.csv'}")

    # curve.png — aggregate lines.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed — skipping curve.png (summary.csv written)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for lib in libs:
        ax.plot(grid, agg[lib], label=f"{lib} (n={len(libs[lib])})", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock (s, log)")
    ax.set_ylabel("mean COCO AP @[.50:.95]")
    ax.set_title("RF100-VL: AP vs training wall-clock")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "curve.png", dpi=150)
    print(f"wrote {results / 'curve.png'}")


if __name__ == "__main__":
    main()
