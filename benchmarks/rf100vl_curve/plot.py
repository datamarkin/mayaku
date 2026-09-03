"""Aggregate per-dataset RF100-VL curves into one mean curve per library.

    python plot.py [--results results] [--intersection] [--points 100] [--band]

INPUT
    <results>/<lib>/<variant>/<dataset>/curve.csv, with columns ``checkpoint``,
    ``wall_clock_s`` and the pycocotools stats; ``ap`` (AP@[.50:.95]) is the only
    stat read.

METHOD
    Datasets are aligned by training progress rather than by wall-clock.

    1. Each dataset's checkpoint list is prepended with the origin
       (wall_clock_s 0, ap 0): before any training step the detection head has
       not been fitted to the dataset's classes, so its AP is 0.
    2. The list is sampled at ``--points`` + 1 evenly spaced fractions of its
       checkpoint index, so index 0 is the origin and index ``--points`` is the
       dataset's last checkpoint. A dataset with exactly ``--points``
       checkpoints lands on each of them and is not interpolated; a dataset with
       fewer is linearly interpolated onto the same index axis.
    3. Point j of a library's curve is the mean across its datasets of the
       wall-clock at index j (x) and of the AP at index j (y).

    Every point therefore averages the same set of datasets, and the last point
    is (mean total training time, mean final-checkpoint AP). Between consecutive
    indices the curve is a straight line through values measured at real
    checkpoints; the span from the origin to a dataset's first checkpoint is
    interpolated for every library.

    Libraries differ in how many datasets they have finished and in how many
    checkpoints they write per run. ``--intersection`` restricts every library
    to the datasets all of them have, so the means cover identical data. The
    figure states the dataset count once beneath the title, or on each legend
    entry when libraries cover different numbers; per-library checkpoint counts
    are in summary.csv.

OUTPUT  (suffixed ``_common`` with --intersection)
    <results>/curve_<variant>.png    mean AP vs mean wall-clock in minutes, one line
                                     per library (the CSVs stay in seconds)
    <results>/curve_points.csv       every index: mean wall-clock, mean AP, n
    <results>/summary.csv            per (variant, library): dataset and checkpoint
                                     counts, mean/median total time, and mean/median
                                     AP at the final and at the best checkpoint
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


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (wall_clock_s, ap) sorted by time, or None if unusable."""
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    t = np.array([float(r["wall_clock_s"]) for r in rows])
    ap = np.array([float(r["ap"]) for r in rows])
    order = np.argsort(t)
    t, ap = t[order], ap[order]
    if t[-1] <= 0:
        return None
    return t, ap


def _load(variant_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {dataset: (times, aps)} for the curves under a <lib>/<variant> dir."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(variant_dir.glob("*/curve.csv")):
        curve = _read_curve(path)
        if curve is not None:
            out[path.parent.name] = curve
    return out


def _resample(t: np.ndarray, ap: np.ndarray, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample one dataset at ``points`` + 1 fractions of its checkpoint index.

    The origin (0, 0) is prepended, so index 0 is wall-clock 0 / AP 0 and index
    ``points`` is the last checkpoint. A dataset with exactly ``points``
    checkpoints lands on each of them, so np.interp returns them unchanged.
    """
    t = np.concatenate(([0.0], t))
    ap = np.concatenate(([0.0], ap))
    idx = np.arange(t.size, dtype=float)
    want = np.linspace(0.0, t.size - 1, points + 1)
    return np.interp(want, idx, t), np.interp(want, idx, ap)


def _mean_curve(curves, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean wall-clock, mean AP, and the AP's standard error at each index."""
    ts, aps = zip(*(_resample(t, ap, points) for t, ap in curves))
    aps = np.array(aps)
    sem = (aps.std(axis=0, ddof=1) / np.sqrt(len(aps)) if len(aps) > 1
           else np.zeros(points + 1))
    return np.mean(ts, axis=0), aps.mean(axis=0), sem


def _plot(plt, curves: dict, labels: dict, band: dict | None, subtitle: str,
          best: dict | None):
    """Mean AP against mean wall-clock, one line per library.

    ``best`` maps a library to (mean wall-clock of each dataset's own best
    checkpoint, mean of those best APs). That point is built the same way as
    every point on the line — mean x, mean y across datasets — but indexed by
    each dataset's own argmax instead of a fixed progress fraction, so it sits
    off the line: the mean of per-dataset maxima always exceeds the maximum of
    the mean curve, because datasets peak at different times.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ends = []
    for i, lib in enumerate(sorted(curves)):
        seconds, y = curves[lib]
        x = seconds / 60.0  # curves are computed in seconds; the axis reads in minutes
        color = f"C{i}"
        ax.plot(x, y, color=color, lw=2, label=labels[lib])
        if band is not None:
            ax.fill_between(x, y - band[lib], y + band[lib], color=color, alpha=0.15, lw=0)
        ax.plot(x[-1], y[-1], "o", color=color, ms=6)
        note = f"final: {y[-1]:.3f}"  # the endpoint dot already marks when
        if best is not None:
            bt, bap = best[lib]
            ax.plot(bt / 60.0, bap, "*", color=color, ms=11,
                    markerfacecolor="none", markeredgewidth=1.4)
            note += f"\nbest: {bap:.3f}"  # the star already marks where
        ends.append((x[-1], y[-1], note, color))

    xmax = max(e[0] for e in ends)
    ax.set_xlim(0, xmax * 1.20)   # room for the endpoint labels, which sit to the right
    ax.set_ylim(bottom=0)
    ax.set_xlabel("mean training wall-clock across datasets (min)")
    ax.set_ylabel("mean COCO AP @[.50:.95] across datasets")

    # Endpoint labels overlap whenever two libraries finish at a similar AP, which is
    # the normal case. Walk them from the highest down and push any that would collide
    # far enough apart to stay legible, with a leader line back to the real point.
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * (0.075 if best is not None else 0.045)
    dx = ax.get_xlim()[1] * 0.015
    placed = None
    for x_end, y_end, note, color in sorted(ends, key=lambda e: -e[1]):
        y_lab = y_end if placed is None else min(y_end, placed - gap)
        placed = y_lab
        ax.annotate(note, xy=(x_end, y_end), xytext=(x_end + dx, y_lab),
                    textcoords="data", va="center", color=color,
                    fontweight="bold", fontsize=9,
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=color, alpha=0.5,
                                    shrinkA=0, shrinkB=3))

    # Line 1 states what is plotted; line 2 carries scope and method in smaller type.
    ax.set_title("RF100-VL: mean AP vs mean wall-clock", fontsize=12, pad=18)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="0.35")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--points", type=int, default=100,
                   help="checkpoint indices per dataset; a dataset with exactly this "
                        "many checkpoints is used without interpolation (default: 100)")
    p.add_argument("--intersection", action="store_true",
                   help="per variant, keep only datasets present in every library, so "
                        "each library's mean covers the same datasets; writes *_common")
    p.add_argument("--no-best", action="store_true",
                   help="omit the best-checkpoint marker (a star at the mean wall-clock "
                        "and mean AP of each dataset's own best checkpoint)")
    p.add_argument("--band", action="store_true",
                   help="shade +/- 1 standard error of the mean AP across datasets")
    args = p.parse_args()

    results = Path(args.results)
    # loaded[variant][lib] = {dataset: (times, aps)} — every curve, read once.
    loaded: dict[str, dict[str, dict]] = defaultdict(dict)
    for lib_dir in sorted(d for d in results.iterdir() if d.is_dir()):
        for variant_dir in sorted(d for d in lib_dir.iterdir() if d.is_dir()):
            found = _load(variant_dir)
            if found:
                loaded[variant_dir.name][lib_dir.name] = found
    present = {v: {lib: set(cs) for lib, cs in per_lib.items()}
               for v, per_lib in loaded.items()}
    if not present:
        raise SystemExit(f"no <lib>/<variant>/<dataset>/curve.csv found under {results}")

    # Per variant, the dataset names to keep. None means "every dataset this
    # library has"; --intersection narrows it to the ones every library has.
    keep: dict[str, set | None] = {
        variant: (set.intersection(*per_lib.values()) if args.intersection else None)
        for variant, per_lib in present.items()
    }

    suffix = "_common" if args.intersection else ""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
        print("matplotlib not installed — skipping figures (CSVs still written)")

    point_rows, summary_rows = [], []
    for variant in sorted(present, key=_variant_key):
        curves, bands, counts, bests = {}, {}, {}, {}
        names = keep[variant]
        for lib, by_dataset in sorted(loaded[variant].items()):
            values = [c for n, c in by_dataset.items() if names is None or n in names]
            if not values:
                continue
            x, y, bands[lib] = _mean_curve(values, args.points)
            curves[lib] = (x, y)
            counts[lib] = len(values)

            ckpts = np.array([t.size for t, _ in values])
            span = f"{ckpts.min()}" if ckpts.min() == ckpts.max() else f"{ckpts.min()}-{ckpts.max()}"

            totals = np.array([t[-1] for t, _ in values])
            finals = np.array([ap[-1] for _, ap in values])
            # Each dataset's own best checkpoint — what a library that ships its
            # best weights (Ultralytics best.pt, RF-DETR checkpoint_best) delivers.
            bestap = np.array([ap.max() for _, ap in values])
            bestt = np.array([t[ap.argmax()] for t, ap in values])
            bests[lib] = (float(bestt.mean()), float(bestap.mean()))
            summary_rows.append([
                variant, lib, len(values), span,
                f"{totals.mean():.1f}", f"{np.median(totals):.1f}",
                f"{finals.mean():.4f}", f"{np.median(finals):.4f}",
                f"{bestap.mean():.4f}", f"{np.median(bestap):.4f}", f"{bestt.mean():.1f}",
            ])
            for j, (xj, yj) in enumerate(zip(x, y)):
                point_rows.append([variant, lib, j, f"{xj:.1f}", f"{yj:.4f}", len(values)])

        # Libraries usually cover the same datasets (always so under --intersection),
        # so state the count once in a subtitle and leave the legend as bare names.
        # While a benchmark is still filling in, the counts differ and each line has to
        # carry its own. Per-library checkpoint counts live in summary.csv.
        scope = "shared datasets" if args.intersection else "datasets"
        if len(set(counts.values())) == 1:
            subtitle = f"{next(iter(counts.values()))} {scope} \u00b7 aligned by training progress"
            labels = {lib: f"{lib}_{variant}" for lib in counts}
        else:
            subtitle = "aligned by training progress"
            labels = {lib: f"{lib}_{variant}  ({counts[lib]} {scope})" for lib in counts}

        if curves and plt is not None:
            fig = _plot(plt, curves, labels, bands if args.band else None, subtitle,
                        None if args.no_best else bests)
            out = results / f"curve_{variant}{suffix}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"wrote {out}")

    points_path = results / f"curve_points{suffix}.csv"
    with points_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "library", "checkpoint_index", "mean_wall_clock_s",
                    "mean_ap", "n_datasets"])
        w.writerows(point_rows)
    print(f"wrote {points_path}")

    summary_path = results / f"summary{suffix}.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "library", "n_datasets", "checkpoints_per_dataset",
                    "mean_total_s", "median_total_s", "mean_final_ap", "median_final_ap",
                    "mean_best_ap", "median_best_ap", "mean_best_s"])
        w.writerows(summary_rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
