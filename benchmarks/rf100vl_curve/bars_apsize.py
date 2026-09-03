"""Compare libraries on size-stratified COCO AP (small / medium / large).

    python bars_apsize.py [--results results] [--intersection] [--final]

INPUT
    <results>/<lib>/<variant>/<dataset>/curve.csv, reading ``ap_small``,
    ``ap_medium`` and ``ap_large`` from one row per dataset: by default the row
    with that dataset's highest overall ``ap`` — the checkpoint a library would
    ship, and the one plot.py's star marks — or its last checkpoint with
    ``--final``. The size-class values are read off that one checkpoint; they
    are not each maximised separately, which would be a different number.

METHOD
    pycocotools writes -1.0 for a size class a dataset's val split has no
    ground-truth objects of, so those datasets carry no information for that
    class and are dropped from it. Each class therefore has its own dataset
    count, printed under its tick label; a dataset dropped from ``small`` still
    counts toward ``medium`` and ``large``.

    A class keeps only datasets where EVERY library has a usable value, so the
    bars within a class are means over identical data. ``--intersection``
    narrows the pool further to datasets every library ran at all. Error bars
    are +/- 1 standard error of the mean across datasets.

OUTPUT  (suffixed ``_common`` with --intersection, ``_final`` with --final)
    <results>/apsize_<variant>.png   grouped bars, one group per size class
    <results>/apsize.csv             per (variant, library, size class): n, mean, sem
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANT_ORDER = ("nano", "small", "medium", "large")
SIZES = (("ap_small", "small"), ("ap_medium", "medium"), ("ap_large", "large"))


def _variant_key(name: str) -> tuple[int, str]:
    return (VARIANT_ORDER.index(name) if name in VARIANT_ORDER else len(VARIANT_ORDER), name)


def _pick_row(path: Path, final: bool) -> dict | None:
    """The dataset's best row by overall ``ap``, or its last row by wall-clock."""
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    if final:
        return max(rows, key=lambda r: float(r["wall_clock_s"]))
    return max(rows, key=lambda r: float(r["ap"]))


def _load(variant_dir: Path, final: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(variant_dir.glob("*/curve.csv")):
        row = _pick_row(path, final)
        if row is not None:
            out[path.parent.name] = row
    return out


def _plot(plt, variant: str, libs: list[str], stats: dict, counts: dict, subtitle: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.8 / len(libs)
    centres = np.arange(len(SIZES))
    for i, lib in enumerate(libs):
        offset = (i - (len(libs) - 1) / 2) * width
        means = [stats[(lib, key)][0] for key, _ in SIZES]
        sems = [stats[(lib, key)][1] for key, _ in SIZES]
        bars = ax.bar(centres + offset, means, width * 0.9, yerr=sems, capsize=3,
                      color=f"C{i}", label=f"{lib}_{variant}",
                      error_kw=dict(lw=1, ecolor="0.3"))
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(centres)
    ax.set_xticklabels([f"{label}\n(n={counts[key]})" for key, label in SIZES])
    ax.set_xlabel("object size class")
    ax.set_ylabel("mean COCO AP @[.50:.95] across datasets")
    ax.set_title("RF100-VL: AP by object size", fontsize=12, pad=18)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="0.35")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--intersection", action="store_true",
                   help="keep only datasets present in every library; writes *_common")
    p.add_argument("--final", action="store_true",
                   help="read each dataset's last checkpoint instead of its best-ap one")
    args = p.parse_args()

    results = Path(args.results)
    loaded: dict[str, dict[str, dict]] = defaultdict(dict)
    for lib_dir in sorted(d for d in results.iterdir() if d.is_dir()):
        for variant_dir in sorted(d for d in lib_dir.iterdir() if d.is_dir()):
            found = _load(variant_dir, args.final)
            if found:
                loaded[variant_dir.name][lib_dir.name] = found
    if not loaded:
        raise SystemExit(f"no <lib>/<variant>/<dataset>/curve.csv found under {results}")

    suffix = ("_common" if args.intersection else "") + ("_final" if args.final else "")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
        print("matplotlib not installed — skipping figures (CSV still written)")

    rows = []
    for variant in sorted(loaded, key=_variant_key):
        per_lib = loaded[variant]
        libs = sorted(per_lib)
        pool = set.intersection(*(set(v) for v in per_lib.values())) if args.intersection \
            else set().union(*(set(v) for v in per_lib.values()))

        stats, counts = {}, {}
        for key, _ in SIZES:
            # A dataset counts for this class only where every library scored it.
            usable = sorted(d for d in pool
                            if all(d in per_lib[l] and float(per_lib[l][d][key]) >= 0
                                   for l in libs))
            counts[key] = len(usable)
            for lib in libs:
                vals = np.array([float(per_lib[lib][d][key]) for d in usable])
                sem = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
                stats[(lib, key)] = (vals.mean() if vals.size else np.nan, sem)
                rows.append([variant, lib, key, len(usable),
                             f"{vals.mean():.4f}" if vals.size else "",
                             f"{sem:.4f}" if vals.size else ""])

        if plt is None or not libs:
            continue
        scope = "shared datasets" if args.intersection else "datasets"
        ckpt = "final checkpoint" if args.final else "best checkpoint (highest overall AP)"
        subtitle = f"{ckpt} · per-class {scope}, sized classes with no ground truth dropped"
        fig = _plot(plt, variant, libs, stats, counts, subtitle)
        out = results / f"apsize_{variant}{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")

    path = results / f"apsize{suffix}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "library", "size_class", "n_datasets", "mean_ap", "sem"])
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
