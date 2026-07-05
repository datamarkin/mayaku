"""Plot arbitrary AP-vs-wall-clock curves on one axis from explicit label:path pairs.

    python plot_series.py --metric ap --title "floating-waste" --out fw.png \
        --series "YOLO26:results_medium/yolo/floating-waste/curve.csv" \
        --series "Mayaku (base):results_medium/mayaku/floating-waste/curve.csv" \
        --series "Mayaku+DFL:results_exp/dfl/floating-waste/curve.csv"
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#e0730a", "#2563eb", "#16a34a", "#9333ea", "#dc2626"]


def load(path: str, metric: str):
    rows = list(csv.DictReader(open(path)))
    return [float(r["wall_clock_s"]) for r in rows], [float(r[metric]) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", action="append", required=True, help="label:path")
    p.add_argument("--metric", default="ap", choices=["ap", "ap50"])
    p.add_argument("--title", default="")
    p.add_argument("--xscale", default="log", choices=["log", "linear"])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, s in enumerate(a.series):
        label, path = s.split(":", 1)
        if not Path(path).exists():
            print(f"[skip] {label}: {path} missing"); continue
        t, y = load(path, a.metric)
        if not y:
            print(f"[skip] {label}: {path} has no rows"); continue
        c = COLORS[i % len(COLORS)]
        ax.plot(t, y, "-o", ms=3, lw=1.8, color=c, label=label)
        bi = max(range(len(y)), key=lambda j: y[j])
        ax.scatter([t[bi]], [y[bi]], s=90, color=c, zorder=5, edgecolor="white", linewidth=1.2)
        ax.annotate(f"{y[bi]:.3f}", (t[bi], y[bi]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color=c, fontweight="bold")
    ax.set_xscale(a.xscale)
    mname = "COCO AP@[.50:.95]" if a.metric == "ap" else "COCO AP@.50"
    ax.set_xlabel(f"training wall-clock (s{', log' if a.xscale == 'log' else ''})")
    ax.set_ylabel(mname)
    ax.set_title(f"{a.title} — {mname} vs wall-clock")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
