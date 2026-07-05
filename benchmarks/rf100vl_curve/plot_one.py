"""Plot ONE dataset's AP-vs-wall-clock curve for all libraries on the same axes.

A fair same-dataset comparison (unlike the aggregate mean in plot.py). Reads
<results>/<lib>/<dataset>/curve.csv for each library present and draws one line
each, with the best-AP point marked.

    python plot_one.py --dataset floating-waste [--results results_medium] [--metric ap]
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIBS = ["mayaku", "yolo", "rfdetr"]
COLORS = {"mayaku": "#2563eb", "yolo": "#e0730a", "rfdetr": "#16a34a"}
LABELS = {"mayaku": "Mayaku-N", "yolo": "YOLO26-N", "rfdetr": "RF-DETR-N"}


def load(path: Path, metric: str):
    rows = list(csv.DictReader(open(path)))
    t = [float(r["wall_clock_s"]) for r in rows]
    y = [float(r[metric]) for r in rows]
    return t, y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--results", default="results_medium")
    p.add_argument("--metric", default="ap", choices=["ap", "ap50"])
    p.add_argument("--out", default="")
    a = p.parse_args()

    res = Path(a.results)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = 0
    for lib in LIBS:
        c = res / lib / a.dataset / "curve.csv"
        if not c.exists():
            print(f"[skip] {lib}: no curve.csv"); continue
        t, y = load(c, a.metric)
        ax.plot(t, y, "-o", ms=3, lw=1.8, color=COLORS[lib], label=LABELS[lib])
        bi = max(range(len(y)), key=lambda i: y[i])
        ax.scatter([t[bi]], [y[bi]], s=90, color=COLORS[lib], zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"{y[bi]:.3f}", (t[bi], y[bi]), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color=COLORS[lib], fontweight="bold")
        plotted += 1

    if not plotted:
        raise SystemExit(f"no curves found for {a.dataset} under {res}")

    ax.set_xscale("log")
    metric_name = "COCO AP@[.50:.95]" if a.metric == "ap" else "COCO AP@.50"
    ax.set_xlabel("training wall-clock (s, log scale)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{a.dataset} — {metric_name} vs wall-clock (defaults)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = a.out or str(res / f"curve_{a.dataset}_{a.metric}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
