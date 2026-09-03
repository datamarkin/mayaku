"""Sequential driver for the RF100-VL AP-vs-wall-clock benchmark — start it and forget.

Runs the configured legs back-to-back over every (variant, dataset), one unit at a
time, on a single machine. Sequential is the point: no two legs ever share the GPU,
so the wall-clock x-axis stays clean.

Each unit is an isolated subprocess: the stock `run_<leg>.py` is invoked on a
one-dataset *view* of the root — a temp dir holding a single symlink to the real
dataset — so `run_<leg>.py` is used **exactly as-is**, unpatched. A crash (OOM on a
large variant, a corrupt image, a library bug) sinks only that one unit, not the week.
Everything is resumable: a unit whose `curve.csv` already exists is skipped, so Ctrl-C,
a reboot, or re-running the whole command all just continue where it left off.

Storage stays bounded independent of each leg's own purge: after every unit this driver
strips all checkpoint files (`weights/`, `*.pt`, `*.ckpt`, `*.pth`) from the run dir,
keeping only `meta.json` + `curve.csv` — the durable artifacts. All 12 COCO stats are
already written to `curve.csv` (`common.COCO_STATS`: 6 AP + 6 AR), so there is nothing
to add on the metric side.

    python run_all.py --datasets <coco_root>                        # default legs, all variants
    python run_all.py --datasets <root> --legs mayaku,yolo,rfdetr   # all three, in that order
    python run_all.py --datasets <root> --variants nano,small
    nohup python run_all.py --datasets <root> > results/logs/driver.out 2>&1 &

Per-unit child output goes to `results/logs/<leg>/<variant>/<dataset>.log`; a one-line
status per unit is appended to `results/logs/ledger.csv` and printed here. Pin the GPU
with `CUDA_VISIBLE_DEVICES` (and `CUDA_DEVICE_ORDER=PCI_BUS_ID`) in the environment —
it is inherited by every child.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # so `import common` works regardless of cwd
import common  # noqa: E402

# Legs this driver knows how to run. Each is a self-contained `run_<leg>.py` in this
# directory sharing the same CLI (--datasets / --out / --variants), so the driver treats
# them uniformly. YOLO first by default: it is the fast leg, so its full curve lands early.
LEGS = ("yolo", "rfdetr", "mayaku")
CKPT_GLOBS = ("*.pt", "*.ckpt", "*.pth")  # checkpoint file shapes across the legs


def purge_checkpoints(run_dir: Path) -> None:
    """Strip every checkpoint from a run dir, keeping only meta.json + curve.csv.

    Runs after *every* unit, so storage stays bounded even if a leg's own purge misses a
    file shape (e.g. an RF-DETR `checkpoint_best_*.pth` its `*.ckpt` glob wouldn't catch)
    or the unit crashed before purging. Only checkpoint files are removed; the durable
    `.csv`/`.json` artifacts are untouched.
    """
    if not run_dir.exists():
        return
    import shutil

    shutil.rmtree(run_dir / "weights", ignore_errors=True)  # Ultralytics' epoch*.pt live here
    for pat in CKPT_GLOBS:
        for f in run_dir.glob(pat):
            f.unlink(missing_ok=True)


def curve_rows(run_dir: Path) -> int | None:
    """Data-row count of the unit's curve.csv (None if absent). 0 = wrote a header but
    scored no checkpoints — worth a warning on an unattended run."""
    f = run_dir / "curve.csv"
    if not f.exists():
        return None
    with f.open() as fh:
        return max(0, sum(1 for _ in fh) - 1)


def run_unit(leg: str, variant: str, name: str, real_dir: Path, out: Path, device: str,
             log: Path, timeout: float) -> int:
    """Invoke `run_<leg>.py` on a one-dataset view of the root; return the child exit code.

    The one-dataset view is a fresh temp dir holding a single symlink `name -> real_dir`,
    passed as `--datasets`. `run_<leg>.py` then discovers exactly this one dataset and does
    its own train -> score -> purge, writing to `out/<leg>/<variant>/<name>/` as always.
    """
    tmp_parent = out / ".driver"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{leg}_{variant}_", dir=tmp_parent))
    (tmp / name).symlink_to(real_dir.resolve(), target_is_directory=True)
    cmd = [sys.executable, str(SCRIPT_DIR / f"run_{leg}.py"),
           "--datasets", str(tmp), "--out", str(out / leg), "--variants", variant]
    if device:
        cmd += ["--device", device]
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("a") as fh:
            fh.write(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}\n")
            fh.flush()
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=(timeout or None)).returncode
    except subprocess.TimeoutExpired:
        return 124  # conventional timeout code; treated as a failed attempt
    finally:
        (tmp / name).unlink(missing_ok=True)  # unlink the symlink, never its target
        tmp.rmdir()


def ledger_append(ledger: Path, row: dict) -> None:
    fields = ["leg", "variant", "dataset", "status", "attempts", "seconds", "rows"]
    new = not ledger.exists()
    with ledger.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", required=True, help="COCO dataset root (one subdir per dataset)")
    p.add_argument("--out", default="results", help="results root; legs write to <out>/<leg>")
    p.add_argument("--legs", default="yolo,rfdetr", help=f"comma-separated, in run order; from {list(LEGS)}")
    p.add_argument("--variants", default="nano,small,medium,large", help="comma-separated size subset")
    p.add_argument("--device", default="", help="passed to each leg's --device (e.g. 0, cuda, auto)")
    p.add_argument("--retries", type=int, default=1, help="extra attempts after a failed unit (default 1)")
    p.add_argument("--unit-timeout", type=float, default=0, help="seconds per attempt, 0 = no limit")
    p.add_argument("--skip", default="", help="comma-separated leg/variant/dataset tags to leave "
                                              "un-run (e.g. known-OOM units); they are reported and "
                                              "never attempted")
    args = p.parse_args()

    root = Path(args.datasets)
    out = Path(args.out)
    legs = [x.strip() for x in args.legs.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if bad := [x for x in legs if x not in LEGS]:
        raise SystemExit(f"unknown legs {bad}; choose from {list(LEGS)}")

    datasets = list(common.iter_datasets(root))  # scan the tree once
    if not datasets:
        raise SystemExit(f"no datasets with a train/{common.COCO_ANN} split under {root}")
    log_dir = out / "logs"
    ledger = log_dir / "ledger.csv"
    units = [(leg, v, name, d) for leg in legs for v in variants for name, d in datasets]
    print(f"[all] {len(units)} units = {len(legs)} legs x {len(variants)} variants x "
          f"{len(datasets)} datasets  ->  {out}")

    skip_tags = {x.strip() for x in args.skip.split(",") if x.strip()}
    if skip_tags:
        print(f"[all] excluding {len(skip_tags)} unit(s) by --skip: {', '.join(sorted(skip_tags))}")

    tallies = {"done": 0, "skip": 0, "fail": 0, "excluded": 0}
    for i, (leg, variant, name, real_dir) in enumerate(units, 1):
        run_dir = out / leg / variant / name
        tag = f"{leg}/{variant}/{name}"
        if tag in skip_tags:
            print(f"[all] ({i}/{len(units)}) excluded {tag} — listed in --skip")
            tallies["excluded"] += 1
            continue
        if (rows := curve_rows(run_dir)) is not None:  # resumable: leg already wrote a curve
            if rows == 0:
                print(f"[all] ({i}/{len(units)}) skip {tag} — curve.csv has 0 rows (scored no checkpoints!)")
            else:
                print(f"[all] ({i}/{len(units)}) skip {tag} — done ({rows} points)")
            tallies["skip"] += 1
            continue

        log = log_dir / leg / variant / f"{name}.log"
        t0 = time.time()
        code, attempt = 1, 0
        for attempt in range(1, args.retries + 2):
            print(f"[all] ({i}/{len(units)}) run  {tag}  attempt {attempt}  (log: {log})", flush=True)
            try:
                code = run_unit(leg, variant, name, real_dir, out, args.device, log, args.unit_timeout)
            except KeyboardInterrupt:
                print("\n[all] interrupted — current unit stopped; re-run the same command to resume.")
                purge_checkpoints(run_dir)  # don't leave a half-written checkpoint trail behind
                raise SystemExit(130)
            if code == 0 and curve_rows(run_dir):
                break

        purge_checkpoints(run_dir)  # bound storage: keep only meta.json + curve.csv
        rows = curve_rows(run_dir)
        ok = code == 0 and bool(rows)
        status = "done" if ok else "fail"
        tallies["done" if ok else "fail"] += 1
        ledger_append(ledger, {"leg": leg, "variant": variant, "dataset": name, "status": status,
                               "attempts": attempt, "seconds": round(time.time() - t0, 1),
                               "rows": rows if rows is not None else 0})
        mark = "done" if ok else f"FAIL (exit {code}) — logged, moving on"
        print(f"[all] ({i}/{len(units)}) {mark}: {tag}  ({round(time.time() - t0)}s, {rows or 0} points)")

    print(f"[all] finished: {tallies['done']} done, {tallies['skip']} skipped, "
          f"{tallies['excluded']} excluded, {tallies['fail']} failed. "
          f"See {ledger} and run: python plot.py --results {out} --intersection --band")


if __name__ == "__main__":
    main()
