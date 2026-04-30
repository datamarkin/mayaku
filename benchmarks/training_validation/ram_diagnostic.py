"""RAM diagnostic for the training pipeline.

Run for a few hundred iters; prints per-iter RSS growth and the top-N
``tracemalloc`` allocators. The output tells us **where** memory is
going, not just that it's growing — without that data any fix is a
guess.

Usage::

    python benchmarks/training_validation/ram_diagnostic.py

Edit the constants below to point at your COCO data; defaults match
``benchmarks/training_validation/tier3.py``. Stop with Ctrl-C once
you've seen 500-1000 iters of output; that's enough to extrapolate
the 90k-iter trajectory.

What to look for in the output:

* **RSS growing linearly with iteration count** → leak in the per-iter
  path (mapper, augmentation, dataloader). The ``tracemalloc`` top-10
  diff will name the file:line that's accumulating.
* **RSS jumps once at startup, then stable** → not a leak, just a
  high baseline. Check what loaded.
* **RSS stable for hundreds of iters then jumps** → a periodic event
  (eval, checkpoint, hook) is allocating but not freeing.
* **Python heap (``tracemalloc``) flat but RSS climbing** → not Python
  objects. Either glibc malloc fragmentation (Linux only) or a C
  extension leak (PIL, pycocotools, torch, numpy).
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import torch

from mayaku.cli._factory import build_detector
from mayaku.config import load_yaml, merge_overrides
from mayaku.data import (
    AspectRatioGroupedDataset,
    DatasetMapper,
    ResizeShortestEdge,
    SerializedList,
    TrainingSampler,
    build_coco_metadata,
    load_coco_json,
)
from mayaku.engine import build_optimizer

# ---------------------------------------------------------------------------
# Configure these.
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/detection/faster_rcnn_R_50_FPN_1x.yaml")
COCO_TRAIN_JSON = Path("/data/coco/annotations/instances_train2017.json")
COCO_TRAIN_IMAGES = Path("/data/coco/train2017")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How many iters to sample. 500 is enough to see whether RSS is growing
# linearly (= leak) or stabilising.
NUM_ITERS = 500
# Print RSS + tracemalloc snapshot every N iters.
PRINT_EVERY = 50
# Top-N tracemalloc lines to print on each snapshot.
TOP_N = 10


def _rss_mb() -> float:
    """Resident-set size in MiB. Linux: /proc/self/status. macOS: ru_maxrss
    (in bytes on macOS, kibibytes on Linux — yes, really)."""
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss_bytes / (1024 * 1024)
    return rss_bytes / 1024  # Linux: ru_maxrss is in KiB


def main() -> int:
    if not COCO_TRAIN_JSON.exists():
        print(
            f"[diag] {COCO_TRAIN_JSON} does not exist — edit the constants at the "
            "top of this file to point at your COCO install.",
            file=sys.stderr,
        )
        return 2

    print(f"[diag] PID = {os.getpid()}")
    print(f"[diag] device = {DEVICE}")
    print(f"[diag] platform = {sys.platform}")
    print(f"[diag] starting RSS = {_rss_mb():.0f} MiB")
    print()

    # Phase 1: load config + dataset. RSS after this tells us the static
    # cost of dataset_dicts.
    cfg = load_yaml(CONFIG_PATH)
    cfg = merge_overrides(cfg, {"solver": {"ims_per_batch": 4}})
    metadata = build_coco_metadata(name="diag", json_path=COCO_TRAIN_JSON)

    rss_before_load = _rss_mb()
    keep_seg = cfg.model.meta_architecture == "mask_rcnn"
    keep_kp = cfg.model.meta_architecture == "keypoint_rcnn"
    raw = load_coco_json(
        COCO_TRAIN_JSON,
        COCO_TRAIN_IMAGES,
        metadata,
        keep_segmentation=keep_seg,
        keep_keypoints=keep_kp,
    )
    rss_after_load = _rss_mb()
    print(
        f"[diag] load_coco_json: {len(raw)} dicts, "
        f"RSS {rss_before_load:.0f} → {rss_after_load:.0f} MiB "
        f"(+{rss_after_load - rss_before_load:.0f} MiB)"
    )

    dataset_dicts = SerializedList(raw)
    del raw
    gc.collect()
    rss_after_serialize = _rss_mb()
    print(
        f"[diag] SerializedList + gc.collect: RSS {rss_after_serialize:.0f} MiB "
        f"({rss_after_load - rss_after_serialize:+.0f} MiB delta)"
    )

    # Phase 2: build model and optimizer. RSS delta ≈ model param count + optimizer state.
    rss_before_model = _rss_mb()
    model = build_detector(cfg, backbone_weights="DEFAULT").to(torch.device(DEVICE))
    optimizer = build_optimizer(model, cfg.solver)
    rss_after_model = _rss_mb()
    print(
        f"[diag] model + optimizer: RSS {rss_before_model:.0f} → "
        f"{rss_after_model:.0f} MiB (+{rss_after_model - rss_before_model:.0f} MiB)"
    )

    # Phase 3: data pipeline.
    augmentations = [
        ResizeShortestEdge(
            cfg.input.min_size_train,
            max_size=cfg.input.max_size_train,
            sample_style=cfg.input.min_size_train_sampling,
        ),
    ]
    mapper = DatasetMapper(
        augmentations,
        is_train=True,
        mask_format=cfg.input.mask_format,
        deepcopy_input=False,
    )

    class _MappedView:
        def __init__(self, dicts, m):
            self._d = dicts
            self._m = m

        def __len__(self):
            return len(self._d)

        def __getitem__(self, idx):
            return self._m(self._d[idx])

    mapped = _MappedView(dataset_dicts, mapper)
    sampler = TrainingSampler(size=len(mapped), shuffle=True, seed=0)
    sampled_iter = iter(sampler)

    class _SamplerView:
        def __init__(self, mapped, it):
            self._mapped = mapped
            self._it = it

        def __iter__(self):
            for idx in self._it:
                yield self._mapped[idx]

    indexed = _SamplerView(mapped, sampled_iter)
    loader = AspectRatioGroupedDataset(indexed, batch_size=cfg.solver.ims_per_batch)
    print()

    # Phase 4: instrument the training loop.
    tracemalloc.start(25)
    snapshot_prev = tracemalloc.take_snapshot()
    rss_at_iter_0 = _rss_mb()
    print(f"[diag] iter   0: RSS {rss_at_iter_0:.0f} MiB (baseline)")

    model.train()
    t0 = time.time()
    loader_iter = iter(loader)
    for i in range(1, NUM_ITERS + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            print(f"[diag] iter {i}: loader exhausted")
            break
        for s in batch:
            s["image"] = s["image"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(batch)
        total = sum(loss_dict.values())
        total.backward()
        optimizer.step()
        del batch, loss_dict, total

        if i % PRINT_EVERY == 0:
            gc.collect()
            rss = _rss_mb()
            wall = time.time() - t0
            it_per_s = i / wall
            print(
                f"[diag] iter {i:4d}: RSS {rss:.0f} MiB "
                f"(+{rss - rss_at_iter_0:.0f} since iter 0, "
                f"+{(rss - rss_at_iter_0) / max(i, 1):.2f} MiB/iter), "
                f"{it_per_s:.2f} it/s"
            )
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.compare_to(snapshot_prev, "lineno")[:TOP_N]
            for s in stats:
                # Skip lines that haven't moved.
                if abs(s.size_diff) < 100 * 1024:  # < 100 KiB
                    continue
                print(f"        {s.size_diff / 1024:+8.0f} KiB  {s.traceback.format()[-1].strip()}")
            snapshot_prev = snapshot

    print()
    print("[diag] done — extrapolate the +N MiB/iter line to your max_iter for")
    print("       a projected final RSS. If most of the growth is in the")
    print("       tracemalloc top-10, that's the file:line to fix. If")
    print("       tracemalloc is flat but RSS keeps climbing, it's glibc")
    print("       fragmentation or a C-extension leak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
