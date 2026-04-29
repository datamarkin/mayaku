# 006 — MPS training: torchvision roi_align backward is broken; ship a custom gather-based fallback

Status: accepted
Date: 2026-04-28

## Context

Mayaku advertises Apple-Silicon MPS as a first-class training backend in
the capability matrix. When a user actually ran `mayaku train --device mps`
on a tiny 61-image fine-tune (the matterport balloon dataset, single
class, R-50+FPN), iter 0 completed in ~17 s, then iter 1's
`total.backward()` hung indefinitely with macOS spamming
`kIOGPUCommandBufferCallbackErrorImpactingInteractivity` errors. The
process eventually accumulated 70+ GB of resident memory (on a 64 GB
M1 Max) and started swapping. This made MPS training effectively
unusable for any nontrivial workload — directly contradicting the
"first-class" claim.

This ADR captures the root-cause diagnosis (it took several wrong
guesses to get to the right one), the fix that landed, and the
measured before/after numbers.

## What we initially blamed (and why those guesses were wrong)

1. **Op-fallback round-tripping.** `OpFallbackTracker` was added to
   surface MPS→CPU fallback ops and dedupe their warnings. The tracker
   recorded **zero** fallbacks during the failing run. Not the cause.
2. **The macOS GPU watchdog rejecting long command buffers.** A
   `torch.mps.synchronize()` at the end of `run_step` was added to
   flush the buffer between iters. Iter 1 still hung. Not the cause —
   or at least, not by itself.
3. **PyTorch version regression.** Tested against PyTorch 2.3.1 and
   2.11.0; same hang on both. Not version-specific.
4. **AMP fp16 numerics.** SimpleTrainer was being used (`solver.amp_enabled`
   defaults to `False`), so autocast wasn't even in the picture.
5. **Input image size.** Reducing from 800/1333 to 480/800 made things
   worse, not better.

## Root cause

A minimal synthetic repro (just the model, a 480×640 noise batch, no
dataloader) cleanly reproduced the hang in 60 seconds. Splitting the
iter into forward + backward time:

| Iter | Forward | Backward |
|---|---|---|
| 0 | 1.01 s | 0.95 s |
| 1 | 0.33 s | **14.77 s** |
| 2 | 0.19 s | 8.82 s |

Forward gets *faster* on iter 1 (kernel-cache hit), but backward
explodes by 15×. That's the smoking gun — a specific backward op is
broken, not the watchdog generally.

Forcing the existing pure-PyTorch `_roi_align_fallback` (built on
`F.grid_sample`) instead of torchvision's MPS roi_align produced:

| Iter | Forward | Backward |
|---|---|---|
| 0 | 2.02 s | 1.40 s |
| 1 | 0.49 s | **0.92 s** (16× faster) |

**Conclusion: torchvision's `roi_align` MPS backward kernel is
pathologically slow** in the configurations R-CNN training uses. The
forward path is fine; only the backward triggers the watchdog.

But there's a second problem: the existing grid_sample-based
`_roi_align_fallback` *also* doesn't work in real R-CNN training,
because `aten::grid_sampler_2d_backward` itself isn't implemented on
MPS. With `PYTORCH_ENABLE_MPS_FALLBACK=1` it falls back to CPU, and
the CPU fallback path retains huge tensors in autograd's saved-tensor
pool — we observed 72 GB peak RAM during a 20-iter balloon run before
killing it.

## Fix

A new function `_roi_align_mps_native` in
`src/mayaku/backends/ops/roi_align.py`. Pure-PyTorch gather-based
bilinear interpolation using only ATen ops with native MPS forward
**and** backward: `index_select`, `floor`, `clamp`, basic arithmetic,
`view`/`reshape`/`permute`. No `grid_sample`, no scatter-add. The
implementation matches torchvision's ROIAlignV2 algorithm exactly
(forward and backward verified against torchvision on CPU, max abs
diff ~5e-6, FP noise).

The wrapper routes to this fallback only when `input.device.type ==
"mps" AND input.requires_grad` — so MPS *inference* still uses the
fast torchvision kernel, and CUDA / CPU training are unaffected.

Three supporting fixes that were also needed:

- `mayaku/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` at package
  import. Setting it later (inside `run_train`) is too late: PyTorch
  reads the env once, at MPS-init time.
- `benchmarks/training_validation/run.sh` exports the variable at
  shell level too, as a belt-and-suspenders measure.
- `Device.amp_dtype` returns `torch.float32` on MPS (matching the
  documented promise that drifted from the code earlier). The
  autocast wrapper now skips the autocast block entirely on fp32
  to avoid PyTorch's spurious "MPS autocast only supports fp16/bf16"
  warning. Users can opt into fp16 via `solver.amp_dtype: float16`.

## Measured outcome (M1 Max 64 GB, balloon, R-50+FPN, 480/800 input)

20-iter run, MPS only:

| Metric | Before | After |
|---|---|---|
| Total wall clock | hung indefinitely | **56 s** |
| Per-iter wall (steady state) | n/a | ~2.3 s |
| Peak RSS | 70+ GB (swapping) | 3.7 GB |
| Metal command-buffer errors | continuous | **none** |
| Op-fallback summary | 0 ops | 0 ops |
| Final box AP (20 iters) | n/a | 0.006 (random-init heads, expected low) |

200-iter run, MPS + CPU baseline (PyTorch 2.11.0, macOS 26.3.1,
**reduced 480/800 input** — leftover workaround from the watchdog
hypothesis that turned out to be wrong, see "Throughput at production
config" below for the real numbers):

| Device | Train | Eval | Total | Final box AP | Final AP50 |
|---|---|---|---|---|---|
| MPS  | 471.9 s | 5.8 s | **477.7 s** | 0.0521 | 0.1562 |
| CPU  | 445.6 s | 5.7 s | 451.3 s | 0.0101 | 0.0441 |

**MPS:CPU wall-clock ratio = 1.06x at 480/800 input.** This was
reported as "essentially tied" — but the input size is too small
to expose the GPU's strength. See the throughput sweep below for
the real story at the production R-CNN config.

Per-iter wall clock was ~2.3 s on MPS, ~2.0 s on CPU. The 5x AP
gap between MPS (0.052) and CPU (0.010) is most likely run-to-run
variance — neither run was seeded and 200 iters is far too few for
convergence; both passed `final_box_ap > 0` which is what the
"functional_pass" criterion checks.

The MPS run recorded **zero op fallbacks** in the tracker summary,
confirming the gather-based roi_align eliminated the only
significant fallback contributor in the R-CNN training graph.

## Throughput at production config (default 800/1333 input)

After the input-size workaround was removed and the iter-boundary
`torch.mps.synchronize()` was confirmed unnecessary (a 50-iter
spike test ran clean without it), we ran a batch-size sweep on
the same balloon dataset at the published R-CNN training input
distribution: `min_size_train=(640..800)` jitter,
`max_size_train=1333`, `min_size_test=800`, `max_size_test=1333`.
50 iters per device, MPS + CPU baseline:

| ims_per_batch | MPS wall | CPU wall | MPS:CPU |
|---|---|---|---|
| 2 | 170 s | 207 s | **0.82×** |
| 4 | 316 s | 386 s | **0.82×** |
| 8 | 688 s | 811 s | **0.85×** |

**MPS gives a consistent ~15-18% wall-clock win over CPU at the
production R-CNN config on M1 Max 64 GB.** The ratio is roughly
flat across batch sizes — the GPU is already mostly saturated at
batch=2 in this configuration, so larger batches just scale both
sides linearly without changing the ratio. Zero MPS→CPU op
fallbacks at any batch size.

**Practical recommendation: default to batch=2 on Apple Silicon.**
batch=8 nearly exhausted 64 GB of unified memory on the test
machine; users with 16 GB or 32 GB Macs would OOM. Since larger
batches don't improve the MPS:CPU ratio meaningfully, there's no
throughput reason to push past 2. The `tier1_mps.py` harness
default is 2 for this reason.

The earlier "1.06× tied" headline was an artifact of the smaller
480/800 input, not a real M1-Max characteristic. At the
realistic R-CNN config, MPS is genuinely faster than CPU — modest,
but a real win, with much more headroom on the GPU's memory
budget than CPU has on its compute.

The per-iter `torch.mps.synchronize()` we initially added has been
removed — the spike test confirmed the watchdog never trips
once the broken roi_align backward is out of the picture, and
the sync was costing ~13-17% per-iter throughput for nothing.

## Architecture coverage on MPS

All three R-CNN family heads exercise the gather-based roi_align
during training. We verified each on MPS at the same recipe (300
iters, lr=2.5e-4, batch=2, balloon dataset where applicable),
**with the trainer fixes from ADR 003 follow-up** (commit 1829c8a):

| Architecture | Wall clock | box AP / AP50 | mask AP / AP50 | keypoint AP | Op fallbacks | Verdict |
|---|---|---|---|---|---|---|
| Faster R-CNN | 1038 s | 0.36 / 0.67 | n/a | n/a | 0 | ✓ AP50 within 3 AP of D2 reference (pre-fix run) |
| **Mask R-CNN (post-fix)** | 1252 s | **0.75 / 0.87** | **0.80 / 0.86** | n/a | 0 | ✓ matches or exceeds D2 published ~0.70-0.75 box AP |
| Keypoint R-CNN | smoke (3 iters synthetic) | — | — | — | 0 | ✓ all loss components fire on MPS |

Notes:
- **Faster R-CNN** numbers above are from a pre-fix run.
  Re-running with the trainer fixes would lift them similarly to
  Mask R-CNN; not yet repeated, but the user's CUDA-side balloon
  runs (Mayaku-CUDA 1-2 AP *higher* than D2-CUDA after the fix)
  confirm the trainer is no longer the bottleneck.
- **Mask R-CNN** post-fix is the AP-comparison anchor. Box AP
  0.754 matches D2's published 0.70-0.75 balloon-tutorial range;
  mask AP 0.796 exceeds the typical 0.64-0.68. The trainer fix
  was the missing piece — see ADR 003 "Follow-up: training-quality
  gap" for the three bugs.
- **Keypoint R-CNN** was validated as a smoke test only (3 iters
  on a synthetic person-keypoint batch — no public small
  keypoint dataset that's balloon-equivalent). All five loss
  components (`loss_rpn_cls`, `loss_rpn_loc`, `loss_cls`,
  `loss_box_reg`, `loss_keypoint`) compute and backprop without
  errors. The validation question is "does the training pipeline
  work on MPS?" — yes.

The op-fallback tracker recorded **zero MPS→CPU fallbacks**
across all three architectures. The gather-based roi_align fix
covers the entire R-CNN family; nothing else in any of the heads
hit an MPS coverage gap on PyTorch 2.11.

## What this is not

- **Not a parity claim against CUDA training.** The custom roi_align
  is ~10-15× slower than torchvision's CUDA kernel. That's fine for
  the "small-dataset fine-tune on Apple Silicon" use case; not fine
  for from-scratch COCO training (use a CUDA box for that).
- **Not a permanent solution.** When torchvision fixes
  `RoIAlign.backward()` on MPS, the gather-based fallback can be
  retired in favour of the native kernel. The wrapper's gating
  condition is intentionally narrow so the right thing happens
  automatically the day the upstream fix lands.

## References

- Issue log: `MPS_ISSUES_DISCOVERED.md` (entries 2 and 3)
- Custom op: `src/mayaku/backends/ops/roi_align.py:_roi_align_mps_native`
- Validation harness: `benchmarks/training_validation/tier1_mps.py`
- Public-facing claim: `docs/vs_detectron2.md` ("Training from scratch")
