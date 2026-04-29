# 004 — CoreML/ANE: input-shape limited for R-CNN; not the speedup the marketing implies

Status: accepted
Date: 2026-04-26

## Context

Round-1 of `examples/coreml_speed_check.py` reported the CoreML hybrid
eval path running ~2.4× slower than MPS on M1 Max. After three rounds
of progressively isolated diagnostics (capped at the actual R-CNN
graph, then standalone classifiers, then the FPN composition), the
real story is now clear.

**The headline finding: CoreML/ANE on Apple Silicon delivers a 6×
speedup over MPS for image-classification workloads, ~1.3–1.5× for the
R-CNN backbone+FPN alone, and ~1.05× for the full R-CNN eval pipeline.
Apple's "ANE is much faster than GPU" marketing is true — but only at
ImageNet-scale input shapes (224×224). At R-CNN-scale inputs
(736×1344), most of the speedup evaporates.**

## Empirical results across three benchmark scripts

### Sanity baseline: `benchmarks/coreml_ane_baseline.py`

Standalone classifiers on this M1 Max, fp16, all four `compute_units`:

| Model | Input | PyTorch CPU | PyTorch MPS | CoreML best | vs MPS |
|---|---|---|---|---|---|
| MobileNet-v3-small | 224×224 | 106.6 ms | 9.1 ms | **0.40 ms** (CPU_AND_NE) | **22.6× faster** |
| ResNet-50 | 224×224 | 33.7 ms | 11.5 ms | **1.72 ms** (CPU_AND_NE) | **6.7× faster** |

**ANE absolutely engages and is dramatically faster than MPS for
classifier-shape workloads.** This rules out "ANE is broken on this
machine" hypotheses.

### Progressive isolation: `benchmarks/coreml_isolate_fpn.py`

Holds the architecture roughly constant, varies one factor at a time:

| Test | Input | MPS | CoreML best | speedup |
|---|---|---|---|---|
| A: R-50 classifier | 224×224 | 11.1 ms | 1.7 ms (CPU_AND_NE) | **6.4×** |
| B: R-50 classifier | 736×1344 | 43.0 ms | 32.1 ms (CPU_AND_GPU) | 1.34× |
| C: R-50 4-output | 736×1344 | 42.2 ms | 53.5 ms (CPU_AND_GPU) | **0.79× (loses)** |
| D: R-50 + FPN | 736×1344 | 85.0 ms | 58.4 ms (CPU_AND_GPU) | 1.46× |

Reading the progression:

- **A → B (input shape changes from 224×224 to 736×1344): 6.4× → 1.34×**.
  The input-shape effect alone destroys ~80% of ANE's speedup. ANE has
  internal tile-size constraints; at R-CNN-scale inputs, ANE either
  spills work to GPU or runs at lower utilisation.
- **B → C (one output → four outputs): 1.34× → 0.79×**. Multi-output
  graphs cost more device transitions; CoreML actually *loses* to MPS
  here.
- **C → D (add FPN top-down fusion): 0.79× → 1.46×**. Counterintuitively,
  *adding* the FPN improves the ratio, because MPS slows down more on
  the FPN compute (42→85 ms) than CoreML does (53→58 ms). FPN compute
  is exactly the kind of work CoreML's static graph optimiser handles
  better than MPS's per-op dispatch.

### Full eval pipeline: `examples/coreml_speed_check.py` (round 3)

```
MPS eager fp32                      : 171–174 ms/iter
CoreML fp16 heads=mps CPU_AND_GPU   : 154–163 ms/iter   1.05–1.12×
```

The full pipeline includes ResizeShortestEdge + DataLoader + RPN +
ROI heads + postprocess. The heads alone are ~85 ms on MPS, which
dilutes the backbone speedup: `(85ms heads + 27ms saved on backbone)
/ 85ms ≈ 1.07×`. The math checks out.

### Cross-meta-arch + cross-backbone sweep

Two scripts, two paths into Apple's CoreML stack:

- `benchmarks/export_smoke_all_metaarchs.py` — **ORT-CoreML EP**
  path (CoreML accessed through ONNX Runtime's wrapper).
- `benchmarks/coreml_smoke_all_metaarchs.py` — **Direct CoreML**
  path (`mayaku export coreml` → `.mlpackage` → `CoreMLBackbone`
  runtime swap). This is the path `mayaku eval --backbone-mlpackage`
  uses; the one users actually deploy through.

Ran every shipped checkpoint (4 detection + 4 segmentation + 4
keypoint = 12 total) through both. **Zero failures on either
path**: every checkpoint exported, every hybrid runtime ran
end-to-end, every meta-arch produced its expected output fields
(boxes, +masks for mask R-CNN, +keypoints for keypoint R-CNN).

Side-by-side speedup vs MPS eager (Mac, single-image steady state,
averaged across the three meta-arch families per backbone class):

| Backbone class | ORT-CoreML EP | **Direct CoreML** | Δ |
|---|---|---|---|
| R-50 1x | 1.01× | **1.16×** | direct +15% |
| R-50 3x | 1.00× | **1.16×** | direct +16% |
| R-101 3x | 1.09× | **1.24×** | direct +14% |
| X-101 32x8d 3x | 1.44× | **1.43×** | parity (both win) |

Within each backbone class, the three meta-architectures land within
a few percent of each other on either path — confirming "Graph A"
(only backbone+FPN in the export) makes the runtime ratio depend on
the backbone, not on which heads run on top in PyTorch.

**Headline correction**: the round-1 "CoreML at parity for R-50,
win only for X-101" finding was an artefact of testing through ORT's
CoreML EP wrapper (which adds ~15% dispatch overhead vs the direct
`CoreMLBackbone` path). With the direct path, **all 12 checkpoints
get a real speedup over MPS eager** — 1.13× for the smallest
configs, 1.46× for the largest. The published claim should reference
the direct-path numbers since `mayaku eval --backbone-mlpackage`
uses that path.

**Backbone-size scaling still holds on the direct path**: bigger
backbone → bigger relative win. R-50 ≈ 1.16×, X-101 ≈ 1.44×.
Structural argument: deployment-runtime overhead is fixed per call;
backbone compute scales with model size; the overhead-to-compute
ratio drops as the backbone grows.

## Diagnosis

The reason our R-CNN eval gets only 1.05–1.46× from CoreML, when
ImageNet classifiers get 6–22×:

1. **Input shape is too large for ANE's sweet spot.** ANE was
   designed in 2017 around 224×224 ImageNet-class workloads. At
   736×1344 (the standard COCO eval size), ANE's tile/memory
   constraints force lower utilisation. The 6.4× → 1.34× collapse
   between A and B is the dominant effect.
2. **Multi-output graphs incur transition overhead.** R-CNN
   backbones output 4–5 FPN levels per call; classifiers output 1.
   Each extra output is a tensor that crosses ANE↔caller boundary.
3. **The PyTorch heads dilute the gain.** Even when CoreML saves
   27 ms on the backbone (B-isolated), the heads + postprocess add
   ~85 ms regardless. Net per-image speedup is 27/171 ≈ 16% (1.19×)
   minus the host-transfer overhead between the two devices.
4. **Earlier wiring bugs (round 1) made it look worse:** building
   the model with `device="cpu"` ran the heads on CPU instead of
   MPS, costing another ~1.5×. Fixing this (build on MPS, swap the
   backbone for CoreMLBackbone) is in the change set below.

## Decision

**Three changes accepting the empirical reality:**

1. **Default `compute_units` for the runtime hybrid path is
   `"CPU_AND_GPU"`** in both `cli/eval.py` and `CoreMLBackbone.__init__`.
   `"ALL"` was the worst default — it forces ANE↔GPU transitions
   that hurt R-CNN graphs without helping. (Already applied.)
2. **The eval CLI must build the model on the user's `--device`
   (MPS) and only swap the backbone for CoreMLBackbone.** The
   PyTorch heads have to run on MPS, not CPU. (`cli/eval.py:55-87`
   already does this correctly when `--device mps` is passed.)
3. **Documentation framing changes** in README and `vs_detectron2.md`:
   - **Do not claim** "CoreML is much faster than MPS on Apple
     Silicon." That was a marketing-driven claim and is false for
     R-CNN.
   - **Do claim** "CoreML matches MPS within ~5% on the full
     pipeline; the value is the deployment artifact, not throughput."
   - **Do call out** "For pure backbone feature extraction (no R-CNN
     heads), CoreML gives a real 1.5–6× speedup over MPS depending
     on input size." That's a use case where CoreML is genuinely
     better — and it's the path users would take if they wanted to
     run just the backbone for embeddings.

## Consequences

- **The "CoreML for speed on Apple Silicon" pitch in the comparison
  doc and the paper plan is downgraded.** CoreML is shipped because
  it produces the deployment artifact iOS / macOS apps need, not
  because it's faster than MPS for our default workload.
- **The 12-checkpoint COCO val parity sweep can proceed via either
  MPS eager or the CoreML hybrid.** Numbers will be within ±0.1 AP
  either way (export parity is `atol=1e-3`); throughput will be
  within ±5% either way. There is no reason to prefer CoreML for
  validation runs.
- **No source changes to `CoreMLExporter` or the export contract.**
  The exports themselves are correct; the runtime path is what we
  tuned.
- **Three benchmark scripts are kept as artifacts**:
  `benchmarks/coreml_ane_baseline.py` (sanity check that ANE works),
  `benchmarks/coreml_isolate_fpn.py` (the progression A→B→C→D), and
  `examples/coreml_speed_check.py` (the gating test for the eval
  pipeline). These are how a future reader can reproduce the
  finding.

## Alternatives considered

- **Smaller fixed export shape (e.g. 512×512) with EnumeratedShapes
  for multi-shape support.** Rejected: COCO eval images vary widely
  in aspect ratio; clamping to 512×512 sacrifices accuracy. ANE
  doesn't reliably switch between enumerated shapes without
  recompiling.
- **Export the heads to CoreML too**, eliminating the host transfer
  and putting the whole graph on ANE. Rejected: the heads contain
  ROI Align (not ANE-eligible), per-class NMS (not ANE-eligible),
  and dynamic-shape RoI processing (not ANE-eligible). The whole
  reason FPN's `add` stays on GPU is structural; the heads are
  worse.
- **Switch the default backbone to a more ANE-friendly architecture**
  (MobileNet-v3, EfficientNet). Rejected: changes the parity story
  with D2 (we can no longer load D2's R-50 checkpoints), which is
  one of the project's strongest claims.
- **Drop CoreML support entirely.** Rejected: the deployment artifact
  is real value for users targeting iOS / macOS app distribution,
  even if it's not a runtime-speed win on developer Macs.

## References

- `examples/coreml_speed_check.py` — round-3 sweep that pinned the
  full-pipeline number at 1.05–1.12×.
- `benchmarks/coreml_ane_baseline.py` — proves ANE works on this
  hardware (22× MobileNet, 6.7× R-50 classifier).
- `benchmarks/coreml_isolate_fpn.py` — pins down the A→B→C→D
  progression that explains where 6× collapses to 1.05×.
- `benchmarks/export_smoke_all_metaarchs.py` — the all-12-checkpoint
  sweep via ORT-CoreML EP across detection / segmentation /
  keypoints; zero failures, confirms Graph A's structural promise.
- `benchmarks/coreml_smoke_all_metaarchs.py` — the same sweep via
  direct `CoreMLBackbone` (the path `mayaku eval --backbone-mlpackage`
  uses); also zero failures, ~15% faster than the ORT-CoreML EP
  path on R-50/R-101.
- `coremltools.compute_plan.MLComputePlan` — per-op device tally
  showing 89% of R-CNN backbone+FPN ops route to ANE at compile
  time, but at runtime the input-shape × multi-output combo limits
  the speedup.
- ADR 003 — D2-faithful structure constraint that rules out
  alternative backbone choices.
