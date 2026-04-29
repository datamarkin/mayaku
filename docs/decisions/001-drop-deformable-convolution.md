# 001 — Drop deformable convolution from the rewrite

- **Status:** accepted
- **Date:** 2026-04-25

## Context

`detectron2/layers/deform_conv.py` ships a custom C++/CUDA `DeformConv` and
`ModulatedDeformConv` op, used by `DeformBottleneckBlock`
(`backbone/resnet.py:213-327`) when
`MODEL.RESNETS.DEFORM_ON_PER_STAGE[i] == True`.

The locked v1 model scope (`03_d2_reimplementation_prompt.md` §"Locked
model scope") is Faster / Mask / Keypoint R-CNN with R-50, R-101, and
ResNeXt-101 32×8d FPN backbones. Two independent sources confirm the
deformable code path is unused by any in-scope config:

- `BACKEND_PORTABILITY_REPORT.md` §3 ("Kernel replacement plan"):
  *"Verified: zero in-scope configs enable `DEFORM_ON_PER_STAGE`. Drop the
  entire deformable code path from the rewrite."*
- `DETECTRON2_TECHNICAL_SPEC.md` §7.3 ("Deformable convolution"):
  *"None of the in-scope zoo configs (R50/R101/X101_32x8d FPN 3x for
  Detection, InstanceSeg, Keypoints) enable deformable convolution. The
  detectron2 model zoo lists separate `*_dconv_*` configs for that."*

Carrying a `deform_conv.py` would also expose new portability work:

- `torchvision.ops.deform_conv2d` covers CPU and CUDA but **MPS support
  is missing as of the pinned versions** (Appendix A of the report).
- ONNX `DeformConv` requires opset 19; runtime support is patchy.
- CoreML has no first-class deformable conv op.

## Decision

We do not implement `src/mayaku/backends/ops/deform_conv.py` at this
time. `Step 3 — Portable ops` ships `roi_align` and `nms` only.

The Step 3 entry in `PROJECT_STATUS.md` records this drop in its open-
questions section so future readers see the deviation from the master
plan immediately.

## Consequences

- Smaller portable-ops surface, fewer MPS / ONNX / CoreML risks to chase
  in Steps 18–20.
- The ResNet implementation in Step 7 will not have a `DeformBottleneckBlock`
  branch; the `freeze_at` and stage-construction code paths simplify.
- A future `*_dconv_*` config (or a research extension) that needs
  deformable conv would require:
    1. Adding `src/mayaku/backends/ops/deform_conv.py` that dispatches to
       `torchvision.ops.deform_conv2d` (with a clear MPS-unsupported
       error until upstream lands it).
    2. Reintroducing `DeformBottleneckBlock` in `models/backbones/resnet.py`.
    3. Wiring `MODEL.RESNETS.DEFORM_ON_PER_STAGE` through the pydantic
       config schema in Step 5.
    4. Re-evaluating ONNX export feasibility at opset 19+.

To revisit: supersede this ADR with `NNN-add-deformable-convolution.md`
documenting the triggering use case.
