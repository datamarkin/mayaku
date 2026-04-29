# 002 — RGB-native image ingestion

Status: accepted
Date: 2026-04-25

## Context

Detectron2 uses BGR throughout, inherited from OpenCV/Caffe lineage. The
modern Python ML ecosystem (torchvision, PIL, timm, Hugging Face,
Ultralytics, MMDetection-recent) is RGB-native. The mayaku rewrite is not
constrained by Detectron2 checkpoint compatibility (per master plan
non-goals). torchvision-pretrained ResNet/ResNeXt weights — the only
realistic starting point for our backbones — are RGB. Export targets
(ONNX, CoreML, OpenVINO) all expect RGB by default; CoreML especially,
since Apple's Vision framework is RGB.

## Decision

mayaku is RGB-native end-to-end:
- Image ingestion uses Pillow as the primary path (PIL.Image.open returns RGB).
- Numpy array inputs are documented as RGB; users with OpenCV-decoded
  arrays must convert before passing.
- All tensors are (C, H, W) with channels in [R, G, B] order.
- Pixel mean/std defaults are written in RGB order:
  mean = [123.675, 116.280, 103.530], std = [58.395, 57.120, 57.375].
- No `INPUT.FORMAT` config knob. Channel order is not configurable.

## Consequences

- Users coming from OpenCV pipelines must convert BGR→RGB at the boundary.
  We provide `mayaku.utils.bgr_to_rgb` for clarity.
- Direct accuracy comparisons against Detectron2 trained models will
  involve a channel-order difference unless Detectron2 is retrained with
  `INPUT.FORMAT="RGB"`. This is acceptable; competitive comparison target
  is Ultralytics/Roboflow (both RGB).
- Pretrained backbone loading is straightforward — torchvision RGB weights
  load directly without channel swapping.

## Alternatives considered

- Configurable INPUT.FORMAT — rejected. Adds complexity throughout the
  codebase for no gain; RGB is the dominant convention.
- BGR-native — rejected. Fights every modern ecosystem touchpoint.

## References

- DETECTRON2_TECHNICAL_SPEC.md §5 (data pipeline) — describes BGR pipeline
  in the original; not inherited.
- BACKEND_PORTABILITY_REPORT.md §6 (CoreML) — RGB expected by Apple Vision.