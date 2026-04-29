"""Detection / segmentation / keypoint head modules."""

from __future__ import annotations

from mayaku.models.heads.box_head import FastRCNNConvFCHead, build_box_head
from mayaku.models.heads.fast_rcnn import (
    FastRCNNOutputLayers,
    fast_rcnn_inference,
    fast_rcnn_inference_single_image,
)
from mayaku.models.heads.keypoint_head import (
    KRCNNConvDeconvUpsampleHead,
    build_keypoint_head,
    keypoint_rcnn_inference,
    keypoint_rcnn_loss,
    select_proposals_with_visible_keypoints,
)
from mayaku.models.heads.mask_head import (
    MaskRCNNConvUpsampleHead,
    build_mask_head,
    mask_rcnn_inference,
    mask_rcnn_loss,
    select_foreground_proposals,
)

__all__ = [
    "FastRCNNConvFCHead",
    "FastRCNNOutputLayers",
    "KRCNNConvDeconvUpsampleHead",
    "MaskRCNNConvUpsampleHead",
    "build_box_head",
    "build_keypoint_head",
    "build_mask_head",
    "fast_rcnn_inference",
    "fast_rcnn_inference_single_image",
    "keypoint_rcnn_inference",
    "keypoint_rcnn_loss",
    "mask_rcnn_inference",
    "mask_rcnn_loss",
    "select_foreground_proposals",
    "select_proposals_with_visible_keypoints",
]
