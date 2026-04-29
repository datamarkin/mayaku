"""Tests for :mod:`mayaku.inference.postprocess`."""

from __future__ import annotations

import torch

from mayaku.inference.postprocess import detector_postprocess
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances


def _bbox_only_inst(device: torch.device) -> Instances:
    inst = Instances(image_size=(32, 64))
    inst.pred_boxes = Boxes(torch.tensor([[4.0, 8.0, 16.0, 24.0]], device=device))
    inst.scores = torch.tensor([0.9], device=device)
    inst.pred_classes = torch.tensor([0], device=device)
    return inst


def test_postprocess_rescales_boxes_to_output_size(device: torch.device) -> None:
    inst = _bbox_only_inst(device)
    out = detector_postprocess(inst, output_height=64, output_width=128)
    # input (32, 64) → output (64, 128); scale_x = scale_y = 2.
    expected = torch.tensor([[8.0, 16.0, 32.0, 48.0]], device=device)
    torch.testing.assert_close(out.pred_boxes.tensor, expected)
    assert out.image_size == (64, 128)


def test_postprocess_clips_to_image_canvas(device: torch.device) -> None:
    inst = Instances(image_size=(32, 32))
    inst.pred_boxes = Boxes(torch.tensor([[-10.0, -5.0, 100.0, 100.0]], device=device))
    inst.scores = torch.tensor([0.9], device=device)
    inst.pred_classes = torch.tensor([0], device=device)
    out = detector_postprocess(inst, output_height=32, output_width=32)
    b = out.pred_boxes.tensor
    assert b[0, 0].item() == 0.0 and b[0, 1].item() == 0.0
    assert b[0, 2].item() == 32.0 and b[0, 3].item() == 32.0


def test_postprocess_pastes_masks_at_output_size(device: torch.device) -> None:
    inst = _bbox_only_inst(device)
    # (R, 1, M, M) soft masks (the mask head's output shape).
    inst.pred_masks = torch.full((1, 1, 28, 28), 0.9, device=device)
    out = detector_postprocess(inst, output_height=32, output_width=64)
    assert out.has("pred_masks")
    # Pasted masks are (R, H, W) bool.
    assert out.pred_masks.shape == (1, 32, 64)
    assert out.pred_masks.dtype == torch.bool
    # All-0.9 soft masks above the 0.5 threshold inside the box yield True.
    box = out.pred_boxes.tensor[0]
    x0, y0, x1, y1 = box.long().tolist()
    assert out.pred_masks[0, y0:y1, x0:x1].all()


def test_postprocess_rescales_keypoints(device: torch.device) -> None:
    inst = _bbox_only_inst(device)
    inst.pred_keypoints = torch.tensor([[[10.0, 16.0, 0.9], [20.0, 24.0, 0.5]]], device=device)
    out = detector_postprocess(inst, output_height=64, output_width=128)
    # scale_x = scale_y = 2; visibility column unchanged.
    torch.testing.assert_close(
        out.pred_keypoints[0, :, :2],
        torch.tensor([[20.0, 32.0], [40.0, 48.0]], device=device),
    )
    torch.testing.assert_close(out.pred_keypoints[0, :, 2], torch.tensor([0.9, 0.5], device=device))


def test_postprocess_passthrough_when_size_matches(device: torch.device) -> None:
    inst = _bbox_only_inst(device)
    out = detector_postprocess(inst, output_height=32, output_width=64)
    torch.testing.assert_close(out.pred_boxes.tensor, inst.pred_boxes.tensor)
    assert out.image_size == (32, 64)


def test_postprocess_handles_empty_predictions(device: torch.device) -> None:
    inst = Instances(image_size=(32, 64))
    inst.pred_boxes = Boxes(torch.zeros(0, 4, device=device))
    inst.scores = torch.zeros(0, device=device)
    inst.pred_classes = torch.zeros(0, dtype=torch.long, device=device)
    out = detector_postprocess(inst, 64, 128)
    assert len(out) == 0
    assert out.image_size == (64, 128)


def test_postprocess_drops_boxes_that_collapse_after_clip(device: torch.device) -> None:
    """Mirrors Detectron2's `results = results[output_boxes.nonempty()]`.

    Boxes predicted entirely outside the image clip to zero width or
    height; without the filter they survive into the top-K and count as
    FPs at every IoU threshold. Regression for the 5.8 AP gap found
    during the D2 parity validation (see ADR 003).
    """
    inst = Instances(image_size=(32, 32))
    inst.pred_boxes = Boxes(
        torch.tensor(
            [
                [4.0, 8.0, 16.0, 24.0],  # valid box, kept
                [-50.0, -50.0, -10.0, -10.0],  # entirely above-left of image → clip → (0,0,0,0)
                [
                    100.0,
                    5.0,
                    200.0,
                    25.0,
                ],  # entirely right of image → clip → (32,5,32,25), zero width
                [
                    5.0,
                    100.0,
                    25.0,
                    200.0,
                ],  # entirely below image → clip → (5,32,25,32), zero height
            ],
            device=device,
        )
    )
    inst.scores = torch.tensor([0.9, 0.8, 0.7, 0.6], device=device)
    inst.pred_classes = torch.tensor([0, 1, 2, 3], device=device)
    out = detector_postprocess(inst, output_height=32, output_width=32)
    assert len(out) == 1
    torch.testing.assert_close(
        out.pred_boxes.tensor,
        torch.tensor([[4.0, 8.0, 16.0, 24.0]], device=device),
    )
    torch.testing.assert_close(out.scores, torch.tensor([0.9], device=device))
    torch.testing.assert_close(
        out.pred_classes, torch.tensor([0], device=device, dtype=out.pred_classes.dtype)
    )


def test_postprocess_filters_masks_and_keypoints_by_nonempty(device: torch.device) -> None:
    """The nonempty filter must apply to masks and keypoints too, not
    just boxes. Otherwise the per-instance fields desynchronise."""
    inst = Instances(image_size=(32, 32))
    inst.pred_boxes = Boxes(
        torch.tensor(
            [
                [4.0, 8.0, 16.0, 24.0],
                [100.0, 5.0, 200.0, 25.0],  # collapses to zero width
            ],
            device=device,
        )
    )
    inst.scores = torch.tensor([0.9, 0.7], device=device)
    inst.pred_classes = torch.tensor([0, 1], device=device)
    inst.pred_masks = torch.stack(
        [
            torch.full((1, 28, 28), 0.9, device=device),
            torch.full((1, 28, 28), 0.1, device=device),
        ],
        dim=0,
    )
    inst.pred_keypoints = torch.tensor(
        [
            [[10.0, 16.0, 0.9]],
            [[150.0, 15.0, 0.5]],
        ],
        device=device,
    )
    out = detector_postprocess(inst, output_height=32, output_width=32)
    assert len(out) == 1
    assert out.pred_masks.shape == (1, 32, 32)
    assert out.pred_keypoints.shape == (1, 1, 3)
    # The kept keypoint corresponds to the kept box (the first one).
    torch.testing.assert_close(
        out.pred_keypoints[0, 0, :2], torch.tensor([10.0, 16.0], device=device)
    )
