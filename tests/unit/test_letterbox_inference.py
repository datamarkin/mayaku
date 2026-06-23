"""Eager letterbox inference: the host letterbox→infer→un-letterbox contract.

Verifies the un-letterbox postprocess and the Predictor's letterbox path map
predictions from the fixed ``infer_size`` canvas back to original-image coords —
no real detector needed (the geometry is what's under test).
"""

from __future__ import annotations

import numpy as np
import torch

from mayaku.config.schemas import MayakuConfig
from mayaku.data.transforms import LetterboxTransform
from mayaku.inference import Predictor
from mayaku.inference.postprocess import unletterbox_instances
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances


def _predictor(cfg: MayakuConfig, model: object) -> Predictor:
    """Deploy a fake/in-code model in tests via the shared internal seam.

    ``from_pretrained`` needs a real checkpoint, so tests use ``_from_cfg``
    directly — the same cfg→Predictor mapping ``from_pretrained`` uses."""
    return Predictor._from_cfg(cfg, model)  # type: ignore[arg-type]


def test_unletterbox_instances_is_exact_inverse() -> None:
    # Wide image → vertical bars (pad_top=160, pad_left=0, scale=640/600).
    t = LetterboxTransform(300, 600, 640)
    orig = np.array([[100.0, 50.0, 400.0, 200.0]], dtype=np.float32)

    # Forward-letterbox the box into canvas space, wrap it as a model output.
    canvas_box = t.apply_box(orig)
    inst = Instances(image_size=(640, 640))
    inst.pred_boxes = Boxes(torch.from_numpy(canvas_box))
    inst.scores = torch.tensor([0.9])
    inst.pred_classes = torch.tensor([1])

    out = unletterbox_instances(inst, t, 300, 600)
    assert out.image_size == (300, 600)
    np.testing.assert_allclose(out.pred_boxes.tensor.numpy(), orig, atol=1e-2)


def test_unletterbox_drops_predictions_in_the_pad_region() -> None:
    # A box living entirely in the bottom bar maps outside the image → clipped
    # to zero area → dropped.
    t = LetterboxTransform(300, 600, 640)  # content rows [160, 480)
    inst = Instances(image_size=(640, 640))
    inst.pred_boxes = Boxes(torch.tensor([[10.0, 500.0, 100.0, 620.0]]))  # below content
    inst.scores = torch.tensor([0.9])
    inst.pred_classes = torch.tensor([0])
    out = unletterbox_instances(inst, t, 300, 600)
    assert len(out) == 0


class _FakeDetector(torch.nn.Module):
    """Records what the Predictor fed it; returns a fixed canvas-space box."""

    def __init__(self, canvas_box: list[float]) -> None:
        super().__init__()
        self.canvas_box = canvas_box
        self.seen: dict[str, object] = {}

    def forward(self, inputs: list[dict[str, object]]) -> list[dict[str, object]]:
        d = inputs[0]
        img = d["image"]
        assert isinstance(img, torch.Tensor)
        self.seen["shape"] = tuple(int(x) for x in img.shape)
        self.seen["hw"] = (int(d["height"]), int(d["width"]))  # type: ignore[arg-type]
        inst = Instances(image_size=(int(d["height"]), int(d["width"])))  # type: ignore[arg-type]
        inst.pred_boxes = Boxes(torch.tensor([self.canvas_box], dtype=torch.float32))
        inst.scores = torch.tensor([0.9])
        inst.pred_classes = torch.tensor([0])
        return [{"instances": inst}]


def test_predictor_letterbox_feeds_square_and_unletterboxes() -> None:
    fake = _FakeDetector([50.0, 200.0, 300.0, 400.0])  # a box on the 640 canvas
    pred = Predictor(fake, resize_mode="letterbox", infer_size=640, device=torch.device("cpu"))

    img = (np.random.default_rng(0).random((300, 600, 3)) * 255).astype(np.uint8)
    out = pred(img)

    # The model saw a square infer_size input + canvas-sized metadata.
    assert fake.seen["shape"] == (3, 640, 640)
    assert fake.seen["hw"] == (640, 640)
    # Output is back in original-image space.
    assert out.image_size == (300, 600)
    b = out.pred_boxes.tensor[0]
    assert 0.0 <= b[0] <= 600.0 and 0.0 <= b[1] <= 300.0
    assert 0.0 <= b[2] <= 600.0 and 0.0 <= b[3] <= 300.0
    # Matches the explicit inverse of the canvas box.
    t = LetterboxTransform(300, 600, 640)
    expected = t.inverse_box(np.array([[50.0, 200.0, 300.0, 400.0]], dtype=np.float32))[0]
    np.testing.assert_allclose(b.numpy(), expected, atol=1e-2)


def test_predictor_letterbox_with_real_detector() -> None:
    # End-to-end with a real faster_rcnn: letterbox in → predictions out in
    # original-image coords, all boxes inside the image.
    from mayaku.config.schemas import (
        BackboneConfig,
        InputConfig,
        MayakuConfig,
        ModelConfig,
        ROIBoxHeadConfig,
        ROIHeadsConfig,
        RPNConfig,
    )
    from mayaku.models.detectors import build_faster_rcnn

    cfg = MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
            backbone=BackboneConfig(name="resnet50", freeze_at=2, norm="FrozenBN"),
            rpn=RPNConfig(pre_nms_topk_test=50, post_nms_topk_test=10),
            roi_heads=ROIHeadsConfig(num_classes=2),
            roi_box_head=ROIBoxHeadConfig(num_fc=1, fc_dim=32),
        ),
        input=InputConfig(resize_mode="letterbox", infer_size=640),
    )
    torch.manual_seed(0)
    pred = _predictor(cfg, build_faster_rcnn(cfg).eval())

    img = (np.random.default_rng(0).random((300, 600, 3)) * 255).astype(np.uint8)
    out = pred(img)

    assert out.image_size == (300, 600)
    if len(out):
        b = out.pred_boxes.tensor
        assert bool((b[:, 0] >= 0).all() and (b[:, 2] <= 600).all())
        assert bool((b[:, 1] >= 0).all() and (b[:, 3] <= 300).all())


def test_predictor_selects_letterbox_from_cfg() -> None:
    from mayaku.config import InputConfig

    cfg = MayakuConfig(input=InputConfig(resize_mode="letterbox", infer_size=640))
    pred = _predictor(cfg, _FakeDetector([0.0, 0.0, 1.0, 1.0]))
    assert pred.resize_mode == "letterbox"
    assert pred.infer_size == 640


def test_eval_mapper_records_letterbox_transform() -> None:
    # The eval mapper letterboxes the image and hands the evaluator the
    # transform (carrying original h,w) + canvas-sized height/width.
    from mayaku.data import DatasetMapper
    from mayaku.data.transforms import LetterboxResize

    mapper = DatasetMapper([LetterboxResize((640,))], is_train=False)
    img = (np.random.default_rng(0).random((300, 600, 3)) * 255).astype(np.uint8)
    dd = mapper({"__image": img, "height": 300, "width": 600, "image_id": 1})

    assert isinstance(dd["letterbox"], LetterboxTransform)
    assert (dd["letterbox"].h, dd["letterbox"].w) == (300, 600)  # original dims preserved
    assert dd["height"] == 640 and dd["width"] == 640  # canvas → model emits canvas boxes
    assert tuple(dd["image"].shape) == (3, 640, 640)


def test_train_mapper_letterboxes_to_square() -> None:
    # Letterbox training: each image is resized+padded to one square size drawn
    # from train_sizes, with GT transformed to match (geometry == eval/deploy).
    from mayaku.data import DatasetMapper
    from mayaku.data.transforms import LetterboxResize, RandomFlip

    sizes = [480, 512, 544, 576, 608, 640]
    mapper = DatasetMapper([LetterboxResize(sizes), RandomFlip(prob=0.0)], is_train=True)
    img = (np.random.default_rng(1).random((300, 600, 3)) * 255).astype(np.uint8)
    dd = mapper(
        {
            "__image": img,
            "height": 300,
            "width": 600,
            "image_id": 1,
            "annotations": [
                {"bbox": [100, 50, 200, 150], "bbox_mode": 0, "category_id": 0, "iscrowd": 0}
            ],
        }
    )
    side = int(dd["image"].shape[1])
    assert side in sizes
    assert tuple(dd["image"].shape) == (3, side, side)  # square canvas
    assert dd["instances"].image_size == (side, side)
    assert len(dd["instances"]) == 1  # GT carried through the transform


def test_letterbox_rejects_gpu_preprocess() -> None:
    import pytest

    with pytest.raises(ValueError, match="gpu_preprocess"):
        Predictor(
            _FakeDetector([0.0, 0.0, 1.0, 1.0]),
            resize_mode="letterbox",
            gpu_preprocess=True,
            device=torch.device("cpu"),
        )
