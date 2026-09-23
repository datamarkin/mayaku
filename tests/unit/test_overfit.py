"""Can the network, the assigner and the loss learn at all?

Each test overfits one synthetic image. That is a weak test of accuracy and a
strong test of wiring: if the anchor ordering, the distance encoding, the
assignment and the loss terms disagree anywhere, a single image will not fit.
The detection runs train quantization-aware, the path the small tiers ship.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mayaku.data.polygons import Polys
from mayaku.engine.loss import DetectionLoss
from mayaku.model import TINY, Detector, enable_qat, split_outputs
from mayaku.model import mask as masklib
from mayaku.model.box import decode_head, iou_xyxy

STEPS = 260
STEPS_PER_EPOCH = 20
SIZE = 256
NC = 4
BOXES = torch.tensor([[20.0, 20.0, 60.0, 60.0],
                      [100.0, 90.0, 190.0, 200.0],
                      [150.0, 30.0, 230.0, 80.0]])

pytestmark = pytest.mark.slow


def scene(seg=False):
    """One image, three objects at three scales, each painted in its own
    channel so the network can key on it. With `seg` every object's mask is
    its filled box: the target table gains a has_mask column and the stride-8
    instance raster comes back too, painted by the dataset's own
    `Polys.raster`."""
    x = torch.zeros(1, 3, SIZE, SIZE)
    for i, (x1, y1, x2, y2) in enumerate(BOXES.long().tolist()):
        x[0, i % 3, y1:y2, x1:x2] = 1.0
    targets = torch.cat((torch.zeros(3, 1), torch.arange(3.0)[:, None], BOXES), 1)
    if not seg:
        return x, targets, None
    polys = Polys.from_coco([[[x1, y1, x2, y1, x2, y2, x1, y2]]
                             for x1, y1, x2, y2 in BOXES.tolist()])
    raster, _ = polys.raster(targets[:, 1:].numpy(), (SIZE, SIZE))
    return x, torch.cat((targets, torch.ones(3, 1)), 1), torch.from_numpy(raster)[None]


def overfit(tier=TINY, cls_loss="bce", warmup=3, qat=True, targets=None, steps=STEPS):
    """Train `steps` AdamW steps on the scene; returns (first total loss,
    first parts), (last total loss, last parts) and the trained model's
    outputs on the scene."""
    torch.manual_seed(0)
    model = Detector(tier, NC)
    if qat:
        enable_qat(model)
    crit = DetectionLoss(nc=NC, reg_max=tier.reg_max, cls_loss=cls_loss, warmup=warmup,
                         seg=tier.seg, kpt=tier.kpt)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    x, scene_targets, raster = scene(seg=tier.seg)
    targets = scene_targets if targets is None else targets
    for step in range(steps):
        total, parts = crit(model(x), targets, epoch=step // STEPS_PER_EPOCH, masks=raster)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step == 0:
            first = total.item(), {k: float(v) for k, v in parts.items()}
    model.eval()
    with torch.no_grad():
        preds = model(x)
    return first, (total.item(), parts), preds


def best_ious(preds, tier=TINY):
    """For each object, the IoU of the top-scoring prediction of its class:
    the number that says the box branch decoded, not just the classifier."""
    head = split_outputs(preds, tier.seg, tier.kpt)["head"]
    cls, _, boxes, _, _, _ = decode_head(head, NC, tier.reg_max)
    return [iou_xyxy(boxes[0, cls[0, :, c].argmax()], BOXES[c]).item() for c in range(3)]


@pytest.mark.parametrize("cls_loss", ["bce", "vfl"])
def test_detection_overfits(cls_loss) -> None:
    (first, _), (last, parts), preds = overfit(cls_loss=cls_loss)
    assert last < first / 4, "loss did not fall"
    assert min(best_ious(preds)) > 0.70
    assert parts["starved"] == 0, "an object was starved in a three-object scene"


@pytest.mark.parametrize("warmup", [0, STEPS], ids=["TAL-only", "ATSS-only"])
def test_either_assigner_alone_learns(warmup) -> None:
    """The ATSS-to-TAL warmup must not be what makes training work."""
    assert min(best_ious(overfit(warmup=warmup)[2])) > 0.70


def test_mask_branch_learns() -> None:
    """The kernels and the mask feature learn to paint each object, not just
    to score it: the assembled mask at each object's best anchor overlaps
    its filled box."""
    tier = dataclasses.replace(TINY, seg=True)
    _, (_, parts), preds = overfit(tier)
    assert min(best_ious(preds, tier)) > 0.70
    assert parts["seg"] < 0.5, "mask loss did not fall"
    raster = scene(seg=True)[2][0]
    grp = split_outputs(preds, seg=True)
    cls, _, boxes, points, stride, _ = decode_head(grp["head"], NC, tier.reg_max)
    ker = masklib.flatten_ker(grp["ker"])[0]
    for c in range(3):
        a = cls[0, :, c].argmax()
        lg = masklib.assemble(grp["mask"][0], ker[a][None], points[a][None],
                              stride[a][None], boxes[0, a][None])[0]
        m = torch.isfinite(lg) & (lg > 0)
        gt = raster == c + 1
        assert ((m & gt).sum() / (m | gt).sum().clamp(min=1)).item() > 0.5, c


def test_keypoint_branch_learns() -> None:
    """Three keypoints per object (corners and centre), no flip pairs; a
    keypoint-only table has no has_mask column."""
    tier = dataclasses.replace(TINY, kpt=3)
    x1, y1, x2, y2 = BOXES.unbind(1)
    v = torch.full_like(x1, 2.0)
    kp = torch.stack((x1, y1, v, (x1 + x2) / 2, (y1 + y2) / 2, v, x2, y2, v), 1)
    targets = torch.cat((scene()[1], kp), 1)
    # a falling OKS loss is the whole claim, and it shows within two epochs
    (_, first), (_, last), _ = overfit(tier, qat=False, targets=targets, steps=40)
    assert float(last["kpt"]) < first["kpt"], "OKS loss did not fall"
