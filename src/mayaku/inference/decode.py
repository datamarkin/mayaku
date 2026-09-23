"""The host decode: raw output maps -> detections in original-image pixels.

Everything after the five-op graph lives here -- the sigmoid, the DFL
expectation, the score threshold, top-k, NMS, the per-instance mask network
and the keypoint snap -- so the Predictor, the exported-artifact runner and the
evaluator all run the same code, and evaluation measures exactly what a
deployment gets. Reading `postprocess` and `decode` with a `Decode` tells a
runtime author what to implement.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from mayaku.backends.ops.nms import batched_nms
from mayaku.data.geometry import unletterbox, unletterbox_maps
from mayaku.model import kpt as kptlib
from mayaku.model import mask as masklib
from mayaku.model.box import decode_head
from mayaku.model.detector import split_outputs

__all__ = ["DEPLOY", "Decode", "Detections", "decode", "decode_sidecar", "postprocess"]


@dataclasses.dataclass(frozen=True)
class Decode:
    """What the host does after the graph. These numbers change AP without
    changing a weight, so they are recorded with every run and every artifact.

    `multi_label` False takes the best class per anchor, one comparison per
    anchor, which is what a deployed decoder does. True emits one detection
    per class above the threshold, which scores slightly higher because a
    second-choice class can still earn recall.
    """

    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    topk: int = 1000
    multi_label: bool = False

    @classmethod
    def from_sidecar(cls, block, **override):
        """From a sidecar's "decode" block (which also carries strides and
        reg_max); `override` replaces fields, e.g. conf."""
        return cls(**{f.name: block[f.name] for f in dataclasses.fields(cls)} | override)


DEPLOY = Decode()


@dataclasses.dataclass
class Detections:
    """One image's detections in its original pixels, on the CPU.

    boxes (n, 4) xyxy float32, scores (n,), labels (n,) int64 class indices;
    masks (n, h, w) bool when the model segments; keypoints (n, K, 3) as
    (x, y, visibility probability) when it has keypoints.
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    masks: torch.Tensor | None = None
    keypoints: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.scores)


def postprocess(preds, nc, reg_max, d=DEPLOY):
    """Head outputs -> (dets, idxs, points, stride): per image an (n, 6)
    [x1, y1, x2, y2, score, class] tensor in canvas pixels and the anchor
    index each surviving detection came from, plus the shared anchor points
    and strides. The indices are how the mask and keypoint heads find a
    detection's kernels / offsets without decoding the head a second time.
    """
    cls, _, boxes, points, stride, _ = decode_head(preds, nc, reg_max)
    floor = math.log(d.conf / (1 - d.conf))
    out, idxs = [], []
    for logits, bx in zip(cls, boxes):
        anchor = torch.arange(len(bx), device=bx.device)
        if d.multi_label:
            i, lab = (logits > floor).nonzero(as_tuple=True)
            box, score, anchor = bx[i], logits[i, lab].sigmoid(), anchor[i]
        else:
            best, lab = logits.max(1)
            m = best > floor
            box, score, lab, anchor = bx[m], best[m].sigmoid(), lab[m], anchor[m]
        if len(score) > d.topk:
            keep = score.topk(d.topk).indices
            box, score, lab, anchor = box[keep], score[keep], lab[keep], anchor[keep]
        keep = batched_nms(box, score, lab, d.iou)[:d.max_det]
        out.append(torch.cat((box[keep], score[keep, None], lab[keep, None].float()), 1))
        idxs.append(anchor[keep])
    return out, idxs, points, stride


@torch.no_grad()
def decode(preds, metas, nc, reg_max, seg=False, kpt=0, d=DEPLOY):
    """The model's flat output list for a batch -> one `Detections` per image.

    `metas` are the letterbox metas ({"ratio", "pad", "shape"}) of the batch's
    images; everything returned is mapped back to those original frames.
    """
    dets, idxs, points, stride = postprocess(preds, nc, reg_max, d)
    grp = split_outputs(preds, seg=seg, kpt=kpt)
    # flatten the aux maps once for the whole batch, not per image
    kflat = masklib.flatten_ker(grp["ker"]) if seg else None
    kpflat = kptlib.flatten_kpt(grp["kpt"], kpt) if kpt else None
    out = []
    for b, meta in enumerate(metas):
        box = dets[b][:, :4].float()           # stays on the device for the aux decode
        det = dets[b].float().cpu()
        r = Detections(unletterbox(det[:, :4], meta["ratio"], meta["pad"], meta["shape"]),
                       det[:, 4], det[:, 5].long())
        ai = idxs[b]
        if seg:
            h, w = meta["shape"]
            if len(det):
                lg = masklib.assemble(grp["mask"][b].float(), kflat[b][ai].float(),
                                      points[ai], stride[ai], box)
                r.masks = (unletterbox_maps(masklib.to_canvas(lg), meta)[:, 0] > 0).cpu()
            else:
                r.masks = torch.zeros(0, h, w, dtype=torch.bool)
        if kpt:
            if len(det):
                heat, off = grp["heat"][b, :kpt].float(), grp["heat"][b, kpt:].float()
                pxy, pv = kptlib.decode(kpflat[b][ai].float(), points[ai], stride[ai])
                vis = pv.sigmoid()
                pxy = kptlib.snap(pxy, vis, heat, off, box)
                pxy, vis = torch.cat((pxy, vis[..., None]), -1).cpu().split((2, 1), -1)
                xy = unletterbox(pxy.reshape(len(det), -1), meta["ratio"], meta["pad"],
                                 meta["shape"]).reshape(len(det), kpt, 2)
                r.keypoints = torch.cat((xy, vis), -1)
            else:
                r.keypoints = torch.zeros(0, kpt, 3)
        out.append(r)
    return out


def decode_sidecar(preds, metas, sidecar, conf=None):
    """`decode` with every parameter read from a model's sidecar (see
    `mayaku.utils.checkpoint.build_sidecar`); `conf` overrides the recorded
    score threshold."""
    dec = sidecar["decode"]
    d = Decode.from_sidecar(dec, **({} if conf is None else {"conf": conf}))
    kp = sidecar["keypoints"]
    return decode(preds, metas, len(sidecar["class_names"]), dec["reg_max"],
                  seg=sidecar["mask"] is not None, kpt=kp["num"] if kp else 0, d=d)
