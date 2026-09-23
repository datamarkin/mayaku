"""Detections from head outputs, and COCO metrics from detections.

Two halves that must not be confused.

`postprocess` is the reference host decode: the sigmoid, the DFL expectation,
the score threshold, the top-k and the NMS. None of it is in the deploy graph;
reading it and the `Decode` object that parameterises it tells a runtime
author exactly what to implement.

`coco_ap` is measurement only and deliberately delegates to `pycocotools`, so
a disagreement is never ambiguous between the model and the ruler. AP-S / AP-M
/ AP-L come out of the same call and are always reported alongside AP.
"""

import contextlib
import dataclasses
import functools
import io
import math

import numpy as np
import torch

from mayaku.backends.ops.nms import batched_nms
from mayaku.data.batch import batch_to, collate
from mayaku.data.geometry import unletterbox
from mayaku.model import kpt as kptlib
from mayaku.model import mask as masklib
from mayaku.model.box import decode_head

# The order pycocotools returns its twelve numbers in. A size bucket with no
# ground-truth objects comes back as -1, meaning not applicable, not zero.
STATS = ("AP", "AP50", "AP75", "AP-S", "AP-M", "AP-L",
         "AR1", "AR10", "AR100", "AR-S", "AR-M", "AR-L")
# keypoint eval has no small bucket and returns ten numbers
KPT_STATS = ("AP", "AP50", "AP75", "AP-M", "AP-L",
             "AR", "AR50", "AR75", "AR-M", "AR-L")


@dataclasses.dataclass(frozen=True)
class Decode:
    """What the host does after the graph. These numbers change AP without
    changing a weight, so they are recorded with every run.

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


DEPLOY = Decode()


def postprocess(preds, nc, reg_max, d=DEPLOY):
    """Head outputs -> (dets, idxs, points, stride): per image an (n, 6)
    [x1, y1, x2, y2, score, class] tensor and the anchor index each surviving
    detection came from, plus the shared anchor points and strides. The
    indices are how the mask and keypoint heads find a detection's kernels /
    offsets without decoding the head a second time.

    Boxes are in letterboxed pixels; `unletterbox` puts them back on the
    original image.
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


def to_coco_segm(det, masks, meta, cat_ids):
    """Detections + their boolean masks in original-image pixels -> COCO segm
    result records (RLE)."""
    from pycocotools import mask as maskutil
    recs = []
    for k in range(len(det)):
        rle = maskutil.encode(np.asfortranarray(masks[k].numpy()))
        rle["counts"] = rle["counts"].decode("ascii")
        recs.append({"image_id": meta["id"], "category_id": cat_ids[int(det[k, 5])],
                     "segmentation": rle, "score": float(det[k, 4])})
    return recs


def to_coco_kpt(det, kpts, meta, cat_ids):
    """Detections + their (K, 3) keypoints in letterbox pixels -> COCO keypoint
    result records, keypoints mapped back to the original image."""
    xy = kpts[:, :, :2].clone()
    xy = unletterbox(xy.reshape(len(det), -1), meta["ratio"], meta["pad"], meta["shape"]
                     ).reshape(len(det), -1, 2)
    recs = []
    for k in range(len(det)):
        flat = torch.cat((xy[k], kpts[k, :, 2:3].sigmoid()), 1).reshape(-1)
        recs.append({"image_id": meta["id"], "category_id": cat_ids[int(det[k, 5])],
                     "keypoints": flat.tolist(), "score": float(det[k, 4])})
    return recs


def to_coco(det, meta, cat_ids):
    """One image's detections -> COCO result records: undo the letterbox,
    convert to xywh, and map the dense class index back to the category id
    the annotation file uses."""
    box = unletterbox(det[:, :4], meta["ratio"], meta["pad"], meta["shape"])
    box[:, 2:] -= box[:, :2]
    return [{"image_id": meta["id"], "category_id": cat_ids[int(c)],
             "bbox": b, "score": s}
            for b, s, c in zip(box.tolist(), det[:, 4].tolist(), det[:, 5].tolist())]


@functools.lru_cache(maxsize=None)
def ground_truth(ann_path):
    """The parsed annotations, once per path. Large annotation files cost
    seconds to parse and index and the trainer evaluates every epoch; they
    are read-only, so one parse serves the whole run."""
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(ann_path)


def coco_ap(ann_path, results, iou_type="bbox", k=0):
    """COCO metrics for a list of result records; `iou_type` is bbox, segm or
    keypoints (with `k` keypoints per instance). Keypoints report ten
    numbers, not twelve."""
    from pycocotools.cocoeval import COCOeval

    stats = KPT_STATS if iou_type == "keypoints" else STATS
    if not results:
        return dict.fromkeys(stats, 0.0)
    gt = ground_truth(ann_path)
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(gt, gt.loadRes(results), iou_type)
        if iou_type == "keypoints":
            # pycocotools defaults its OKS sigmas to the 17 COCO person
            # keypoints; a model with a different K needs its own, the same
            # table the loss uses
            e.params.kpt_oks_sigmas = kptlib.sigmas(k).numpy()
        e.evaluate()
        e.accumulate()
        e.summarize()
    return dict(zip(stats, [float(v) for v in e.stats]))


@torch.no_grad()
def evaluate(model, dataset, device="cpu", batch=16, workers=0, d=DEPLOY):
    """Run a model over a dataset and return the COCO metrics.

    The class count and `reg_max` come from the model, because the model is
    what fixed the head's channel split. The dataset is asserted against it,
    so a head and an annotation file that disagree fail here instead of
    decoding plausible nonsense.
    """
    from torch.utils.data import DataLoader

    assert model.nc == dataset.nc, (model.nc, dataset.nc)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=workers, collate_fn=collate,
                        pin_memory=device.startswith("cuda"))
    was_training = model.training
    model.eval()
    seg, kpt = model.cfg.seg, model.cfg.kpt
    results, segm, kpts = [], [], []
    for imgs, _, metas, _ in loader:
        preds = model(batch_to(imgs, device))
        dets, idxs, points, stride = postprocess(preds, model.nc, model.cfg.reg_max, d)
        grp = model.split(preds)
        # flatten the aux maps once for the whole batch, not per image
        kflat = masklib.flatten_ker(grp["ker"]) if seg else None
        kpflat = kptlib.flatten_kpt(grp["kpt"], kpt) if kpt else None
        for b, meta in enumerate(metas):
            box = dets[b][:, :4].float()      # stays on the device for the aux decode
            det = dets[b].float().cpu()
            results += to_coco(det, meta, dataset.cat_ids)
            if not len(det):
                continue
            ai = idxs[b]
            if seg:
                lg = masklib.assemble(grp["mask"][b].float(), kflat[b][ai].float(),
                                      points[ai], stride[ai], box)
                m = masklib.to_image(lg, meta, dataset.imgsz).cpu()
                segm += to_coco_segm(det, m, meta, dataset.cat_ids)
            if kpt:
                heat, off = grp["heat"][b, :kpt].float(), grp["heat"][b, kpt:].float()
                pxy, pv = kptlib.decode(kpflat[b][ai].float(), points[ai], stride[ai])
                pxy = kptlib.snap(pxy, pv.sigmoid(), heat, off, box)
                allk = torch.cat((pxy, pv[..., None]), -1).cpu()
                kpts += to_coco_kpt(det, allk, meta, dataset.cat_ids)
    model.train(was_training)
    out = coco_ap(dataset.ann_path, results)
    if seg:
        out.update({"segm_" + k: v for k, v in coco_ap(dataset.ann_path, segm, "segm").items()})
    if kpt:
        kann = getattr(dataset, "kpt_ann_path", None) or dataset.ann_path
        out.update({"kpt_" + k: v
                    for k, v in coco_ap(kann, kpts, "keypoints", k=kpt).items()})
    return out


def summary(stats, prefix=""):
    """One line, AP first, size split always present."""
    return "%sAP %.4f  AP50 %.4f  AP75 %.4f  S %.4f  M %.4f  L %.4f" % (
        prefix, stats["AP"], stats["AP50"], stats["AP75"],
        stats["AP-S"], stats["AP-M"], stats["AP-L"])
