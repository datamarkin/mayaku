"""COCO metrics for a model on a dataset.

Detections come from `mayaku.inference.decode`, the same host decode a
deployment runs, so evaluation measures exactly what ships. `coco_ap` is
measurement only and deliberately delegates to `pycocotools`, so a
disagreement is never ambiguous between the model and the ruler. AP-S / AP-M
/ AP-L come out of the same call and are always reported alongside AP.
"""

import contextlib
import functools
import io

import numpy as np
import torch

from mayaku.data.batch import batch_to, collate
from mayaku.inference.decode import DEPLOY, decode
from mayaku.model import kpt as kptlib

# The order pycocotools returns its twelve numbers in. A size bucket with no
# ground-truth objects comes back as -1, meaning not applicable, not zero.
STATS = ("AP", "AP50", "AP75", "AP-S", "AP-M", "AP-L",
         "AR1", "AR10", "AR100", "AR-S", "AR-M", "AR-L")
# keypoint eval has no small bucket and returns ten numbers
KPT_STATS = ("AP", "AP50", "AP75", "AP-M", "AP-L",
             "AR", "AR50", "AR75", "AR-M", "AR-L")


def to_coco(det, image_id, cat_ids):
    """One image's `Detections` -> COCO bbox result records: xywh, and the
    dense class index mapped back to the annotation file's category id."""
    box = det.boxes.clone()
    box[:, 2:] -= box[:, :2]
    return [{"image_id": image_id, "category_id": cat_ids[int(c)], "bbox": b, "score": s}
            for b, s, c in zip(box.tolist(), det.scores.tolist(), det.labels.tolist())]


def to_coco_segm(det, image_id, cat_ids):
    """One image's `Detections` with masks -> COCO segm result records (RLE)."""
    from pycocotools import mask as maskutil
    recs = []
    for k in range(len(det)):
        rle = maskutil.encode(np.asfortranarray(det.masks[k].numpy()))
        rle["counts"] = rle["counts"].decode("ascii")
        recs.append({"image_id": image_id, "category_id": cat_ids[int(det.labels[k])],
                     "segmentation": rle, "score": float(det.scores[k])})
    return recs


def to_coco_kpt(det, image_id, cat_ids):
    """One image's `Detections` with keypoints -> COCO keypoint result records."""
    return [{"image_id": image_id, "category_id": cat_ids[int(det.labels[k])],
             "keypoints": det.keypoints[k].reshape(-1).tolist(), "score": float(det.scores[k])}
            for k in range(len(det))]


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
        for det, meta in zip(decode(preds, metas, model.nc, model.cfg.reg_max, seg, kpt, d),
                             metas):
            results += to_coco(det, meta["id"], dataset.cat_ids)
            if seg:
                segm += to_coco_segm(det, meta["id"], dataset.cat_ids)
            if kpt:
                kpts += to_coco_kpt(det, meta["id"], dataset.cat_ids)
    model.train(was_training)
    out = coco_ap(dataset.ann_path, results)
    if seg:
        out.update({"segm_" + k: v for k, v in coco_ap(dataset.ann_path, segm, "segm").items()})
    if kpt:
        out.update({"kpt_" + k: v
                    for k, v in coco_ap(dataset.kpt_ann_path, kpts, "keypoints", k=kpt).items()})
    return out


def summary(stats, prefix=""):
    """One line, AP first, size split always present."""
    return "%sAP %.4f  AP50 %.4f  AP75 %.4f  S %.4f  M %.4f  L %.4f" % (
        prefix, stats["AP"], stats["AP50"], stats["AP75"],
        stats["AP-S"], stats["AP-M"], stats["AP-L"])
