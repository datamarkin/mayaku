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
            for b, s, c in zip(box.tolist(), det.scores.tolist(), det.labels.tolist(), strict=True)]


def rle(mask):
    """An (h, w) bool mask -> COCO run-length encoding, JSON-ready."""
    from pycocotools import mask as maskutil
    r = maskutil.encode(np.asfortranarray(mask.numpy()))
    r["counts"] = r["counts"].decode("ascii")
    return r


def to_coco_segm(det, image_id, cat_ids):
    """One image's `Detections` with masks -> COCO segm result records (RLE)."""
    return [{"image_id": image_id, "category_id": cat_ids[int(det.labels[k])],
             "segmentation": rle(det.masks[k]), "score": float(det.scores[k])}
            for k in range(len(det))]


def to_coco_kpt(det, image_id, cat_ids):
    """One image's `Detections` with keypoints -> COCO keypoint result records."""
    return [{"image_id": image_id, "category_id": cat_ids[int(det.labels[k])],
             "keypoints": det.keypoints[k].reshape(-1).tolist(), "score": float(det.scores[k])}
            for k in range(len(det))]


@functools.cache
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
    return dict(zip(stats, [float(v) for v in e.stats], strict=True))


def score(ann_path, detections, cat_ids, seg=False, kpt=0, kpt_ann_path=None):
    """COCO metrics for `(image_id, Detections)` pairs: box AP, plus mask AP
    under "segm_" and keypoint AP under "kpt_" when the model has those heads.
    Keypoints are scored against `kpt_ann_path` when the file is separate."""
    results, segm, kpts = [], [], []
    for image_id, det in detections:
        results += to_coco(det, image_id, cat_ids)
        if seg:
            segm += to_coco_segm(det, image_id, cat_ids)
        if kpt:
            kpts += to_coco_kpt(det, image_id, cat_ids)
    out = coco_ap(ann_path, results)
    if seg:
        out.update({"segm_" + k: v for k, v in coco_ap(ann_path, segm, "segm").items()})
    if kpt:
        out.update({"kpt_" + k: v for k, v in
                    coco_ap(kpt_ann_path or ann_path, kpts, "keypoints", k=kpt).items()})
    return out


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

    def detections():
        for imgs, _, metas, _ in loader:
            preds = model(batch_to(imgs, device))
            dets = decode(preds, metas, model.nc, model.cfg.reg_max, seg, kpt, d)
            yield from ((meta["id"], det) for meta, det in zip(metas, dets, strict=True))

    try:
        return score(dataset.ann_path, detections(), dataset.cat_ids, seg, kpt,
                     dataset.kpt_ann_path)
    finally:
        model.train(was_training)


def evaluate_runner(runner, images, annotations, batch=16, log=None) -> dict[str, float]:
    """COCO metrics for a deployed detector -- a `mayaku.inference.Runner`,
    so a checkpoint and its exported artifacts score through their own
    preprocessing, graph and precision -- on the split `annotations` /
    `images`. `log` receives a progress line every 100 batches."""
    import os

    ann = str(annotations)
    gt = ground_truth(ann)             # the index scoring reads, parsed once
    cat_ids = sorted(gt.getCatIds())   # dense class order, as `load_coco` has it
    if len(runner.class_names) != len(cat_ids):
        raise ValueError(f"the model has {len(runner.class_names)} classes and "
                         f"{annotations} has {len(cat_ids)}")
    ids = list(gt.imgs)
    files = [os.path.join(images, gt.imgs[i]["file_name"]) for i in ids]
    kp = runner.sidecar["keypoints"]

    def detections():
        for i in range(0, len(files), batch):
            if log and i and i % (100 * batch) == 0:
                log("eval %d/%d" % (i, len(files)))
            yield from zip(ids[i:i + batch], runner.batch(files[i:i + batch]), strict=True)

    return score(ann, detections(), cat_ids,
                 seg=runner.sidecar["mask"] is not None, kpt=kp["num"] if kp else 0)


def summary(stats, prefix=""):
    """One line, AP first, size split always present."""
    return "%sAP %.4f  AP50 %.4f  AP75 %.4f  S %.4f  M %.4f  L %.4f" % (
        prefix, stats["AP"], stats["AP50"], stats["AP75"],
        stats["AP-S"], stats["AP-M"], stats["AP-L"])
