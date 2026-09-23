"""The detection objective, plus the mask and keypoint terms.

Three detection terms:

  classification  BCE against the assigner's soft target, or Varifocal. The
                  target is not a hard one-hot: TAL rescales it so the best
                  anchor of an object carries that object's IoU, which makes
                  confidence track localisation quality.
  box             complete IoU on positives, weighted by that same target
                  score, so badly localised anchors pull less.
  distribution    cross-entropy over the two bins straddling the true
                  distance, which is what lets the box branch express
                  uncertainty about an edge instead of guessing a number.

The loss reads the training graph. Nothing here exists at deploy: the
sigmoid, the softmax expectation and the decode run on the host.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mayaku.engine.assign import AssignInput, WarmupAssigner
from mayaku.model import box as boxlib
from mayaku.model import kpt as kptlib
from mayaku.model import mask as masklib
from mayaku.model.box import decode_head
from mayaku.model.detector import split_outputs

# The target table is [image, class, x1, y1, x2, y2] then a variable tail the
# dataset appends in a fixed order: has_mask (one column) when it carries
# masks, then the 3K keypoint values when it carries keypoints. `pad_targets`
# splits that tail off as `extras`; `split_extras` owns the order.


def split_extras(extras, seg, k):
    """(..., E) extras -> (has_mask (...) or None, keypoints (..., 3k) or None).
    A keypoint-only table has no has_mask column, so keypoints start at 0."""
    ko = 1 if seg else 0
    has_mask = extras[..., 0] if seg and extras.shape[-1] >= 1 else None
    kpts = extras[..., ko:ko + 3 * k] if k and extras.shape[-1] >= ko + 3 * k else None
    return has_mask, kpts


def pad_targets(targets, batch):
    """(n, 6 + E) rows of [image, class, x1, y1, x2, y2, extras...] -> padded
    per image: labels (B, M, 1), boxes (B, M, 4), mask (B, M, 1), extras
    (B, M, E).

    A flat table is what a collate function can produce without knowing the
    maximum object count; the assigner wants a rectangle. Empty images get a
    zero row masked off. Slot j of image i is the j-th row of image i, which
    is the index the mask raster and the keypoint table use too.
    """
    device = targets.device
    e = max(0, targets.shape[1] - 6) if targets.ndim == 2 else 0
    if targets.numel() == 0:
        z = torch.zeros(batch, 1, 1, device=device)
        return z, torch.zeros(batch, 1, 4, device=device), z, torch.zeros(batch, 1, e, device=device)
    idx = targets[:, 0].long()
    counts = torch.bincount(idx, minlength=batch)
    m = int(counts.max())
    labels = torch.zeros(batch, m, 1, device=device)
    bboxes = torch.zeros(batch, m, 4, device=device)
    mask = torch.zeros(batch, m, 1, device=device)
    extras = torch.zeros(batch, m, e, device=device)
    for i in range(batch):
        sel = targets[idx == i]
        if len(sel):
            labels[i, :len(sel), 0] = sel[:, 1]
            bboxes[i, :len(sel)] = sel[:, 2:6]
            mask[i, :len(sel), 0] = 1
            extras[i, :len(sel)] = sel[:, 6:]
    return labels, bboxes, mask, extras


class SegLoss(nn.Module):
    """Dice on every box positive's mask, through the dynamic kernels.

    As in RTMDet-Ins: the loss is computed on the full stride-8 map, not a box
    crop, so the kernels also learn to say "not me" around the object; all
    task-aligned positives feed it; at most `cap` positives per batch are
    kept (at random) so a crowded batch cannot blow the step's memory (the
    gathered features are P x 8 x H8*W8). Instances without a mask
    (`has_mask` 0: crowd RLE, box-only data) are left out.
    """

    def __init__(self, cap=250):
        super().__init__()
        self.cap = cap

    def forward(self, mask_feat, kers, a, points, stride, masks, has_mask):
        _, c, h, w = mask_feat.shape
        kflat = masklib.flatten_ker(kers)
        owner = a.owner
        pos = a.fg & has_mask.gather(1, owner).bool()
        bi, ni = pos.nonzero(as_tuple=True)
        if len(bi) > self.cap:
            keep = torch.randperm(len(bi), device=bi.device)[:self.cap]
            bi, ni = bi[keep], ni[keep]
        if len(bi) == 0:
            return mask_feat.sum() * 0, 0
        k = kflat[bi, ni]                                        # (P, 169)
        feat = mask_feat[bi].reshape(len(bi), c, -1)             # (P, 8, Q)
        coords = masklib.rel_coords(points[ni], stride[ni], h, w, dtype=feat.dtype)
        logits = masklib.dyn_conv(feat, coords, k)               # (P, Q)
        slot = owner[bi, ni]
        gt = masks[bi].reshape(len(bi), -1) == (slot + 1)[:, None]
        return masklib.dice(logits, gt).mean(), len(bi)


class KeypointLoss(nn.Module):
    """OKS on the regressed points, BCE on visibility, focal on the heatmaps,
    L1 on the peak offsets."""

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.register_buffer("sig", kptlib.sigmas(k), persistent=False)

    def forward(self, heat, kps, a, points, stride, gt_boxes, gt_kpts, mask_gt):
        b = heat.shape[0]
        k = self.k
        gk = gt_kpts.view(b, -1, k, 3)                           # (B, M, K, 3)
        gvis = (gk[..., 2] > 0) & mask_gt.bool()                 # (B, M, K)
        zero = heat.sum() * 0
        # regression + visibility on the positives whose object has keypoints
        pos = a.fg & gvis.any(-1).gather(1, a.owner)
        bi, ni = pos.nonzero(as_tuple=True)
        kflat = kptlib.flatten_kpt(kps, k)                        # (B, N, K, 3)
        if len(bi):
            slot = a.owner[bi, ni]
            pred = kflat[bi, ni].float()                          # (P, K, 3)
            pxy, pvis = kptlib.decode(pred, points[ni], stride[ni])
            g = gk[bi, slot]
            valid = gvis[bi, slot]
            box = gt_boxes[bi, slot]
            area = ((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])).clamp(min=1)
            oks = kptlib.oks_loss(pxy, g[..., :2], valid, area, self.sig.to(pxy.device))
            vis = F.binary_cross_entropy_with_logits(pvis, valid.float())
        else:
            oks = vis = zero
        # dense heatmap + offsets on every labelled keypoint of the batch
        _, _, h, w = heat.shape
        tgt, toff, at = kptlib.gaussian_targets(gk[..., :2], gvis, gt_boxes, k, h, w)
        hl = kptlib.focal(heat[:, :k], tgt)
        # offset L1 on the cells that hold a keypoint, as a mask-weighted mean
        # so there is no data-dependent branch or boolean gather (both force a
        # host sync on the training step)
        m = at[:, None].float()
        denom = m.sum().clamp(min=1)
        ol = (F.l1_loss(heat[:, k:].float(), toff, reduction="none") * m).sum() / (denom * 2)
        return oks, vis, hl, ol, len(bi)


def dfl_loss(dist_logits, target_dist, reg_max):
    """Cross-entropy split between the two bins straddling the target.

    GFL's Distribution Focal Loss (Li et al. 2020, arXiv 2006.04388, Eq. 6):
    -((y_hi - y) log S_lo + (y - y_lo) log S_hi), averaged over the four
    sides, which is the paper's 1/4 weighting.
    """
    t = target_dist.clamp_(0, reg_max - 1 - 0.01)
    lo = t.floor().long()
    hi = lo + 1
    w_hi = t - lo.float()
    w_lo = 1 - w_hi
    logits = dist_logits.view(-1, reg_max)
    lo, hi = lo.view(-1), hi.clamp_(max=reg_max - 1).view(-1)
    return (F.cross_entropy(logits, lo, reduction="none") * w_lo.view(-1)
            + F.cross_entropy(logits, hi, reduction="none") * w_hi.view(-1)
            ).view(target_dist.shape).mean(-1, keepdim=True)


def varifocal(pred_logits, target_scores, alpha=0.75, gamma=2.0):
    """VarifocalNet's asymmetric weighting: down-weight easy negatives, leave
    positives weighted by their own IoU target.

    Zhang et al. 2020 (arXiv 2008.13367) Eq. 2: -q(q log p + (1-q) log(1-p))
    where q > 0, -alpha p^gamma log(1-p) where q = 0; the paper's alpha 0.75,
    gamma 2.0.
    """
    pos = target_scores > 0
    weight = torch.where(pos, target_scores, alpha * pred_logits.sigmoid().pow(gamma))
    return F.binary_cross_entropy_with_logits(
        pred_logits, target_scores, reduction="none") * weight


class DetectionLoss(nn.Module):
    """TOOD's objective (Feng et al. 2021, arXiv 2108.07755): BCE against the
    normalised alignment target t_hat (Eq. 10), and a box loss weighted by
    t_hat on the positives (Eq. 12), with GFL's DFL alongside.

    Departures from the papers: CIoU, not TOOD's GIoU; gains 7.5 / 0.5 / 1.5
    for box / cls / DFL.
    """

    def __init__(self, nc=80, reg_max=16, cls_loss="bce",
                 warmup=5, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5,
                 atss_soft=False, tal_beta=6.0, loc_weight_floor=0.0,
                 seg=False, seg_gain=2.0, seg_cap=250, kpt=0,
                 kpt_gains=(12.0, 1.0, 1.0, 1.0)):
        super().__init__()
        assert cls_loss in ("bce", "vfl"), cls_loss
        self.nc, self.reg_max = nc, reg_max
        self.cls_loss = cls_loss
        self.gains = (box_gain, cls_gain, dfl_gain)
        self.loc_weight_floor = loc_weight_floor
        self.assigner = WarmupAssigner(nc, warmup=warmup, atss_soft=atss_soft, beta=tal_beta)
        # auxiliary heads, each reading the model's extra tensors in the order
        # the detector appends them: [mask, ker*] then [heat, kpt*]
        self.seg = SegLoss(seg_cap) if seg else None
        self.seg_gain = seg_gain
        self.kpt = KeypointLoss(kpt) if kpt else None
        self.kpt_gains = kpt_gains

    def forward(self, preds, targets, epoch=0, masks=None):
        """preds: the model's tensors, head first. targets: (n, 6 + E) rows of
        [image, class, x1, y1, x2, y2, has_mask?, keypoints...] in pixels.
        masks: (B, H/8, W/8) uint8 instance index map (row + 1) when the model
        has a mask branch. Returns (total, parts)."""
        pd_cls, pd_dist, pd_boxes, points, stride, shapes = decode_head(
            preds, self.nc, self.reg_max)
        b = pd_cls.shape[0]
        device = pd_cls.device

        # padded before the transfer, not after: `counts.max()` on the device
        # would be one more host sync per step, and the label table is a few
        # hundred bytes either way
        gt_labels, gt_boxes, mask_gt, extras = (
            t.to(device, non_blocking=True) for t in pad_targets(targets, b))
        with torch.no_grad():
            a = self.assigner(AssignInput(
                pd_cls.detach().sigmoid(), pd_boxes.detach(), points, stride,
                tuple(h * w for h, w in shapes), gt_labels, gt_boxes, mask_gt),
                epoch)
        tgt_boxes, tgt_scores, fg, starved = a.boxes, a.scores, a.fg, a.starved

        norm = tgt_scores.sum().clamp_(min=1)
        if self.cls_loss == "vfl":
            cls = varifocal(pd_cls, tgt_scores).sum() / norm
        else:
            cls = F.binary_cross_entropy_with_logits(
                pd_cls, tgt_scores, reduction="sum") / norm

        # one nonzero (one host sync) instead of a boolean gather per tensor
        bi, ni = fg.nonzero(as_tuple=True)
        if len(bi):
            weight = tgt_scores.sum(-1)[bi, ni][..., None]
            if self.loc_weight_floor:
                weight = weight.clamp(min=self.loc_weight_floor)
            pos_boxes = tgt_boxes[bi, ni]
            iou = boxlib.iou_xyxy(pd_boxes[bi, ni], pos_boxes, ciou=True)[..., None]
            box = ((1 - iou) * weight).sum() / norm
            tgt_ltrb = boxlib.xyxy_to_ltrb(pos_boxes, points[ni], stride[ni], self.reg_max)
            dfl = (dfl_loss(pd_dist[bi, ni].view(-1, 4, self.reg_max),
                            tgt_ltrb, self.reg_max) * weight).sum() / norm
        else:
            # no positives this step; a zero that still carries a graph, so
            # backward() does not fail on an empty batch
            box = dfl = pd_cls.sum() * 0

        parts = {"box": box.detach(), "cls": cls.detach(), "dfl": dfl.detach(),
                 "positives": fg.sum(), "starved": starved.sum()}
        aux = pd_cls.sum() * 0
        grp = split_outputs(preds, seg=self.seg is not None, kpt=self.kpt is not None)
        has_mask, gt_kpts = split_extras(extras, self.seg is not None,
                                         self.kpt.k if self.kpt is not None else 0)
        if self.seg is not None and masks is not None and has_mask is not None:
            seg, n = self.seg(grp["mask"], grp["ker"], a, points, stride,
                              masks.to(device, non_blocking=True), has_mask)
            aux = aux + self.seg_gain * seg
            parts["seg"], parts["mask_pos"] = seg.detach(), n
        if self.kpt is not None and gt_kpts is not None:
            oks, vis, hl, ol, n = self.kpt(grp["heat"], grp["kpt"], a, points, stride,
                                           gt_boxes, gt_kpts, mask_gt)
            g = self.kpt_gains
            aux = aux + g[0] * oks + g[1] * vis + g[2] * hl + g[3] * ol
            parts.update(kpt=oks.detach(), kvis=vis.detach(), heat=hl.detach(),
                         koff=ol.detach(), kpt_pos=n)

        gb, gc, gd = self.gains
        # Scaled by the batch size, i.e. a per-image sum. The terms are
        # already normalised by the target score sum, so this does not change
        # their balance or the logged `parts`. What it changes is the
        # gradient's magnitude relative to the fixed clip norm (10): without
        # it the gradient falls under the clip early in training and the
        # steps shrink with it, which costs substantial AP.
        total = (gb * box + gc * cls + gd * dfl + aux) * b
        # every part stays a device tensor; converting here would sync the
        # host against the GPU several times per step
        return total, parts
