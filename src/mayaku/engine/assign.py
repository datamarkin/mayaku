"""Which anchor point is responsible for which object.

Two assigners behind one interface:

  TaskAligned  ranks candidates by s^alpha * u^beta, so it needs the
               classifier to already mean something. The strongest on short
               schedules (PP-YOLOE, arXiv 2203.16250: ATSS 43.1, SimOTA 44.3,
               TAL 45.2 at 36 epochs).
  ATSS         a statistics-only assigner that ignores the classifier
               entirely. Used to warm TAL up, because the alignment metric is
               noise while the predictions are random, which is exactly the
               regime a from-scratch run starts in.

Both report starvation: real objects that received zero positive anchors and
therefore contribute nothing to the box loss. Starvation is a sub-stride
phenomenon; above one stride it is zero.

Shapes throughout: B batch, N anchors, M padded ground-truth slots.
"""

import dataclasses

import torch
import torch.nn.functional as F

from mayaku.model.blocks import STRIDES
from mayaku.model.box import iou_xyxy


@dataclasses.dataclass
class AssignInput:
    """Everything any assigner could need, so they share one signature and a
    new assigner drops in as another class with the same `__call__`."""

    pd_scores: torch.Tensor    # (B, N, nc) sigmoid probabilities
    pd_boxes: torch.Tensor     # (B, N, 4) xyxy pixels
    points: torch.Tensor       # (N, 2) anchor centres in pixels
    stride: torch.Tensor       # (N, 1)
    n_per_level: tuple         # anchors per pyramid level, fine to coarse
    gt_labels: torch.Tensor    # (B, M, 1)
    gt_boxes: torch.Tensor     # (B, M, 4) xyxy pixels
    mask_gt: torch.Tensor      # (B, M, 1), 1 where the slot holds a real object


@dataclasses.dataclass
class Assignment:
    labels: torch.Tensor       # (B, N)
    boxes: torch.Tensor        # (B, N, 4)
    scores: torch.Tensor       # (B, N, nc) soft classification target
    fg: torch.Tensor           # (B, N) bool
    starved: torch.Tensor      # (B, M) bool, real objects with no anchor
    owner: torch.Tensor = None # (B, N) long, the padded GT slot each anchor
                               # serves; meaningful where `fg`. The mask and
                               # keypoint losses read their targets through it.


def candidates_in_gt(points, gt_boxes, eps=1e-9, finest=STRIDES[0], floor=STRIDES[1]):
    """(B, M, N) bool: is this anchor centre inside this box?

    The hard precondition both assigners share. A box narrower than one
    stride contains no cell centre at all, so it would be assigned nothing
    and contribute nothing to any loss. So any side under the finest stride
    is widened to the middle stride before the test, keeping the centre
    fixed, and a three-pixel object still owns a cell. The widening is for
    the test only; the loss still regresses the true box.
    """
    centre = (gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2
    wh = gt_boxes[..., 2:] - gt_boxes[..., :2]
    wh = torch.where(wh < finest, torch.full_like(wh, float(floor)), wh)
    lt, rb = (centre - wh / 2), (centre + wh / 2)
    deltas = torch.cat((points[None, None] - lt[..., None, :],
                        rb[..., None, :] - points[None, None]), -1)
    return deltas.amin(-1) > eps


def masked_mean_std(x, mask, eps=1e-9):
    """Mean and sample standard deviation over the True entries of `mask`,
    without two (B, M, N) NaN-filled temporaries."""
    m = mask.to(x.dtype)
    n = m.sum(-1, keepdim=True).clamp_(min=2)
    mean = (x * m).sum(-1, keepdim=True) / n
    var = (((x - mean) * m) ** 2).sum(-1, keepdim=True) / (n - 1)
    return mean, (var + eps).sqrt()


def resolve_conflicts(mask_pos, tiebreak):
    """One anchor may be selected by several objects. Give it to the one that
    wins `tiebreak`, so every anchor carries at most one target.

    Unconditional on purpose: where nothing is contested the `where` returns
    `mask_pos` unchanged, and testing first would sync the host every step."""
    contested = (mask_pos.sum(-2) > 1)[:, None, :].expand_as(mask_pos)
    winner = tiebreak.argmax(-2)
    keep = F.one_hot(winner, mask_pos.shape[-2]).permute(0, 2, 1).to(mask_pos.dtype)
    return torch.where(contested, keep, mask_pos)


def gather_targets(mask_pos, tiebreak, inp, nc):
    """Collapse an (B, M, N) selection into per-anchor targets.

    Returns the Assignment with a one-hot score, plus the resolved selection,
    which the caller needs to weight that score.
    """
    b, m, _ = mask_pos.shape
    mask_pos = resolve_conflicts(mask_pos, tiebreak)
    fg = mask_pos.sum(-2) > 0
    index = torch.arange(m, device=mask_pos.device)[None, :, None]
    owner = (mask_pos * index).amax(-2).long()
    flat = owner + torch.arange(b, device=mask_pos.device)[:, None] * m
    labels = inp.gt_labels.long().flatten()[flat]
    boxes = inp.gt_boxes.reshape(-1, 4)[flat]
    scores = (F.one_hot(labels.clamp(0, nc - 1), nc) * fg[..., None]).to(boxes.dtype)
    starved = (mask_pos.sum(-1) == 0) & inp.mask_gt[..., 0].bool()
    return Assignment(labels, boxes, scores, fg, starved, owner), mask_pos


class TaskAlignedAssigner:
    """TOOD's alignment metric (Feng et al. 2021, arXiv 2108.07755).

    Eq. 9: t = s^alpha * u^beta, the m anchors with the largest t per object
    are positive. The class target is t_hat = t / max(t) * max(u) per
    object, so its best anchor carries the object's best IoU, which couples
    classification confidence to localisation quality.

    Departures from the paper:
      defaults      alpha 0.5 and m = 10, not the paper's 1.0 and 13 (beta 6
                    is the paper's).
      u             CIoU, not plain IoU, so the metric ranks by the same
                    overlap the box loss optimises.
      candidates    restricted to anchors whose centre lies in the (widened)
                    box: ATSS's centre rule, which bounds the top-m search to
                    plausible cells.
      conflicts     an anchor claimed by several objects goes to the one its
                    predicted box overlaps most (ATSS's highest-IoU rule).
    """

    def __init__(self, nc, topk=10, alpha=0.5, beta=6.0, eps=1e-9):
        self.nc, self.topk, self.alpha, self.beta, self.eps = nc, topk, alpha, beta, eps

    def __call__(self, inp):
        # CIoU, clamped at zero. With beta 6 the overlap term dominates the
        # metric, so CIoU's centre and aspect penalties change which anchors
        # are selected, not merely how they are weighted.
        overlaps = iou_xyxy(inp.gt_boxes[:, :, None, :],
                            inp.pd_boxes[:, None, :, :], ciou=True).clamp_(0)
        idx = inp.gt_labels.long().squeeze(-1).clamp(0, self.nc - 1)
        scores = inp.pd_scores.permute(0, 2, 1).gather(
            1, idx[..., None].expand(-1, -1, inp.pd_scores.shape[1]))
        align = scores.pow(self.alpha) * overlaps.pow(self.beta)

        inside = candidates_in_gt(inp.points, inp.gt_boxes)
        align = align * inside * inp.mask_gt
        top = align.topk(min(self.topk, align.shape[-1]), dim=-1)
        mask_topk = torch.zeros_like(align, dtype=torch.bool).scatter_(
            -1, top.indices, top.values > self.eps)
        mask_pos = (mask_topk & inside & inp.mask_gt.bool()).to(align.dtype)

        out, mask_pos = gather_targets(mask_pos, overlaps, inp, self.nc)

        # rescale: the best anchor of each object gets that object's best IoU
        align = align * mask_pos
        best_iou = (overlaps * mask_pos).amax(-1, keepdim=True)
        weight = (align * best_iou / (align.amax(-1, keepdim=True) + self.eps)).amax(-2)
        out.scores = out.scores * weight[..., None]
        return out


class ATSSAssigner:
    """Adaptive Training Sample Selection, anchor-free form.

    Zhang et al. 2020 (arXiv 1912.02424), Algorithm 1: per object and per
    level, take the k (paper: 9) anchors whose centres are nearest the
    object's centre, then keep those whose IoU clears mean + std of that
    candidate set and whose centre lies inside the object; an anchor claimed
    twice goes to the higher IoU. The anchor-free form scores a square
    pseudo-anchor of side `scale` * stride around each point; the paper uses
    8 and reports 5 to 9 all stable, this uses 5. The std is the sample one.

    It reads no prediction at all, and it breaks ties on the same
    pseudo-anchor overlap it selects on. That is the point: it is well
    defined at epoch zero, where TAL's metric is two noise terms raised to a
    power.

    One guard not in the paper. The threshold comes from the candidates
    themselves, so a set like IoU {0, 1, 1} yields mean + std = 1.14 and
    admits nothing. Capping it at the best candidate's IoU keeps at least one
    anchor per object whenever the centre prior admits one.
    """

    def __init__(self, nc, topk=9, scale=5.0, eps=1e-9, soft=False):
        self.nc, self.topk, self.scale, self.eps = nc, topk, scale, eps
        self.soft = soft

    def __call__(self, inp):
        half = self.scale * inp.stride / 2
        anchors = torch.cat((inp.points - half, inp.points + half), -1)
        overlaps = iou_xyxy(inp.gt_boxes[:, :, None, :], anchors[None, None]).clamp_(0)

        centres = (inp.gt_boxes[..., :2] + inp.gt_boxes[..., 2:]) / 2
        dist = (centres[:, :, None, :] - inp.points[None, None]).pow(2).sum(-1)

        near = torch.zeros_like(dist, dtype=torch.bool)
        start = 0
        for n in inp.n_per_level:
            level = dist[..., start:start + n]
            near[..., start:start + n].scatter_(
                -1, level.topk(min(self.topk, n), dim=-1, largest=False).indices, True)
            start += n

        mean, std = masked_mean_std(overlaps, near)
        thr = torch.minimum(mean + std,
                            overlaps.masked_fill(~near, -1).amax(-1, keepdim=True))
        mask_pos = (near & (overlaps >= thr)
                    & candidates_in_gt(inp.points, inp.gt_boxes)
                    & inp.mask_gt.bool()).to(overlaps.dtype)

        # a flat target of 1: the alignment metric is meaningless this early,
        # which is the reason this assigner is running at all
        out, mask_pos = gather_targets(mask_pos, overlaps, inp, self.nc)
        if self.soft:
            # Rescale the class target by the object's best IoU against a
            # predicted box. Without it the classifier is trained to answer
            # 1.0 on every positive regardless of how well the box fits, so
            # confidence carries no ordering, and AP is a ranking metric.
            pred = iou_xyxy(inp.gt_boxes[:, :, None, :],
                            inp.pd_boxes[:, None, :, :]).clamp_(0)
            best = (pred * mask_pos).amax(-1)
            out.scores = out.scores * (mask_pos * best[..., None]).amax(-2)[..., None]
        return out


class WarmupAssigner:
    """ATSS for the first `warmup` epochs, then TAL (the schedule YOLOv6
    reports, Li et al. 2022, arXiv 2209.02976).

    The epoch is an argument, not state: training-loop position does not
    belong to a loss object, and passing it keeps a run reproducible from its
    inputs alone.
    """

    def __init__(self, nc, warmup=5, topk=10, alpha=0.5, beta=6.0,
                 atss_topk=9, atss_scale=5.0, atss_soft=False):
        self.warmup = warmup
        self.atss = ATSSAssigner(nc, topk=atss_topk, scale=atss_scale, soft=atss_soft)
        self.tal = TaskAlignedAssigner(nc, topk=topk, alpha=alpha, beta=beta)

    def __call__(self, inp, epoch):
        return self.atss(inp) if epoch < self.warmup else self.tal(inp)
