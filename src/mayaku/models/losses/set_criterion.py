"""Hungarian-matching set prediction loss for UniQuery.

Faithfully follows the original Sparse R-CNN (PeizeSun/SparseR-CNN)
loss computation: sigmoid focal loss with sum reduction, L1 on
image-size-normalized boxes, GIoU on absolute xyxy, external weight
application, and DDP-aware num_boxes normalization.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["SetCriterion"]


class SetCriterion(nn.Module):
    """Set prediction loss with Hungarian matching and deep supervision.

    Loss weights (class_weight, l1_weight, giou_weight) are applied by
    the detector's forward(), NOT inside this module — matching the
    original Sparse R-CNN's weight_dict pattern.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cascade_iou_thresholds: tuple[float, ...] = (),
        cls_loss_type: str = "focal",
        mal_gamma: float = 1.5,
        mal_alpha: float | None = None,
        distill_labels: bool = False,
        distill_conf_weight: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cascade_iou_thresholds = cascade_iou_thresholds
        if cls_loss_type not in ("focal", "vfl", "mal"):
            raise ValueError(
                f"cls_loss_type must be one of focal/vfl/mal, got {cls_loss_type!r}"
            )
        # ``focal`` trains every matched query toward a hard 1.0 regardless of
        # how well its box fits, so a well-localized box and a barely-matched
        # one get the same target and AP — a *ranking* metric — cannot tell them
        # apart. ``vfl`` and ``mal`` replace that target with the prediction's
        # own IoU, making the score a localization-quality estimate.
        self.cls_loss_type = cls_loss_type
        self.mal_gamma = mal_gamma
        # Focal damps its negative branch by alpha=0.25. MAL, as DEIM ships it,
        # does not — and with num_proposals x num_classes slots per image being
        # almost all negative, dropping alpha raises the negative term ~4x and
        # so the effective weight of loss_ce against the box losses. ``None`` is
        # DEIM's default; set 0.25 to hold the classification/box balance at
        # focal's, isolating the IoU-target change from a loss-weight change.
        self.mal_alpha = mal_alpha
        # Stage-wise classification self-distillation (see _distill_labels).
        self.distill_labels = distill_labels
        self.distill_conf_weight = distill_conf_weight

    def _distill_labels(
        self,
        outputs_list: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Distil the last stage's class posterior into every earlier stage.

        D-FINE's GO-LSD distils the final layer's *localization* distribution
        backward. Here the measured deficit is elsewhere: extra stages buy
        score calibration and duplicate suppression, while box geometry is
        already close after one pass. So the quantity worth transferring
        backward is the class posterior, not the box.

        The teacher is the final stage's own sigmoid output, detached. It
        already encodes the two things depth buys — the winning query is
        confident, its near-duplicate neighbours are not — so a stage trained
        toward it inherits that ordering without running the extra passes.
        Training-only: nothing is added to the graph at inference, and a
        shallower ``inference_num_stages`` becomes a better predictor rather
        than a worse one.

        ``distill_conf_weight`` scales each query's term by the teacher's own
        peak confidence, so an uncertain teacher does not drag the student.
        This mirrors GO-LSD's decoupled weighting without needing its matched /
        unmatched split, which does not apply to a pure classification target.

        Two deliberate choices about scale:

        * The term is a **KL**, i.e. BCE minus the teacher's own entropy, not
          raw BCE. The two have identical gradients (the entropy is constant in
          the student), but KL is 0 exactly when the student matches the
          teacher, so the logged value reads as "distance still to close"
          rather than an uninterpretable offset.
        * It is normalised by the **weighted query count**, not ``num_boxes``.
          The other losses sum over ``B*N*K`` slots and divide by the GT count,
          which works for focal only because ``(1-p)^gamma`` crushes the easy
          negatives. Soft-target BCE has no such damping, so that normaliser
          put this term at ~950 against a ``loss_ce`` of ~2.4 and it would have
          swamped every other gradient. Dividing by the weight sum makes it a
          weighted mean over queries of the per-query class-sum — independent
          of ``num_proposals`` and of the weighting scheme.
        """
        teacher_logits = outputs_list[-1]["pred_logits"].detach().float()
        teacher = teacher_logits.sigmoid()
        # Teacher entropy, subtracted to turn BCE into KL. clamp keeps
        # log(0) out of the 0*log(0) corners.
        t = teacher.clamp(1e-6, 1 - 1e-6)
        entropy = -(t * t.log() + (1 - t) * (1 - t).log())

        if self.distill_conf_weight:
            weight = teacher.amax(dim=-1, keepdim=True)
        else:
            weight = torch.ones_like(teacher[..., :1])
        denom = weight.sum().clamp(min=1e-6)

        losses: dict[str, Tensor] = {}
        for i, outputs in enumerate(outputs_list[:-1]):
            student = outputs["pred_logits"].float()
            kl = F.binary_cross_entropy_with_logits(student, teacher, reduction="none") - entropy
            losses[f"loss_distill_{i}"] = (kl * weight).sum() / denom
        return losses

    def forward(
        self,
        outputs_list: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
        num_boxes: float | None = None,
    ) -> dict[str, Tensor]:
        """Compute raw (unweighted) losses with deep supervision.

        Returns loss dict with keys: loss_ce_{i}, loss_bbox_{i}, loss_giou_{i}
        for each stage i. The caller applies weight_dict to these.

        ``num_boxes`` normalises the losses (DETR convention). Pass the
        *effective-batch* box count — summed over all gradient-accumulation
        micro-batches and DDP ranks — so the loss scale is invariant to how the
        batch is split. When ``None`` (the non-accumulating path) it is derived
        from this call's ``targets``, which is correct only at
        ``grad_accum_steps == 1``.
        """
        if num_boxes is None:
            num_boxes = self.reduce_num_boxes(
                sum(len(t["labels"]) for t in targets), outputs_list[0]["pred_logits"].device
            )

        losses: dict[str, Tensor] = {}
        for stage_idx, outputs in enumerate(outputs_list):
            stage_losses = self._single_stage_loss(outputs, targets, num_boxes, stage_idx)
            for k, v in stage_losses.items():
                losses[f"{k}_{stage_idx}"] = v
        # Needs >=2 stages to have a teacher distinct from the student.
        if self.distill_labels and len(outputs_list) > 1:
            losses.update(self._distill_labels(outputs_list))
        return losses

    @staticmethod
    def reduce_num_boxes(num_boxes: int, device: torch.device) -> float:
        """Turn a raw GT box count into the normalizer, DDP-reduced (DETR
        convention): all-reduce-sum then ÷ world_size → the per-rank-average
        count over the global batch, clamped ≥ 1.

        The trainer passes the count summed over all grad-accum micro-batches,
        so the normalizer is the *effective-batch* count and accumulation
        matches a real batch.
        """
        num_boxes_t = torch.as_tensor([float(num_boxes)], dtype=torch.float32, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes_t)
            num_boxes_t = num_boxes_t / torch.distributed.get_world_size()
        return float(torch.clamp(num_boxes_t, min=1).item())

    def _single_stage_loss(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        num_boxes: float,
        stage_idx: int = 0,
    ) -> dict[str, Tensor]:
        pred_logits = outputs["pred_logits"]  # (B, N, K)
        pred_boxes = outputs["pred_boxes"]  # (B, N, 4) absolute xyxy

        indices = self._hungarian_match(pred_logits, pred_boxes, targets, stage_idx)

        loss_ce = self._loss_labels(pred_logits, pred_boxes, targets, indices, num_boxes)
        loss_bbox, loss_giou = self._loss_boxes(pred_boxes, targets, indices, num_boxes)

        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    @torch.no_grad()
    def match(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        stage_idx: int = 0,
    ) -> list[tuple[Tensor, Tensor]]:
        """Run Hungarian matching on a single stage's outputs."""
        return self._hungarian_match(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            targets,
            stage_idx,
        )

    def denoising_loss(
        self,
        dn: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        num_boxes: float | None = None,
    ) -> dict[str, Tensor]:
        """Box-only DN loss (L1 + GIoU) with deep supervision.

        Each DN query's target is the clean GT it was noised from — no
        matching needed. L1 is on image-size-normalized boxes (matching the
        main box loss); both are normalized by the number of DN queries.
        Returns ``loss_dn_bbox_{i}`` / ``loss_dn_giou_{i}`` per stage.

        ``num_boxes`` is the effective-batch GT count (same as
        :meth:`forward`). DN queries are a fixed multiple of GT
        (``num_dn = dn_groups × num_GT``), so the local DN count is rescaled to
        the effective GT count — making grad accumulation and DDP match a real
        batch (per-micro ``valid.sum()`` is also never DDP-reduced). ``None``
        falls back to this micro-batch's own DN count.
        """
        tgt = dn["tgt_boxes"].float()  # (B, M, 4)
        valid = dn["valid"]  # (B, M) bool
        img = torch.stack([t["image_size_xyxy"] for t in targets]).float().unsqueeze(1)  # (B, 1, 4)
        # Keep num_dn a 0-dim tensor (no host sync). When num_boxes is given,
        # scale local DN count by effective_GT / local_GT (= keep dn_groups, swap
        # the per-micro GT count for the effective one).
        if num_boxes is None:
            num_dn = valid.sum().clamp(min=1)
        else:
            local_gt = max(sum(len(t["labels"]) for t in targets), 1)
            num_dn = (valid.sum() * (num_boxes / local_gt)).clamp(min=1)
        vmask = valid.unsqueeze(-1).float()  # (B, M, 1)
        tgt_valid = tgt[valid]  # (V, 4) — loop-invariant

        losses: dict[str, Tensor] = {}
        for i, pred in enumerate(dn["pred_boxes"]):
            pred = pred.float()
            l1 = F.l1_loss(pred / img, tgt / img, reduction="none") * vmask
            # Empty valid -> empty gather -> giou sum is a clean 0; no guard needed.
            giou = paired_generalized_box_iou(pred[valid], tgt_valid)
            losses[f"loss_dn_bbox_{i}"] = l1.sum() / num_dn
            losses[f"loss_dn_giou_{i}"] = (1 - giou).sum() / num_dn
        return losses

    @torch.no_grad()
    def _hungarian_match(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
        stage_idx: int = 0,
    ) -> list[tuple[Tensor, Tensor]]:
        # Force fp32 — fp16 overflows on absolute-xyxy GIoU and focal log
        pred_logits = pred_logits.float()
        pred_boxes = pred_boxes.float()

        batch_size, _ = pred_logits.shape[:2]
        indices = []

        iou_floor = 0.0
        if self.cascade_iou_thresholds:
            idx = min(stage_idx, len(self.cascade_iou_thresholds) - 1)
            iou_floor = self.cascade_iou_thresholds[idx]

        for b in range(batch_size):
            tgt_labels = targets[b]["labels"]
            tgt_boxes_xyxy = targets[b]["boxes_xyxy"]  # absolute xyxy

            if tgt_labels.shape[0] == 0:
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=pred_logits.device),
                        torch.tensor([], dtype=torch.long, device=pred_logits.device),
                    )
                )
                continue

            out_prob = pred_logits[b].sigmoid()
            alpha, gamma = self.focal_alpha, self.focal_gamma
            neg_cost = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]

            # L1 cost on image-size-normalized boxes
            image_size_xyxy = targets[b]["image_size_xyxy"]  # (4,)
            out_bbox_norm = pred_boxes[b] / image_size_xyxy.unsqueeze(0)
            tgt_bbox_norm = tgt_boxes_xyxy / targets[b]["image_size_xyxy_tgt"]
            cost_bbox = torch.cdist(out_bbox_norm, tgt_bbox_norm, p=1)

            # GIoU cost on absolute xyxy
            cost_giou = -generalized_box_iou(pred_boxes[b], tgt_boxes_xyxy)

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            if iou_floor > 0.0:
                iou = _pairwise_iou(pred_boxes[b], tgt_boxes_xyxy)
                cost = cost + (iou < iou_floor).float() * 1e6

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device),
                    torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device),
                )
            )
        return indices

    def _loss_labels(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> Tensor:
        """Classification loss — sum reduction, divided by num_boxes.

        ``focal`` uses one-hot targets. ``vfl``/``mal`` replace the positive
        target with the matched prediction's own IoU against its GT, so the
        score a query emits is trained to estimate how well it is localized.
        """
        pred_logits = pred_logits.float()
        batch_size, num_queries, num_classes = pred_logits.shape
        target_classes = torch.full(
            (batch_size, num_queries),
            num_classes,
            dtype=torch.long,
            device=pred_logits.device,
        )
        # IoU of each matched query's box against the GT it was matched to.
        # Detached: this is a *target*, not a path for box gradients (the box
        # is already supervised by L1 + GIoU).
        target_iou = pred_logits.new_zeros((batch_size, num_queries))
        needs_iou = self.cls_loss_type != "focal"
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.shape[0] == 0:
                continue
            target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]
            if needs_iou:
                target_iou[b, src_idx] = _paired_iou(
                    pred_boxes[b, src_idx].detach().float(),
                    targets[b]["boxes_xyxy"][tgt_idx].float(),
                ).clamp(0.0, 1.0)

        # Flatten to (B*N, K) — matching original's flatten(0, 1)
        src_logits = pred_logits.flatten(0, 1)
        target_classes_flat = target_classes.flatten(0, 1)

        labels = torch.zeros_like(src_logits)
        pos_inds = (target_classes_flat != num_classes).nonzero(as_tuple=True)[0]
        pos_cls = target_classes_flat[pos_inds]

        if self.cls_loss_type == "focal":
            labels[pos_inds, pos_cls] = 1.0
            loss = sigmoid_focal_loss(
                src_logits, labels, alpha=self.focal_alpha, gamma=self.focal_gamma
            )
        else:
            # ``labels`` doubles as the soft regression target *and* (via its
            # own positive mask) the positive/negative selector, so a matched
            # query with IoU 0 still counts as a positive slot rather than
            # silently falling through to the negative branch.
            q = target_iou.flatten(0, 1)[pos_inds]
            is_pos = torch.zeros_like(src_logits, dtype=torch.bool)
            is_pos[pos_inds, pos_cls] = True
            if self.cls_loss_type == "mal":
                labels[pos_inds, pos_cls] = q.pow(self.mal_gamma)
                loss = matchability_aware_loss(
                    src_logits,
                    labels,
                    is_pos,
                    gamma=self.mal_gamma,
                    alpha=self.mal_alpha,
                )
            else:
                labels[pos_inds, pos_cls] = q
                loss = varifocal_loss(
                    src_logits,
                    labels,
                    is_pos,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )
        return loss / num_boxes

    def _loss_boxes(
        self,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> tuple[Tensor, Tensor]:
        """L1 (on normalized boxes) + GIoU (on absolute xyxy)."""
        pred_boxes = pred_boxes.float()
        src_list, tgt_list, tgt_norm_list, src_norm_list = [], [], [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.shape[0] == 0:
                continue
            src_list.append(pred_boxes[b, src_idx])
            tgt_list.append(targets[b]["boxes_xyxy"][tgt_idx])
            # Normalize by image size for L1 (matches original)
            tgt_norm_list.append(
                targets[b]["boxes_xyxy"][tgt_idx] / targets[b]["image_size_xyxy_tgt"][tgt_idx]
            )
            image_size = targets[b]["image_size_xyxy"]
            src_norm_list.append(pred_boxes[b, src_idx] / image_size.unsqueeze(0))

        if not src_list:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        src_boxes = torch.cat(src_list, dim=0)
        tgt_boxes = torch.cat(tgt_list, dim=0)
        src_norm = torch.cat(src_norm_list, dim=0)
        tgt_norm = torch.cat(tgt_norm_list, dim=0)

        # L1 on normalized boxes
        loss_bbox = F.l1_loss(src_norm, tgt_norm, reduction="none").sum() / num_boxes

        # GIoU on absolute xyxy (paired, not full NxN matrix)
        loss_giou = (1 - paired_generalized_box_iou(src_boxes, tgt_boxes)).sum() / num_boxes

        return loss_bbox, loss_giou


# ---------------------------------------------------------------------------
# Focal loss (matches fvcore's sigmoid_focal_loss with reduction="sum")
# ---------------------------------------------------------------------------


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Sigmoid focal loss — sum over all elements."""
    p = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    total: Tensor = loss.sum()
    return total


def matchability_aware_loss(
    inputs: Tensor,
    target_score: Tensor,
    is_pos: Tensor,
    gamma: float = 1.5,
    alpha: float | None = None,
) -> Tensor:
    """Matchability-Aware Loss (DEIM, CVPR 2025) — sum over all elements.

        MAL(p, q, y) = -q^g log(p) - (1 - q^g) log(1 - p)    y = 1
                     = -a p^g log(1 - p)                      y = 0

    ``alpha`` (``a``) is ``None`` in DEIM's default, i.e. 1.0.

    ``target_score`` already carries ``q^gamma`` at the positive slots and 0
    everywhere else, so the positive branch is plain BCE against it. The
    negative branch is focal with alpha folded away (DEIM drops VFL's class
    balance term). The ``p^gamma`` modulator is detached, matching the
    reference implementation: it is a weight, not a gradient path.

    Versus VFL, raising the target to ``gamma`` pulls the target down hard for
    low-IoU matches, so a confident prediction on a poorly localized box is
    penalised steeply instead of being left nearly unchanged.
    """
    p = inputs.sigmoid()
    ce_pos = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    ce_neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    neg = p.detach().pow(gamma) * ce_neg
    if alpha is not None:
        neg = alpha * neg
    loss = torch.where(is_pos, ce_pos, neg)
    total: Tensor = loss.sum()
    return total


def varifocal_loss(
    inputs: Tensor,
    target_score: Tensor,
    is_pos: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Varifocal loss (VarifocalNet) — sum over all elements.

        VFL(p, q, y) = -q (q log(p) + (1 - q) log(1 - p))   y = 1
                     = -alpha p^g log(1 - p)                 y = 0

    ``target_score`` carries the raw IoU ``q`` at positive slots. Kept beside
    :func:`matchability_aware_loss` as the ablation control — MAL is VFL with
    the target powered and the alpha dropped.
    """
    p = inputs.sigmoid()
    ce_pos = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    ce_neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    loss = torch.where(
        is_pos, target_score * ce_pos, alpha * p.detach().pow(gamma) * ce_neg
    )
    total: Tensor = loss.sum()
    return total


def _paired_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Element-wise IoU between paired (N, 4) xyxy boxes. Returns (N,)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------


def paired_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Element-wise GIoU between paired (N, 4) boxes. Returns (N,)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    return iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """GIoU between (N, 4) and (M, 4) boxes in xyxy absolute format. Returns (N, M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    return iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)


def _pairwise_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Standard IoU (N, 4) vs (M, 4) in xyxy. Returns (N, M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)
