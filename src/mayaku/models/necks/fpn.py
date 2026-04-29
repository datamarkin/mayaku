"""Feature Pyramid Network (FPN).

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.2:

* For each input level ``res{i}`` from a bottom-up :class:`Backbone`,
  build a 1x1 *lateral* conv (channels → ``out_channels``) and a 3x3
  *output* conv (``out_channels`` → ``out_channels``).
* Top-down pass: start at ``res5``'s lateral; for each lower level
  add nearest-neighbour upsampled higher level + lateral, then run
  the output conv.
* Optionally append ``LastLevelMaxPool`` (a 1x1 stride-2 max-pool on
  ``p5``) to produce ``p6`` — required for the in-scope FPN-RCNN
  configs.

Channel / stride table for the standard ``in_features=
("res2","res3","res4","res5")`` and ``LastLevelMaxPool`` top block:

| name | channels    | stride |
|------|-------------|-------:|
| p2   | out_channels|      4 |
| p3   | out_channels|      8 |
| p4   | out_channels|     16 |
| p5   | out_channels|     32 |
| p6   | out_channels|     64 |

``size_divisibility`` is the max *bottom-up* input stride (32 by
default), not the max output stride. This matches Detectron2's
behaviour (`spec §2.2`, `fpn.py:113`): the top block is a strided
pool that tolerates any input shape, so we don't need to pad to a
multiple of 64.
"""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from mayaku.config.schemas import FPNConfig
from mayaku.models.backbones._base import Backbone
from mayaku.models.backbones._frozen_bn import FrozenBatchNorm2d

__all__ = ["FPN", "LastLevelMaxPool", "build_fpn"]

NormChoice = Literal["", "BN", "GN", "FrozenBN"]
FuseType = Literal["sum", "avg"]


class LastLevelMaxPool(nn.Module):
    """Append ``p6`` to FPN outputs by max-pooling ``p5`` (`spec §2.2`).

    Used by the FPN-RCNN family. The pool kernel is ``1`` with stride
    ``2``, so ``p6`` has shape ``ceil(p5 / 2)`` per spatial dim — no
    padding constraint on the input. ``in_feature="p5"`` and
    ``num_levels=1`` are read by the parent :class:`FPN` to decide
    where to pull the input from and how many extra outputs to expect.
    """

    in_feature: str = "p5"
    num_levels: int = 1

    def forward(self, p5: Tensor) -> list[Tensor]:
        return [F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)]


class FPN(Backbone):
    """Feature Pyramid Network on top of any bottom-up :class:`Backbone`.

    Args:
        bottom_up: The wrapped backbone producing ``in_features``. Its
            ``output_shape()`` is consulted to size the lateral convs.
        in_features: Names from ``bottom_up.output_shape()`` to fuse.
            Order is **bottom-up** (e.g. ``("res2", "res3", "res4",
            "res5")``); the top-down loop iterates in reverse.
        out_channels: Channels for every FPN output (typically 256).
        norm: Optional norm applied after each lateral and output conv.
            ``""`` for none (the spec default for in-scope configs);
            ``"BN"``, ``"GN"``, or ``"FrozenBN"`` are supported.
        fuse_type: ``"sum"`` (default — element-wise add of lateral +
            upsampled higher level) or ``"avg"`` (divide the sum by 2,
            sometimes used for stability with very small features).
        top_block: Optional module producing extra coarser levels from
            the highest FPN output. Pass ``LastLevelMaxPool()`` for
            FPN-RCNN; ``None`` to skip ``p6``.
    """

    def __init__(
        self,
        bottom_up: Backbone,
        in_features: tuple[str, ...] = ("res2", "res3", "res4", "res5"),
        out_channels: int = 256,
        norm: NormChoice = "",
        fuse_type: FuseType = "sum",
        top_block: LastLevelMaxPool | None = None,
    ) -> None:
        super().__init__()
        if len(in_features) == 0:
            raise ValueError("FPN requires at least one input feature")
        if fuse_type not in ("sum", "avg"):
            raise ValueError(f"fuse_type must be 'sum' or 'avg'; got {fuse_type!r}")
        bu_shapes = bottom_up.output_shape()
        for name in in_features:
            if name not in bu_shapes:
                raise ValueError(
                    f"in_feature {name!r} not in bottom_up output_shape ({tuple(bu_shapes)})"
                )

        self.bottom_up = bottom_up
        self.in_features = tuple(in_features)
        self.out_channels = out_channels
        self.fuse_type: FuseType = fuse_type
        self.top_block = top_block
        self._norm: NormChoice = norm

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for name in self.in_features:
            in_channels = bu_shapes[name].channels
            self.lateral_convs.append(_conv1x1_with_norm(in_channels, out_channels, norm))
            self.output_convs.append(_conv3x3_with_norm(out_channels, out_channels, norm))

        # Output naming: res{i} -> p{i}; top block extends with p{i+1}.
        out_names = [f"p{name[len('res') :]}" for name in self.in_features]
        out_strides = [bu_shapes[name].stride for name in self.in_features]
        if top_block is not None:
            for _ in range(top_block.num_levels):
                last_stride = out_strides[-1]
                out_names.append(f"p{int(out_names[-1][1:]) + 1}")
                out_strides.append(last_stride * 2)
        self._out_features = tuple(out_names)
        self._out_feature_channels = {n: out_channels for n in out_names}
        self._out_feature_strides = dict(zip(out_names, out_strides, strict=False))
        # The bottom-up's largest stride dictates how much padding the
        # input needs; the top block tolerates anything (strided pool).
        self._size_divisibility = max(bu_shapes[name].stride for name in self.in_features)

        self._init_weights()

    # ------------------------------------------------------------------
    # Backbone overrides
    # ------------------------------------------------------------------

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        bu = self.bottom_up(x)

        # Top-down: start from the deepest input.
        # We index laterals/outputs from deepest (last) to shallowest (first).
        prev = self.lateral_convs[-1](bu[self.in_features[-1]])
        results: list[Tensor] = [self.output_convs[-1](prev)]
        for i in range(len(self.in_features) - 2, -1, -1):
            lateral = self.lateral_convs[i](bu[self.in_features[i]])
            # Nearest-neighbour upsample, factor 2. Using
            # `interpolate(scale_factor=2)` is portable across CPU/MPS/CUDA.
            upsampled = F.interpolate(prev, scale_factor=2.0, mode="nearest")
            prev = lateral + upsampled
            if self.fuse_type == "avg":
                prev = prev * 0.5
            results.insert(0, self.output_convs[i](prev))

        if self.top_block is not None:
            top_input = results[-1]
            results.extend(self.top_block(top_input))

        return dict(zip(self._out_features, results, strict=True))

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Detectron2 uses Xavier init for FPN convs (`fpn.py:80-90`).
        # Plain Xavier on the conv weight; biases zeroed.
        for conv_list in (self.lateral_convs, self.output_convs):
            for module in conv_list:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Conv + norm helpers
# ---------------------------------------------------------------------------


def _make_norm(norm: NormChoice, channels: int) -> nn.Module | None:
    if norm == "":
        return None
    if norm == "BN":
        return nn.BatchNorm2d(channels)
    if norm == "FrozenBN":
        return FrozenBatchNorm2d(channels)
    if norm == "GN":
        # Spec §6.1 uses GN with 32 groups; clamp to channels for tiny
        # widths so the divisibility check passes.
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm {norm!r}; expected '', 'BN', 'GN', 'FrozenBN'")


def _conv1x1_with_norm(in_channels: int, out_channels: int, norm: NormChoice) -> nn.Module:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=norm == "")]
    n = _make_norm(norm, out_channels)
    if n is not None:
        layers.append(n)
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


def _conv3x3_with_norm(in_channels: int, out_channels: int, norm: NormChoice) -> nn.Module:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=norm == "")
    ]
    n = _make_norm(norm, out_channels)
    if n is not None:
        layers.append(n)
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


# ---------------------------------------------------------------------------
# Build factory from FPNConfig (Step 5)
# ---------------------------------------------------------------------------


def build_fpn(cfg: FPNConfig, bottom_up: Backbone, *, with_top_block: bool = True) -> FPN:
    """Construct an :class:`FPN` from a typed config + bottom-up backbone.

    ``with_top_block=True`` (default) appends :class:`LastLevelMaxPool`
    so the output set matches the FPN-RCNN convention ``(p2..p6)``.
    Pass ``False`` for retina-style configs that supply their own
    ``LastLevelP6P7`` (out of scope for v1; the kwarg is here so the
    plumbing doesn't have to change later).
    """
    norm = cfg.norm if cfg.norm in ("", "BN", "GN", "FrozenBN") else ""
    return FPN(
        bottom_up=bottom_up,
        in_features=cfg.in_features,
        out_channels=cfg.out_channels,
        norm=norm,  # type: ignore[arg-type]
        fuse_type=cfg.fuse_type,
        top_block=LastLevelMaxPool() if with_top_block else None,
    )
