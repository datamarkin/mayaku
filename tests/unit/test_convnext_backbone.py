"""Tests for :mod:`mayaku.models.backbones.convnext`.

Heavy module — instantiating ConvNeXt takes a moment. Where the test only
checks structural invariants, we use the lightest variant ("convnext_tiny")
and skip the forward pass. The mirror of :mod:`tests.unit.test_resnet_backbone`.
"""

from __future__ import annotations

from mayaku.models.backbones.convnext import ConvNeXtBackbone


def test_res_downs_res2_is_parameterless_identity() -> None:
    """``_res_downs.res2`` is an :class:`~torch.nn.Identity` (no params).

    This invariant is load-bearing for the LLRD adapter in
    :mod:`mayaku.engine.optim`: our ``_res_downs.res2`` has no MMDet
    counterpart, so the adapter raises if it ever sees a parameter
    there. A regression that promotes ``_res_downs.res2`` to a real
    module with parameters would silently violate the adapter contract.
    Pin it here so the contract is checked at the source (backbone),
    not at the use site (optimizer).
    """
    backbone = ConvNeXtBackbone(name="convnext_tiny")
    assert list(backbone._res_downs["res2"].parameters()) == []
