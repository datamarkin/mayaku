"""Tests for :mod:`mayaku.models.backbones.convnext`.

Heavy module — instantiating ConvNeXt takes a moment. Where the test only
checks structural invariants, we use the lightest variant ("convnext_tiny")
and skip the forward pass. The mirror of :mod:`tests.unit.test_resnet_backbone`.
"""

from __future__ import annotations

import torch

from mayaku.models.backbones.convnext import (
    ConvNeXtBackbone,
    _remap_torchvision_convnext_state_dict,
)


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
    backbone = ConvNeXtBackbone(name="convnext_tiny", weights=None, weights_path=None)
    assert list(backbone._res_downs["res2"].parameters()) == []


def test_torchvision_remap_includes_downsamples() -> None:
    """torchvision-native ConvNeXt checkpoints (``features.X`` — what
    torchvision.models / torch.hub / lightly_train export) remap onto the
    carved layout, *including the downsample layers*.

    The Facebook-only path used to leave ``features.*`` keys untouched, so the
    downsamples (``_res_downs.resN``) silently went missing and the backbone
    kept random weights — exactly the kind of partial load that invalidates a
    backbone comparison. Pin the full mapping here.
    """
    src = {
        "features.0.0.weight": torch.zeros(128, 3, 4, 4),  # stem conv
        "features.0.1.weight": torch.zeros(128),  # stem LayerNorm2d
        "features.1.0.layer_scale": torch.zeros(128, 1, 1),  # res2 block0
        "features.1.0.block.0.weight": torch.zeros(128, 1, 7, 7),  # dwconv
        "features.1.0.block.3.weight": torch.zeros(512, 128),  # pwconv1 (Linear)
        "features.2.0.weight": torch.zeros(128),  # downsample LN -> res3
        "features.2.1.weight": torch.zeros(256, 128, 2, 2),  # downsample conv -> res3
        "features.7.0.block.0.weight": torch.zeros(1024, 1, 7, 7),  # res5 block0
        "classifier.2.weight": torch.zeros(1000, 1024),  # head -> passthrough
    }
    out = _remap_torchvision_convnext_state_dict(src)

    assert {"stem.0.weight", "stem.1.weight"} <= out.keys()
    assert "_res_stages.res2.0.layer_scale" in out
    assert "_res_stages.res2.0.block.0.weight" in out
    assert "_res_stages.res2.0.block.3.weight" in out
    assert "_res_stages.res5.0.block.0.weight" in out  # features.7 -> res5
    # the downsamples the old path dropped:
    assert "_res_downs.res3.0.weight" in out
    assert "_res_downs.res3.1.weight" in out
    # non-features keys pass through for the strict-load filter to drop:
    assert "classifier.2.weight" in out
    # nothing leaks under the raw torchvision prefix:
    assert not any(k.startswith("features.") for k in out)
