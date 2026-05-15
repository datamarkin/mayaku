"""End-to-end tests for the ``mayaku train`` auto-config wiring.

Verifies that the auto-config layer in :func:`mayaku.cli.train.run_train`:

* Runs when ``auto_config.enabled`` is true and the dataset is large
  enough to derive a recipe.
* Preserves YAML fields the user explicitly set.
* Skips entirely when ``auto_config.enabled`` is false.
* Skips on datasets below the ``MIN_IMAGES_FOR_AUTO_CONFIG`` threshold.

Uses a 20-image synthetic COCO fixture with ``num_workers=0`` so the
test stays free of the multiprocessing-pickle issues that affect the
multi-worker tests on some hosts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mayaku.cli._factory import build_detector
from mayaku.cli.train import run_train
from mayaku.config import (
    AutoConfig,
    BackboneConfig,
    DataLoaderConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    RPNConfig,
    SolverConfig,
    dump_yaml,
    load_yaml,
)


def _fake_coco(num_images: int, num_classes: int, tmp_path: Path) -> dict[str, Path]:
    """Build an N-image synthetic COCO dataset for auto-config exercise."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    images = []
    annotations = []
    ann_id = 1
    for i in range(num_images):
        rgb = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        fname = f"img_{i}.png"
        Image.fromarray(rgb).save(images_dir / fname)
        images.append({"id": i, "file_name": fname, "height": 64, "width": 64})
        # Two boxes per image — one of class 0 (common), one rotating
        # across the rest. Total of ~num_images*2 boxes drives the
        # k-means anchor path on >50 boxes.
        annotations.append(
            {
                "id": ann_id,
                "image_id": i,
                "category_id": 1,
                "bbox": [5.0, 5.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            }
        )
        ann_id += 1
        annotations.append(
            {
                "id": ann_id,
                "image_id": i,
                "category_id": 1 + (i % max(1, num_classes - 1)),
                "bbox": [25.0, 25.0, 30.0, 15.0],
                "area": 450.0,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    coco = {
        "images": images,
        "categories": [
            {"id": c + 1, "name": f"c{c}", "supercategory": "thing"} for c in range(num_classes)
        ],
        "annotations": annotations,
    }
    json_path = tmp_path / "gt.json"
    json_path.write_text(json.dumps(coco))
    return {"images": images_dir, "json": json_path}


def _base_cfg(num_classes: int = 5, *, auto_config_enabled: bool = True) -> MayakuConfig:
    """Tiny detector config that builds quickly on CPU."""
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
            backbone=BackboneConfig(name="resnet50", freeze_at=2, norm="FrozenBN"),
            rpn=RPNConfig(
                pre_nms_topk_train=100,
                pre_nms_topk_test=50,
                post_nms_topk_train=20,
                post_nms_topk_test=10,
                batch_size_per_image=16,
            ),
            roi_heads=ROIHeadsConfig(num_classes=num_classes, batch_size_per_image=8),
            roi_box_head=ROIBoxHeadConfig(num_fc=1, fc_dim=32),
        ),
        solver=SolverConfig(
            base_lr=1e-4,
            momentum=0.0,
            ims_per_batch=1,
            max_iter=5,
            warmup_iters=1,
            warmup_factor=0.5,
            steps=(1,),
            checkpoint_period=2,
        ),
        # num_workers=0 sidesteps multiprocessing pickle issues that
        # bite memoryview-backed SerializedList on some hosts.
        dataloader=DataLoaderConfig(num_workers=0),
        auto_config=AutoConfig(enabled=auto_config_enabled),
    )


def _prebuild_weights(cfg: MayakuConfig, path: Path) -> None:
    """Save a state-dict matching ``cfg`` for the no-pretrain warning path."""
    model = build_detector(cfg)
    torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# Auto-config runs end-to-end
# ---------------------------------------------------------------------------


def test_auto_config_overrides_num_classes_and_anchors_via_yaml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """30-image dataset with auto-config on → resolved config has the
    auto-derived num_classes and anchor sizes (k-means output).

    30 images × 2 boxes = 60 boxes, above MIN_BOXES_FOR_ANCHOR_TUNE.
    """
    fixture = _fake_coco(num_images=30, num_classes=4, tmp_path=tmp_path)

    cfg = _base_cfg(num_classes=99)  # wrong on purpose; auto-config should fix it
    cfg_path = tmp_path / "cfg.yaml"
    dump_yaml(cfg, cfg_path)
    # Dumping writes EVERY field — so by user-set-paths semantics, the
    # whole config is "set." To exercise auto-config, write a minimal
    # YAML that omits num_classes and anchor_generator so they're
    # eligible for override.
    minimal = {
        "model": {
            "meta_architecture": "faster_rcnn",
            "backbone": {"name": "resnet50", "freeze_at": 2, "norm": "FrozenBN"},
            "rpn": {
                "pre_nms_topk_train": 100,
                "pre_nms_topk_test": 50,
                "post_nms_topk_train": 20,
                "post_nms_topk_test": 10,
                "batch_size_per_image": 16,
            },
            "roi_heads": {"batch_size_per_image": 8},  # NB: no num_classes
            "roi_box_head": {"num_fc": 1, "fc_dim": 32},
        },
        "solver": {
            "base_lr": 1e-4,  # user-set, must survive
            "momentum": 0.0,
            "ims_per_batch": 1,
            "max_iter": 5,
            "warmup_iters": 0,
            "warmup_factor": 0.5,
            "steps": [3],
            "checkpoint_period": 5,
        },
        "dataloader": {"num_workers": 0},
        "auto_config": {"enabled": True},
    }
    import yaml as _yaml

    cfg_path.write_text(_yaml.safe_dump(minimal))

    out = tmp_path / "run"
    # We don't care about the train loop succeeding; we care that
    # auto-config rewrote the cfg before model build. max_iter=1 keeps
    # the actual training cheap.
    with pytest.warns(UserWarning, match="freeze_at"):
        run_train(
            cfg_path,
            coco_gt_json=fixture["json"],
            image_root=fixture["images"],
            output_dir=out,
            device="cpu",
            max_iter=5,
        )

    # Resolved config got dumped — read it back and confirm the
    # auto-config decisions landed.
    resolved = load_yaml(out / "config.yaml")
    # num_classes auto-derived from the dataset (4 categories).
    assert resolved.model.roi_heads.num_classes == 4
    # Anchors were k-means-tuned: sizes differ from schema default
    # ((32,), (64,), (128,), (256,), (512,)).
    assert resolved.model.anchor_generator.sizes != (
        (32,),
        (64,),
        (128,),
        (256,),
        (512,),
    )
    # User-set base_lr SURVIVED.
    assert resolved.solver.base_lr == 1e-4
    # max_iter was CLI-overridden to 5 — preserved.
    assert resolved.solver.max_iter == 5

    # The [auto-config] report fired in stdout.
    captured = capsys.readouterr().out
    assert "[auto-config]" in captured


def test_auto_config_skipped_when_disabled(tmp_path: Path) -> None:
    """``auto_config.enabled=False`` short-circuits the whole pass."""
    fixture = _fake_coco(num_images=20, num_classes=4, tmp_path=tmp_path)

    minimal = {
        "model": {
            "meta_architecture": "faster_rcnn",
            "backbone": {"name": "resnet50", "freeze_at": 2, "norm": "FrozenBN"},
            "rpn": {
                "pre_nms_topk_train": 100,
                "pre_nms_topk_test": 50,
                "post_nms_topk_train": 20,
                "post_nms_topk_test": 10,
                "batch_size_per_image": 16,
            },
            # roi_heads.num_classes deliberately omitted — schema default
            # is 80. With auto-config OFF, num_classes stays at 80.
            "roi_heads": {"batch_size_per_image": 8},
            "roi_box_head": {"num_fc": 1, "fc_dim": 32},
        },
        "solver": {
            "base_lr": 1e-4,
            "momentum": 0.0,
            "ims_per_batch": 1,
            "max_iter": 5,
            "warmup_iters": 0,
            "warmup_factor": 0.5,
            "steps": [3],
            "checkpoint_period": 5,
        },
        "dataloader": {"num_workers": 0},
        "auto_config": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    import yaml as _yaml

    cfg_path.write_text(_yaml.safe_dump(minimal))

    out = tmp_path / "run"
    with pytest.warns(UserWarning, match="freeze_at"):
        run_train(
            cfg_path,
            coco_gt_json=fixture["json"],
            image_root=fixture["images"],
            output_dir=out,
            device="cpu",
            max_iter=5,
        )

    resolved = load_yaml(out / "config.yaml")
    # Schema default survived — auto-config didn't run.
    assert resolved.model.roi_heads.num_classes == 80
    # Anchors are still the schema default ladder.
    assert resolved.model.anchor_generator.sizes == ((32,), (64,), (128,), (256,), (512,))


def test_auto_config_skipped_below_min_images_threshold(tmp_path: Path) -> None:
    """Tiny datasets (<10 images) short-circuit without modifying cfg."""
    fixture = _fake_coco(num_images=3, num_classes=2, tmp_path=tmp_path)

    minimal = {
        "model": {
            "meta_architecture": "faster_rcnn",
            "backbone": {"name": "resnet50", "freeze_at": 2, "norm": "FrozenBN"},
            "rpn": {
                "pre_nms_topk_train": 100,
                "pre_nms_topk_test": 50,
                "post_nms_topk_train": 20,
                "post_nms_topk_test": 10,
                "batch_size_per_image": 16,
            },
            "roi_heads": {"batch_size_per_image": 8},
            "roi_box_head": {"num_fc": 1, "fc_dim": 32},
        },
        "solver": {
            "base_lr": 1e-4,
            "momentum": 0.0,
            "ims_per_batch": 1,
            "max_iter": 5,
            "warmup_iters": 0,
            "warmup_factor": 0.5,
            "steps": [3],
            "checkpoint_period": 5,
        },
        "dataloader": {"num_workers": 0},
    }
    cfg_path = tmp_path / "cfg.yaml"
    import yaml as _yaml

    cfg_path.write_text(_yaml.safe_dump(minimal))

    out = tmp_path / "run"
    with pytest.warns(UserWarning, match="freeze_at"):
        run_train(
            cfg_path,
            coco_gt_json=fixture["json"],
            image_root=fixture["images"],
            output_dir=out,
            device="cpu",
            max_iter=5,
        )

    resolved = load_yaml(out / "config.yaml")
    # With only 3 images auto-config refuses to run; defaults unchanged.
    assert resolved.model.roi_heads.num_classes == 80
    assert resolved.model.anchor_generator.sizes == ((32,), (64,), (128,), (256,), (512,))
