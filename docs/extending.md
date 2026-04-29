# Extending Mayaku

How to add a backbone, head, dataset, augmentation, or export target. Each
recipe lists the files you'll touch, the contracts to satisfy, and the
test surface that proves the change is wired through.

The repo intentionally has *no string-keyed registries* (Detectron2's
`Registry` / `BACKBONE_REGISTRY` etc.). Adding a new component means
extending a typed `Literal`, wiring an `if`/`elif` in the relevant
factory, and writing a normal Python module — refactor-friendly, no
import-side-effects, fully visible to mypy.

## Add a backbone

1. **Implement** the `nn.Module` under
   `src/mayaku/models/backbones/<your_backbone>.py`. Subclass
   `mayaku.models.backbones._base.Backbone` and populate:

   ```python
   self._out_features = ("res2", "res3", "res4", "res5")
   self._out_feature_channels = {"res2": 256, ..., "res5": 2048}
   self._out_feature_strides  = {"res2": 4,   ..., "res5": 32}
   def forward(self, x: Tensor) -> dict[str, Tensor]: ...
   ```

   Override `size_divisibility` if your backbone needs alignment beyond
   the largest output stride (e.g. ViT patch size).

2. **Extend the schema** in `src/mayaku/config/schemas.py`:

   ```python
   BackboneName = Literal["resnet50", "resnet101", "resnext101_32x8d",
                          "your_backbone"]
   ```

   If the new backbone needs new knobs (groups, depth multiplier, …),
   add them to `BackboneConfig` rather than smuggling them through
   `name`. Keep field types tight (`Annotated[int, Field(gt=0)]`,
   `Literal[...]`) so configs fail at load time rather than at first
   forward.

3. **Wire it into the build dispatcher** under
   `src/mayaku/models/backbones/__init__.py` (look at how `resnet50`
   dispatches today and follow the same pattern). The detector
   factories pick up the new name through this path automatically.

4. **Test** under `tests/unit/test_backbones_<your_backbone>.py`:
   - Output shape against `Backbone.dummy_input(...)`.
   - `size_divisibility` matches what FPN expects.
   - Forward on the active `MAYAKU_DEVICE` (CPU + MPS + CUDA).
   - Optional: parameter count vs the upstream reference.

## Add an ROI head (e.g. a new mask head variant)

1. **Implement** the head under `src/mayaku/models/heads/<name>.py`.
   Mirror the existing `MaskRCNNConvUpsampleHead` shape:
   `forward(features, instances)` returns either a loss dict (training)
   or the predicted field on `instances` (inference). Use
   `mayaku.models.poolers.ROIPooler` for FPN-aware pooling.

2. **Extend the head config** in `src/mayaku/config/schemas.py`. If the
   new head replaces an existing one (e.g. a different mask head),
   widen the relevant `*Config` `Literal`. If it's a *new* head type,
   add a new `*Config` model and a corresponding optional field on
   `ModelConfig`, plus a `model_validator` that ensures it's set
   exactly when the meta-architecture asks for it.

3. **Wire it into `StandardROIHeads`** at
   `src/mayaku/models/roi_heads/standard.py` — that's the dispatcher
   that builds box / mask / keypoint heads from the config. New head
   types extend the existing branch ladder.

4. **Test** parity with the head it replaces (loss values within fp32
   tolerance for a fixed-seed forward) and ensure the COCO evaluator
   still ingests its outputs (`tests/unit/test_evaluator.py` is the
   end-to-end exerciser).

## Add a dataset

The contract is the Detectron2 dataset-dict idiom: each sample is a
`dict[str, Any]` with `image_id`, `file_name`, `height`, `width`, and
`annotations: list[dict]`.

1. **Loader** under `src/mayaku/data/datasets/<name>.py` exposes a
   function `load_<name>(json_path, image_root, metadata) -> list[dict]`.
   `load_coco_json` in `coco.py` is the reference shape.

2. **Metadata** is a plain `mayaku.data.catalog.Metadata` — class
   names, evaluator type (`"coco"` is currently the only one), and
   keypoint connectivity if applicable. Build it via
   `build_<name>_metadata(...)` and register it with
   `MetadataCatalog.set(name, metadata)`.

3. **Mapper compatibility** — `DatasetMapper` consumes the
   dataset-dict shape directly, so if your loader produces the standard
   keys you don't need a custom mapper.

4. **Test** that `load_<name>(...)` round-trips through `DatasetMapper`
   and `AspectRatioGroupedDataset` to a batch the detector accepts. A
   smoke training step (a few iterations on a tiny config) catches
   most wiring bugs.

5. **Set `model.roi_heads.num_classes` in your YAML** to match the
   dataset's class count. The schema default is 80 (COCO); the value
   is *not* auto-derived from the dataset metadata. Mismatches don't
   raise — they just train an over-/under-sized classifier that
   underperforms silently.

6. **Pass `--pretrained-backbone`** to `mayaku train` for fine-tuning
   on a custom dataset. The default `backbone.freeze_at=2` freezes
   the ResNet stem + res2; that's only useful if those stages start
   from ImageNet weights. The CLI warns if you skip the flag while
   `freeze_at >= 1`, because the silent "freeze random init" path
   produces a model that detects nothing at inference.

## Add an augmentation

1. **Deterministic transform** under
   `src/mayaku/data/transforms/geometry.py` (or `pixel.py` if you add
   one). Subclass `mayaku.data.transforms.base.Transform` and implement
   `apply_image`, `apply_box`, `apply_coords`, `apply_segmentation` as
   needed.

2. **Augmentation wrapper** under
   `src/mayaku/data/transforms/augmentation.py` — subclass
   `Augmentation` and implement `get_transform(image)` returning the
   deterministic transform. The split keeps random sampling (the
   augmentation) separate from the recorded operation (the transform);
   `TransformList.apply_keypoints` handles flip-pair swaps centrally so
   the augmentation never has to know about keypoint indices.

3. **Wire it into `cli/train.py`** if it should be on by default;
   user scripts can pass any list of augmentations to `DatasetMapper`
   directly.

4. **Test** that the transform's box / mask / keypoint application
   commutes with the image transformation and that keypoint
   flip-pairs work for both vertical and horizontal flips.

## Add an export target

1. **Implement** `src/mayaku/inference/export/<target>.py` exposing a
   class that satisfies the `Exporter` protocol from `base.py`:

   ```python
   class YourExporter:
       name: str = "your_target"
       def export(self, model, sample, out_path, **opts) -> ExportResult: ...
       def parity_check(self, model, exported_path, sample, *, atol, rtol)
           -> ParityResult: ...
   ```

   Re-export it from `inference/export/__init__.py`. The convention is
   to serialise the **Graph A body only** (backbone + FPN producing
   `{p2..p6}`); per-image Python glue stays out of the exported graph
   — see [`docs/export/`](export/) for what each existing target does
   and why.

2. **Optional dependency** in `pyproject.toml`:

   ```toml
   [project.optional-dependencies]
   your_target = ["your-runtime>=X.Y"]
   ```

   Use a PEP 508 marker if the dependency only exists on certain
   platforms — the `tensorrt` extra is the example
   (`tensorrt>=10.0; sys_platform != 'darwin'`).

3. **CLI dispatch** — add a branch in `src/mayaku/cli/export.py`:
   `_AVAILABLE_TARGETS = (..., "your_target")` and an `elif target ==`
   case that returns `YourExporter().export(model, sample, output)`.

4. **pytest marker** in `pyproject.toml` (`[tool.pytest.ini_options]
   markers`) and the auto-skip in `tests/conftest.py`. Mirror how the
   `openvino` or `tensorrt` markers are wired.

5. **Tests** under `tests/unit/test_<target>_export.py`:
   - Export writes the artefact and the file is non-empty.
   - Parity check passes within whatever tolerance is honest for the
     runtime (1e-3 for fp32 paths like ONNX/CPU; 1e-2 if the runtime
     uses fused / lower-precision kernels).
   - CLI dispatch reaches the new exporter.

   Mark all of them with `pytest.mark.<your_target>` so they auto-skip
   when the dependency isn't installed. Heavy builds should also wear
   `pytest.mark.slow` (TensorRT does this — engine builds take a few
   seconds each).

If a best-effort target turns out to require model-distorting changes
to make work, write `docs/decisions/NNN-drop-<target>.md` describing
the specific blocker and drop the target from CI rather than
contorting the model.

## Add a config field

Configs are pydantic v2 models in `src/mayaku/config/schemas.py`.
Adding a field is just adding an attribute with a typed default:

```python
class RPNConfig(_BaseModel):
    ...
    your_new_knob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
```

The model is `frozen=True`, `extra="forbid"`, `validate_default=True`
(via `_BaseModel`), so:
- Defaults are validated at import time (catches typos in the schema
  itself).
- YAML configs that mention an unknown field fail at load with a clear
  `ValidationError` instead of silently ignoring it.
- `cfg.model_copy(update={"rpn": cfg.rpn.model_copy(update={"your_new_knob":
  0.7})})` is the canonical way to derive a variant.

## Quality gates

Per `03_d2_reimplementation_prompt.md` §"Quality gates per phase",
every change ships with:

1. `ruff check src tests` clean.
2. `mypy --strict` clean.
3. `MAYAKU_DEVICE={cpu,mps,cuda} pytest -q` green on each available
   backend.
4. Public API has docstrings (one-line minimum; module-level docstrings
   carry the *why*).
5. `PROJECT_STATUS.md` updated with what was built and which backends
   it was tested on.
6. Any new MPS issue goes into `MPS_ISSUES_DISCOVERED.md` with
   `file.py:line`, symptom, and workaround.

Cross-machine verification is manual — there is no CI. Run the suite
on each physical host before claiming a step done.
