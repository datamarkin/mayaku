"""The ``mayaku`` command line: argument plumbing over the Python API.

    mayaku train [CONFIG] [--weights W] --annotations A --images I
                 [--val-annotations VA --val-images VI] [--output DIR]
                 [--epochs N] [--size-budget S] [--set key=value ...]
                 [--device D] [--resume STATE]
    mayaku eval WEIGHTS --annotations A --images I [--output DIR] [--device D]
    mayaku predict WEIGHTS IMAGE [--conf C] [--output FILE] [--device D]
    mayaku export TARGET WEIGHTS [--output FILE]
    mayaku download [NAME] [--list] [--all]

``--set`` takes any config field by its dotted path, e.g.
``--set train.lr=0.005 --set model.tier=s``; the value is read as YAML.
WEIGHTS is a checkpoint, a hosted model name, or (eval, predict) an exported
artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pydantic
import typer

from mayaku.api import evaluate, train
from mayaku.config import parse_assignments
from mayaku.engine.evaluation import rle
from mayaku.inference import Predictor, from_pretrained
from mayaku.inference.export import TARGETS
from mayaku.utils.download import (
    DEFAULT_MANIFEST_URL,
    download_model,
    list_models,
    resolve_weights,
)

app = typer.Typer(
    name="mayaku",
    help="Train, evaluate, run and export mayaku detectors.",
    no_args_is_help=True,
    add_completion=False,
)

_WEIGHTS_HELP = "A checkpoint, a hosted model name, or an exported artifact."
_JSON = {"exists": True, "dir_okay": False}
_DIR = {"exists": True, "file_okay": False}


@app.command("train")
def _train(
    config: Path | None = typer.Argument(
        None, exists=True, dir_okay=False,
        help="YAML config. Optional with --weights, which then defines the architecture."),
    weights: str | None = typer.Option(
        None, "--weights", help="Checkpoint or hosted model name to warm-start from."),
    annotations: Path = typer.Option(
        ..., "--annotations", **_JSON, help="Train COCO annotation JSON."),
    images: Path = typer.Option(..., "--images", **_DIR, help="Train image directory."),
    val_annotations: Path | None = typer.Option(
        None, "--val-annotations", **_JSON,
        help="Validation COCO JSON; enables per-epoch evaluation."),
    val_images: Path | None = typer.Option(
        None, "--val-images", **_DIR, help="Validation images."),
    output: Path | None = typer.Option(
        None, "--output", file_okay=False, help="Run directory; default ./runs/<name>."),
    epochs: int | None = typer.Option(None, "--epochs", min=1, help="Sets train.epochs."),
    size_budget: int | None = typer.Option(
        None, "--size-budget", help="Sets input.size_budget: the canvas's square-equivalent side."),
    assignments: list[str] = typer.Option(
        [], "--set", help="Set a config field: --set train.lr=0.005 (repeatable)."),
    device: str = typer.Option("auto", "--device", help="cuda, mps or cpu; default auto."),
    num_gpus: int = typer.Option(1, "--num-gpus", min=1, help="GPUs to train on."),
    resume: Path | None = typer.Option(
        None, "--resume", exists=True,
        help="Continue a run from its state.pt (or its train/ directory)."),
) -> None:
    """Train a detector; the command-line form of `mayaku.train`."""
    try:
        overrides = parse_assignments(assignments)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--set") from exc
    try:
        result = train(
            config, weights=weights,
            train_annotations=annotations, train_images=images,
            val_annotations=val_annotations, val_images=val_images,
            output_dir=output, size_budget=size_budget, num_epochs=epochs,
            overrides=overrides, device=device, num_gpus=num_gpus, resume=resume,
        )
    except pydantic.ValidationError as exc:     # a config value the schema rejects
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"final weights: {result['final_weights']}")


@app.command("eval")
def _eval(
    weights: str = typer.Argument(..., help=_WEIGHTS_HELP),
    annotations: Path = typer.Option(..., "--annotations", **_JSON, help="COCO annotation JSON."),
    images: Path = typer.Option(..., "--images", **_DIR, help="Image directory."),
    output: Path | None = typer.Option(
        None, "--output", file_okay=False, help="Directory to write metrics.json to."),
    device: str = typer.Option("auto", "--device", help="cuda, mps or cpu; default auto."),
) -> None:
    """COCO metrics on a split; the command-line form of `mayaku.evaluate`."""
    metrics = evaluate(weights, annotations=annotations, images=images,
                       output_dir=output, device=device)
    typer.echo(json.dumps(metrics, indent=2))


@app.command("predict")
def _predict(
    weights: str = typer.Argument(..., help=_WEIGHTS_HELP),
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    conf: float | None = typer.Option(
        None, "--conf", help="Score threshold; default the model's recorded one."),
    output: Path | None = typer.Option(None, "--output", help="Write the JSON here."),
    device: str = typer.Option("auto", "--device", help="cuda, mps or cpu; default auto."),
) -> None:
    """Detect objects in one image and print (or write) them as JSON."""
    runner = from_pretrained(weights, device=device)
    dets = runner(image, conf)
    payload = {"image": str(image), "detections": _to_json(dets, runner.class_names)}
    text = json.dumps(payload, indent=2)
    if output is None:
        typer.echo(text)
    else:
        output.write_text(text)


@app.command("export")
def _export(
    target: str = typer.Argument(..., help="Deployment target: onnx."),
    weights: str = typer.Argument(..., help="A checkpoint or a hosted model name."),
    output: Path | None = typer.Option(None, "--output", help="Artifact path."),
) -> None:
    """Export a trained model to a deployment target, sidecar embedded."""
    if target not in TARGETS:
        raise typer.BadParameter(f"unknown target {target!r}; available: {', '.join(TARGETS)}")
    path = resolve_weights(weights)
    predictor = Predictor.from_checkpoint(path, device="cpu")
    typer.echo(str(predictor.export(target, output or path.with_suffix(TARGETS[target]))))


@app.command("download")
def _download(
    name: str | None = typer.Argument(None, help="Model name from the hosted manifest."),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Download directory; default the current directory."),
    manifest_url: str = typer.Option(
        DEFAULT_MANIFEST_URL, "--manifest-url", help="Manifest URL, for a mirror."),
    list_models_: bool = typer.Option(False, "--list", help="List the hosted models."),
    download_all: bool = typer.Option(False, "--all", help="Fetch every hosted model."),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip the SHA256 check."),
) -> None:
    """Fetch hosted model checkpoints: one by name, or --all; --list shows them."""
    if list_models_ or download_all:
        index = list_models(manifest_url=manifest_url)
        if list_models_:
            for task in sorted(index):
                typer.echo(f"{task}:\n" + "".join(f"  {n}\n" for n in index[task]))
            return
        names = [n for task in index.values() for n in task]
    elif name is not None:
        names = [name]
    else:
        raise typer.BadParameter("pass a model name, --list or --all")
    for n in names:
        path = download_model(n, cache_dir=cache_dir, manifest_url=manifest_url,
                              verify_sha256=not no_verify)
        typer.echo(f"{n}: {path}")


def _to_json(dets, class_names: list[str]) -> list[dict[str, Any]]:
    """`Detections` as one JSON object per detection: an xyxy box in the
    image's pixels, keypoints as (x, y, visibility), a mask as COCO RLE."""
    out = []
    for k in range(len(dets)):
        c = int(dets.labels[k])
        d: dict[str, Any] = {"class_id": c, "class": class_names[c],
                             "score": round(float(dets.scores[k]), 4),
                             "box_xyxy": [round(v, 2) for v in dets.boxes[k].tolist()]}
        if dets.keypoints is not None:
            d["keypoints"] = [[round(v, 2) for v in p] for p in dets.keypoints[k].tolist()]
        if dets.masks is not None:
            d["segmentation"] = rle(dets.masks[k])
        out.append(d)
    return out


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
