"""Typer entry point for the ``mayaku`` console script.

The console-script entry in ``pyproject.toml`` (`mayaku =
"mayaku.cli.__main__:app"`) imports the :data:`app` symbol below.
Each subcommand is a thin wrapper around the corresponding
``cli/<name>.py`` ``run_*`` function so the implementation stays
testable in-process via direct calls (and the Typer layer is just
argument plumbing).

Subcommands:

* ``mayaku train [CONFIG] [--weights] --annotations --images [--val-annotations --val-images] [--output] [--device] [--epochs] [--num-gpus]``
* ``mayaku eval CONFIG --weights --annotations --images [--output] [--device]``
* ``mayaku predict CONFIG IMAGE [--weights] [--output] [--device]``
* ``mayaku export TARGET CONFIG --weights --output``
  (TARGET ∈ ``onnx`` | ``coreml`` | ``openvino`` | ``tensorrt``)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer

from mayaku.backends.mps import track_mps_fallbacks
from mayaku.cli.download import render_index, run_download
from mayaku.cli.eval import run_eval
from mayaku.cli.export import run_export
from mayaku.cli.predict import run_predict
from mayaku.config.schemas import DeviceSetting
from mayaku.utils.download import DEFAULT_MANIFEST_URL

app = typer.Typer(
    name="mayaku",
    help="Backend-portable detection / segmentation / keypoint CLI.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("train")
def _train(
    config: str | None = typer.Argument(
        None,
        help="YAML path or bundled config name. Omit when --weights defines the architecture.",
    ),
    weights: str | None = typer.Option(
        None,
        "--weights",
        help=(
            "Bundled model name or a .pth path. Defines the architecture when "
            "CONFIG is omitted, and seeds training (the class-specific head "
            "re-initialises when the dataset's class count differs)."
        ),
    ),
    annotations: Path | None = typer.Option(
        None,
        "--annotations",
        exists=True,
        dir_okay=False,
        help="Train COCO annotation JSON (with --images).",
    ),
    images: Path | None = typer.Option(
        None,
        "--images",
        exists=True,
        file_okay=False,
        help="Train image directory (with --annotations).",
    ),
    val_annotations: Path | None = typer.Option(
        None,
        "--val-annotations",
        exists=True,
        dir_okay=False,
        help="Val COCO annotation JSON for final eval.",
    ),
    val_images: Path | None = typer.Option(
        None, "--val-images", exists=True, file_okay=False, help="Val image directory."
    ),
    output: Path | None = typer.Option(
        None, "--output", file_okay=False, help="Run directory. Default ./runs/<config_stem>/."
    ),
    device: str = typer.Option("auto", "--device", help="cpu/mps/cuda; default = auto"),
    epochs: int | None = typer.Option(
        None, "--epochs", help="Number of passes over the dataset (overrides the recipe)."
    ),
    num_gpus: int = typer.Option(
        1,
        "--num-gpus",
        min=1,
        help=(
            "Number of GPUs to train on (DDP). Default 1. Multiply "
            "`solver.base_lr` by --num-gpus (linear scaling rule) when scaling "
            "up. MPS is single-device only."
        ),
    ),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        exists=True,
        dir_okay=False,
        readable=True,
        help=(
            "Resume training from a `model_iter_*.pth` checkpoint: restores "
            "weights + optimizer + LR-schedule position (+ EMA shadow) and "
            "continues at the checkpoint's iteration. Use the same CONFIG the "
            "checkpoint was trained with. Mutually exclusive with --weights."
        ),
    ),
) -> None:
    """Train a detector — the CLI mirror of :func:`mayaku.train`.

    Define the model with ``CONFIG`` (YAML path or bundled name) or
    ``--weights`` (bundled name or trained .pth). Point at the dataset
    with ``--annotations`` (a COCO JSON) and ``--images`` (its image
    directory). Picks the best checkpoint, runs final eval when a val
    split is present, and writes ``metadata.json``.
    """
    # Deferred: `mayaku.api` imports from `mayaku.cli` (resolve_weights, run_train),
    # so importing it at module load would create a cycle (api → cli → __main__ → api).
    # `train` is only needed when this command runs, so import it here.
    from mayaku.api import train

    # Install the MPS op-fallback tracker at the shell-CLI boundary only;
    # library callers (``mayaku.train`` from Python) manage it themselves.
    with track_mps_fallbacks(label="train"):
        try:
            result = train(
                config=config,
                weights=weights,
                train_annotations=annotations,
                train_images=images,
                val_annotations=val_annotations,
                val_images=val_images,
                output_dir=output,
                num_epochs=epochs,
                device=cast(DeviceSetting, device),
                num_gpus=num_gpus,
                resume=resume,
            )
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    ap = result["final_box_ap"]
    if ap is not None:
        typer.echo(f"box AP: {ap * 100:.2f}")
    typer.echo(f"final weights: {result['final_weights']}")


@app.command("eval")
def _eval(
    weights: str = typer.Argument(
        ...,
        help=(
            "Trained model: a .pth checkpoint (its embedded sidecar defines the "
            "architecture), OR a bare model name from `mayaku download --list` "
            "(e.g. `faster_rcnn_R_50_FPN_3x`) — names auto-fetch from the hosted "
            "manifest on first use."
        ),
    ),
    annotations: Path = typer.Option(..., "--annotations", exists=True, dir_okay=False),
    images: Path = typer.Option(..., "--images", exists=True, file_okay=False),
    output: Path | None = typer.Option(None, "--output", file_okay=False),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    """Run COCO evaluation; print the per-task metrics dict."""
    with track_mps_fallbacks(label="eval"):
        metrics = run_eval(
            weights,
            coco_gt_json=annotations,
            image_root=images,
            output_dir=output,
            device=device,
        )
    typer.echo(json.dumps(metrics, indent=2))


@app.command("predict")
def _predict(
    weights: str = typer.Argument(
        ...,
        help=(
            "Trained model: a .pth checkpoint (its embedded sidecar defines the "
            "architecture), OR a bare model name from `mayaku download --list` — "
            "names auto-fetch from the manifest."
        ),
    ),
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", file_okay=True),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    """Run inference on a single image; print or save the detections."""
    payload = run_predict(weights, image, output=output, device=device)
    if output is None:
        typer.echo(json.dumps(payload, indent=2))


@app.command("export")
def _export(
    target: str = typer.Argument(..., help="onnx / coreml / openvino / tensorrt"),
    weights: str = typer.Argument(
        ...,
        help=(
            "Trained model: a .pth checkpoint (its embedded sidecar defines the "
            "architecture to export), OR a bare model name from "
            "`mayaku download --list`."
        ),
    ),
    output: Path = typer.Option(..., "--output"),
    sample_height: int = typer.Option(640, "--sample-height"),
    sample_width: int = typer.Option(640, "--sample-width"),
    coreml_precision: str = typer.Option(
        "fp32",
        "--coreml-precision",
        help=(
            "fp32 / fp16. CoreML target only. Default fp32 keeps "
            "single-image parity_check tight; pass fp16 to enable "
            "Apple Silicon Neural Engine execution at deployment."
        ),
    ),
    onnx_dynamic_input_shape: bool = typer.Option(
        True,
        "--onnx-dynamic-shapes/--no-onnx-dynamic-shapes",
        help=(
            "ONNX target only. When --onnx-dynamic-shapes (default), "
            "the exported graph supports any (N, 3, H, W) input. When "
            "--no-onnx-dynamic-shapes, the graph is exported at the "
            "literal --sample-height/--sample-width and any other "
            "input shape is rejected. Use --no-onnx-dynamic-shapes "
            "when targeting TensorRT — TRT throughput on R-CNN graphs "
            "degrades dramatically with dynamic shapes (see ADR 005)."
        ),
    ),
) -> None:
    """Export a trained model to a deployment target.

    ONNX is the required target; CoreML / OpenVINO / TensorRT are
    best-effort. All four are live — see ``docs/export/<target>.md``
    for what's in each exported graph and how to run the resulting
    artefact.
    """
    result = run_export(
        target,
        weights,
        output=output,
        sample_height=sample_height,
        sample_width=sample_width,
        coreml_precision=coreml_precision,
        onnx_dynamic_input_shape=onnx_dynamic_input_shape,
    )
    typer.echo(
        json.dumps(
            {
                "target": result.target,
                "path": str(result.path),
                "opset": result.opset,
                "input_names": list(result.input_names),
                "output_names": list(result.output_names),
            },
            indent=2,
        )
    )


@app.command("download")
def _download(
    name: str | None = typer.Argument(
        None,
        help=(
            "Model name from the hosted manifest (e.g. faster_rcnn_R_50_FPN_3x). "
            "Omit and pass --list / --all instead."
        ),
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Override the download directory (default: the current directory).",
    ),
    manifest_url: str = typer.Option(
        DEFAULT_MANIFEST_URL,
        "--manifest-url",
        help="Override the manifest URL — useful for self-hosted mirrors.",
    ),
    list_models: bool = typer.Option(
        False, "--list", help="Print every model name in the manifest, grouped by task."
    ),
    download_all: bool = typer.Option(
        False, "--all", help="Fetch every model's checkpoint. Used for fresh-deploy setup."
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Skip the SHA256 verification step (not recommended)."
    ),
) -> None:
    """Fetch hosted Mayaku model checkpoints from the manifest."""
    payload = run_download(
        name=name,
        cache_dir=cache_dir,
        manifest_url=manifest_url,
        do_list=list_models,
        do_all=download_all,
        verify_sha256=not no_verify,
    )
    if "models" in payload:
        typer.echo(render_index(payload["models"]))
    else:
        typer.echo(json.dumps(payload, indent=2, default=str))


def main() -> None:
    """Console-script entry point — equivalent to ``app()``."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
