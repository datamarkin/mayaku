"""Typer entry point for the ``mayaku`` console script.

The console-script entry in ``pyproject.toml`` (`mayaku =
"mayaku.cli.__main__:app"`) imports the :data:`app` symbol below.
Each subcommand is a thin wrapper around the corresponding
``cli/<name>.py`` ``run_*`` function so the implementation stays
testable in-process via direct calls (and the Typer layer is just
argument plumbing).

Subcommands:

* ``mayaku train CONFIG --json --images --output [--weights] [--device] [--max-iter] [--val-json --val-images]``
* ``mayaku eval CONFIG --weights --json --images [--output] [--device]``
* ``mayaku predict CONFIG IMAGE [--weights] [--output] [--device]``
* ``mayaku export TARGET CONFIG --weights --output``
  (TARGET ∈ ``onnx`` | ``coreml`` | ``openvino`` | ``tensorrt``)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from mayaku.backends.mps import track_mps_fallbacks
from mayaku.cli.download import render_index, run_download
from mayaku.cli.eval import run_eval
from mayaku.cli.export import run_export
from mayaku.cli.predict import run_predict
from mayaku.cli.train import run_train
from mayaku.utils.download import DEFAULT_MANIFEST_URL, VARIANTS

app = typer.Typer(
    name="mayaku",
    help="Backend-portable detection / segmentation / keypoint CLI.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("train")
def _train(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    json_path: Path = typer.Option(..., "--json", exists=True, dir_okay=False),
    images: Path = typer.Option(..., "--images", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output", file_okay=False),
    weights: Path | None = typer.Option(None, "--weights", exists=True, dir_okay=False),
    pretrained_backbone: bool = typer.Option(
        False,
        "--pretrained-backbone",
        help=(
            "Initialise the backbone from torchvision's ImageNet pretrained "
            "weights (DEFAULT). Strongly recommended for fine-tuning — the "
            "schema's freeze_at=2 default freezes the early stages, which is "
            "only meaningful when those stages already carry useful features."
        ),
    ),
    device: str | None = typer.Option(None, "--device", help="cpu/mps/cuda; default = auto"),
    max_iter: int | None = typer.Option(
        None, "--max-iter", help="Override SolverConfig.max_iter (smoke runs)."
    ),
    log_period: int = typer.Option(20, "--log-period", help="Print loss/lr every N iterations."),
    val_json: Path | None = typer.Option(
        None,
        "--val-json",
        exists=True,
        dir_okay=False,
        help=(
            "Held-out COCO ground-truth JSON for periodic mid-training "
            "evaluation. Required when test.eval_period > 0 in the config."
        ),
    ),
    val_images: Path | None = typer.Option(
        None,
        "--val-images",
        exists=True,
        file_okay=False,
        help="Image directory matching --val-json.",
    ),
) -> None:
    """Train a detector. ``CONFIG`` is a YAML loadable by :func:`load_yaml`."""
    # Install the MPS op-fallback tracker at the shell-CLI boundary
    # only. Library callers (``run_train`` from Python) are free to
    # install the tracker themselves or skip it. ``track_mps_fallbacks``
    # is a no-op when MAYAKU_VERBOSE_MPS=1 or when the device isn't MPS.
    with track_mps_fallbacks(label="train"):
        run_train(
            config,
            coco_gt_json=json_path,
            image_root=images,
            output_dir=output,
            weights=weights,
            pretrained_backbone=pretrained_backbone,
            device=device,
            max_iter=max_iter,
            log_period=log_period,
            val_json=val_json,
            val_image_root=val_images,
        )


@app.command("eval")
def _eval(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    weights: str = typer.Option(
        ...,
        "--weights",
        help=(
            "Path to a .pth checkpoint, OR a bare model name from "
            "`mayaku download --list` (e.g. `faster_rcnn_R_50_FPN_3x`) — "
            "names auto-fetch from the hosted manifest on first use."
        ),
    ),
    json_path: Path = typer.Option(..., "--json", exists=True, dir_okay=False),
    images: Path = typer.Option(..., "--images", exists=True, file_okay=False),
    output: Path | None = typer.Option(None, "--output", file_okay=False),
    device: str | None = typer.Option(None, "--device"),
    backbone_mlpackage: Path | None = typer.Option(
        None,
        "--backbone-mlpackage",
        exists=True,
        file_okay=False,
        help=(
            "Path to a CoreML .mlpackage exported by `mayaku export coreml`. "
            "Replaces the eager backbone+FPN at eval time; RPN/ROI heads "
            "still run in PyTorch. macOS only."
        ),
    ),
    coreml_compute_units: str = typer.Option(
        "CPU_AND_GPU",
        "--coreml-compute-units",
        help=(
            "ALL / CPU_ONLY / CPU_AND_GPU / CPU_AND_NE. Only used with "
            "--backbone-mlpackage. Default CPU_AND_GPU is fastest on R-CNN "
            "graphs because the FPN's top-down `add` ops force ANE↔GPU "
            "transitions whose cost exceeds ANE's per-conv speedup. See "
            "docs/decisions/004-coreml-export-positioning.md."
        ),
    ),
    backbone_onnx: Path | None = typer.Option(
        None,
        "--backbone-onnx",
        exists=True,
        dir_okay=False,
        help=(
            "Path to a .onnx file exported by `mayaku export onnx`. "
            "Replaces the eager backbone+FPN at eval time; RPN/ROI heads "
            "still run in PyTorch. Cross-platform via onnxruntime."
        ),
    ),
    onnx_providers: str | None = typer.Option(
        None,
        "--onnx-providers",
        help=(
            "Comma-separated ORT execution providers in order of "
            "preference (e.g. 'CoreMLExecutionProvider,CPUExecutionProvider'). "
            "Defaults to onnxruntime's auto-selection. Only used with --backbone-onnx."
        ),
    ),
) -> None:
    """Run COCO evaluation; print the per-task metrics dict."""
    with track_mps_fallbacks(label="eval"):
        metrics = run_eval(
            config,
            weights=weights,
            coco_gt_json=json_path,
            image_root=images,
            output_dir=output,
            device=device,
            backbone_mlpackage=backbone_mlpackage,
            coreml_compute_units=coreml_compute_units,
            backbone_onnx=backbone_onnx,
            onnx_providers=onnx_providers,
        )
    typer.echo(json.dumps(metrics, indent=2))


@app.command("predict")
def _predict(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    weights: str | None = typer.Option(
        None,
        "--weights",
        help=(
            "Path to a .pth checkpoint, OR a bare model name from "
            "`mayaku download --list` — names auto-fetch from the manifest."
        ),
    ),
    output: Path | None = typer.Option(None, "--output", file_okay=True),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    """Run inference on a single image; print or save the detections."""
    payload = run_predict(config, image, weights=weights, output=output, device=device)
    if output is None:
        typer.echo(json.dumps(payload, indent=2))


@app.command("export")
def _export(
    target: str = typer.Argument(..., help="onnx / coreml / openvino / tensorrt"),
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    weights: Path = typer.Option(..., "--weights", exists=True, dir_okay=False),
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
        config,
        weights=weights,
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
    target: str | None = typer.Option(
        None,
        "--target",
        help=f"Variant to fetch. One of {', '.join(VARIANTS)}. Omit to fetch all variants for NAME.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Override the cache root (default: $XDG_CACHE_HOME/mayaku/v1/models).",
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
        False, "--all", help="Fetch every variant of every model. Used for fresh-deploy setup."
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Skip the SHA256 verification step (not recommended)."
    ),
) -> None:
    """Fetch hosted Mayaku model artifacts from the manifest."""
    payload = run_download(
        name=name,
        target=target,
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
