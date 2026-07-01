"""Command line interface for running body-eye-sync experiments non-interactively."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from body_eye_sync.experiment.config import (
    DEFAULT_EXPERIMENT_FILENAME,
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
    VideoInput,
)
from body_eye_sync.experiment.run import run_experiment


@click.group(invoke_without_command=True)
@click.version_option(package_name="body-eye-sync", prog_name="body-eye-sync")
@click.pass_context
def main(ctx):
    """Run body-eye-sync experiments from the command line."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument(
    "experiment",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the Parquet outputs. Defaults to the experiment's 'outputs/'.",
)
@click.option(
    "--device",
    default=None,
    help="Compute device, e.g. 'cpu', 'mps', '0'. Auto-detected when unset.",
)
@click.option(
    "--providers",
    default=None,
    help="Comma-separated ONNX providers for face detection, e.g. 'CPUExecutionProvider'.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-run inputs whose output already exists, overwriting it.",
)
def run(experiment, output_dir, device, providers, force):
    """Run the pipeline for EXPERIMENT (a folder or an experiment.yaml file)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    exp = (
        Experiment.from_dir(experiment)
        if experiment.is_dir()
        else Experiment.from_yaml(experiment)
    )
    provider_list = [p.strip() for p in providers.split(",")] if providers else None
    results = run_experiment(
        exp,
        output_dir=output_dir,
        device=device,
        providers=provider_list,
        force=force,
    )
    for input_id, path in results.items():
        click.echo(f"{input_id}: {path}")


@main.command()
@click.argument("directory", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--name", default=None, help="Experiment name. Defaults to the folder name."
)
@click.option("--force", is_flag=True, help="Overwrite an existing experiment.yaml.")
def init(directory, name, force):
    """Scaffold a template experiment folder at DIRECTORY."""
    config_file = directory / DEFAULT_EXPERIMENT_FILENAME
    if config_file.exists() and not force:
        raise click.ClickException(
            f"{config_file} already exists (use --force to overwrite)"
        )
    template = Experiment(
        name=name or directory.resolve().name,
        inputs=[VideoInput(id="cam1", path=Path("videos/example.mp4"))],
        object_tracking=ObjectTrackingStep(),
        face_detection=FaceDetectionStep(),
        body_pose=BodyPoseStep(),
    )
    template.to_dir(directory)
    click.echo(f"wrote {config_file}")


if __name__ == "__main__":
    main()
