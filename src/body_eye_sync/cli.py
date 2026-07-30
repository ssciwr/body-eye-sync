"""Command line interface for running body-eye-sync experiments non-interactively."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.prepare import prepare_experiment
from body_eye_sync.experiment.run import run_experiment

_FOLDER = click.argument(
    "folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)


@click.group()
@click.version_option(package_name="body-eye-sync", prog_name="body-eye-sync")
def main():
    """Prepare and run body-eye-sync experiments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@_FOLDER
def prepare(folder):
    """Put the recordings of the experiment in FOLDER on one clock.

    Aligns the inputs, then corrects clock drift and recording gaps, and writes
    the result back to the experiment. Run this before ``run``: everything that
    relates one recording to another reads the timeline it writes.
    """
    exp = Experiment.load(folder)
    prepare_experiment(exp)
    exp.save()
    for data in exp.inputs:
        click.echo(
            f"{data.id}: offset {data.time_offset:+.3f} s, "
            f"scale {data.time_scale:.9f}, {len(data.time_shifts)} gap(s)"
        )


@main.command()
@_FOLDER
@click.option(
    "--force",
    is_flag=True,
    help="Re-run inputs whose output already exists, overwriting it.",
)
def run(folder, force):
    """Run the body-eye-sync pipeline for the experiment in FOLDER."""
    exp = Experiment.load(folder)
    results = run_experiment(exp, force=force)
    for input_id, path in results.items():
        click.echo(f"{input_id}: {path}")


if __name__ == "__main__":
    main()
