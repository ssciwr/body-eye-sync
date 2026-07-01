"""Command line interface for running body-eye-sync experiments non-interactively."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.run import run_experiment


@click.command()
@click.version_option(package_name="body-eye-sync", prog_name="body-eye-sync")
@click.argument(
    "folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-run inputs whose output already exists, overwriting it.",
)
def main(folder, force):
    """Run the body-eye-sync pipeline for the experiment in FOLDER."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    exp = Experiment.load(folder)
    results = run_experiment(exp, force=force)
    for input_id, path in results.items():
        click.echo(f"{input_id}: {path}")


if __name__ == "__main__":
    main()
