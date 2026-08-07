#!/usr/bin/env python
"""Fit and plot clock drift and missing audio across several recordings.

The first stage uses landmark fingerprints to recover each recording's start
offset. The second stage measures spectral alignment in short windows and fits
the timeline model used by body-eye-sync::

    experiment_time = offset + scale * local_time + cumulative missing content

The upper panel shows the measured offset and the complete fitted model. The
lower panel removes the continuous clock-rate term, leaving the discrete
missing-content shifts and their fitted step function.

Pass ``--step-only`` to constrain the clock rate to 1.0 and plot a model made
entirely from discrete positive jumps.

    plot_alignment_timeline.py out.png room=room.wav glasses=glasses.mp4
    plot_alignment_timeline.py out.png --reference room room=... glasses=...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from body_eye_sync.pipeline.alignment import (
    SPECTRAL,
    DriftPoint,
    TimelineFit,
    align_media,
    drift_curve,
    fit_timeline,
    missing_before_all,
)


@dataclass
class Result:
    """Local alignment observations and their fitted timeline."""

    points: list[DriftPoint]
    fit: TimelineFit


def _input(value: str) -> tuple[str, Path]:
    """Parse one ``NAME=PATH`` command-line input."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must have the form NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("input name cannot be empty")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input does not exist: {path}")
    return name, path


def _shift_experiment_times(fit: TimelineFit) -> list[float]:
    """Start of every missing stretch on the shared experiment clock."""
    times = []
    prior = 0.0
    for shift in sorted(fit.shifts, key=lambda value: value.at):
        times.append(fit.offset + fit.scale * shift.at + prior)
        prior += shift.seconds
    return times


def _print_fit(name: str, fit: TimelineFit, points: list[DriftPoint]) -> None:
    """Print the numerical model represented in the plot."""
    ppm = (fit.scale - 1.0) * 1e6
    print(
        f"{name:<10} {len(points):>4} windows  offset {fit.offset:>9.3f} s  "
        f"rate {ppm:>+8.1f} ppm  residual {fit.residual * 1000:>6.1f} ms"
    )
    if not fit.shifts:
        print(" " * 13 + "no missing-content shifts")
    for shift, experiment_time in zip(fit.shifts, _shift_experiment_times(fit)):
        print(
            " " * 13
            + f"local {shift.at:>8.2f} s / experiment {experiment_time:>8.2f} s: "
            + f"missing {shift.seconds * 1000:>7.1f} ms"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="PNG to write")
    parser.add_argument("inputs", nargs="+", type=_input, metavar="NAME=PATH")
    parser.add_argument("--reference", help="input whose clock defines experiment time")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--search", type=float, default=12.0)
    parser.add_argument("--min-shift", type=float, default=0.08)
    parser.add_argument(
        "--step-only",
        action="store_true",
        help="fix clock scale at 1.0 and fit only discrete positive jumps",
    )
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = dict(args.inputs)
    if len(paths) != len(args.inputs):
        parser.error("input names must be unique")
    reference = args.reference or next(iter(paths))
    if reference not in paths:
        parser.error(f"unknown reference input: {reference!r}")

    print("extracting landmark fingerprints and solving start offsets...")
    coarse = align_media(paths, reference=reference)
    if reference not in coarse.offsets:
        print("reference could not be aligned", file=sys.stderr)
        return 1
    print(f"landmark graph residual: {coarse.residual * 1000:.1f} ms")
    for name in paths:
        if name in coarse.offsets:
            print(f"  {name:<10} {coarse.offsets[name]:>10.3f} s")
        else:
            print(f"  {name:<10} not connected to {reference}", file=sys.stderr)

    print(f"\nextracting spectral features for reference {reference!r}...")
    reference_features = SPECTRAL.extract(paths[reference])
    results: dict[str, Result] = {}
    for name, path in paths.items():
        if name == reference or name not in coarse.offsets:
            continue
        print(f"measuring local alignment for {name!r}...")
        other_features = SPECTRAL.extract(path)
        points = drift_curve(
            reference_features,
            other_features,
            coarse.offsets[name],
            window=args.window,
            step=args.step,
            search=args.search,
            method=SPECTRAL,
        )
        del other_features
        if not points:
            print(f"  no spectral windows locked for {name}", file=sys.stderr)
            continue
        fit = fit_timeline(
            points,
            min_shift=args.min_shift,
            time_scale=1.0 if args.step_only else None,
        )
        results[name] = Result(points, fit)
        _print_fit(name, fit, points)

    if not results:
        print("no input produced a timeline fit", file=sys.stderr)
        return 1

    if args.step_only:
        figure, raw_axis = plt.subplots(figsize=(14, 7))
        shift_axis = None
    else:
        figure, (raw_axis, shift_axis) = plt.subplots(
            2, 1, figsize=(14, 9), sharex=True, height_ratios=(3, 2)
        )
    for colour_index, (name, result) in enumerate(results.items()):
        colour = f"C{colour_index}"
        fit = result.fit
        experiment = np.asarray([point.time for point in result.points])
        measured_offset = np.asarray([point.offset for point in result.points])
        local = experiment - measured_offset
        missing = missing_before_all(fit.shifts, local)
        fitted_offset = fit.offset + (fit.scale - 1.0) * local + missing
        detrended = experiment - fit.scale * local - fit.offset

        minutes = experiment / 60.0
        raw_axis.scatter(
            minutes,
            (measured_offset - fit.offset) * 1000,
            s=8,
            alpha=0.45,
            color=colour,
        )
        raw_axis.plot(
            minutes,
            (fitted_offset - fit.offset) * 1000,
            lw=1.5,
            color=colour,
            label=(
                f"{name} ({len(fit.shifts)} jumps)"
                if args.step_only
                else f"{name} ({(fit.scale - 1) * 1e6:+.0f} ppm)"
            ),
        )
        annotation_axis = raw_axis
        if shift_axis is not None:
            shift_axis.scatter(
                minutes,
                detrended * 1000,
                s=8,
                alpha=0.45,
                color=colour,
            )
            shift_axis.step(
                minutes,
                missing * 1000,
                where="post",
                lw=1.5,
                color=colour,
                label=name,
            )
            annotation_axis = shift_axis
        for shift, when in zip(fit.shifts, _shift_experiment_times(fit)):
            annotation_axis.axvline(when / 60.0, color=colour, alpha=0.2, lw=1)
            annotation_axis.annotate(
                f"+{shift.seconds * 1000:.0f} ms",
                (when / 60.0, missing.max() * 1000),
                color=colour,
                fontsize=7,
                rotation=90,
                va="top",
                ha="right",
            )

    raw_axis.axhline(0, color="black", lw=0.8)
    raw_axis.set_ylabel("offset change from recording start (ms)")
    raw_axis.set_title(
        "Measured local offsets and fitted discrete jumps (clock rate fixed)"
        if args.step_only
        else "Measured local offsets and fitted clock + dropout model"
    )
    raw_axis.legend(fontsize=9)
    if shift_axis is None:
        raw_axis.set_xlabel(f"experiment time on {reference} clock (minutes)")
        raw_axis.grid(alpha=0.25)
    else:
        shift_axis.axhline(0, color="black", lw=0.8)
        shift_axis.set_ylabel("cumulative missing content (ms)")
        shift_axis.set_xlabel(f"experiment time on {reference} clock (minutes)")
        shift_axis.set_title("After removing continuous clock-rate drift")
        shift_axis.legend(fontsize=9)
        for axis in (raw_axis, shift_axis):
            axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
