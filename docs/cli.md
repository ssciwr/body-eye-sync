# Command Line

The CLI works on a saved experiment folder non-interactively, in two steps:
`prepare` puts the recordings on a shared timeline, then `run` processes them.

```bash
body-eye-sync-cli prepare path/to/experiment
body-eye-sync-cli run path/to/experiment
```

From a development checkout:

```bash
uv run body-eye-sync-cli run path/to/experiment
```

## prepare

`prepare` aligns the recordings against each other, then measures clock drift
and recording gaps and corrects them. It writes what it found back into
`experiment.yaml`, as each input's `time_offset`, `time_scale` and
`time_shifts`, and prints them.

Everything that relates one recording to another reads that timeline, so run
this before `run`. It only needs running again when the inputs change: the
corrections are saved with the experiment, and the GUI's Alignment and Timing
correction tabs write the same fields.

## run

`run` loads `experiment.yaml`, runs every configured input through the pipeline,
and writes one Parquet result per input under `outputs/`. It takes the timeline
as it finds it: an experiment that has never been prepared is processed with
every input still at offset `0.0`.

Existing outputs are skipped by default. Re-run all inputs and overwrite the
existing output files with:

```bash
body-eye-sync-cli run path/to/experiment --force
```

The command prints the input IDs and output paths it wrote or reused.
