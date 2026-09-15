# Command Line

Use the CLI to run a saved experiment folder non-interactively:

```bash
body-eye-sync-cli path/to/experiment
```

From a development checkout:

```bash
uv run body-eye-sync-cli path/to/experiment
```

The CLI loads `experiment.yaml`, runs every configured input, and writes one
Parquet result per input under `outputs/`.

After the input stages finish, the CLI runs speech attribution and tracklet
clustering. When all glasses videos have face-detection results and retained
recognition embeddings, clustering writes `outputs/identities.parquet`. Otherwise
it logs why clustering was skipped. Fixed videos are ignored by clustering.

Existing outputs are skipped by default:

```bash
body-eye-sync-cli path/to/experiment
```

Re-run all inputs and overwrite existing output files with:

```bash
body-eye-sync-cli path/to/experiment --force
```

The command prints the input IDs and output paths it wrote or reused.
