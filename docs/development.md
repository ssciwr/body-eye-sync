# Development

## Environment

Install the development environment with `uv`:

```bash
uv sync
```

Install pre-commit hooks with `prek`:

```bash
prek install
```

## Run the Application

```bash
uv run body-eye-sync
```

Open a saved experiment directly:

```bash
uv run body-eye-sync path/to/experiment
```

## Run Tests

```bash
uv run pytest
```

## Documentation

Zensical and `mkdocstrings-python` are installed as development dependencies.
Preview the docs locally:

```bash
uv run zensical serve
```

Build the static site:

```bash
uv run zensical build --clean
```

The generated site is written to `site/`. The GitHub Actions documentation
workflow publishes that directory to GitHub Pages on pushes to `main`.

API reference pages live under `docs/api/` and use mkdocstrings directives:

```md
::: body_eye_sync.experiment.config
```

Add new public modules to the API pages and navigation in `zensical.toml`.
