# Development

## Environment

Install the development environment with `uv`:

```bash
uv sync
```

### Updating Dependencies

The committed `uv.lock` is the single dependency lockfile used by local
development, CI and the Windows installer. Windows selects the CUDA 12.6 builds
of Torch and Torchvision through the platform-specific sources in
`pyproject.toml`; the other platforms use PyPI.

After adding or changing a dependency in `pyproject.toml`, update and check the
lockfile with:

```bash
uv lock
uv lock --check
```

To update one dependency while keeping the rest of the locked versions where
possible, use:

```bash
uv lock --upgrade-package <package>
```

Commit `pyproject.toml` and `uv.lock` together. CI uses `uv run --locked`, and
the Windows installer uses `uv export --locked`, so both fail instead of
silently resolving different dependencies when the lockfile is stale.

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
