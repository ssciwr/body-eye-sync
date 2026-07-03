# Body Eye Sync

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/body-eye-sync/ci.yml?branch=main)](https://github.com/ssciwr/body-eye-sync/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ssciwr/body-eye-sync/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/body-eye-sync)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://ssciwr.github.io/body-eye-sync/)

WIP

## User installation

If you already have Python and pip:

```
pip install body-eye-sync
```

Then launch the GUI:

```
body-eye-sync
```

## Windows installation

Alternatively on Windows we also provide an executable installer.

## Developer setup

After cloning the repo, ensure you have [prek](https://prek.j178.dev/) installed, then do

```
prek install
```

This will ensure pre-commit hooks are ran whenever you do a git commit.

To run the GUI:

```
uv run body-eye-sync
```

To run the CLI:

```
uv run body-eye-sync-cli --help
```

To run the tests:

```
uv run pytest
```

To preview the documentation:

```
uv run zensical serve
```
