# Body Eye Sync

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/body-eye-sync/ci.yml?branch=main)](https://github.com/ssciwr/body-eye-sync/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ssciwr/body-eye-sync/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/body-eye-sync)

WIP

## User installation

Clone this repository, e.g.

```
git clone https://github.com/ssciwr/body-eye-sync.git
cd body-eye-sync
```

Install the package:

```
pip install .
```

Then launch the GUI:

```
body-eye-sync
```

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

To run the tests:

```
uv run pytest
```
