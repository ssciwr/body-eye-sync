# Installation

Body Eye Sync requires Python 3.11 or newer.

## From Source

Clone the repository and install the package:

```bash
git clone https://github.com/ssciwr/body-eye-sync.git
cd body-eye-sync
pip install .
```

Launch the GUI:

```bash
body-eye-sync
```

## Development Install

The repository uses `uv` for development:

```bash
git clone https://github.com/ssciwr/body-eye-sync.git
cd body-eye-sync
uv sync
```

Run the GUI from the development environment:

```bash
uv run body-eye-sync
```

Run the CLI:

```bash
uv run body-eye-sync-cli --help
```

## Model Downloads

The first run of a pipeline step may download model weights for Ultralytics,
BoxMOT, or InsightFace. Body Eye Sync routes bare Ultralytics model names into a
shared user cache instead of writing weights into the current working directory.

The application chooses an available accelerator when possible:

- CUDA through PyTorch or ONNX Runtime when available.
- Apple MPS for supported PyTorch workloads.
- CPU as the fallback.

## Windows Media Feature Pack

On Windows, OpenCV requires Media Foundation. If the application reports that
`mfplat.dll` is missing, install the Windows Media Feature Pack from:

`Settings -> Apps -> Optional features -> Add a feature -> Media Feature Pack`
