# Installation

To install using pip:

```bash
pip install body-eye-sync
```

Then you can launch the GUI:

```bash
body-eye-sync
```

Alternatively on windows you can download the standalone installer
[body-eye-sync.exe](https://github.com/ssciwr/body-eye-sync/releases/latest/download/body-eye-sync.exe)
and run it, then launch Body Eye Sync from the Start menu or desktop shortcut.

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
