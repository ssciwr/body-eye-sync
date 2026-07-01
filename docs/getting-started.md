# Getting Started

## Open a Video

Start the desktop app:

```bash
body-eye-sync
```

Choose **Open video...** and select a video file. This creates an unsaved
single-input experiment using the video file name as the experiment and input
identifier.

## Configure the Pipeline

The pipeline editor is shown in the dock on the right side of the window.

Object tracking is always present, because face detection and body-pose detection
run on tracked person boxes. The optional steps can be enabled or disabled in the
pipeline editor.

Useful defaults:

- `object_classes = [0]` tracks people using COCO class IDs.
- `embeddings_per_track = 32` keeps the best body or face embeddings for each
  tracklet.
- Face detection and body-pose detection can be run after object tracking
  results exist.

## Run Steps

Use **Run all** to run every enabled step in order. You can also run an
individual step:

- Object tracking can run once a video is open.
- Face detection can run once object tracking results exist.
- Body-pose detection can run once object tracking results exist.

The viewer shows live overlays while a step runs. Use **Cancel** to stop a
running step; partial results from the cancelled step are discarded.

## Save the Experiment

Use **File -> Save** to choose an experiment folder. Body Eye Sync writes:

- `experiment.yaml` with the experiment definition.
- `outputs/<input-id>.parquet` for completed model outputs.
- Optional companion embedding files when embeddings were collected.

Saved experiments can be reopened in the GUI or processed through the CLI.
