# Body Eye Sync

Body Eye Sync is a Python desktop application for extracting synchronized
computer-vision measurements from video. It tracks people in a recording, can
run face detection and body-pose detection on the tracked person boxes, and
saves the results in analysis-friendly Parquet files.

The project has two entry points:

- `body-eye-sync` launches the Qt desktop application.
- `body-eye-sync-cli` prepares and runs a saved experiment folder without opening the GUI.

The GUI is the primary workflow for opening a video, tuning the pipeline, running
steps interactively, previewing overlays, and saving an experiment. The CLI uses
the same experiment format for reproducible or batch processing.

## Pipeline

The current pipeline is centered on video inputs:

1. Object tracking detects and tracks objects across frames. By default it tracks
   COCO class `0`, which is `person`.
2. Face detection optionally finds one face per tracked person box and stores the
   face box, confidence, landmarks, and selected embeddings.
3. Body-pose detection optionally estimates COCO keypoints inside tracked person
   boxes.

Outputs are written per input under the experiment's `outputs/` directory.
