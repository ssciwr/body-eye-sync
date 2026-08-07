# Outputs

Body Eye Sync writes results as Parquet files so they can be loaded directly by
Python analysis tools such as pandas, Polars, or PyArrow.

## Synchronized Video

The Data export tab creates an MP4 containing the checked video inputs in a
labelled grid on the shared experiment timeline. Every input starts checked;
checked audio-only inputs contribute audio without adding a grid cell. The
output is always 25 fps; source frames are selected by timestamp, so both 25 fps
and 50 fps recordings remain synchronized. A cell is black before or after its
recording and wherever the input lost content.

Every checked input that carries audio becomes a separately selectable,
full-length audio track named after the input id. Its offset, clock scale and
shifts are applied independently, with silence filling time not covered by that
recording. Selecting **Include merged audio track** also adds a default playback
track that mixes all the checked sources while retaining their individual
tracks.

The same export is available through `construct_video_grid`; omit `input_ids`
to include every input.

```python
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.video_grid import construct_video_grid

experiment = Experiment.load("my-experiment")
construct_video_grid(
    experiment,
    "my-experiment/synchronized-grid.mp4",
    input_ids=["room", "glasses_1", "microphone_1"],
    include_merged_audio=True,
)
```

## Main Output

Each video input receives one main output:

```text
outputs/<input-id>/results.parquet
```

Each `<input-id>` directory is managed by Body Eye Sync. Files placed directly
in `outputs/` are left alone.

Audio inputs have no pipeline stages yet, so they produce no output.

The core tracking columns are:

| Column | Meaning |
| --- | --- |
| `frame` | Zero-based video frame index |
| `track_id` | Stable track identifier |
| `x1`, `y1`, `x2`, `y2` | Tracked object box in video pixels |
| `conf` | Object detection confidence |

When face detection is enabled, face columns are merged onto matching
`(frame, track_id)` rows:

- `face_score`
- `face_x1`, `face_y1`, `face_x2`, `face_y2`
- `left_eye_x`, `left_eye_y`
- `right_eye_x`, `right_eye_y`
- `nose_x`, `nose_y`
- `mouth_left_x`, `mouth_left_y`
- `mouth_right_x`, `mouth_right_y`

When body-pose detection is enabled, pose columns are merged onto matching rows:

- `pose_score`
- `pose_x1`, `pose_y1`, `pose_x2`, `pose_y2`
- Per-keypoint `pose_<name>_x`, `pose_<name>_y`, and `pose_<name>_score` columns
  for the COCO keypoints.

## Embedding Outputs

If embeddings are collected, companion files are written beside the main output:

```text
outputs/<input-id>/body_embeddings.parquet
outputs/<input-id>/face_embeddings.parquet
```

Embedding files contain:

| Column | Meaning |
| --- | --- |
| `track_id` | Track the embedding belongs to |
| `frame` | Frame the embedding came from |
| `score` | Detection score used for top-k selection |
| `embedding` | Fixed-size float16 vector |

Only the best `embeddings_per_track` vectors are kept for each tracklet.

## Reading Results

```python
import pandas as pd

tracks = pd.read_parquet("outputs/camera-1/results.parquet")
body_embeddings = pd.read_parquet("outputs/camera-1/body_embeddings.parquet")
```
