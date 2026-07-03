# Outputs

Body Eye Sync writes results as Parquet files so they can be loaded directly by
Python analysis tools such as pandas, Polars, or PyArrow.

## Main Output

Each input receives one main output:

```text
outputs/<input-id>.parquet
```

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
outputs/<input-id>.body_embeddings.parquet
outputs/<input-id>.face_embeddings.parquet
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

tracks = pd.read_parquet("outputs/camera-1.parquet")
body_embeddings = pd.read_parquet("outputs/camera-1.body_embeddings.parquet")
```
