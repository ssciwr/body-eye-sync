# Outputs

Body Eye Sync writes results as Parquet files so they can be loaded directly by
Python analysis tools such as pandas, Polars, or PyArrow.

## Main Output

Each input receives one main output:

```text
outputs/<input-id>.parquet
```

Video and audio inputs use different rows: a video row is one tracked box in one
frame, an audio row is one speech turn.

### Video Columns

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

### Audio Columns

For an audio input each row is one speech turn found by diarization:

| Column | Meaning |
| --- | --- |
| `segment_id` | Zero-based speech turn index, in start-time order |
| `start`, `end` | Turn bounds in seconds, on the recording's own clock |
| `speaker` | Speaker identifier, stable within this recording only |

Times are in seconds rather than frames, and the input's `time_offset` is what
places them on the shared experiment timeline. Turns by different speakers may
overlap: that is how simultaneous speech is represented.

When transcription is enabled, one more column is merged onto each turn:

- `text` — the words spoken in that turn, empty if none were attributed to it.

## Word Outputs

If transcription runs, per-word timings are written beside the main output:

```text
outputs/<input-id>.words.parquet
```

| Column | Meaning |
| --- | --- |
| `segment_id` | Speech turn the word was attributed to |
| `word_index` | Position of the word within that turn |
| `start`, `end` | Word bounds in seconds |
| `word` | The word itself |
| `score` | Whisper's confidence in the word |
| `speaker` | Speaker of the turn the word belongs to |

Words that fall outside every speech turn — text Whisper produced over a stretch
diarization heard no speech in — are kept with `segment_id` and `speaker` set to
`-1`, so the transcript never silently loses text.

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

turns = pd.read_parquet("outputs/p1-mic.parquet")
words = pd.read_parquet("outputs/p1-mic.words.parquet")
```
