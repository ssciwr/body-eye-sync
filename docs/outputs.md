# Outputs

Body Eye Sync writes outputs for each input as Parquet files in the `outputs/<input-id>`
subdirectory of the experiment folder, and experiment-level outputs in `outputs/`.

## Video outputs

```text
outputs/<input-id>/results.parquet
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

## Embedding outputs

```text
outputs/<input-id>/body_embeddings.parquet
outputs/<input-id>/face_embeddings.parquet
```

| Column | Meaning |
| --- | --- |
| `track_id` | Track the embedding belongs to |
| `frame` | Frame the embedding came from |
| `score` | Detection score used for top-k selection |
| `embedding` | Fixed-size float16 vector |

Only the best `embeddings_per_track` vectors are kept for each tracklet.

## Speech outputs

```text
outputs/<input-id>/transcript_segments.parquet
```

| Column | Meaning |
| --- | --- |
| `segment_id` | Zero-based segment index, in start-time order |
| `start`, `end` | Segment bounds in seconds, on the recording's own clock |
| `text` | The words spoken in that segment |

## Word outputs

```text
outputs/<input-id>/transcript_words.parquet
```

| Column | Meaning |
| --- | --- |
| `segment_id` | Segment the word belongs to |
| `word_index` | Position of the word within that segment |
| `start`, `end` | Word bounds in seconds, on the recording's own clock |
| `word` | The word itself |
| `score` | Whisper's confidence in the word |

## Speech turns

This is the main output of the speech post-processing step, which combines all
the transcripts from the glasses videos into a single table of who spoke when.

```text
outputs/speech_turns.parquet
```

| Column | Meaning |
| --- | --- |
| `turn_id` | Zero-based turn index, in start-time order |
| `start`, `end` | Turn bounds in seconds, on the **experiment** clock |
| `speaker` | Input id of the wearer who was speaking |
| `source_segment_id` | Segment of the speaker's own transcript the text came from, used to retain its word timings |
| `text` | What they said |
