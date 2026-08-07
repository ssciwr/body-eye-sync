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

Each input gets a directory, holding a file per kind of result it has. Each
`<input-id>` directory is managed by Body Eye Sync; files placed directly in
`outputs/` are left alone.

```text
outputs/<input-id>/results.parquet        # video: one row per tracked box per frame
outputs/<input-id>/transcript_segments.parquet  # speech: one row per segment
```

The rows are shaped differently because they count different things, which is
why they are separate files rather than one table. A video has `results.parquet`
and, if its camera recorded audio, speech files as well; an audio input has only
the speech files, since speech is all it has.

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

For an input with audio each row is one segment of the transcript, as Whisper
split it:

| Column | Meaning |
| --- | --- |
| `segment_id` | Zero-based segment index, in start-time order |
| `start`, `end` | Segment bounds in seconds, on the recording's own clock |
| `text` | The words spoken in that segment |

Times are in seconds rather than frames, on the recording's own clock. Its
`time_offset`, `time_scale` and `time_shifts` together place them on the shared
experiment timeline -- `Timeline.to_experiment_time` applies all three, so
prefer it to adding the offset by hand.

These results say what was said, not who said it: one recording cannot tell the
voices apart, and every microphone in the room hears everybody. Speakers are
worked out later, by comparing the experiment's recordings with each other.

## Word Outputs

Per-word timings are written beside the transcript:

```text
outputs/<input-id>/transcript_words.parquet
```

| Column | Meaning |
| --- | --- |
| `segment_id` | Segment the word belongs to |
| `word_index` | Position of the word within that segment |
| `start`, `end` | Word bounds in seconds |
| `word` | The word itself |
| `score` | Whisper's confidence in the word |

## Embedding Outputs

If embeddings are collected, companion files are written in the same directory:

```text
outputs/<input-id>/body_embeddings.parquet
outputs/<input-id>/face_embeddings.parquet
```

They exist to relate identities *across* inputs, which `track_id` cannot do on
its own:

| Column | Meaning |
| --- | --- |
| `track_id` | Track the embedding belongs to |
| `frame` | Frame the embedding came from |
| `score` | Detection score used for top-k selection |
| `embedding` | Fixed-size float16 vector |

Only the best `embeddings_per_track` vectors are kept for each tracklet.

## Speech Turns

Who spoke when, for the whole experiment. This is the one result that belongs to
no single input -- it can only be worked out by comparing the recordings with
each other -- so it sits beside the per-input directories:

```text
outputs/speech_turns.parquet
```

| Column | Meaning |
| --- | --- |
| `turn_id` | Zero-based turn index, in start-time order |
| `start`, `end` | Turn bounds in seconds, on the **experiment** clock |
| `speaker` | Input id of the wearer who was speaking |
| `source` | Input the text was taken from |
| `source_segment_id` | Whisper segment in that source, used to retain its word timings |
| `text` | What they said |

Unlike every other output these times are already on the shared experiment
timeline, since the turns were worked out by comparing recordings on it. Turns
may overlap: two people talking at once are two turns over the same stretch.

`speaker` names an input rather than a number, so it means the same thing across
the whole experiment -- there is nothing to match up afterwards.

## ELAN Annotations

Exporting the combined video also writes the experiment's speech turns beside
it, for [ELAN](https://archive.mpi.nl/tools/tla-tools/elan/), whenever there are
any to write:

```text
<combined video>.eaf
```

Each speaker has a tier named for their input, holding readable turns with their
text. A dependent `<speaker>-words` tier contains each word as its own
time-aligned annotation within the parent turn. Times are the experiment's,
shifted to where the video starts, which the file records as a
`body-eye-sync:experiment-start-seconds` property in its header. Turns the video
does not cover are left out.

Speakers talking at once are separate tiers, so their turns overlap freely; one
speaker's own turns never do, since nobody talks over themselves.

```python
from body_eye_sync.export.elan import export_elan
from body_eye_sync.export.video_grid import construct_video_grid

video = construct_video_grid(experiment, "my-experiment/combined.mp4")
export_elan(experiment, video)  # writes my-experiment/combined.eaf
```

## Reading Results

```python
import pandas as pd

tracks = pd.read_parquet("outputs/camera-1/results.parquet")
body_embeddings = pd.read_parquet("outputs/camera-1/body_embeddings.parquet")

transcript = pd.read_parquet("outputs/p1-mic/transcript_segments.parquet")
words = pd.read_parquet("outputs/p1-mic/transcript_words.parquet")

# A camera that recorded audio has a transcript of its own, under its own id.
room_transcript = pd.read_parquet("outputs/camera-1/transcript_segments.parquet")

# Who spoke when, across the whole experiment.
turns = pd.read_parquet("outputs/speech_turns.parquet")
```
