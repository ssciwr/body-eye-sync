# Outputs

Body Eye Sync writes results as Parquet files so they can be loaded directly by
Python analysis tools such as pandas, Polars, or PyArrow.

## Synchronized Video and ELAN Annotations

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

The tab also writes a same-stem ELAN `.eaf` file beside the MP4. Each selected
input's available speech results become tiers named
`<input-id>:speaker_<local-id>`. Turn boundaries are converted from source-local
time to the exported video's zero-based timeline, and annotation values contain
transcript text when transcription has run. Speaker ids deliberately remain
local to each input until post-processing provides experiment-wide identities.

The same export is available through `construct_video_grid`; omit `input_ids`
to include every input.

```python
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.elan import export_elan
from body_eye_sync.export.video_grid import construct_video_grid

experiment = Experiment.load("my-experiment")
video = construct_video_grid(
    experiment,
    "my-experiment/synchronized-grid.mp4",
    input_ids=["room", "glasses_1", "microphone_1"],
    include_merged_audio=True,
)
export_elan(
    experiment,
    video,
    input_ids=["room", "glasses_1", "microphone_1"],
)
```

## Main Output

Each input gets a directory, holding a file per kind of result it has. Each
`<input-id>` directory is managed by Body Eye Sync; files placed directly in
`outputs/` are left alone.

```text
outputs/<input-id>/results.parquet        # video: one row per tracked box per frame
outputs/<input-id>/speaker_turns.parquet  # speech: one row per speech turn
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

For an audio input each row is one speech turn found by diarization:

| Column | Meaning |
| --- | --- |
| `segment_id` | Zero-based speech turn index, in start-time order |
| `start`, `end` | Turn bounds in seconds, on the recording's own clock |
| `speaker` | Speaker identifier, stable within this recording only |

Times are in seconds rather than frames, on the recording's own clock. Its
`time_offset`, `time_scale` and `time_shifts` together place them on the shared
experiment timeline -- `Timeline.to_experiment_time` applies all three, so
prefer it to adding the offset by hand. Turns by different speakers may overlap:
that is how simultaneous speech is represented.

When transcription is enabled, one more column is merged onto each turn:

- `text` — the words spoken in that turn, empty if none were attributed to it.

## Word Outputs

If transcription runs, per-word timings are written beside the speech turns:

```text
outputs/<input-id>/speaker_words.parquet
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

If embeddings are collected, companion files are written in the same directory:

```text
outputs/<input-id>/body_embeddings.parquet
outputs/<input-id>/face_embeddings.parquet
outputs/<input-id>/speaker_embeddings.parquet
```

They exist to relate identities *across* inputs, which neither `track_id` nor
`speaker` can do on their own. The video files contain:

| Column | Meaning |
| --- | --- |
| `track_id` | Track the embedding belongs to |
| `frame` | Frame the embedding came from |
| `score` | Detection score used for top-k selection |
| `embedding` | Fixed-size float16 vector |

Only the best `embeddings_per_track` vectors are kept for each tracklet.

The speaker file has the same shape, with the identity and its source renamed
for what they are in audio:

| Column | Meaning |
| --- | --- |
| `speaker` | Speaker the embedding belongs to |
| `segment_id` | Speech turn the embedding was computed from |
| `duration` | Length of that turn in seconds, used for top-k selection |
| `embedding` | Fixed-size float16 vector |

Only the best `embeddings_per_speaker` vectors are kept for each speaker, the
longest turns first, since a longer turn gives the model more voice to work
with. Turns under 0.2 seconds are not embedded at all.

## Reading Results

```python
import pandas as pd

tracks = pd.read_parquet("outputs/camera-1/results.parquet")
body_embeddings = pd.read_parquet("outputs/camera-1/body_embeddings.parquet")

turns = pd.read_parquet("outputs/p1-mic/speaker_turns.parquet")
words = pd.read_parquet("outputs/p1-mic/speaker_words.parquet")

# A camera that recorded audio has speech results of its own, under its own id.
room_turns = pd.read_parquet("outputs/camera-1/speaker_turns.parquet")
```
