# Experiments

An experiment is a folder containing an `experiment.yaml` file and, after a run,
an `outputs/` directory.

## Folder Layout

```text
my-experiment/
├─ experiment.yaml
└─ outputs/
   └─ camera-1/
      ├─ results.parquet
      ├─ body_embeddings.parquet
      └─ face_embeddings.parquet
```

## Configuration Format

The current experiment format is versioned. Input paths may be absolute or
relative to the experiment folder. An experiment is identified by the folder it
lives in, and so carries no name of its own.

```yaml
version: 1
glasses_videos:
  - id: p1-glasses
    path: videos/p1.mp4
    gaze_path: videos/p1-gaze.tsv
    time_offset: 0.0
fixed_videos:
  - id: camera-1
    path: videos/camera-1.mp4
    time_offset: -1.25
    time_scale: 1.000031
    time_shifts:
      - at: 512.4
        seconds: 0.18
audio:
  - id: p1-mic
    path: audio/p1.wav
    glasses_video: p1-glasses
    time_offset: 0.0
pipeline:
  glasses_video:
    object_tracking:
      detector: yolo26m
      reid: osnet_x1_0_msmt17
      tracker: botsort
      object_classes:
        - 0
      embeddings_per_track: 32
    face_detection:
      model_name: antelopev2
      det_size: 640
      det_thresh: 0.5
      embeddings_per_track: 32
    body_pose:
      model_name: yolo26m-pose.pt
      conf: 0.25
  fixed_video:
    object_tracking:
      detector: yolo26m
    face_detection: null
    body_pose: null
  audio: {}
```

## Inputs

Inputs are grouped by type, each in its own list. Every input needs an `id` that
is unique across *all* the lists, since it names that input's output directory;
for the same reason it has to be usable as a filename, so it cannot be empty or
contain a path separator.

Each input also carries where it sits on the experiment's shared timeline. All
three fields are optional, and the defaults describe a device that started with
the experiment and kept time:

- `time_offset`: seconds added to this input's own clock to reach experiment
  time, since every device was switched on at its own moment. Defaults to `0.0`.
- `time_scale`: experiment seconds per second of this input's own clock. Values
  slightly away from `1.0` compensate a clock that ran fast or slow. Defaults to
  `1.0`.
- `time_shifts`: content the recording lost partway through, if any. Each entry
  has an `at`, where the loss falls on the recording's own clock, and `seconds`,
  how much is missing there, which is added to the offset from that point on.
  Defaults to none.

The Alignment tab fills in `time_offset`; the Timing correction tab measures
`time_shifts`, and `time_scale` too when its Fit drift box is ticked, and
refines the offset of any input that needs them. Its Clear corrections button
puts both back to their defaults. All three can also
be edited by hand. Because a lost stretch was never
recorded, the mapping is not symmetric: every moment an input holds has an
experiment time, but a stretch of experiment time covered by a `time_shift` has
nowhere in that recording to map to, and `Timeline.to_local_time` returns `None`
there rather than the nearest frame it does have.

This file is the on-disk form. At runtime each input is a `GlassesVideo`,
`FixedVideo` or `Audio` that owns these settings alongside its results, reached
through `Experiment.glasses_videos`, `.fixed_videos` and `.audio`. Inputs are
added, removed and renamed through `Experiment` so their ids stay unique.

- `glasses_videos`: video recorded by a participant's glasses-mounted camera.
  This is the input that carries eye tracking, so it needs a `gaze_path` as well
  as a `path`: the gaze samples the same device recorded, as a TSV file. They
  share the video's clock, and so its `time_offset`.
- `fixed_videos`: video from a camera at a fixed position in the room.
- `audio`: audio recorded on another device, such as a directional microphone
  aimed at one participant, or a single microphone recording the whole group.
  If the audio is from a specific participant, the optional `glasses_video`
  field identifies the id of the glasses worn by the participant.
  The video inputs carry their own audio, so this is
  for separately recorded audio only.

Every list may be empty: an experiment with no inputs at all is valid, which is
what a new one starts as before any files have been added to it.

## Pipeline

Each type of input has its own block under `pipeline`, so e.g. a room camera can
be tracked with a different detector than the glasses cameras, or skip a stage
they run. Omit a block to use its defaults.

For the video blocks — `glasses_video` and `fixed_video` — `object_tracking` is
required, while `face_detection` and `body_pose` are optional; omit either key or
set it to `null` to skip that stage. The `audio` block has no stages yet, so
audio inputs are currently only loaded and placed on the timeline.

## Object Tracking

Object tracking combines object detection, re-identification, and a tracker:

- `detector`: object detector model reference.
- `reid`: re-identification model used to keep track IDs stable.
- `tracker`: BoxMOT tracking algorithm.
- `object_classes`: COCO class IDs to detect.
- `embeddings_per_track`: number of best body-appearance embeddings to keep per
  tracklet.

## Face Detection

Face detection runs InsightFace inside each tracked person box and keeps the best
face above the configured threshold:

- `model_name`: InsightFace model pack.
- `det_size`: square detector input size.
- `det_thresh`: minimum face detection confidence.
- `embeddings_per_track`: number of best face embeddings to keep per tracklet.

## Body Pose

Body-pose detection runs an Ultralytics YOLO pose model inside each tracked
person box:

- `model_name`: YOLO pose weights.
- `conf`: minimum pose confidence.
