# Experiments

An experiment is a folder containing an `experiment.yaml` file and, after a run,
an `outputs/` directory.

## Folder Layout

```text
my-experiment/
├─ experiment.yaml
└─ outputs/
   ├─ camera-1.parquet
   ├─ camera-1.body_embeddings.parquet
   └─ camera-1.face_embeddings.parquet
```

## Configuration Format

The current experiment format is versioned. Input paths may be absolute or
relative to the experiment folder.

```yaml
version: 1
name: demo
glasses_videos:
  - id: p1-glasses
    path: videos/p1.mp4
    time_offset: 0.0
fixed_videos:
  - id: camera-1
    path: videos/camera-1.mp4
    time_offset: -1.25
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
is unique across *all* the lists, since it names that input's output file. Each
also has a `time_offset` in seconds, which is added to that input's own clock to
place it on the experiment's shared timeline; it defaults to `0.0`.

This file is the on-disk form. At runtime each input is a `GlassesVideo`,
`FixedVideo` or `Audio` that owns these settings alongside its results, reached
through `Experiment.glasses_videos`, `.fixed_videos` and `.audio`. Inputs are
added, removed and renamed through `Experiment` so their ids stay unique.

- `glasses_videos`: video recorded by a participant's glasses-mounted camera.
  This is the input that carries eye tracking.
- `fixed_videos`: video from a camera at a fixed position in the room.
- `audio`: audio recorded on its own device, such as a directional microphone
  aimed at one participant. The video inputs carry their own audio, so this is
  for separately recorded audio only. An audio input may name the
  `glasses_video` worn by the participant it captures.

At least one input, of any type, is required.

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
