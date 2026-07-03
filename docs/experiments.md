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

The current experiment format is versioned. Video inputs may use absolute paths
or paths relative to the experiment folder.

```yaml
version: 1
name: demo
inputs:
  - kind: video
    id: camera-1
    path: videos/camera-1.mp4
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
```

`object_tracking` is required. `face_detection` and `body_pose` are optional; omit
either key or set it to `null` to skip that step.

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
