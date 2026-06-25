# AMGAZE Tracklet Clustering

This repository contains a small pipeline for assigning BoxMoT person tracklets
from an egocentric video to stable person identities. It combines body
re-identification, face recognition, clustering, and an optional fixed-layout
constraint for scenes where people stay in the same relative seating positions.

The main scripts are:

- `amgaze.cluster_tracklets`: cluster tracklets into person IDs and write output
  files.
- `amgaze.extract_face_landmarks`: run face landmark detection over the tracked
  faces and write a CSV.
- `amgaze.align_gaze_timestamps`: extract the overlaid gaze cursor from the video,
  estimate the TSV/video timestamp offset, and write an aligned gaze TSV.
- `amgaze.compute_gaze_on_faces`: intersect aligned gaze samples with detected face
  boxes and write the gazed-at person/nearest facial feature per timepoint.
- `amgaze.pipeline_backend`: reusable tracking, alignment, face preprocessing,
  clustering, and visualization stages.
- `amgaze.gui`: run the complete pipeline in a four-tab Qt GUI and
  inspect the video output after every stage.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

The project depends on BoxMOT, Ultralytics, PyTorch, Torchreid, InsightFace,
ONNX Runtime, OpenCV, NumPy, scikit-learn, qtpy, and PySide6. If CUDA is
available, embedding extraction uses it automatically; otherwise it runs on
CPU.

Launch the application with:

```bash
uv run amgaze
```

## BoxMOT

BoxMOT does multi-object tracking, which tracks people and objects as they enter
and leave the video, resulting in a "tracklet".

This works by collecting per-frame features:

- a YOLO (You Only Look Once) detector identifies objects (including people) and their bounding boxes
- a ReID (Re-Identification) model identifies people, mostly by their clothing

Then linking these together between frames:

- the BoT-SORT tracker combines this information with previous state and motion
  to assign stable identities to objects across frames ("tracklets").

Note that if a person leaves and then enters the video again it will be given a new tracklet ID, so the output of this step is likely hundreds of tracklets for a few people.

The first GUI tab runs BoxMOT through its Python API, restricted to person
detections. Its editable defaults are `yolo26l`, `osnet_x0_25_msmt17`,
BoT-SORT, and CUDA device `0`. Model weights may be downloaded on the first
run.


## GUI Workflow

Choose an input video, gaze TSV, and run directory in the first tab. Then run
the tabs in order:

1. **Tracking** runs BoxMOT and previews tracklet boxes and IDs.
2. **Gaze alignment** aligns the eye-tracking timestamps and adds the red gaze
   cross to the preview.
3. **Faces / ReID** extracts reusable body/face embeddings and facial
   landmarks. Its preview shows facial features without person clustering.
4. **Clustering** reads the embedding cache, applies the selected options, and
   previews final person IDs, facial features, and gaze. This stage performs no
   model inference, so options such as fixed-layout constraints and person count
   can be changed and rerun cheaply.

The run directory contains these principal artifacts:

```text
tracks.txt
tracks.mp4
gaze_aligned.tsv
video_gaze_marker.csv
embeddings.pkl
face_landmarks_unclustered.csv
clustering/track_to_person.csv
clustering/tracks_with_person.txt
clustering/face_landmarks.csv
clustering/gaze_on_faces.tsv
```

## Clustering Pipeline

Run the default clustering pipeline:

```bash
uv run python -m amgaze.cluster_tracklets \
  --tracks runs/track/quartet2/tracks.txt \
  --video Archiv/quartet.mp4 \
  --out-dir runs/track/quartet2/clustering
```

The script performs these steps:

1. Parses `tracks.txt` into per-tracklet detections.
2. Samples high-quality detections from each tracklet, spread across time.
3. Crops each sampled detection from the original video.
4. Computes body embeddings with Torchreid OSNet.
5. Computes face embeddings with InsightFace when a face is detected.
6. Averages embeddings per tracklet.
7. Clusters face-bearing tracklets by face distance with agglomerative
   clustering.
8. Attaches faceless tracklets to the nearest face cluster by body distance.
9. Writes person assignments, an annotated tracks file, a summary, and montage
   images.

## Models Used

Body re-identification:

- Library: Torchreid
- Default model: `osnet_x1_0`
- Default weights: `weights/osnet_x1_0_msmt17.pt`
- Feature: 512-dimensional body appearance embedding

Face recognition:

- Library: InsightFace
- Default model pack: `buffalo_l`
- Feature: 512-dimensional ArcFace embedding
- Face detector threshold: controlled by `--face-det-thresh`

Clustering:

- Constrained average-linkage agglomerative clustering
- Default linkage: average
- Default distance: precomputed cosine distance
- Tracklets visible in the same frame have an always-on hard cannot-link
  constraint and can never receive the same person ID.
- Number of people is selected by silhouette score unless `--n-clusters` or
  `--distance-threshold` is provided.

## Clustering Modes

Clustering is always face-primary, which is suited to egocentric or dim footage
where body crops are less reliable:

```bash
uv run python -m amgaze.cluster_tracklets \
  --tracks runs/track/quartet2/tracks.txt \
  --video Archiv/quartet.mp4 \
  --out-dir runs/track/quartet2/clustering_facep
```

This clusters tracklets that have face embeddings using face distance first,
then attaches tracklets without faces to the nearest face cluster using body
embeddings.

### Fixed-Layout Mode

If people remain in fixed relative positions, for example seated around a table,
add `--fixed-layout`:

```bash
uv run python -m amgaze.cluster_tracklets \
  --tracks runs/track/quartet2/tracks.txt \
  --video Archiv/quartet.mp4 \
  --out-dir runs/track/quartet2/clustering_facep_fixed_layout \
  --fixed-layout
```

This adds spatial consistency to the clustering:

- Tracklets with contradictory left/right relationships to shared neighboring
  tracklets receive an extra distance penalty.

This is useful when a single face embedding occasionally pulls a tracklet into
the wrong identity cluster, but the relative seating order makes that assignment
impossible or unlikely.

For a known number of people, force the cluster count:

```bash
uv run python -m amgaze.cluster_tracklets \
  --tracks runs/track/quartet2/tracks.txt \
  --video Archiv/quartet.mp4 \
  --out-dir runs/track/quartet2/clustering_facep_fixed_layout \
  --fixed-layout \
  --n-clusters 5
```

The requested count must be compatible with the hard co-presence constraints.
Clustering fails with a clear error if more tracklets are visible in one frame
than the requested number of people, rather than assigning duplicate IDs.

Person IDs from clustering are arbitrary. To make a new result reuse the IDs
from an existing result directory, add `--align-labels-to`:

```bash
uv run python -m amgaze.cluster_tracklets \
  --tracks runs/track/quartet2/tracks.txt \
  --video Archiv/quartet.mp4 \
  --out-dir runs/track/quartet2/clustering_facep_fixed_layout \
  --fixed-layout \
  --align-labels-to runs/track/quartet2/clustering_facep
```

This remaps output person IDs by maximizing tracklet overlap with the reference
result. The inspector uses this automatically when it prepares fixed-layout
comparison data.

## Reusing Embeddings

Embedding extraction is the slow step. Reuse cached embeddings when comparing
different clustering options:

```bash
uv run python -m amgaze.cluster_tracklets \
  --tracks runs/track/quartet2/tracks.txt \
  --video Archiv/quartet.mp4 \
  --out-dir runs/track/quartet2/clustering_facep_fixed_layout \
  --fixed-layout \
  --embedding-cache runs/track/quartet2/clustering_facep/embeddings.pkl \
  --reuse-embeddings
```

If `--embedding-cache` is omitted, the cache defaults to:

```text
<out-dir>/embeddings.pkl
```

## Outputs

Each output directory contains:

- `track_to_person.csv`: one row per tracklet with the assigned `person_id`,
  number of detections, sample counts, and whether a face was found.
- `tracks_with_person.txt`: the original tracks file with `person_id` appended
  as an extra final column.
- `summary.json`: clustering metadata and the tracklets assigned to each
  person.
- `embeddings.pkl`: cached body/face embeddings and thumbnails, unless a shared
  `--embedding-cache` path was used.
- `montages/`: one image per person cluster, useful for quick visual checking.
- `labelled.mp4`: only written when `--render-video` is used.

Render a labelled verification video:

```bash
uv run python -m amgaze.cluster_tracklets \
  --tracks runs/track/quartet2/tracks.txt \
  --video Archiv/quartet.mp4 \
  --out-dir runs/track/quartet2/clustering_facep \
  --reuse-embeddings \
  --render-video
```

## Face Landmark Extraction

Facial landmarks are extracted as a separate offline stage. The viewer does not
run any model inference; it only displays landmarks already written to CSV.

Run landmark extraction after the fixed-layout result exists:

```bash
uv run python -m amgaze.extract_face_landmarks \
  --video Archiv/quartet.mp4 \
  --tracks runs/track/quartet2/clustering_facep_fixed_layout/tracks_with_person.txt \
  --out runs/track/quartet2/clustering_facep_fixed_layout/face_landmarks.csv
```

The script uses InsightFace `buffalo_l` to detect faces inside each tracked
person box and writes:

- frame and video frame index
- track ID and person ID
- tracklet box coordinates
- detected face box and confidence
- five facial landmarks: left eye, right eye, nose, left mouth corner, right
  mouth corner

By default it processes every tracklet detection in the video. To make a faster
preview CSV, process every Nth MOT frame:

```bash
uv run python -m amgaze.extract_face_landmarks --stride 5
```

The default output path is:

```text
runs/track/quartet2/clustering_facep_fixed_layout/face_landmarks.csv
```

## Gaze Timestamp Alignment

The raw gaze TSV timestamps may not start at the same point as the video clip.
The original exported video contains a magenta gaze cursor. The alignment stage
uses this cursor as a reference:

1. Detect the magenta gaze cursor in the video.
2. Compare those video-derived cursor positions to the TSV `gaze_x/gaze_y`
   positions under candidate timestamp offsets.
3. Pick the offset that minimizes the median pixel distance.
4. Write a shifted TSV with video-aligned timestamps.

Run:

```bash
uv run python -m amgaze.align_gaze_timestamps \
  --video Archiv/quartet.mp4 \
  --gaze-tsv Archiv/G3_1401mp4.tsv \
  --out runs/track/quartet2/gaze_aligned.tsv
```

Outputs:

- `runs/track/quartet2/video_gaze_marker.csv`: gaze cursor positions extracted
  from the video.
- `runs/track/quartet2/gaze_aligned.tsv`: original TSV rows plus
  `video_time_ms`, `video_frame`, and `alignment_offset_ms`.
- `runs/track/quartet2/gaze_alignment_summary.json`: estimated offset and
  alignment error.

The offset definition is:

```text
video_time_ms = gaze_video_time + offset_ms
```

For a faster rough pass, process fewer video frames:

```bash
uv run python -m amgaze.align_gaze_timestamps --marker-stride 5
```

## Gaze-On-Face Assignment

After gaze timestamps are aligned and face landmarks have been extracted, assign
each gaze sample to the detected face box it falls inside:

```bash
uv run python -m amgaze.compute_gaze_on_faces \
  --gaze runs/track/quartet2/gaze_aligned.tsv \
  --landmarks runs/track/quartet2/clustering_facep_fixed_layout/face_landmarks.csv \
  --out runs/track/quartet2/clustering_facep_fixed_layout/gaze_on_faces.tsv
```

The output keeps all gaze TSV rows and appends:

- `gaze_on_face`: `1` if the gaze point is inside a face box, otherwise `0`
- `gaze_person_id`: clustered person ID for the gazed-at face
- `gaze_track_id`: tracklet ID for the gazed-at face
- `face_x1`, `face_y1`, `face_x2`, `face_y2`: selected face box
- `nearest_face_feature`: nearest of `left_eye`, `right_eye`, `nose`,
  `mouth_left`, `mouth_right`
- `nearest_feature_distance`: pixel distance from gaze to that feature

The default output path is:

```text
runs/track/quartet2/clustering_facep_fixed_layout/gaze_on_faces.tsv
```

## Pipeline GUI

Start the GUI with:

```bash
uv run amgaze
```

Each tab has its own stage settings, run button, log, and embedded video player.
Long-running work executes outside the GUI thread. The video controls support:

- Play/pause.
- Frame-by-frame seeking with the slider.
- Automatic loading of existing artifacts from the default run directory at
  startup.

To reopen another run, select its original video and run directory in the
Tracking tab, then click **Load existing workspace**. Every cached stage found
in that directory is loaded into its corresponding tab; missing stages remain
available to run normally.

## Main `amgaze.cluster_tracklets` Arguments

Input/output:

- `--tracks PATH`: input BoxMoT/MOT tracks file.
- `--video PATH`: original video.
- `--out-dir PATH`: output directory.
- `--embedding-cache PATH`: cache path for embeddings.
- `--reuse-embeddings`: load cached embeddings instead of recomputing them.

Sampling:

- `--max-samples INT`: maximum sampled detections per tracklet. Default: `30`.
- `--min-conf FLOAT`: minimum detection confidence for preferred samples.
  Default: `0.5`.
- `--min-w FLOAT`: minimum box width for preferred samples. Default: `40`.
- `--min-h FLOAT`: minimum box height for preferred samples. Default: `80`.
- `--frame-offset INT`: maps MOT frame numbers to video frame indices.
  Default: `-1`.
- `--batch-size INT`: body embedding batch size. Default: `256`.

Models:

- `--reid-model NAME`: Torchreid model name. Default: `osnet_x1_0`.
- `--reid-weights PATH`: Torchreid weights path.
  Default: `weights/osnet_x1_0_msmt17.pt`.
- `--face-model NAME`: InsightFace model pack. Default: `buffalo_l`.
- `--face-det-thresh FLOAT`: minimum face detection score. Default: `0.55`.

Clustering:

- Face-bearing tracklets are always clustered by face first; faceless tracklets
  are then attached by body distance.
- Co-present tracklets are always assigned different person IDs, independently
  of fixed-layout mode.
- `--n-clusters INT`: force a fixed number of people.
- `--distance-threshold FLOAT`: use a cosine distance cut instead of silhouette
  auto-selection.
- `--kmax INT`: maximum number of people considered by silhouette search.
  Default: `10`.
- `--align-labels-to PATH`: remap output person IDs to best match an existing
  result directory, `track_to_person.csv`, or `tracks_with_person.txt`.

Fixed-layout constraints:

- `--fixed-layout`: enable fixed left/right seating constraints.
- `--layout-weight FLOAT`: extra distance added for contradictory left/right
  evidence. Default: `0.35`.
- `--layout-min-x-gap FLOAT`: ignore left/right evidence when box centers are
  closer than this many pixels. Default: `20.0`.
- `--layout-min-shared-anchors INT`: minimum shared third-party tracklets needed
  before adding a left/right contradiction penalty. Default: `1`.

Montages and rendered videos:

- `--montage-source body|face`: thumbnail source for montages. Default: `body`.
- `--montage-per-track INT`: maximum thumbnails per tracklet. Default: `4`.
- `--render-video`: write an annotated MP4.
- `--render-name NAME`: annotated video filename inside `out-dir`.
  Default: `labelled.mp4`.
- `--render-scale FLOAT`: scale factor for annotated video output. Default:
  `1.0`.

## Main `amgaze.extract_face_landmarks` Arguments

- `--video PATH`: original video. Default: `Archiv/quartet.mp4`.
- `--tracks PATH`: tracks file to process. This can be raw `tracks.txt` or a
  `tracks_with_person.txt` result file. Default:
  `runs/track/quartet2/clustering_facep_fixed_layout/tracks_with_person.txt`.
- `--out PATH`: output CSV path. Default:
  `runs/track/quartet2/clustering_facep_fixed_layout/face_landmarks.csv`.
- `--face-model NAME`: InsightFace model pack. Default: `buffalo_l`.
- `--face-det-thresh FLOAT`: minimum face detection score. Default: `0.55`.
- `--det-size INT`: InsightFace detector input size. Default: `640`.
- `--frame-offset INT`: maps MOT frame numbers to video frame indices.
  Default: `-1`.
- `--stride INT`: process every Nth MOT frame. Default: `1`.
- `--min-w FLOAT`: minimum tracklet box width to process. Default: `40`.
- `--min-h FLOAT`: minimum tracklet box height to process. Default: `80`.

## Main `amgaze.align_gaze_timestamps` Arguments

- `--video PATH`: video containing the overlaid magenta gaze cursor. Default:
  `Archiv/quartet.mp4`.
- `--gaze-tsv PATH`: raw gaze TSV. Default: `Archiv/G3_1401mp4.tsv`.
- `--out PATH`: shifted gaze TSV output. Default:
  `runs/track/quartet2/gaze_aligned.tsv`.
- `--marker-csv PATH`: extracted video cursor positions. Default:
  `runs/track/quartet2/video_gaze_marker.csv`.
- `--summary PATH`: alignment summary JSON. Default:
  `runs/track/quartet2/gaze_alignment_summary.json`.
- `--time-col NAME`: TSV timestamp column to shift, in milliseconds. Default:
  `gaze_video_time`.
- `--reuse-marker-csv`: reuse an existing marker CSV instead of extracting from
  video again.
- `--marker-stride INT`: extract video marker every Nth frame. Default: `1`.
- `--align-marker-stride INT`: use every Nth detected marker during offset
  search. Default: `5`.
- `--offset-min-ms FLOAT`: minimum candidate offset. Default: `-2000000`.
- `--offset-max-ms FLOAT`: maximum candidate offset. Default: `200000`.
- `--coarse-step-ms FLOAT`: coarse offset search step. Default: `1000`.
- `--fine-step-ms FLOAT`: fine offset search step. Default: `20`.
- `--tsv-smooth-ms FLOAT`: smooth TSV gaze before alignment. Default: `200`.

The remaining HSV and size arguments tune detection of the magenta video cursor:
`--hue-min`, `--hue-max`, `--sat-min`, `--val-min`, `--min-area`,
`--max-area`, `--min-diameter`, `--max-diameter`, `--min-aspect`,
`--max-aspect`, and `--morph-kernel`.

## Main `amgaze.compute_gaze_on_faces` Arguments

- `--gaze PATH`: aligned gaze TSV. Default:
  `runs/track/quartet2/gaze_aligned.tsv`.
- `--landmarks PATH`: face landmark CSV. Default:
  `runs/track/quartet2/clustering_facep_fixed_layout/face_landmarks.csv`.
- `--out PATH`: gaze-on-face TSV output. Default:
  `runs/track/quartet2/clustering_facep_fixed_layout/gaze_on_faces.tsv`.
- `--summary PATH`: summary JSON output. Default:
  `runs/track/quartet2/clustering_facep_fixed_layout/gaze_on_faces_summary.json`.
- `--frame-tolerance INT`: use face boxes from a nearby video frame if the exact
  frame has no landmarks. Default: `0`.

`amgaze.gui` is GUI-only. The individual processing modules retain
their command-line interfaces for automation and debugging.
