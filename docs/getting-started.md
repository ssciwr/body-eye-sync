# Getting Started

The window is a set of tabs, one per stage of working with an experiment:
**Input files**, **Alignment**, **Timing correction**, **Video processing**,
**Audio processing**, **Post processing** and **Data export**. Audio
processing, post processing and data export do nothing so far.

The title bar names the folder the open experiment is saved to, or says
`[unsaved experiment]` until it has one.

## Add Input Files

Start the desktop app:

```bash
body-eye-sync
```

It opens on a new, empty and unsaved experiment. In the **Input files** tab, add
the recordings it is made of: glasses videos, fixed videos and separately
recorded audio. Each input is identified by an id, taken from its filename and
editable in the table.

A glasses video also needs the gaze file the same device recorded. If it sits
beside the video it is picked up automatically; otherwise you are asked for it,
and a video with no gaze file is not added. The **Gaze file** column shows which
one is in use, and changes it.

## Place the Inputs on One Timeline

Every recording starts whenever its device was switched on, so nothing relates
them until each input's offset is known. In the **Alignment** tab, **Automatic
alignment** finds those offsets from the audio every input shares, and writes
them into the inputs. It needs at least two inputs that have a file.

The **Timing correction** tab then checks whether a single offset actually held
for the whole recording. **Analyse and correct timing** measures each input
against whichever one overlaps the others most, and reports two further things
per input:

- **Drift** -- a device whose clock ran slightly fast or slow, in parts per
  million.
- **Gaps** -- stretches a device stalled and never wrote, listed as the time
  each one falls at and how much content is missing.

Only inputs that need one of those corrections are changed; an input that kept
time keeps the offset alignment gave it. Drift and gaps too small to tell apart
from measurement noise are ignored, which is the usual outcome. An input whose
audio could not be lined up against the reference reads **Couldn't match**
rather than being reported as having kept time -- it may overlap too little, be
too quiet, or need aligning first.

## Configure the Pipeline

The **Video processing** tab shows one video input at a time, chosen with the
selector above the viewer, with the pipeline editor beside it. Each video type
has its own pipeline settings, so a fixed camera can be tracked differently to
the glasses cameras.

Object tracking is always present, because face detection and body-pose detection
run on tracked person boxes. The optional steps can be enabled or disabled in the
pipeline editor.

Useful defaults:

- `object_classes = [0]` tracks people using COCO class IDs.
- `embeddings_per_track = 32` keeps the best body or face embeddings for each
  tracklet.
- Face detection and body-pose detection can be run after object tracking
  results exist.

## Run Steps

Use **Run all** to run every enabled step in order. You can also run an
individual step:

- Object tracking can run once a video input has been added.
- Face detection can run once object tracking results exist.
- Body-pose detection can run once object tracking results exist.

The viewer shows live overlays while a step runs. Use **Cancel** to stop a
running step; partial results from the cancelled step are discarded.

## Save the Experiment

Use **File -> Save**. The first save asks where to put the experiment and what
to call its folder, which is created for you; leave the name empty to save into
the folder you chose. Later saves go straight there. Body Eye Sync writes:

- `experiment.yaml` with the experiment definition.
- `outputs/<input-id>/results.parquet` for completed model outputs.
- Optional companion embedding files when embeddings were collected.

Saved experiments can be reopened in the GUI or processed through the CLI.

Anything that would drop the open experiment -- closing the window, **File ->
New**, **File -> Open** -- asks first if it has unsaved changes, offering to
save them, discard them, or stay where you are.
