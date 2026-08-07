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

Five settings beside the buttons control how the measurement is made, and the
defaults suit most recordings:

- **Window** -- how long each measurement window is. Shorter places a gap more
  precisely; longer locks onto quiet or sparse recordings more often. The
  windows are always spaced at half a window, which is what the gap fit expects.
- **Search** -- how far either side of the current offset each window looks for
  its lag. Widen it when the inputs are only roughly aligned, since a lag
  further out than this is never found.
- **Min quality** -- how far a window's best lag has to stand above the rest
  before it is believed. Lower it to get a reading out of a difficult
  recording, at the risk of a wrong lag being read as a gap.
- **Min gap** -- the smallest step in the offset worth calling lost content.
  Lower it to find more, smaller gaps, at the risk of reading measurement noise
  as one.
- **Fit drift** -- whether a recording may have a clock rate of its own. Off by
  default, because a device that stalls is far commoner than one whose crystal
  is out, and a free rate will absorb a run of small gaps into a slope no
  oscillator could produce. Turn it on for a recording whose offset really does
  climb steadily between its steps; the Drift column reads `0 ppm` for
  everything while it is off.

If a plot shows a staircase but the gaps found do not account for all of it,
those last two are the ones to reach for: lower **Min gap** until the steps are
picked up. It moves the whole fit, since the same threshold decides what the
drift estimate hands over to the gap detector.

They apply to the next run and are not saved with the experiment.

**Clear corrections** puts every input back on an unstretched, ungapped clock,
which is where a recording that needs no correction already sits. The offsets
are left alone: those say where each recording starts, which is the Alignment
tab's answer rather than this one's.

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
