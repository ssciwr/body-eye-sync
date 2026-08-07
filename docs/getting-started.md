# Getting Started

The window is a set of tabs, one per stage of working with an experiment:
**Input files**, **Alignment**, **Timing correction**, **Video processing**,
**Audio processing**, **Speech post processing**, **Post processing** and
**Data export**. Post processing does nothing so far.

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

## Configure the Video Pipeline

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

## Transcribe the Speech

The **Audio processing** tab transcribes one recording at a time. The selector
above the results lists every input that carries audio -- the separately recorded
audio inputs, and the video inputs, which are transcribed from their own audio
track. Unlike the video pipeline, there is a single set of speech settings that
every input shares.

- `model_name` picks the Whisper model. The default
  `primeline/whisper-large-v3-turbo-german` is fine-tuned for highly accurate
  German transcription. `primeline/whisper-large-v3-german` is the full-size
  German alternative, while `large-v3` is the strongest general multilingual
  choice. The primeLine checkpoints are converted once on first use and cached;
  allow extra time and disk space for that first run.
- `language` takes an ISO 639-1 code and defaults to `de`, matching the German
  model. Change it when selecting a multilingual model for another language, or
  leave it empty to detect the language from the first 30 seconds.
- `device` defaults to `auto`, which uses a GPU when the machine has one, and can
  be set to `cpu` or `cuda` to choose.
- `vad_filter` skips silent stretches. It speeds the pass up and suppresses text
  Whisper otherwise invents over silence, but it also drops quiet speech, so
  switch it off when the recording has to catch people further from the
  microphone.

Transcript segments appear with their times and text as Whisper produces them.
They are kept as results only when the run finishes successfully; cancelling or
failing discards the provisional rows. Each input keeps its own completed
transcript, so switching recordings shows that recording's.

This says what was said, not who said it. Every microphone in the room hears
everybody, so a single recording cannot tell the voices apart; speakers are
worked out afterwards, by comparing all of the experiment's recordings with each
other.

## Work Out Who Said What

The **Speech post processing** tab turns those transcripts into one table of
speech turns for the whole experiment, with a speaker against each. It acts on
the experiment rather than on one input, because that is the only way the
question can be answered: a glasses microphone hears its own wearer far louder
than anyone else in the room, so whoever's recording is loudest at a given
moment is whoever is speaking -- and the speaker's name is simply whose
recording won.

**Attribute speech to speakers** measures how loud each glasses recording is,
places them all on the experiment clock, and gives each transcribed segment to
the wearer who was loudest for most of it. Whole segments are attributed rather
than single words, so a sentence stays intact and its text always comes from one
recording.

The button stays disabled, saying why, until the experiment is ready for it:

- at least two glasses recordings, since they are compared against each other,
- each of them transcribed in **Audio processing**,
- and all of them aligned, since attribution compares them moment by moment.

The last one matters more than it looks. Attribution reads the recordings'
offsets rather than measuring them, so recordings that were never aligned
produce confident nonsense instead of an obvious failure.

Turns may overlap: two people talking at once are two turns covering the same
stretch of time. Speech nobody clearly owns is left out, and so is everyone who
is not wearing glasses -- a participant without a headset has no recording of
their own to win, so their speech is attributed to whichever wearer sat closest.

## Export a Combined Video

The **Data export** tab lists every experiment input, initially checked. Checked
video inputs appear as labelled cells in a synchronized 25 fps grid; checked
audio-only inputs contribute audio without adding a cell. Each source carrying
audio gets its own selectable track.

Select **Include merged audio track** to append a default playback track that
mixes the synchronized audio from every checked source while retaining the
individual tracks. **Export combined video** asks where to write the MP4 and
shows progress while it is rendered. Missing recording intervals become black
video and silence.

If the experiment has speech turns, they are written beside the video as it is
exported, in a `.eaf` file named after it. Each speaker has a tier of readable
speech turns and a dependent `<speaker>-words` tier carrying Whisper's precise
word timings. ELAN offers the annotations when the video is opened, and the two
line up without any further work because both are on the experiment clock. Run
**Speech post processing** before exporting to have them; without any turns the
video is written on its own.

## Save the Experiment

Use **File -> Save**. The first save asks where to put the experiment and what
to call its folder, which is created for you; leave the name empty to save into
the folder you chose. Later saves go straight there. Body Eye Sync writes:

- `experiment.yaml` with the experiment definition.
- `outputs/<input-id>/results.parquet` for completed model outputs.
- `outputs/speech_turns.parquet` for the experiment's speech turns, which belong
  to no single input.
- Optional companion embedding files when embeddings were collected.

Saved experiments can be reopened in the GUI or processed through the CLI.

Anything that would drop the open experiment -- closing the window, **File ->
New**, **File -> Open** -- asks first if it has unsaved changes, offering to
save them, discard them, or stay where you are.
