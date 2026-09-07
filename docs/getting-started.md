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
them until each input's offset is known. In the **Alignment** tab there is a button
to automatically align the inputs, and you can also manually set the offset for each video.

The **Timing correction** tab then checks whether a single offset actually held
for the whole recording, and detects and corrects:

- **Gaps** -- stretches where a device stalled and didn't write leading to missing content

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

The **Audio processing** tab transcribes the speech in every input that carries audio.

- `model_name` picks the Whisper model. The default
  `primeline/whisper-large-v3-turbo-german` is fine-tuned for highly accurate
  German transcription. `primeline/whisper-large-v3-german` is the full-size
  German alternative, while `large-v3` is the strongest general multilingual
  choice.
- `language` takes an ISO 639-1 code and defaults to `de`, matching the German
  model. Change it when selecting a multilingual model for another language, or
  leave it empty to detect the language from the first 30 seconds.
- `vad_filter` skips silent stretches.

## Work Out Who Said What

The **Speech post processing** tab turns those transcripts into one table of
speech turns for the whole experiment, with a speaker against each.
It uses the loudness of each glasses recording to decide who was speaking at any
given moment, since the glasses microphone hears its own wearer far louder than anyone else in the room.
If multiple recordings contain the same speech segment, only the loudest one is taken to be the speaker.
However if the recordings contain different speech segments, they are all kept and attributed to the loudest
speaker for each segment.

## Export a Combined Video

You can export a single video that shows every aligned video input in a grid,
with an audio track for each input, and optionally an additional merge audio track
that combines the audio from all inputs. And missing recording intervals become black
video and silence.

If the experiment has speech turns, they are written beside the video as it is
exported, in a `.eaf` ELAN file named after it. Each speaker has a tier of readable
speech turns and a dependent `<speaker>-words` tier carrying precise
word timings.

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
