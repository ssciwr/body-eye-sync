# Speaker attribution from the loudest microphone

**Status:** steps 1-3 implemented; the post-processing tab and bleed rejection
are not (see the order of work at the end). The measurements come from
`/data/body-eye-sync/5min` (four glasses recordings and one GoPro, 5 minutes of
four people talking about food), using `large-v3-turbo` on a GPU.

## Why change anything

The speech pipeline diarizes each recording on its own, then attributes Whisper's
words to the turns it found. On the glasses recordings that fails badly:

| pipeline | speakers found | what the turns table shows |
| --- | --- | --- |
| diarization, current defaults | 24 | text in fragments, phantom speakers |
| diarization, `num_speakers=4` | 4 | 835 words, 44% of turns empty |
| loudest microphone, per segment | 4, named | 968 words, reads as a conversation |

Whisper itself is not the problem. Stock faster-whisper on `G3_1401.mp4` detects
German with probability 1.00 and returns clean sentences; our own call returns
the same text. Three separate things degrade it on the way to the table:

1. **Over-clustering.** `threshold=0.5` with `num_speakers=-1` gives 24 clusters
   for four people. A head-mounted microphone hears its wearer loudly and
   everyone else faintly, and the wearer's own level swings as they move, so
   speaker embeddings from one recording do not cluster into one speaker per
   person. The embedding model is also English (VoxCeleb) on German audio.
2. **Overlapping turns starve each other.** `assign_words` gives each word to the
   turn it overlaps *most* in absolute seconds, so a 0.4 s backchannel nested
   inside a 10 s turn can never win a word: it stays empty and the long turn
   absorbs words another person said.
3. **`vad_filter=True`** drops 429 of 813 words on this material -- it removes
   quiet speech, which mostly means everyone who is not the wearer.

Fixing all three would still leave the fundamental problem: diarization has to
*discover* speakers that we already know, and then those speakers have to be
matched across recordings afterwards.

## The idea

Each glasses recording belongs to a known wearer, and the wearer is much louder
on their own microphone than anyone else is. So attribution can come from
comparing recordings rather than clustering within one: whoever's microphone is
loudest is whoever is speaking, and their identity is which recording it is.

This is what close-talking microphone arrays do, and this setup is a better case
than most: the recordings are already on one clock, with drift and gaps
corrected, and each maps to exactly one participant.

### Measured separation

Over the 290 s all five recordings cover, at 50 ms resolution:

- Median gap between the loudest microphone and the runner-up: **10.6 dB**.
  85% of speech frames are decided by more than 3 dB, 71% by more than 6 dB.
- The GoPro wins **0.3%** of frames. A distant microphone never beats a close one.
- Dominance splits 19% / 18% / 26% / 37% across the four wearers, which is a
  plausible turn distribution rather than noise.
- Cut into turns: 272 turns, median 0.6 s, each wearer holding 43-86 s.

### Granularity matters

Deciding per **word** does not work. Momentary flips -- a breath, a backchannel,
a head turn -- let another microphone win for 200 ms, so single words defect
mid-sentence. Worse, words then get stitched together from four different
transcripts, which produces duplicates and overlaps:

```
[  4.8-  5.2] G2_1404: Absolut,
[  5.4-  5.5] G2_1403: glaube,
[  5.5-  8.8] G2_1404: gar nicht so viel habe. Ich glaube, es wird noch mehr...
[  8.8-  9.0] G3_1402: es
```

Deciding per **Whisper segment** works. The sentence goes whole to whoever wins
most of its span, so its text always comes from one recording:

```
[  0.0-  1.5] G3_1401: Aber Nudeln denn echt nice?
[  2.0-  4.2] G2_1403: Ja, ich glaube, wir essen alle so viele Nudeln.
[  4.8- 10.5] G2_1404: Absolut, obwohl ich gar nicht so viel habe. Ich glaube, es
                       wird noch mehr, wenn ich dann wirklich so in die
                       Endexamsphase komme...
[ 10.5- 13.4] G2_1403: Das geht so schnell und du kannst Pesto draufhauen.
[ 12.0- 12.4] G3_1401: Ja.
```

147 segments, 968 words, and the backchannels land on someone other than the
person holding the floor.

## Two people talking at once

Picking one winner per segment drops the quieter speaker, which is wrong: people
do talk over each other. The question is whether a second loud microphone is a
second speaker, or the first speaker bleeding into someone else's microphone.

Comparing what each recording transcribed inside the same stretch of experiment
time answers it. Bleed produces the *same words*; two speakers produce
*different words*:

```
bleed         sim 0.63   "ich hab sicherlich schneller mit der presse als ich"
                     vs  "ich glaube, du bist sicherlich schneller mit der presse"

both talking  sim 0.00   "ja, ja, ja, echt so, ja, ich will un..."
                     vs  "wir haben voll aufgegeben, nach einem satz"
```

Word agreement has to be measured over a **shared time window**, not between
segment pairs: Whisper segments each recording differently, so the same speech
lands in different windows and segment-to-segment comparison understates the
agreement. This pair is plainly bleed but scored only 0.45 segment-to-segment:

```
louder : Absolut, obwohl ich gar nicht so viel habe. Ich glaube, es wird noch mehr...
quieter: Ich glaube, es wird noch mehr, wenn ich das in die Endex-Ausbau...
```

### Signals that do not work

Recorded so nobody spends time re-testing them.

| signal | bleed | both talking | verdict |
| --- | --- | --- | --- |
| Whisper word probability | 0.78 | 0.70 | **backwards**, unusable |
| Whisper `avg_logprob` | −0.43 | −0.48 | negligible |
| envelope correlation | 0.09 | 0.09 | **no separation at all** |
| quieter mic above own floor, inside joint windows | 19.2 dB | 19.3 dB | none (see below) |
| **level margin between the two** | **9.6 dB** | **5.4 dB** | usable, secondary |
| **word agreement in a shared window** | high | low | **primary signal** |

Confidence is anti-correlated because bleed is a clean copy of loud, clear
speech, merely attenuated, and so is *easy* to transcribe, while somebody
speaking under another person is genuinely hard audio.

Envelope correlation was expected to be high for bleed, since a voice reaching
another microphone should track the original scaled down. At 50 ms resolution
over short windows it does not: median 0.09 in both classes, quartiles
indistinguishable.

An earlier measurement suggested the quieter microphone's level above its own
floor separated the classes (9 dB vs 13 dB). That was a **selection artifact** --
it compared all overlapping segment pairs, whereas the windowed analysis requires
both microphones to be loud by construction. Level above floor is the right test
for *whether a microphone is live at all*, not for what a second live microphone
means.

## Proposed design

### Where it sits

Two stages, split by what each one needs to see.

**Audio processing, per input.** Runs models over one recording at a time, which
is now just Whisper: transcribe, keep the segments and the word timings. No
speaker appears anywhere, because a recording on its own cannot say who was
speaking -- it can only say what was said.

**Speech post processing, experiment-wide.** Takes every recording's transcript
together with the levels, and produces one table of speech turns for the whole
experiment: times on the experiment clock, speakers identified by the glasses
input each belongs to, and turns allowed to overlap because people talk over each
other.

This mirrors the video side, where per-input passes find boxes and faces and a
later cross-input stage has to decide which of them are the same person. Speech
now has the same shape: per-input models, then one stage that relates the
recordings to each other.

It also removes an ordering constraint the current pipeline has. Today
transcription cannot run until diarization has produced turns for its words to be
attributed to; `Speech.finish_transcription` merges text onto the diarized turns
and does nothing at all if `data` is `None`. With attribution moved out, Whisper
runs on its own and needs nothing before it.

### Results

Per input, from Audio processing -- the transcript, on the recording's own clock:

| file | columns |
| --- | --- |
| `transcript_segments.parquet` | `segment_id`, `start`, `end`, `text` |
| `transcript_words.parquet` | `segment_id`, `word_index`, `start`, `end`, `word`, `score` |

Experiment-wide, from Speech post processing -- the speech turns, on the
experiment clock:

| column | meaning |
| --- | --- |
| `turn_id` | identifies the turn |
| `start`, `end` | experiment time; turns may overlap |
| `speaker` | the glasses input id of whoever was speaking |
| `source` | the input the text was taken from, normally the same recording |
| `text` | what they said |

The experiment-wide table has no per-input home, so it needs somewhere new to
live -- `outputs/speech_turns.parquet` beside the per-input directories is the
obvious spot, and the same place a cross-input person-identity table would go.

The current per-input `speaker_turns.parquet` and `speaker_embeddings.parquet`
are diarization outputs and disappear with it, along with `Speech`'s turn
handling; what is left of `Speech` is the transcript.

### Algorithm

1. For every input with audio, decode to 16 kHz mono and compute frame RMS in dB
   at a 50 ms hop.
2. Map frame times onto the experiment clock with the input's `Timeline`
   (`to_experiment_times`), and resample onto a common grid.
3. Per recording, take the 10th percentile of its own levels as its floor. This
   absorbs fixed gain differences between devices. A frame is *live* when it is
   more than 12 dB above that floor.
4. Transcribe each glasses recording independently, with word timestamps.
5. For each transcribed segment, look at the frames it spans. If this recording
   is live and the loudest for more than half of them, the wearer keeps the whole
   segment.
6. Where two or more recordings are live together for at least ~0.6 s, compare
   the words each placed in that window. High agreement means bleed: keep only
   the louder. Low agreement means both are talking: keep both.
7. Fixed cameras have no wearer, so they take the attribution computed from the
   glasses rather than diarizing their own audio.

### Parameters

| parameter | proposed | basis |
| --- | --- | --- |
| frame hop | 50 ms | shorter than the shortest word |
| floor percentile | 10th | robust to how much a wearer talks |
| live threshold | 12 dB above floor | separates speech from room noise here |
| segment ownership | >50% of frames | whole-sentence decision, resists flips |
| joint window | ≥0.6 s | shorter windows hold too few words to compare |
| word agreement for bleed | ~0.5 | **unvalidated**, see open questions |

### What changes in the code

- `SpeechPipeline` keeps transcription alone, so the Audio processing tab runs
  one model and its "Run all" is a single pass. The tab's structure is otherwise
  unchanged: it already runs per-input models with a pipeline editor beside the
  results, and its turns table becomes a transcript table.
- `Speech` sheds turns, speakers and voice embeddings, and becomes the
  transcript: segments and words, on the recording's own clock. What is now
  `finish_transcription` -- attributing words to turns -- moves to the new stage.
- A new module for the levels and the attribution, e.g.
  `preprocessing/attribution.py`. It belongs with alignment rather than with the
  model wrappers in `pipeline/`: it is arithmetic on timelines and text, and it
  loads no models.
- A new experiment-wide results object for the turns table, held by `Experiment`
  rather than by any input, with its own save/load.
- A new `SpeechPostProcessingTab`, sitting after Audio processing. It acts on the
  whole experiment rather than a selected input, so it looks more like the
  Alignment and Timing correction tabs -- one button, a results table, a
  progress bar -- than like the per-input tabs. It must refuse to run, saying
  why, until every input has a transcript and the recordings have been aligned.
- `run.py` gains a stage that runs once per experiment, after the per-input loop.
- `TranscriptSegment` does **not** need Whisper's confidence fields after all --
  they were the reason to add them, and they turned out not to discriminate.

### Diarization is removed

Decided: it goes entirely, rather than being kept for recordings no glasses
cover. Speakers come from whose microphone was loudest, and nothing else finds
speakers any more.

The accepted consequence is that an experiment with no glasses inputs gets a
transcript with no speakers at all. That is a real loss of capability, taken
deliberately in exchange for having one way of doing this instead of two.

What goes with it:

- `pipeline/diarization.py`, `DiarizationStep`, `DiarizationWorker`, and their
  tests.
- Voice embeddings per speaker: `Speech`'s use of `TopK`, and
  `speaker_embeddings.parquet`. `embeddings.py` itself stays -- the video stages
  keep body and face embeddings through the same machinery.
- The **`sherpa-onnx` and `sherpa-onnx-core` dependencies**. Nothing else in the
  project uses them: alignment and timing correction do their own signal
  processing, and the remaining models are Whisper, YOLO and InsightFace.
- `pipeline/transcription.py`'s dependency on diarization. It currently imports
  `SpeakerSegment` for `assign_words` and `transcript_to_dataframe`; that
  attribution logic moves wholesale into the new stage, and transcription is left
  doing nothing but transcribing.

### Loading experiments saved before this

`_Model` sets `extra="forbid"`, so an `experiment.yaml` written today -- which
has a `speech.diarization` block, as the 5-minute test experiment does -- will
fail validation outright once the field is gone, taking the whole experiment with
it rather than just the stale setting.

So removal needs a migration: bump `CURRENT_VERSION`, and drop unknown keys under
`speech` when loading anything older. The per-input results have the same problem
in a milder form -- `speaker_turns.parquet` and `speaker_embeddings.parquet` are
simply ignored, which is fine, but they will sit in output directories forever
unless load-time cleanup removes them.

### What it removes

- No speaker embedding model, no clustering threshold, no `num_speakers`.
- No cross-recording speaker matching later: the wearer *is* the identity, so
  "speaker 2 on this recording is speaker 5 on that one" stops being a problem to
  solve.
- Each word is transcribed from the closest microphone that heard it, instead of
  every recording transcribing everybody.

## Limits

- **Alignment errors become attribution errors.** Words are a few hundred
  milliseconds long, so an offset error of that size silently assigns speech to
  the wrong person. It fails quietly, which is worse than failing loudly.
- **Anyone without glasses is invisible.** Their speech is claimed by whichever
  wearer sits closest. Nothing in the level data reveals this, and with
  diarization gone there is no longer a fallback that would have found them.
- **An experiment with no glasses gets no speakers**, only a transcript.
- **Short similar interjections are indistinguishable from bleed.** Two people
  saying "ja" over each other look like one person's "ja" in two microphones.
  Probably acceptable, but it is a known loss, not an accident.
- **Head movement and gain changes** are only partly handled by the per-recording
  floor.
- **It needs every recording**, so it cannot run on a single input, and a missing
  or unreadable recording degrades everyone's attribution, not just its own.

## Open questions

- The word-agreement threshold needs labelled overlap to set properly. Everything
  above rests on the transcripts reading coherently, which is suggestive but is
  not ground truth. A few minutes of hand-labelled speaker turns would settle
  both this and the live threshold.
- Only the glasses transcripts are used for attribution, so what are the fixed
  cameras' and standalone microphones' transcripts *for*? They are still produced
  and saved by the Audio processing tab, and nothing consumes them. Either they
  earn a use in the post-processing stage, or transcribing them is work nobody
  asked for.
- Does the existing (empty) Post processing tab become the video counterpart of
  this one -- matching people across cameras from their embeddings -- or should
  both cross-input stages share a single tab?
- `G3_1402` transcribed only 27 segments where the others gave 116-155, and its
  text is visibly worse (no punctuation, garbled words), despite winning 42 s of
  dominance. Unexplained, and worth understanding before trusting per-recording
  results.

## Suggested order of work

1. ~~**Remove diarization and split the per-input stage.**~~ Done: `Speech` is
   the transcript, the Audio processing tab runs Whisper alone, and the
   sherpa-onnx dependencies are gone.
2. ~~**Levels and dominance** on the experiment clock.~~ Done:
   `preprocessing/attribution.py`, tested on synthetic audio whose speaker is
   known by construction.
3. ~~**Whole-segment attribution** into the experiment-wide turns table.~~ Done:
   `experiment/speech_turns.py` holds the results, `experiment/attribution.py`
   is the glue, and `Experiment` saves and loads them.
4. **Speech post processing tab** over that table, plus `run.py` and docs.
   Nothing in the app produces the turns yet -- the stage has to be called
   directly.
5. **Joint-window bleed rejection**, so simultaneous speech survives. Until it
   lands the table keeps only the loudest speaker in an overlap.

Steps 2 and 3 are what produced the coherent transcript above. Step 5 is the part
that still needs a threshold chosen against labelled data, and until it lands the
table simply keeps the loudest speaker in an overlap -- which is what the current
pipeline effectively does too, so it is not a regression.
