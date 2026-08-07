"""Write the experiment's speech turns and words as an ELAN annotation file.

Each speaker gets a turn tier and a dependent more fine-grained word tier named
``<speaker> [words]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from pympi.Elan import Eaf

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.video_grid import VideoGridResult

# labels for the turn and word tiers
_SPEECH_TYPE = "speech"
_WORD_TYPE = "words"


@dataclass(frozen=True)
class _Annotation:
    start_ms: int
    end_ms: int
    text: str


def _speech_tiers(
    turns: pd.DataFrame, video: VideoGridResult
) -> dict[str, list[_Annotation]]:
    """The turns as annotations on the video's clock, keyed by speaker."""
    duration_ms = round((video.experiment_end - video.experiment_start) * 1000)
    tiers: dict[str, list[_Annotation]] = {}
    for turn in turns.sort_values(["speaker", "start", "end"]).itertuples(index=False):
        # ELAN works in whole milliseconds so round to the nearest millisecond
        start_ms = round((float(turn.start) - video.experiment_start) * 1000)
        end_ms = round((float(turn.end) - video.experiment_start) * 1000)
        start_ms, end_ms = max(0, start_ms), min(duration_ms, end_ms)
        if end_ms <= start_ms:
            # the video does not cover this turn at all.
            continue
        text = "" if pd.isna(turn.text) else str(turn.text)
        tiers.setdefault(str(turn.speaker), []).append(
            _Annotation(start_ms, end_ms, text)
        )

    for spoken in tiers.values():
        spoken.sort(key=lambda item: (item.start_ms, item.end_ms))
    return tiers


def _word_tiers(
    experiment: Experiment,
    turns: pd.DataFrame,
    video: VideoGridResult,
) -> dict[str, list[_Annotation]]:
    """Each individual word, on the video clock, keyed by speaker."""
    duration_ms = round((video.experiment_end - video.experiment_start) * 1000)
    speakers = {video.id: video for video in experiment.glasses_videos}
    tiers: dict[str, list[_Annotation]] = {}

    for turn in turns.sort_values(["speaker", "start", "end"]).itertuples(index=False):
        speaker = speakers.get(str(turn.speaker))
        if speaker is None:
            continue
        words = speaker.speech.words
        if words is None or words.empty:
            continue
        segment_words = words[
            words["segment_id"] == int(turn.source_segment_id)
        ].sort_values("word_index")
        for word in segment_words.itertuples(index=False):
            start = max(
                float(turn.start),
                speaker.timeline.to_experiment_time(float(word.start)),
            )
            end = min(
                float(turn.end), speaker.timeline.to_experiment_time(float(word.end))
            )
            start_ms = round((start - video.experiment_start) * 1000)
            end_ms = round((end - video.experiment_start) * 1000)
            start_ms, end_ms = max(0, start_ms), min(duration_ms, end_ms)
            if end_ms <= start_ms:
                continue
            text = "" if pd.isna(word.word) else str(word.word).strip()
            tiers.setdefault(str(turn.speaker), []).append(
                _Annotation(start_ms, end_ms, text)
            )

    for spoken in tiers.values():
        spoken.sort(key=lambda item: (item.start_ms, item.end_ms))
    return tiers


def _document(
    video: VideoGridResult,
    output_path: Path,
    speech_tiers: dict[str, list[_Annotation]],
    word_tiers: dict[str, list[_Annotation]],
    author: str,
) -> Eaf:
    document = Eaf(author=author)
    document.header.update({"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"})
    document.remove_tier("default")
    document.remove_linguistic_type("default-lt")
    document.remove_property()

    media = video.path.resolve()
    relative = Path(os.path.relpath(media, output_path.parent.resolve())).as_posix()
    relative_url = quote(relative)
    if not relative_url.startswith("../"):
        relative_url = f"./{relative_url}"
    document.add_linked_file(media.as_uri(), relpath=relative_url, mimetype="video/mp4")
    document.add_linguistic_type(_SPEECH_TYPE)
    document.add_linguistic_type(_WORD_TYPE, constraints="Included_In")

    for speaker in sorted(speech_tiers):
        document.add_tier(speaker, ling=_SPEECH_TYPE, part=speaker)
        for annotation in speech_tiers[speaker]:
            document.add_annotation(
                speaker,
                annotation.start_ms,
                annotation.end_ms,
                annotation.text,
            )

    for speaker in sorted(word_tiers):
        tier_id = f"{speaker} [words]"
        document.add_tier(
            tier_id,
            ling=_WORD_TYPE,
            parent=speaker,
            part=speaker,
        )
        for annotation in word_tiers[speaker]:
            document.add_annotation(
                tier_id,
                annotation.start_ms,
                annotation.end_ms,
                annotation.text,
            )

    return document


def export_elan(
    experiment: Experiment,
    video: VideoGridResult,
    *,
    overwrite: bool = False,
    author: str = "body-eye-sync",
) -> Path:
    """Write the experiment's speech turns and words beside the synchronized video."""
    turns = experiment.speech_turns.data
    if turns is None:
        raise ValueError(
            "this experiment has no speech turns; run speech post processing first"
        )
    if video.experiment_end <= video.experiment_start:
        raise ValueError("video result has an invalid experiment interval")

    output = video.path.with_suffix(".eaf")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")

    speech_tiers = _speech_tiers(turns, video)
    word_tiers = _word_tiers(experiment, turns, video)
    document = _document(
        video,
        output,
        speech_tiers,
        word_tiers,
        author,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        output.unlink(missing_ok=True)
    document.to_file(str(output))

    return output
