"""Write the experiment's speech turns and words as an ELAN annotation file.

The turns say who spoke when across the whole experiment, on the experiment
clock, and the synchronized grid video runs on that same clock -- so the two
line up with nothing but a shift to where the video starts. The annotations are
written beside that video, named after it, which is where ELAN looks for them.

Each speaker gets a turn tier and, where word timings are available, a dependent
word tier named ``<speaker>-words``.
"""

from __future__ import annotations

import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.video_grid import VideoGridResult

_EAF_VERSION = "3.0"
_EAF_SCHEMA = "https://www.mpi.nl/tools/elan/EAFv3.0.xsd"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"

#: Linguistic types used by the turn and word tiers.
_SPEECH_TYPE = "speech"
_WORD_TYPE = "words"


@dataclass(frozen=True)
class ElanExportResult:
    """What an ELAN export wrote."""

    path: Path
    media_path: Path
    #: Speaker turn tier ids; dependent word tiers are derived from these.
    tiers: tuple[str, ...]
    #: Total turn and word annotations written.
    annotations: int
    #: Turns that fall outside the video and so have nowhere to go.
    skipped: int


@dataclass(frozen=True)
class _Annotation:
    start_ms: int
    end_ms: int
    text: str


def _speech_tiers(
    turns: pd.DataFrame, video: VideoGridResult
) -> tuple[dict[str, list[_Annotation]], int]:
    """The turns as annotations on the video's clock, keyed by speaker."""
    duration_ms = round((video.experiment_end - video.experiment_start) * 1000)
    tiers: dict[str, list[_Annotation]] = {}
    skipped = 0
    for turn in turns.sort_values(["speaker", "start", "end"]).itertuples(index=False):
        # Rounded to the nearest millisecond rather than outwards at both ends:
        # ELAN works in whole milliseconds, and widening every turn would make
        # two that merely meet at a boundary overlap by one.
        start_ms = round((float(turn.start) - video.experiment_start) * 1000)
        end_ms = round((float(turn.end) - video.experiment_start) * 1000)
        start_ms, end_ms = max(0, start_ms), min(duration_ms, end_ms)
        if end_ms <= start_ms:
            # The video does not cover this turn at all.
            skipped += 1
            continue
        text = "" if pd.isna(turn.text) else str(turn.text)
        tiers.setdefault(str(turn.speaker), []).append(
            _Annotation(start_ms, end_ms, text)
        )

    for speaker, spoken in tiers.items():
        spoken.sort(key=lambda item: (item.start_ms, item.end_ms))
        for previous, current in zip(spoken[:-1], spoken[1:]):
            if current.start_ms < previous.end_ms:
                # ELAN cannot hold two annotations over one another on a tier,
                # and nobody talks over themselves, so this means the turns are
                # not what they claim to be.
                raise ValueError(
                    f"speaker {speaker!r} has overlapping turns at "
                    f"{previous.start_ms}-{previous.end_ms} ms and "
                    f"{current.start_ms}-{current.end_ms} ms"
                )
    return tiers, skipped


def _word_tiers(
    experiment: Experiment,
    turns: pd.DataFrame,
    video: VideoGridResult,
) -> dict[str, list[_Annotation]]:
    """Accepted turns' Whisper words, on the video clock, keyed by speaker."""
    duration_ms = round((video.experiment_end - video.experiment_start) * 1000)
    sources = {source.id: source for source in experiment.glasses_videos}
    tiers: dict[str, list[_Annotation]] = {}

    for turn in turns.sort_values(["speaker", "start", "end"]).itertuples(index=False):
        source = sources.get(str(turn.source))
        if source is None:
            continue
        words = source.speech.words
        if words is None or words.empty:
            continue
        segment_words = words[
            words["segment_id"] == int(turn.source_segment_id)
        ].sort_values("word_index")
        for word in segment_words.itertuples(index=False):
            # Word timestamps belong to the source recording. Use the same
            # timeline mapping as its parent turn, then keep the word within
            # that turn and within the exported video.
            start = max(float(turn.start), source.to_experiment_time(float(word.start)))
            end = min(float(turn.end), source.to_experiment_time(float(word.end)))
            start_ms = round((start - video.experiment_start) * 1000)
            end_ms = round((end - video.experiment_start) * 1000)
            start_ms, end_ms = max(0, start_ms), min(duration_ms, end_ms)
            if end_ms <= start_ms:
                continue
            text = "" if pd.isna(word.word) else str(word.word).strip()
            tiers.setdefault(str(turn.speaker), []).append(
                _Annotation(start_ms, end_ms, text)
            )

    for speaker, spoken in tiers.items():
        spoken.sort(key=lambda item: (item.start_ms, item.end_ms))
        for previous, current in zip(spoken[:-1], spoken[1:]):
            if current.start_ms < previous.end_ms:
                raise ValueError(
                    f"speaker {speaker!r} has overlapping words at "
                    f"{previous.start_ms}-{previous.end_ms} ms and "
                    f"{current.start_ms}-{current.end_ms} ms"
                )
    return tiers


def _word_tier_ids(speakers: set[str]) -> dict[str, str]:
    """Unique, readable tier ids for speakers' dependent word tiers."""
    occupied = set(speakers)
    ids = {}
    for speaker in sorted(speakers):
        candidate = f"{speaker}-words"
        while candidate in occupied:
            candidate += "-words"
        occupied.add(candidate)
        ids[speaker] = candidate
    return ids


def _document(
    video: VideoGridResult,
    output_path: Path,
    speech_tiers: dict[str, list[_Annotation]],
    word_tiers: dict[str, list[_Annotation]],
    author: str,
    date: datetime,
) -> ET.ElementTree:
    """The EAF document itself, ready to write."""
    ET.register_namespace("xsi", _XSI)
    root = ET.Element(
        "ANNOTATION_DOCUMENT",
        {
            "AUTHOR": author,
            "DATE": date.isoformat(timespec="seconds"),
            "FORMAT": _EAF_VERSION,
            "VERSION": _EAF_VERSION,
            f"{{{_XSI}}}noNamespaceSchemaLocation": _EAF_SCHEMA,
        },
    )
    header = ET.SubElement(
        root, "HEADER", {"MEDIA_FILE": "", "TIME_UNITS": "milliseconds"}
    )
    media = video.path.resolve()
    relative = Path(os.path.relpath(media, output_path.parent.resolve())).as_posix()
    relative_url = quote(relative)
    if not relative_url.startswith("../"):
        relative_url = f"./{relative_url}"
    ET.SubElement(
        header,
        "MEDIA_DESCRIPTOR",
        {
            "MEDIA_URL": media.as_uri(),
            "RELATIVE_MEDIA_URL": relative_url,
            "MIME_TYPE": "video/mp4",
        },
    )
    # Where the video sits on the experiment clock, so annotation times can be
    # put back onto it by anything reading this file.
    ET.SubElement(
        header, "PROPERTY", {"NAME": "body-eye-sync:experiment-start-seconds"}
    ).text = f"{video.experiment_start:.12g}"

    time_order = ET.SubElement(root, "TIME_ORDER")
    slots: list[tuple[str, int]] = []
    prepared: list[tuple[ET.Element, _Annotation, int, str, str]] = []
    for speaker in sorted(speech_tiers):
        tier = ET.SubElement(
            root,
            "TIER",
            {
                "TIER_ID": speaker,
                "PARTICIPANT": speaker,
                "LINGUISTIC_TYPE_REF": _SPEECH_TYPE,
            },
        )
        for annotation in speech_tiers[speaker]:
            start_id = f"ts{len(slots) + 1}"
            slots.append((start_id, annotation.start_ms))
            end_id = f"ts{len(slots) + 1}"
            slots.append((end_id, annotation.end_ms))
            prepared.append((tier, annotation, len(prepared) + 1, start_id, end_id))

    word_tier_ids = _word_tier_ids(set(speech_tiers))
    for speaker in sorted(word_tiers):
        tier = ET.SubElement(
            root,
            "TIER",
            {
                "TIER_ID": word_tier_ids[speaker],
                "PARTICIPANT": speaker,
                "LINGUISTIC_TYPE_REF": _WORD_TYPE,
                "PARENT_REF": speaker,
            },
        )
        for annotation in word_tiers[speaker]:
            start_id = f"ts{len(slots) + 1}"
            slots.append((start_id, annotation.start_ms))
            end_id = f"ts{len(slots) + 1}"
            slots.append((end_id, annotation.end_ms))
            prepared.append((tier, annotation, len(prepared) + 1, start_id, end_id))

    for slot_id, value in sorted(slots, key=lambda slot: slot[1]):
        ET.SubElement(
            time_order,
            "TIME_SLOT",
            {"TIME_SLOT_ID": slot_id, "TIME_VALUE": str(value)},
        )

    for tier, annotation, index, start_id, end_id in prepared:
        wrapper = ET.SubElement(tier, "ANNOTATION")
        alignable = ET.SubElement(
            wrapper,
            "ALIGNABLE_ANNOTATION",
            {
                "ANNOTATION_ID": f"a{index}",
                "TIME_SLOT_REF1": start_id,
                "TIME_SLOT_REF2": end_id,
            },
        )
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = annotation.text

    ET.SubElement(
        root,
        "LINGUISTIC_TYPE",
        {
            "LINGUISTIC_TYPE_ID": _SPEECH_TYPE,
            "TIME_ALIGNABLE": "true",
            "GRAPHIC_REFERENCES": "false",
        },
    )
    ET.SubElement(
        root,
        "LINGUISTIC_TYPE",
        {
            "LINGUISTIC_TYPE_ID": _WORD_TYPE,
            "TIME_ALIGNABLE": "true",
            "GRAPHIC_REFERENCES": "false",
            "CONSTRAINTS": "Included_In",
        },
    )
    ET.SubElement(
        root,
        "CONSTRAINT",
        {
            "STEREOTYPE": "Included_In",
            "DESCRIPTION": (
                "Time alignable annotations within the parent annotation's time "
                "interval, gaps are allowed"
            ),
        },
    )
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def export_elan(
    experiment: Experiment,
    video: VideoGridResult,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    author: str = "body-eye-sync",
    date: datetime | None = None,
) -> ElanExportResult:
    """Write the experiment's speech turns and words beside the synchronized video.

    The file is named after the video unless ``output_path`` says otherwise, so
    ELAN offers it when the video is opened. Times are the experiment's, shifted
    to where the video starts; turns the video does not cover are left out.

    Each speaker's words are written on a dependent ``<speaker>-words`` tier.
    Raises :class:`ValueError` if the experiment has no speech turns, or if a
    speaker's turns or words overlap one another, which ELAN cannot represent.
    """
    turns = experiment.speech_turns.data
    if turns is None:
        raise ValueError(
            "this experiment has no speech turns; run speech post processing first"
        )
    if video.experiment_end <= video.experiment_start:
        raise ValueError("video result has an invalid experiment interval")

    output = (
        Path(output_path) if output_path is not None else video.path.with_suffix(".eaf")
    )
    if output.suffix.lower() != ".eaf":
        raise ValueError("ELAN output path must have an .eaf extension")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")

    speech_tiers, skipped = _speech_tiers(turns, video)
    word_tiers = _word_tiers(experiment, turns, video)
    document = _document(
        video,
        output,
        speech_tiers,
        word_tiers,
        author,
        date or datetime.now().astimezone(),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place, so a failure part way through cannot
    # leave half a file where ELAN expects a whole one.
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.eaf")
    try:
        document.write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    return ElanExportResult(
        path=output,
        media_path=video.path,
        tiers=tuple(sorted(speech_tiers)),
        annotations=sum(len(items) for items in speech_tiers.values())
        + sum(len(items) for items in word_tiers.values()),
        skipped=skipped,
    )
