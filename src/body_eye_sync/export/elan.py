"""Write speech annotations accompanying a synchronized video in ELAN EAF."""

from __future__ import annotations

import math
import os
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.export.video_grid import VideoGridResult

_EAF_VERSION = "3.0"
_EAF_SCHEMA = "https://www.mpi.nl/tools/elan/EAFv3.0.xsd"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_SPEECH_COLUMNS = {"segment_id", "start", "end", "speaker"}


@dataclass(frozen=True)
class ElanExportResult:
    """Description of a completed ELAN annotation export."""

    path: Path
    media_path: Path
    tiers: tuple[str, ...]
    annotations: int


@dataclass(frozen=True)
class _Annotation:
    start_ms: int
    end_ms: int
    text: str


def _selected_inputs(experiment: Experiment, input_ids: Iterable[str] | None):
    available = {data.id: data for data in experiment.inputs}
    if input_ids is None:
        selected = set(available)
    else:
        if isinstance(input_ids, (str, bytes)):
            raise TypeError("input_ids must be an iterable of input ids, not a string")
        selected = set(input_ids)
        unknown = selected - set(available)
        if unknown:
            raise ValueError(f"unknown input ids: {sorted(unknown)}")
    return [data for data in experiment.inputs if data.id in selected]


def _local_pieces(timeline: Timeline, start: float, end: float):
    """Map one local interval to observed experiment-time pieces.

    A turn spanning a recording loss is split so its ELAN annotation does not
    claim speech over the silence inserted into the synchronized video.
    """
    boundaries = [start]
    boundaries.extend(
        sorted({shift.at for shift in timeline.time_shifts if start < shift.at < end})
    )
    boundaries.append(end)
    for local_start, local_end in zip(boundaries[:-1], boundaries[1:]):
        missing_at_start = sum(
            shift.seconds for shift in timeline.time_shifts if shift.at <= local_start
        )
        # A loss exactly at the end belongs after this observed piece.
        missing_before_end = sum(
            shift.seconds for shift in timeline.time_shifts if shift.at < local_end
        )
        yield (
            timeline.time_scale * local_start + timeline.time_offset + missing_at_start,
            timeline.time_scale * local_end + timeline.time_offset + missing_before_end,
        )


def _speech_tiers(
    experiment: Experiment,
    input_ids: Iterable[str] | None,
    experiment_start: float,
    experiment_end: float,
) -> list[tuple[str, list[_Annotation]]]:
    tiers: list[tuple[str, list[_Annotation]]] = []
    output_duration_ms = math.ceil((experiment_end - experiment_start) * 1000)
    for data in _selected_inputs(experiment, input_ids):
        speech = data.speech.data
        if speech is None or speech.empty:
            continue
        missing = _SPEECH_COLUMNS - set(speech.columns)
        if missing:
            raise ValueError(
                f"speech results for input {data.id!r} lack columns: {sorted(missing)}"
            )
        ordered = speech.sort_values(["speaker", "start", "end", "segment_id"])
        for speaker, turns in ordered.groupby("speaker", sort=True):
            annotations: list[_Annotation] = []
            for row in turns.itertuples(index=False):
                local_start, local_end = float(row.start), float(row.end)
                if (
                    not math.isfinite(local_start)
                    or not math.isfinite(local_end)
                    or local_end <= local_start
                ):
                    raise ValueError(
                        f"input {data.id!r} has an invalid speech turn "
                        f"{local_start}-{local_end}"
                    )
                value = getattr(row, "text", "")
                text = "" if value is None or pd.isna(value) else str(value)
                for piece_start, piece_end in _local_pieces(
                    data, local_start, local_end
                ):
                    start_ms = max(
                        0, math.floor((piece_start - experiment_start) * 1000)
                    )
                    end_ms = min(
                        output_duration_ms,
                        math.ceil((piece_end - experiment_start) * 1000),
                    )
                    if end_ms > start_ms:
                        annotations.append(_Annotation(start_ms, end_ms, text))
            if not annotations:
                continue
            annotations.sort(key=lambda item: (item.start_ms, item.end_ms))
            for previous, current in zip(annotations[:-1], annotations[1:]):
                if current.start_ms < previous.end_ms:
                    raise ValueError(
                        f"speech turns overlap on input {data.id!r}, "
                        f"local speaker {int(speaker)}"
                    )
            tiers.append((f"{data.id}:speaker_{int(speaker)}", annotations))
    return tiers


def _document(
    video: VideoGridResult,
    output_path: Path,
    tiers: list[tuple[str, list[_Annotation]]],
    author: str,
    date: datetime,
) -> ET.ElementTree:
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
    media_path = video.path.resolve()
    relative_media = Path(os.path.relpath(media_path, output_path.parent.resolve()))
    relative_url = quote(relative_media.as_posix())
    if not relative_url.startswith("../"):
        relative_url = f"./{relative_url}"
    ET.SubElement(
        header,
        "MEDIA_DESCRIPTOR",
        {
            "MEDIA_URL": media_path.as_uri(),
            "RELATIVE_MEDIA_URL": relative_url,
            "MIME_TYPE": "video/mp4",
        },
    )
    property_element = ET.SubElement(
        header, "PROPERTY", {"NAME": "body-eye-sync:experiment-start-seconds"}
    )
    property_element.text = f"{video.experiment_start:.12g}"

    time_order = ET.SubElement(root, "TIME_ORDER")
    tier_elements: list[tuple[ET.Element, list[_Annotation]]] = []
    for tier_id, tier_annotations in tiers:
        tier_elements.append(
            (
                ET.SubElement(
                    root,
                    "TIER",
                    {
                        "TIER_ID": tier_id,
                        "PARTICIPANT": tier_id,
                        "LINGUISTIC_TYPE_REF": "speech",
                    },
                ),
                tier_annotations,
            )
        )

    annotation_id = 0
    time_slot_id = 0
    prepared: list[tuple[ET.Element, _Annotation, int, str, str]] = []
    time_slots: list[tuple[int, int, str]] = []
    for tier, tier_annotations in tier_elements:
        for item in tier_annotations:
            annotation_id += 1
            time_slot_id += 1
            start_id = f"ts{time_slot_id}"
            time_slots.append((item.start_ms, time_slot_id, start_id))
            time_slot_id += 1
            end_id = f"ts{time_slot_id}"
            time_slots.append((item.end_ms, time_slot_id, end_id))
            prepared.append((tier, item, annotation_id, start_id, end_id))

    for time_value, _order, slot_id in sorted(time_slots):
        ET.SubElement(
            time_order,
            "TIME_SLOT",
            {"TIME_SLOT_ID": slot_id, "TIME_VALUE": str(time_value)},
        )

    for tier, item, item_id, start_id, end_id in prepared:
        wrapper = ET.SubElement(tier, "ANNOTATION")
        alignable = ET.SubElement(
            wrapper,
            "ALIGNABLE_ANNOTATION",
            {
                "ANNOTATION_ID": f"a{item_id}",
                "TIME_SLOT_REF1": start_id,
                "TIME_SLOT_REF2": end_id,
            },
        )
        ET.SubElement(alignable, "ANNOTATION_VALUE").text = item.text

    ET.SubElement(
        root,
        "LINGUISTIC_TYPE",
        {
            "LINGUISTIC_TYPE_ID": "speech",
            "TIME_ALIGNABLE": "true",
            "GRAPHIC_REFERENCES": "false",
        },
    )
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def export_elan(
    experiment: Experiment,
    video: VideoGridResult,
    output_path: str | Path | None = None,
    *,
    input_ids: Iterable[str] | None = None,
    overwrite: bool = False,
    author: str = "body-eye-sync",
    date: datetime | None = None,
) -> ElanExportResult:
    """Write selected inputs' local-speaker speech turns beside ``video``.

    Times are converted from each source's local clock to the synchronized
    video's zero-based timeline. Each ``(input id, local speaker id)`` becomes
    one time-alignable tier; transcript text is used when available, otherwise
    annotation values are empty. This deliberately preserves source-local
    speaker identities until post-processing provides experiment-wide people.
    """
    if video.experiment_end <= video.experiment_start:
        raise ValueError("video result has an invalid experiment interval")
    output = (
        Path(output_path) if output_path is not None else video.path.with_suffix(".eaf")
    )
    if output.suffix.lower() != ".eaf":
        raise ValueError("ELAN output path must have an .eaf extension")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tiers = _speech_tiers(
        experiment, input_ids, video.experiment_start, video.experiment_end
    )
    document = _document(
        video, output, tiers, author, date or datetime.now().astimezone()
    )
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.eaf")
    try:
        document.write(temporary, encoding="utf-8", xml_declaration=True)
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return ElanExportResult(
        path=output,
        media_path=video.path,
        tiers=tuple(tier_id for tier_id, _tier_annotations in tiers),
        annotations=sum(len(tier_annotations) for _tier_id, tier_annotations in tiers),
    )
