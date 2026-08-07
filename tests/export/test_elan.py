"""The experiment's speech turns, written for ELAN beside the combined video."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    ExperimentConfig,
    GlassesVideoInput,
    TimelineConfig,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.elan import export_elan
from body_eye_sync.export.video_grid import VideoGridResult
from body_eye_sync.pipeline.transcription import TranscriptSegment, Word


def _turns(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [(index, *row[:-1], index, row[-1]) for index, row in enumerate(rows)],
        columns=[
            "turn_id",
            "start",
            "end",
            "speaker",
            "source_segment_id",
            "text",
        ],
    )


def _experiment(turns: pd.DataFrame | None = None) -> Experiment:
    experiment = Experiment(ExperimentConfig())
    if turns is not None:
        experiment.speech_turns.set_data(turns)
    return experiment


def _video(tmp_path, start=0.0, end=60.0) -> VideoGridResult:
    path = tmp_path / "combined_video.mp4"
    path.write_bytes(b"")
    return VideoGridResult(
        path=path,
        experiment_start=start,
        experiment_end=end,
    )


def _parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _annotations(root: ET.Element) -> dict[str, list[tuple[str, str, str]]]:
    """Each tier's annotations as (start ms, end ms, text), in document order."""
    slots = {
        slot.get("TIME_SLOT_ID"): slot.get("TIME_VALUE")
        for slot in root.iter("TIME_SLOT")
    }
    out = {}
    for tier in root.iter("TIER"):
        items = []
        for alignable in tier.iter("ALIGNABLE_ANNOTATION"):
            items.append(
                (
                    slots[alignable.get("TIME_SLOT_REF1")],
                    slots[alignable.get("TIME_SLOT_REF2")],
                    alignable.find("ANNOTATION_VALUE").text or "",
                )
            )
        out[tier.get("TIER_ID")] = items
    return out


def test_the_file_is_named_after_the_video(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "hallo")))

    result = export_elan(experiment, _video(tmp_path))

    assert result == tmp_path / "combined_video.eaf"
    assert result.exists()


def test_one_tier_per_speaker(tmp_path):
    experiment = _experiment(
        _turns(
            (1.0, 2.0, "p1", "hallo"),
            (2.0, 3.0, "p2", "wie geht es"),
            (4.0, 5.0, "p1", "gut"),
        )
    )

    result = export_elan(experiment, _video(tmp_path))

    tiers = _annotations(_parse(result))
    assert list(tiers) == ["p1", "p2"]
    assert [text for _s, _e, text in tiers["p1"]] == ["hallo", "gut"]
    assert [text for _s, _e, text in tiers["p2"]] == ["wie geht es"]


def test_words_are_exported_on_a_dependent_tier(tmp_path):
    gaze = tmp_path / "gaze.tsv"
    gaze.write_text("")
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id="p1",
                    path=tmp_path / "p1.mp4",
                    gaze_path=gaze,
                    timeline=TimelineConfig(offset=10.0),
                )
            ]
        )
    )
    speech = experiment.glasses_videos[0].speech
    speech.begin_transcription()
    speech.add_transcription_segment(
        TranscriptSegment(0.0, 0.5, "not this segment", [Word(0.1, 0.4, "not", 0.9)])
    )
    speech.add_transcription_segment(
        TranscriptSegment(
            1.0,
            3.0,
            "guten morgen",
            # Whisper can put a word just outside its segment boundary. The
            # dependent ELAN annotation must remain within its parent turn.
            [Word(0.9, 1.7, "guten", 0.9), Word(2.0, 3.1, "morgen", 0.8)],
        )
    )
    speech.finish_transcription()
    experiment.speech_turns.set_data(
        pd.DataFrame(
            [(0, 11.0, 13.0, "p1", 1, "guten morgen")],
            columns=[
                "turn_id",
                "start",
                "end",
                "speaker",
                "source_segment_id",
                "text",
            ],
        )
    )

    result = export_elan(experiment, _video(tmp_path, start=10.0, end=20.0))

    root = _parse(result)
    tiers = _annotations(root)
    assert tiers["p1"] == [("1000", "3000", "guten morgen")]
    assert tiers["p1 [words]"] == [
        ("1000", "1700", "guten"),
        ("2000", "3000", "morgen"),
    ]
    word_tier = root.find("./TIER[@TIER_ID='p1 [words]']")
    assert word_tier.get("PARENT_REF") == "p1"
    assert word_tier.get("PARTICIPANT") == "p1"
    assert word_tier.get("LINGUISTIC_TYPE_REF") == "words"
    word_type = root.find("./LINGUISTIC_TYPE[@LINGUISTIC_TYPE_ID='words']")
    assert word_type.get("TIME_ALIGNABLE") == "true"
    assert word_type.get("CONSTRAINTS") == "Included_In"


def test_times_are_shifted_to_where_the_video_starts(tmp_path):
    experiment = _experiment(_turns((11.0, 12.5, "p1", "hallo")))

    # The video begins ten seconds into the experiment.
    result = export_elan(experiment, _video(tmp_path, start=10.0))

    assert _annotations(_parse(result))["p1"] == [("1000", "2500", "hallo")]


def test_two_speakers_may_talk_at_once(tmp_path):
    experiment = _experiment(
        _turns(
            (1.0, 4.0, "p1", "the pasta was excellent"),
            (2.0, 3.0, "p2", "no I disagree"),
        )
    )

    result = export_elan(experiment, _video(tmp_path))

    # Overlapping turns are fine as long as they are different speakers, which
    # is exactly what separate tiers are for.
    tiers = _annotations(_parse(result))
    assert tiers["p1"] == [("1000", "4000", "the pasta was excellent")]
    assert tiers["p2"] == [("2000", "3000", "no I disagree")]


def test_turns_outside_the_video_are_left_out(tmp_path):
    experiment = _experiment(
        _turns(
            (1.0, 2.0, "p1", "before the video"),
            (15.0, 16.0, "p1", "inside"),
            (70.0, 71.0, "p1", "after the video"),
        )
    )

    result = export_elan(experiment, _video(tmp_path, start=10.0, end=30.0))

    assert [text for _s, _e, text in _annotations(_parse(result))["p1"]] == ["inside"]


def test_a_turn_hanging_over_the_end_is_trimmed(tmp_path):
    experiment = _experiment(_turns((8.0, 30.0, "p1", "runs past the end")))

    result = export_elan(experiment, _video(tmp_path, start=0.0, end=10.0))

    assert _annotations(_parse(result))["p1"] == [
        ("8000", "10000", "runs past the end")
    ]


def test_the_media_is_referenced_next_to_the_annotations(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "hallo")))
    video = _video(tmp_path)

    result = export_elan(experiment, video)

    descriptor = _parse(result).find("./HEADER/MEDIA_DESCRIPTOR")
    assert descriptor.get("RELATIVE_MEDIA_URL") == "./combined_video.mp4"
    assert descriptor.get("MEDIA_URL") == video.path.resolve().as_uri()


def test_an_experiment_with_no_turns_cannot_be_exported(tmp_path):
    with pytest.raises(ValueError, match="no speech turns"):
        export_elan(_experiment(), _video(tmp_path))


def test_an_existing_file_is_kept_unless_overwriting_is_asked_for(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "hallo")))
    export_elan(experiment, _video(tmp_path))

    with pytest.raises(FileExistsError):
        export_elan(experiment, _video(tmp_path))

    experiment.speech_turns.set_data(_turns((1.0, 2.0, "p1", "neu")))
    result = export_elan(experiment, _video(tmp_path), overwrite=True)

    assert _annotations(_parse(result))["p1"] == [("1000", "2000", "neu")]


def test_the_document_declares_what_elan_expects(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "hallo")))

    result = export_elan(experiment, _video(tmp_path))

    root = _parse(result)
    assert root.tag == "ANNOTATION_DOCUMENT"
    assert root.get("FORMAT") == "2.8"
    assert root.get("VERSION") == "2.8"
    assert root.get(
        "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"
    ).endswith("EAFv2.8.xsd")
    assert root.get("DATE")
    assert root.find("./HEADER").get("TIME_UNITS") == "milliseconds"
    linguistic = root.find("LINGUISTIC_TYPE")
    assert linguistic.get("LINGUISTIC_TYPE_ID") == "speech"
    assert linguistic.get("TIME_ALIGNABLE") == "true"
    # Every tier refers to a type the document defines.
    for tier in root.iter("TIER"):
        assert tier.get("LINGUISTIC_TYPE_REF") == "speech"


def test_time_slots_are_written_in_time_order(tmp_path):
    experiment = _experiment(
        _turns(
            (5.0, 6.0, "p2", "later"),
            (1.0, 2.0, "p1", "earlier"),
        )
    )

    result = export_elan(experiment, _video(tmp_path))

    values = [int(slot.get("TIME_VALUE")) for slot in _parse(result).iter("TIME_SLOT")]
    assert values == sorted(values)


def test_turns_that_meet_at_a_boundary_do_not_overlap(tmp_path):
    """Rounding must not widen turns into each other.

    One turn ending exactly where the next begins is ordinary -- speech is cut
    into segments that abut -- and rounding each end outwards would turn that
    into a one millisecond overlap that ELAN refuses.
    """
    experiment = _experiment(
        _turns(
            (52.6054, 54.1255, "p1", "first"),
            (54.1255, 56.0465, "p1", "second"),
        )
    )

    result = export_elan(experiment, _video(tmp_path))

    assert _annotations(_parse(result))["p1"] == [
        ("52605", "54126", "first"),
        # The two meet exactly, rather than running into one another.
        ("54126", "56046", "second"),
    ]


def test_a_turn_shorter_than_a_millisecond_is_left_out(tmp_path):
    experiment = _experiment(
        _turns(
            (1.00000, 1.00002, "p1", "blink"),
            (2.0, 3.0, "p1", "real"),
        )
    )

    result = export_elan(experiment, _video(tmp_path))

    assert [text for _s, _e, text in _annotations(_parse(result))["p1"]] == ["real"]
