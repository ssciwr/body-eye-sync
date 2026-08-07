"""The experiment's speech turns, written for ELAN beside the combined video."""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from body_eye_sync.experiment.config import ExperimentConfig, GlassesVideoInput
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.elan import export_elan
from body_eye_sync.export.video_grid import VideoGridResult
from body_eye_sync.pipeline.transcription import TranscriptSegment, Word

DATE = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


def _turns(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [(index, *row[:-1], index, row[-1]) for index, row in enumerate(rows)],
        columns=[
            "turn_id",
            "start",
            "end",
            "speaker",
            "source",
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
        columns=2,
        rows=1,
        audio_tracks=("p1", "p2"),
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
    experiment = _experiment(_turns((1.0, 2.0, "p1", "p1", "hallo")))

    result = export_elan(experiment, _video(tmp_path), date=DATE)

    assert result.path == tmp_path / "combined_video.eaf"
    assert result.path.exists()
    assert result.media_path == tmp_path / "combined_video.mp4"


def test_one_tier_per_speaker(tmp_path):
    experiment = _experiment(
        _turns(
            (1.0, 2.0, "p1", "p1", "hallo"),
            (2.0, 3.0, "p2", "p2", "wie geht es"),
            (4.0, 5.0, "p1", "p1", "gut"),
        )
    )

    result = export_elan(experiment, _video(tmp_path), date=DATE)

    assert result.tiers == ("p1", "p2")
    assert result.annotations == 3
    tiers = _annotations(_parse(result.path))
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
                    time_offset=10.0,
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
            [(0, 11.0, 13.0, "p1", "p1", 1, "guten morgen")],
            columns=[
                "turn_id",
                "start",
                "end",
                "speaker",
                "source",
                "source_segment_id",
                "text",
            ],
        )
    )

    result = export_elan(experiment, _video(tmp_path, start=10.0, end=20.0), date=DATE)

    root = _parse(result.path)
    tiers = _annotations(root)
    assert result.tiers == ("p1",)
    assert result.annotations == 3
    assert tiers["p1"] == [("1000", "3000", "guten morgen")]
    assert tiers["p1-words"] == [
        ("1000", "1700", "guten"),
        ("2000", "3000", "morgen"),
    ]
    word_tier = root.find("./TIER[@TIER_ID='p1-words']")
    assert word_tier.get("PARENT_REF") == "p1"
    assert word_tier.get("PARTICIPANT") == "p1"
    assert word_tier.get("LINGUISTIC_TYPE_REF") == "words"
    word_type = root.find("./LINGUISTIC_TYPE[@LINGUISTIC_TYPE_ID='words']")
    assert word_type.get("TIME_ALIGNABLE") == "true"
    assert word_type.get("CONSTRAINTS") == "Included_In"


def test_times_are_shifted_to_where_the_video_starts(tmp_path):
    experiment = _experiment(_turns((11.0, 12.5, "p1", "p1", "hallo")))

    # The video begins ten seconds into the experiment.
    result = export_elan(experiment, _video(tmp_path, start=10.0), date=DATE)

    assert _annotations(_parse(result.path))["p1"] == [("1000", "2500", "hallo")]


def test_two_speakers_may_talk_at_once(tmp_path):
    experiment = _experiment(
        _turns(
            (1.0, 4.0, "p1", "p1", "the pasta was excellent"),
            (2.0, 3.0, "p2", "p2", "no I disagree"),
        )
    )

    result = export_elan(experiment, _video(tmp_path), date=DATE)

    # Overlapping turns are fine as long as they are different speakers, which
    # is exactly what separate tiers are for.
    tiers = _annotations(_parse(result.path))
    assert tiers["p1"] == [("1000", "4000", "the pasta was excellent")]
    assert tiers["p2"] == [("2000", "3000", "no I disagree")]


def test_a_speaker_talking_over_themselves_is_refused(tmp_path):
    experiment = _experiment(
        _turns(
            (1.0, 4.0, "p1", "p1", "first"),
            (2.0, 3.0, "p1", "p1", "second"),
        )
    )

    with pytest.raises(ValueError, match="overlapping turns"):
        export_elan(experiment, _video(tmp_path), date=DATE)


def test_turns_outside_the_video_are_left_out(tmp_path):
    experiment = _experiment(
        _turns(
            (1.0, 2.0, "p1", "p1", "before the video"),
            (15.0, 16.0, "p1", "p1", "inside"),
            (70.0, 71.0, "p1", "p1", "after the video"),
        )
    )

    result = export_elan(experiment, _video(tmp_path, start=10.0, end=30.0), date=DATE)

    assert result.skipped == 2
    assert [text for _s, _e, text in _annotations(_parse(result.path))["p1"]] == [
        "inside"
    ]


def test_a_turn_hanging_over_the_end_is_trimmed(tmp_path):
    experiment = _experiment(_turns((8.0, 30.0, "p1", "p1", "runs past the end")))

    result = export_elan(experiment, _video(tmp_path, start=0.0, end=10.0), date=DATE)

    assert _annotations(_parse(result.path))["p1"] == [
        ("8000", "10000", "runs past the end")
    ]


def test_the_media_is_referenced_next_to_the_annotations(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "p1", "hallo")))
    video = _video(tmp_path)

    result = export_elan(experiment, video, date=DATE)

    descriptor = _parse(result.path).find("./HEADER/MEDIA_DESCRIPTOR")
    assert descriptor.get("RELATIVE_MEDIA_URL") == "./combined_video.mp4"
    assert descriptor.get("MEDIA_URL") == video.path.resolve().as_uri()


def test_where_the_video_starts_is_recorded(tmp_path):
    experiment = _experiment(_turns((11.0, 12.0, "p1", "p1", "hallo")))

    result = export_elan(experiment, _video(tmp_path, start=10.25), date=DATE)

    header = _parse(result.path).find("./HEADER")
    property_element = header.find("PROPERTY")
    assert property_element.get("NAME") == "body-eye-sync:experiment-start-seconds"
    assert float(property_element.text) == 10.25


def test_an_experiment_with_no_turns_cannot_be_exported(tmp_path):
    with pytest.raises(ValueError, match="no speech turns"):
        export_elan(_experiment(), _video(tmp_path), date=DATE)


def test_an_existing_file_is_kept_unless_overwriting_is_asked_for(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "p1", "hallo")))
    export_elan(experiment, _video(tmp_path), date=DATE)

    with pytest.raises(FileExistsError):
        export_elan(experiment, _video(tmp_path), date=DATE)

    experiment.speech_turns.set_data(_turns((1.0, 2.0, "p1", "p1", "neu")))
    result = export_elan(experiment, _video(tmp_path), overwrite=True, date=DATE)

    assert _annotations(_parse(result.path))["p1"] == [("1000", "2000", "neu")]


def test_the_output_must_be_an_eaf(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "p1", "hallo")))

    with pytest.raises(ValueError, match=".eaf"):
        export_elan(experiment, _video(tmp_path), tmp_path / "turns.xml", date=DATE)


def test_a_failed_write_leaves_nothing_behind(tmp_path, monkeypatch):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "p1", "hallo")))

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ET.ElementTree, "write", boom)

    with pytest.raises(OSError):
        export_elan(experiment, _video(tmp_path), date=DATE)

    # No half-written annotations, and no temporary file left over either.
    assert not (tmp_path / "combined_video.eaf").exists()
    assert list(tmp_path.glob(".*")) == []


def test_the_document_declares_what_elan_expects(tmp_path):
    experiment = _experiment(_turns((1.0, 2.0, "p1", "p1", "hallo")))

    result = export_elan(experiment, _video(tmp_path), date=DATE)

    root = _parse(result.path)
    assert root.tag == "ANNOTATION_DOCUMENT"
    assert root.get("FORMAT") == "3.0"
    assert root.get("DATE") == "2026-08-06T12:00:00+00:00"
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
            (5.0, 6.0, "p2", "p2", "later"),
            (1.0, 2.0, "p1", "p1", "earlier"),
        )
    )

    result = export_elan(experiment, _video(tmp_path), date=DATE)

    values = [
        int(slot.get("TIME_VALUE")) for slot in _parse(result.path).iter("TIME_SLOT")
    ]
    assert values == sorted(values)


def test_turns_that_meet_at_a_boundary_do_not_overlap(tmp_path):
    """Rounding must not widen turns into each other.

    One turn ending exactly where the next begins is ordinary -- speech is cut
    into segments that abut -- and rounding each end outwards would turn that
    into a one millisecond overlap that ELAN refuses.
    """
    experiment = _experiment(
        _turns(
            (52.6054, 54.1255, "p1", "p1", "first"),
            (54.1255, 56.0465, "p1", "p1", "second"),
        )
    )

    result = export_elan(experiment, _video(tmp_path), date=DATE)

    assert _annotations(_parse(result.path))["p1"] == [
        ("52605", "54126", "first"),
        # The two meet exactly, rather than running into one another.
        ("54126", "56046", "second"),
    ]


def test_a_turn_shorter_than_a_millisecond_is_left_out(tmp_path):
    experiment = _experiment(
        _turns(
            (1.00000, 1.00002, "p1", "p1", "blink"),
            (2.0, 3.0, "p1", "p1", "real"),
        )
    )

    result = export_elan(experiment, _video(tmp_path), date=DATE)

    assert result.skipped == 1
    assert [text for _s, _e, text in _annotations(_parse(result.path))["p1"]] == [
        "real"
    ]
