import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
    TimeShift,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.export.elan import export_elan
from body_eye_sync.export.video_grid import VideoGridResult


def _video(path, start=9.5, end=11.5):
    return VideoGridResult(
        path=path,
        experiment_start=start,
        experiment_end=end,
        columns=1,
        rows=1,
        audio_tracks=(),
    )


def _annotations(root):
    slots = {
        slot.attrib["TIME_SLOT_ID"]: int(slot.attrib["TIME_VALUE"])
        for slot in root.find("TIME_ORDER")
    }
    found = {}
    for tier in root.findall("TIER"):
        values = []
        for annotation in tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
            values.append(
                (
                    slots[annotation.attrib["TIME_SLOT_REF1"]],
                    slots[annotation.attrib["TIME_SLOT_REF2"]],
                    annotation.findtext("ANNOTATION_VALUE") or "",
                )
            )
        found[tier.attrib["TIER_ID"]] = values
    return found


def test_elan_export_maps_selected_local_speakers_onto_the_video_timeline(tmp_path):
    experiment = Experiment(
        ExperimentConfig(
            fixed_videos=[
                FixedVideoInput(id="room", path="room.mp4", time_offset=10.0)
            ],
            audio=[
                AudioInput(id="microphone", path="mic.wav", time_offset=9.0),
                AudioInput(id="excluded", path="other.wav"),
            ],
        )
    )
    experiment.fixed_videos[0].speech.set_data(
        pd.DataFrame(
            {
                "segment_id": [0],
                "start": [0.0],
                "end": [1.0],
                "speaker": [0],
                "text": ["Hello & <welcome>"],
            }
        )
    )
    experiment.audio[0].speech.set_data(
        pd.DataFrame(
            {
                "segment_id": [0],
                "start": [0.0],
                "end": [3.0],
                "speaker": [2],
            }
        )
    )
    experiment.audio[1].speech.set_data(
        pd.DataFrame(
            {
                "segment_id": [0],
                "start": [9.5],
                "end": [10.0],
                "speaker": [7],
                "text": ["not selected"],
            }
        )
    )
    video_path = tmp_path / "combined video.mp4"
    result = export_elan(
        experiment,
        _video(video_path),
        input_ids=["room", "microphone"],
        author="test",
        date=datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
    )

    assert result.path == video_path.with_suffix(".eaf")
    assert result.media_path == video_path
    assert result.tiers == ("room:speaker_0", "microphone:speaker_2")
    assert result.annotations == 2
    root = ET.parse(result.path).getroot()
    assert root.attrib["AUTHOR"] == "test"
    assert root.attrib["VERSION"] == "3.0"
    descriptor = root.find("HEADER/MEDIA_DESCRIPTOR")
    assert descriptor.attrib["MEDIA_URL"] == video_path.resolve().as_uri()
    assert descriptor.attrib["RELATIVE_MEDIA_URL"] == "./combined%20video.mp4"
    assert descriptor.attrib["MIME_TYPE"] == "video/mp4"
    assert root.findtext("HEADER/PROPERTY") == "9.5"
    assert _annotations(root) == {
        "room:speaker_0": [(500, 1500, "Hello & <welcome>")],
        # The turn extends beyond both ends of the video and is clipped.
        "microphone:speaker_2": [(0, 2000, "")],
    }
    assert [
        int(slot.attrib["TIME_VALUE"]) for slot in root.find("TIME_ORDER")
    ] == sorted(int(slot.attrib["TIME_VALUE"]) for slot in root.find("TIME_ORDER"))
    linguistic_type = root.find("LINGUISTIC_TYPE")
    assert linguistic_type.attrib == {
        "LINGUISTIC_TYPE_ID": "speech",
        "TIME_ALIGNABLE": "true",
        "GRAPHIC_REFERENCES": "false",
    }


def test_speech_turns_are_split_across_recording_gaps(tmp_path):
    experiment = Experiment(
        ExperimentConfig(
            audio=[
                AudioInput(
                    id="headset",
                    path="headset.wav",
                    time_shifts=[TimeShift(at=1.0, seconds=0.5)],
                )
            ]
        )
    )
    experiment.audio[0].speech.set_data(
        pd.DataFrame(
            {
                "segment_id": [0],
                "start": [0.5],
                "end": [1.5],
                "speaker": [3],
                "text": ["one turn"],
            }
        )
    )

    result = export_elan(
        experiment,
        _video(tmp_path / "grid.mp4", start=0.0, end=2.0),
    )

    assert result.annotations == 2
    assert _annotations(ET.parse(result.path).getroot()) == {
        "headset:speaker_3": [
            (500, 1000, "one turn"),
            (1500, 2000, "one turn"),
        ]
    }


def test_elan_export_does_not_overwrite_by_default(tmp_path):
    output = tmp_path / "grid.eaf"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        export_elan(
            Experiment(ExperimentConfig()),
            _video(tmp_path / "grid.mp4"),
        )

    assert output.read_text(encoding="utf-8") == "existing"
