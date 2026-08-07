"""Attributing an experiment's speech, over synthetic recordings."""

import wave

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.attribution import (
    attribute_experiment_speech,
    wearers,
)
from body_eye_sync.experiment.config import (
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.media import SAMPLE_RATE

DURATION = 12.0


def _write(path, spans, gain=1.0, seed=0):
    """A recording that is quiet throughout except for loud bursts over ``spans``."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.001, int(DURATION * SAMPLE_RATE))
    times = np.arange(samples.size) / SAMPLE_RATE
    for start, end in spans:
        span = (times >= start) & (times < end)
        samples[span] += gain * 0.3 * np.sin(2 * np.pi * 220 * times[span])
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _transcript(*rows):
    return pd.DataFrame(
        [(index, *row) for index, row in enumerate(rows)],
        columns=["segment_id", "start", "end", "text"],
    )


@pytest.fixture
def experiment(tmp_path):
    """Two wearers who take turns, each transcribed from their own recording."""
    _write(tmp_path / "p1.wav", [(1.0, 4.0)], seed=1)
    _write(tmp_path / "p2.wav", [(6.0, 9.0)], gain=0.5, seed=2)
    gaze = tmp_path / "gaze.tsv"
    gaze.write_text("")
    exp = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(id="p1", path=tmp_path / "p1.wav", gaze_path=gaze),
                GlassesVideoInput(id="p2", path=tmp_path / "p2.wav", gaze_path=gaze),
            ]
        ),
        tmp_path / "experiment",
    )
    # Both microphones heard both speakers, so both transcribed both.
    exp.glasses_videos[0].speech.set_data(
        _transcript((1.0, 4.0, "p1 speaking"), (6.0, 9.0, "p2 as p1 heard them"))
    )
    exp.glasses_videos[1].speech.set_data(
        _transcript((1.0, 4.0, "p1 as p2 heard them"), (6.0, 9.0, "p2 speaking"))
    )
    return exp


def test_wearers_are_the_glasses_that_have_a_transcript(experiment):
    assert sorted(wearers(experiment)) == ["p1", "p2"]

    experiment.glasses_videos[1].speech.clear()

    # A recording nobody transcribed cannot contribute text to anyone.
    assert sorted(wearers(experiment)) == ["p1"]


def test_each_turn_goes_to_the_wearer_who_was_loudest(experiment):
    turns = attribute_experiment_speech(experiment)

    assert turns.data["speaker"].tolist() == ["p1", "p2"]
    assert turns.data["text"].tolist() == ["p1 speaking", "p2 speaking"]
    assert turns.speakers == ["p1", "p2"]


def test_the_turns_are_left_on_the_experiment(experiment):
    turns = attribute_experiment_speech(experiment)

    assert experiment.speech_turns is turns
    assert experiment.speech_turns.has_data()


def test_a_fixed_camera_is_not_attributed_to_anyone(experiment, tmp_path):
    _write(tmp_path / "room.wav", [(1.0, 9.0)], seed=3)
    room = experiment.add_fixed_video(
        FixedVideoInput(id="room", path=tmp_path / "room.wav")
    )
    room.speech.set_data(_transcript((1.0, 9.0, "everything the room heard")))

    turns = attribute_experiment_speech(experiment)

    # Nobody wears a fixed camera, so its transcript names no speaker.
    assert "room" not in turns.data["speaker"].tolist()
    assert "room" not in turns.data["source"].tolist()


def test_one_wearer_alone_has_nothing_to_be_compared_against(experiment):
    experiment.glasses_videos[1].speech.clear()

    turns = attribute_experiment_speech(experiment)

    assert turns.data is None


def test_turns_are_saved_and_loaded_with_the_experiment(experiment):
    attribute_experiment_speech(experiment)

    experiment.save()
    reloaded = Experiment.load(experiment.folder)

    assert reloaded.speech_turns.data["speaker"].tolist() == ["p1", "p2"]
    # They belong to the experiment, not to any one input.
    assert (experiment.output_dir / "speech_turns.parquet").exists()


def test_attribution_survives_a_recording_that_started_late(experiment):
    # p2's device was started two seconds before p1's, so its own clock runs
    # ahead of experiment time.
    experiment.glasses_videos[1].time_offset = -2.0
    experiment.glasses_videos[1].speech.set_data(
        _transcript((3.0, 6.0, "p1 as p2 heard them"), (8.0, 11.0, "p2 speaking"))
    )

    turns = attribute_experiment_speech(experiment)

    assert turns.data["speaker"].tolist() == ["p1", "p2"]
    assert turns.data["start"].tolist() == [1.0, 6.0]
