from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import (
    CURRENT_VERSION,
    AudioInput,
    ExperimentConfig,
    FaceDetectionStep,
    FixedVideoInput,
    GlassesVideoInput,
    ObjectTrackingStep,
    Pipeline,
    TimeShift,
    VideoPipeline,
)
from body_eye_sync.preprocessing.alignment import Shift
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import FixedVideo, GlassesVideo


def _config(**overrides):
    kwargs = dict(
        glasses_videos=[
            GlassesVideoInput(
                id="cam1", path="videos/session1.mp4", gaze_path="videos/session1.tsv"
            )
        ],
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _tracks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 0],
            "track_id": [1, 2],
            "x1": [0.0, 5.0],
            "y1": [0.0, 5.0],
            "x2": [1.0, 6.0],
            "y2": [1.0, 6.0],
            "conf": [0.9, 0.9],
        }
    )


def test_save_load_round_trips_the_experiment(tmp_path):
    original = _config(
        pipeline=Pipeline(
            glasses_video=VideoPipeline(
                object_tracking=ObjectTrackingStep(detector="yolov8m"),
                face_detection=FaceDetectionStep(det_thresh=0.7),
            ),
            fixed_video=VideoPipeline(
                object_tracking=ObjectTrackingStep(detector="yolo26x")
            ),
        ),
        fixed_videos=[FixedVideoInput(id="room", path="videos/room.mp4")],
        audio=[
            AudioInput(
                id="mic1", path="audio/p1.wav", glasses_video="cam1", time_offset=-1.5
            )
        ],
    )
    Experiment(original, tmp_path).save()

    loaded = Experiment.load(tmp_path)
    assert loaded.folder == tmp_path
    assert loaded.config().model_dump() == original.model_dump()


def test_inputs_own_their_settings(tmp_path):
    exp = Experiment(
        _config(
            audio=[
                AudioInput(
                    id="mic1", path="p1.wav", glasses_video="cam1", time_offset=-1.5
                )
            ]
        ),
        tmp_path,
    )
    video = exp.glasses_videos[0]
    audio = exp.audio[0]

    assert (video.id, video.time_offset) == ("cam1", 0.0)
    assert (audio.id, audio.time_offset) == ("mic1", -1.5)
    # An audio input points at the glasses video object itself, not just its id.
    assert audio.glasses_video is video


def test_settings_edited_at_runtime_are_saved(tmp_path):
    exp = Experiment(_config(), tmp_path)
    exp.glasses_videos[0].time_offset = 0.75
    exp.pipeline.glasses_video.object_tracking.detector = "yolo26x"
    exp.save()

    reloaded = Experiment.load(tmp_path)
    assert reloaded.glasses_videos[0].time_offset == 0.75
    assert reloaded.pipeline.glasses_video.object_tracking.detector == "yolo26x"


def test_save_sets_the_folder_when_given(tmp_path):
    exp = Experiment(_config())
    exp.save(tmp_path)  # first save sets the folder
    assert exp.folder == tmp_path
    assert (tmp_path / "experiment.yaml").exists()
    exp.glasses_videos[0].time_offset = 2.5
    exp.save()  # subsequent save reuses the folder
    assert Experiment.load(tmp_path).glasses_videos[0].time_offset == 2.5


def test_relative_paths_resolve_against_folder(tmp_path):
    exp = Experiment(_config(), tmp_path)
    resolved = (tmp_path / "videos" / "session1.mp4").resolve()
    assert exp.glasses_videos[0].video_path == resolved


def test_the_gaze_file_travels_with_its_video(tmp_path):
    exp = Experiment(_config(), tmp_path)
    video = exp.glasses_videos[0]
    # Resolved against the folder on the way in, stored relative on the way out.
    assert video.gaze_path == (tmp_path / "videos" / "session1.tsv").resolve()
    assert exp.config().glasses_videos[0].gaze_path == Path("videos/session1.tsv")

    video.set_gaze(tmp_path / "videos" / "corrected.tsv")
    exp.save()
    assert (
        Experiment.load(tmp_path).glasses_videos[0].gaze_path
        == (tmp_path / "videos" / "corrected.tsv").resolve()
    )


def test_relative_paths_are_written_back_relative(tmp_path):
    Experiment(_config(), tmp_path).save()
    assert Experiment.load(tmp_path).config().glasses_videos[0].path == (
        tmp_path / "videos" / "session1.mp4"
    ).resolve().relative_to(tmp_path)


def test_absolute_paths_outside_the_folder_pass_through(tmp_path):
    absolute = tmp_path.parent / "clip.mp4"
    exp = Experiment(
        _config(
            glasses_videos=[
                GlassesVideoInput(
                    id="cam1", path=absolute, gaze_path=absolute.with_suffix(".tsv")
                )
            ]
        ),
        tmp_path,
    )
    assert exp.glasses_videos[0].video_path == absolute
    assert exp.config().glasses_videos[0].path == absolute


def test_output_paths_are_under_outputs_dir(tmp_path):
    exp = Experiment(_config(), tmp_path)
    assert exp.output_dir == tmp_path / "outputs"
    # The experiment decides where an input's results go, not what they are called.
    assert exp.output_dir_for(exp.glasses_videos[0]) == tmp_path / "outputs" / "cam1"


def test_output_paths_require_a_folder():
    exp = Experiment(_config())
    with pytest.raises(ValueError, match="no folder"):
        _ = exp.output_dir


def test_input_types_follow_the_config_lists(tmp_path):
    exp = Experiment(
        _config(
            fixed_videos=[FixedVideoInput(id="room", path="b.mp4")],
            audio=[AudioInput(id="mic1", path="p1.wav")],
        ),
        tmp_path,
    )
    assert isinstance(exp.glasses_videos[0], GlassesVideo)
    assert isinstance(exp.fixed_videos[0], FixedVideo)
    assert isinstance(exp.audio[0], Audio)
    assert [i.id for i in exp.inputs] == ["cam1", "room", "mic1"]


def test_add_inputs(tmp_path):
    exp = Experiment(_config(), tmp_path)
    room = exp.add_fixed_video(FixedVideoInput(id="room", path="room.mp4"))
    mic = exp.add_audio(AudioInput(id="mic1", path="p1.wav", glasses_video="cam1"))

    assert isinstance(room, FixedVideo)
    assert exp.fixed_videos == [room]
    assert mic.glasses_video is exp.glasses_videos[0]
    config = exp.config()
    assert [video.id for video in config.glasses_videos] == ["cam1"]
    assert [video.id for video in config.fixed_videos] == ["room"]
    assert [audio.id for audio in config.audio] == ["mic1"]


def test_add_input_with_a_duplicate_id_rejected(tmp_path):
    exp = Experiment(_config(), tmp_path)
    with pytest.raises(ValueError, match="duplicate input id"):
        exp.add_fixed_video(FixedVideoInput(id="cam1", path="room.mp4"))
    with pytest.raises(ValueError, match="duplicate input id"):
        exp.add_audio(AudioInput(id="cam1", path="p1.wav"))


def test_adding_an_input_discards_outputs_left_under_its_id(tmp_path):
    exp = Experiment(_config(), tmp_path)
    exp.glasses_videos[0].set_data(_tracks())
    exp.save()
    note = exp.output_dir / "cam1.notes.txt"
    note.write_text("user-owned")
    exp.remove_input(exp.glasses_videos[0])

    # A different recording that happens to be given the id the old one had:
    # it starts empty rather than adopting the results left behind.
    video = exp.add_glasses_video(
        GlassesVideoInput(id="cam1", path="other.mp4", gaze_path="other.tsv")
    )

    assert video.data is None
    assert list(exp.output_dir.iterdir()) == [note]


def test_add_audio_for_an_unknown_glasses_video_rejected(tmp_path):
    exp = Experiment(_config(), tmp_path)
    with pytest.raises(ValueError, match="unknown glasses video id"):
        exp.add_audio(AudioInput(id="mic1", path="p1.wav", glasses_video="nope"))


def test_remove_input(tmp_path):
    exp = Experiment(
        _config(fixed_videos=[FixedVideoInput(id="room", path="room.mp4")]), tmp_path
    )
    exp.remove_input(exp.fixed_videos[0])
    assert exp.fixed_videos == []
    assert [i.id for i in exp.inputs] == ["cam1"]


def test_remove_glasses_video_still_used_by_audio_rejected(tmp_path):
    exp = Experiment(
        _config(audio=[AudioInput(id="mic1", path="p1.wav", glasses_video="cam1")]),
        tmp_path,
    )
    with pytest.raises(ValueError, match="still used by audio inputs"):
        exp.remove_input(exp.glasses_videos[0])

    exp.remove_input(exp.audio[0])
    exp.remove_input(exp.glasses_videos[0])  # free once the audio is gone
    assert exp.inputs == []


def test_rename_input_moves_its_outputs(tmp_path):
    exp = Experiment(_config(), tmp_path)
    video = exp.glasses_videos[0]
    video.set_data(_tracks())
    exp.save()
    pd.DataFrame({"track_id": [1], "frame": [0], "score": [0.5]}).to_parquet(
        exp.output_dir_for(video) / "body_embeddings.parquet"
    )

    exp.rename_input(video, "cam2")

    assert video.id == "cam2"
    assert sorted(p.name for p in (exp.output_dir / "cam2").iterdir()) == [
        "body_embeddings.parquet",
        "results.parquet",
    ]
    assert Experiment.load(tmp_path).glasses_videos[0].id == "cam1"  # not saved yet
    exp.save()
    assert Experiment.load(tmp_path).glasses_videos[0].data is not None


def test_rename_input_leaves_other_inputs_outputs_alone(tmp_path):
    exp = Experiment(
        _config(
            glasses_videos=[
                GlassesVideoInput(id="cam[1]", path="a.mp4", gaze_path="a.tsv")
            ]
        ),
        tmp_path,
    )
    other = exp.add_fixed_video(FixedVideoInput(id="cam1", path="b.mp4"))
    exp.glasses_videos[0].set_data(_tracks())
    other.set_data(_tracks())
    exp.save()
    note = exp.output_dir / "cam[1].notes.txt"
    note.write_text("user-owned")

    exp.rename_input(exp.glasses_videos[0], "renamed")

    assert (exp.output_dir / "cam1" / "results.parquet").exists()
    assert (exp.output_dir / "renamed" / "results.parquet").exists()
    assert note.read_text() == "user-owned"


def test_rename_input_to_a_taken_id_rejected(tmp_path):
    exp = Experiment(
        _config(fixed_videos=[FixedVideoInput(id="room", path="room.mp4")]), tmp_path
    )
    with pytest.raises(ValueError, match="duplicate input id"):
        exp.rename_input(exp.glasses_videos[0], "room")


def test_rename_input_does_not_overwrite_existing_outputs(tmp_path):
    exp = Experiment(_config(), tmp_path)
    exp.glasses_videos[0].set_data(_tracks())
    exp.save()
    existing = exp.output_dir / "cam2"
    existing.mkdir()
    (existing / "notes.txt").write_text("keep me")

    with pytest.raises(ValueError, match="outputs already exist"):
        exp.rename_input(exp.glasses_videos[0], "cam2")

    assert exp.glasses_videos[0].id == "cam1"
    assert (existing / "notes.txt").read_text() == "keep me"


@pytest.mark.parametrize("bad_id", ["", "..", "a/../b", "sub/cam1", "back\\cam1"])
def test_an_id_that_is_not_a_filename_is_rejected(tmp_path, bad_id):
    # Ids name output directories, so one of these would escape outputs/.
    exp = Experiment(_config(), tmp_path)
    with pytest.raises(ValueError, match="input id cannot"):
        exp.rename_input(exp.glasses_videos[0], bad_id)
    with pytest.raises(ValidationError, match="input id cannot"):
        exp.add_fixed_video(FixedVideoInput(id=bad_id, path="room.mp4"))
    assert [i.id for i in exp.inputs] == ["cam1"]


def test_save_writes_and_load_rehydrates_video_results(tmp_path):
    exp = Experiment(_config(), tmp_path)
    exp.glasses_videos[0].set_data(_tracks())
    exp.save()
    assert (tmp_path / "outputs" / "cam1" / "results.parquet").exists()

    video = Experiment.load(tmp_path).glasses_videos[0]
    assert video.data is not None
    assert video.data["track_id"].nunique() == 2


def test_save_writes_results_for_every_input_type(tmp_path):
    exp = Experiment(
        _config(
            fixed_videos=[FixedVideoInput(id="room", path="room.mp4")],
            audio=[AudioInput(id="mic1", path="p1.wav")],
        ),
        tmp_path,
    )
    exp.glasses_videos[0].set_data(_tracks())
    exp.fixed_videos[0].set_data(_tracks())
    exp.audio[0].speech.set_data(
        pd.DataFrame({"segment_id": [0], "start": [0.0], "end": [1.0], "speaker": [0]})
    )
    exp.save()

    # Each input's main output is named for what that output holds.
    assert sorted(
        p.relative_to(tmp_path / "outputs").as_posix()
        for p in (tmp_path / "outputs").glob("*/*.parquet")
    ) == [
        "cam1/results.parquet",
        "mic1/speaker_turns.parquet",
        "room/results.parquet",
    ]
    assert Experiment.load(tmp_path).audio[0].speech.data["end"].tolist() == [1.0]


def test_video_speech_results_survive_a_save_and_load(tmp_path):
    exp = Experiment(_config(), tmp_path)
    video = exp.glasses_videos[0]
    video.set_data(_tracks())
    video.speech.set_data(
        pd.DataFrame({"segment_id": [0], "start": [0.0], "end": [1.0], "speaker": [3]})
    )

    exp.save()

    reloaded = Experiment.load(tmp_path).glasses_videos[0]
    assert reloaded.data["track_id"].nunique() == 2
    assert reloaded.speech.data["speaker"].tolist() == [3]


def test_unreadable_results_are_skipped_rather_than_failing_the_load(tmp_path, caplog):
    exp = Experiment(
        _config(fixed_videos=[FixedVideoInput(id="room", path="room.mp4")]), tmp_path
    )
    exp.glasses_videos[0].set_data(_tracks())
    exp.fixed_videos[0].set_data(_tracks())
    exp.save()
    (exp.output_dir_for(exp.glasses_videos[0]) / "results.parquet").write_bytes(
        b"not a parquet file"
    )

    reloaded = Experiment.load(tmp_path)

    # The bad file is ignored, and the input it belongs to starts empty; the
    # other input's results are unaffected.
    assert reloaded.glasses_videos[0].data is None
    assert reloaded.fixed_videos[0].data is not None
    assert "ignoring unreadable results for input 'cam1'" in caplog.text


def test_results_that_are_parquet_but_not_ours_are_skipped(tmp_path):
    exp = Experiment(_config(), tmp_path)
    exp.glasses_videos[0].set_data(_tracks())
    exp.save()
    # A valid Parquet file without the columns a video's results need.
    pd.DataFrame({"nothing": [1]}).to_parquet(
        exp.output_dir_for(exp.glasses_videos[0]) / "results.parquet"
    )

    assert Experiment.load(tmp_path).glasses_videos[0].data is None


def test_inputs_without_results_write_nothing(tmp_path):
    exp = Experiment(_config(audio=[AudioInput(id="mic1", path="p1.wav")]), tmp_path)
    exp.save()
    assert not (tmp_path / "outputs").exists()


def test_newer_file_version_rejected(tmp_path):
    Experiment(_config(), tmp_path).save()
    path = tmp_path / "experiment.yaml"
    path.write_text(
        path.read_text().replace(
            f"version: {CURRENT_VERSION}", f"version: {CURRENT_VERSION + 1}"
        )
    )
    with pytest.raises(ValueError, match="newer than supported"):
        Experiment.load(tmp_path)


def test_time_shifts_survive_a_save_and_load(tmp_path):
    exp = Experiment(_config(audio=[AudioInput(id="mic1", path="p1.wav")]), tmp_path)
    exp.glasses_videos[0].time_offset = 12.5
    exp.glasses_videos[0].time_scale = 1.0001
    exp.glasses_videos[0].time_shifts = [Shift(at=100.0, seconds=0.4)]
    exp.save()

    reloaded = Experiment.load(tmp_path)

    video = reloaded.glasses_videos[0]
    assert video.time_offset == pytest.approx(12.5)
    assert video.time_scale == pytest.approx(1.0001)
    assert [(s.at, s.seconds) for s in video.time_shifts] == [(100.0, 0.4)]
    # An input that kept time stores nothing extra.
    assert reloaded.audio[0].time_shifts == []


def test_an_input_places_its_own_clock_on_the_experiment(tmp_path):
    exp = Experiment(_config(), tmp_path)
    video = exp.glasses_videos[0]
    video.time_offset = 20.0
    video.time_scale = 1.0001
    video.time_shifts = [Shift(at=100.0, seconds=0.4)]

    # Before the loss only the offset applies; after it, the missing content too.
    assert video.to_experiment_time(50.0) == pytest.approx(70.005)
    assert video.to_experiment_time(150.0) == pytest.approx(170.415)
    # And back again.
    assert video.to_local_time(170.415) == pytest.approx(150.0)
    # The experiment ran on through the loss; this video has nothing for it.
    assert video.to_local_time(120.21) is None
    assert video.unobserved() == [pytest.approx((120.01, 120.41))]


def test_an_input_that_kept_time_is_just_its_offset(tmp_path):
    exp = Experiment(_config(), tmp_path)
    video = exp.glasses_videos[0]
    video.time_offset = 7.0

    assert video.to_experiment_time(30.0) == pytest.approx(37.0)
    assert video.to_local_time(37.0) == pytest.approx(30.0)
    assert video.unobserved() == []


def test_missing_content_duration_must_be_positive():
    with pytest.raises(ValidationError):
        TimeShift(at=10.0, seconds=-0.1)
