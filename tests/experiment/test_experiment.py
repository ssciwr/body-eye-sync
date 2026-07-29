import pandas as pd
import pytest

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
    VideoPipeline,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import FixedVideo, GlassesVideo


def _config(**overrides):
    kwargs = dict(
        name="demo",
        glasses_videos=[GlassesVideoInput(id="cam1", path="videos/session1.mp4")],
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
    exp.name = "renamed"
    exp.save()

    reloaded = Experiment.load(tmp_path)
    assert reloaded.glasses_videos[0].time_offset == 0.75
    assert reloaded.name == "renamed"


def test_save_sets_the_folder_when_given(tmp_path):
    exp = Experiment(_config())
    exp.save(tmp_path)  # first save sets the folder
    assert exp.folder == tmp_path
    assert (tmp_path / "experiment.yaml").exists()
    exp.name = "renamed"
    exp.save()  # subsequent save reuses the folder
    assert Experiment.load(tmp_path).name == "renamed"


def test_relative_paths_resolve_against_folder(tmp_path):
    exp = Experiment(_config(), tmp_path)
    resolved = (tmp_path / "videos" / "session1.mp4").resolve()
    assert exp.glasses_videos[0].video_path == resolved


def test_relative_paths_are_written_back_relative(tmp_path):
    Experiment(_config(), tmp_path).save()
    assert Experiment.load(tmp_path).config().glasses_videos[0].path == (
        tmp_path / "videos" / "session1.mp4"
    ).resolve().relative_to(tmp_path)


def test_absolute_paths_outside_the_folder_pass_through(tmp_path):
    absolute = tmp_path.parent / "clip.mp4"
    exp = Experiment(
        _config(glasses_videos=[GlassesVideoInput(id="cam1", path=absolute)]), tmp_path
    )
    assert exp.glasses_videos[0].video_path == absolute
    assert exp.config().glasses_videos[0].path == absolute


def test_output_paths_are_under_outputs_dir(tmp_path):
    exp = Experiment(_config(), tmp_path)
    assert exp.output_dir == tmp_path / "outputs"
    assert (
        exp.output_path(exp.glasses_videos[0]) == tmp_path / "outputs" / "cam1.parquet"
    )


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
    assert [i.id for i in exp.config().inputs] == ["cam1", "room", "mic1"]


def test_add_input_with_a_duplicate_id_rejected(tmp_path):
    exp = Experiment(_config(), tmp_path)
    with pytest.raises(ValueError, match="duplicate input id"):
        exp.add_fixed_video(FixedVideoInput(id="cam1", path="room.mp4"))
    with pytest.raises(ValueError, match="duplicate input id"):
        exp.add_audio(AudioInput(id="cam1", path="p1.wav"))


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
        exp.output_dir / "cam1.body_embeddings.parquet"
    )

    exp.rename_input(video, "cam2")

    assert video.id == "cam2"
    assert sorted(p.name for p in exp.output_dir.glob("*")) == [
        "cam2.body_embeddings.parquet",
        "cam2.parquet",
    ]
    assert Experiment.load(tmp_path).glasses_videos[0].id == "cam1"  # not saved yet
    exp.save()
    assert Experiment.load(tmp_path).glasses_videos[0].data is not None


def test_rename_input_to_a_taken_id_rejected(tmp_path):
    exp = Experiment(
        _config(fixed_videos=[FixedVideoInput(id="room", path="room.mp4")]), tmp_path
    )
    with pytest.raises(ValueError, match="duplicate input id"):
        exp.rename_input(exp.glasses_videos[0], "room")


def test_save_writes_and_load_rehydrates_video_results(tmp_path):
    exp = Experiment(_config(), tmp_path)
    exp.glasses_videos[0].set_data(_tracks())
    exp.save()
    assert (tmp_path / "outputs" / "cam1.parquet").exists()

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
    exp.audio[0].set_data(pd.DataFrame({"start": [0.0], "end": [1.0]}))
    exp.save()

    assert sorted(p.name for p in (tmp_path / "outputs").glob("*.parquet")) == [
        "cam1.parquet",
        "mic1.parquet",
        "room.parquet",
    ]
    assert Experiment.load(tmp_path).audio[0].data["end"].tolist() == [1.0]


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
