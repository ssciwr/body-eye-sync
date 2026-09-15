import numpy as np
import pandas as pd
import pytest
from qtpy.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QMessageBox,
)

from body_eye_sync.experiment.config import (
    ClusterPostProcessingSettings,
    ExperimentConfig,
    GlassesVideoInput,
    Pipeline,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.tabs import ParticipantIdentificationTab, SpeechPostProcessingTab
from body_eye_sync.gui.tabs import cluster_post_processing as tab_module
from body_eye_sync.pipeline.face_detection import FaceBox, FaceFrameResult
from body_eye_sync.pipeline.body_pose import BodyPose, PoseFrameResult
from body_eye_sync.pipeline.object_tracking import BoundingBox


@pytest.fixture
def experiment(tmp_path, data_dir):
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id=video_id,
                    path=data_dir / "three-people.mp4",
                    gaze_path=f"{video_id}.tsv",
                )
                for video_id in ["a", "b"]
            ],
            pipeline=Pipeline(speech=None),
        ),
        tmp_path,
    )
    for index, video in enumerate(experiment.glasses_videos):
        video.set_data(
            pd.DataFrame(
                {
                    "frame": range(30),
                    "track_id": 1,
                    "x1": 0.0,
                    "y1": 0.0,
                    "x2": 1.0,
                    "y2": 1.0,
                    "conf": 0.9,
                }
            )
        )
        video.begin_face_detection(embeddings_per_track=1)
        for frame in range(30):
            video.add_face_detection_frame(
                FaceFrameResult(
                    frame,
                    [
                        FaceBox(
                            BoundingBox(0.0, 0.0, 1.0, 1.0, 1),
                            0.9,
                            landmarks=[(0.0, 0.0)] * 5,
                            embedding=np.eye(2)[index],
                        )
                    ],
                )
            )
        video.finish_face_detection()
    return experiment


def overlay_labels(tab):
    return [
        item.text()
        for item in tab.video_viewer._overlay_items
        if isinstance(item, QGraphicsSimpleTextItem)
    ]


def overlay_colors(tab):
    return {
        item.pen().color().name()
        if isinstance(item, (QGraphicsRectItem, QGraphicsLineItem))
        else item.brush().color().name()
        for item in tab.video_viewer._overlay_items
        if isinstance(
            item,
            (
                QGraphicsRectItem,
                QGraphicsLineItem,
                QGraphicsSimpleTextItem,
                QGraphicsEllipseItem,
            ),
        )
    }


def test_participant_colors_match_audio_across_videos(qtbot, experiment, monkeypatch):
    experiment.add_glasses_video(
        GlassesVideoInput(
            id="c", path=experiment.glasses_videos[0].path, gaze_path="c.tsv"
        )
    )
    tracks = experiment.glasses_videos[1].data.copy()
    tracks["track_id"] = 7
    experiment.glasses_videos[1].set_data(tracks)
    for video, track_id in zip(experiment.glasses_videos[:2], [1, 7]):
        video.begin_body_pose_detection()
        video.add_body_pose_frame(
            PoseFrameResult(
                0,
                [
                    BodyPose(
                        BoundingBox(0, 0, 1, 1, track_id), 0.9, [(0.5, 0.5, 0.9)] * 17
                    )
                ],
            )
        )
        video.finish_body_pose_detection()
    experiment.identities.set_data(
        pd.DataFrame(
            [("a", 1, "c"), ("b", 7, "c")],
            columns=["video_id", "track_id", "participant_id"],
        )
    )
    # Skipping an input without audio must not shift participant colours.
    monkeypatch.setattr(Video, "has_audio_track", lambda video: video.id != "a")
    speech_tab = SpeechPostProcessingTab(experiment)
    qtbot.addWidget(speech_tab)
    tab = ParticipantIdentificationTab(experiment)
    qtbot.addWidget(tab)
    color = speech_tab.audio_player.color_for("c").name()
    assert overlay_labels(tab) == ["c"]
    assert overlay_colors(tab) == {color}
    tab.video_selector.setCurrentIndex(1)
    assert overlay_labels(tab) == ["c"]
    assert overlay_colors(tab) == {color}


def test_button_runs_clustering_and_shows_glasses_ids(qtbot, experiment):
    tab = ParticipantIdentificationTab(experiment)
    qtbot.addWidget(tab)
    changed, busy = [], []
    tab.experiment_changed.connect(lambda: changed.append(True))
    tab.busy_changed.connect(busy.append)
    assert tab.cluster_button.isEnabled()
    assert overlay_labels(tab) == ["Unidentified"]
    tab.video_viewer.set_frame(3)

    tab.cluster_button.click()
    assert not tab.settings_group.isEnabled()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)

    assert busy == [True, False]
    assert changed == [True]
    assert tab.video_viewer.current_frame == 3
    assert overlay_labels(tab) == ["b"]
    tab.video_selector.setCurrentIndex(1)
    assert tab.video_viewer.video is experiment.glasses_videos[1]
    assert overlay_labels(tab) == ["a"]
    assert tab.cluster_button.isEnabled()
    assert tab.settings_group.isEnabled()
    assert "2 of 2 tracklets identified" in tab.summary_label.text()
    experiment.save()
    assert Experiment.load(experiment.folder).identities.participants == ["a", "b"]


def test_settings_are_saved_loaded_and_used_for_clustering(qtbot, experiment):
    tab = ParticipantIdentificationTab(experiment)
    qtbot.addWidget(tab)
    changed = []
    tab.experiment_changed.connect(lambda: changed.append(True))
    settings = ClusterPostProcessingSettings(
        face_distance_threshold=0.4,
        body_distance_threshold=0.2,
        min_face_detections=2,
        min_body_detections=4,
    )
    for name, value in settings.model_dump().items():
        tab.settings_form._widgets[name].setValue(value)
    assert len(changed) == 4
    assert experiment.pipeline.cluster_post_processing == settings
    experiment.save()
    reloaded = Experiment.load(experiment.folder)
    assert reloaded.pipeline.cluster_post_processing == settings

    tab.set_experiment(reloaded)
    assert len(changed) == 4
    for name, value in settings.model_dump().items():
        assert tab.settings_form._widgets[name].value() == pytest.approx(value)
    tab.cluster_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)
    assert reloaded.identities.data["participant_id"].notna().all()
    assert overlay_labels(tab) == ["b"]


def test_missing_face_processing_blocks_the_button(qtbot, experiment):
    experiment.glasses_videos[1].clear()
    tab = ParticipantIdentificationTab(experiment)
    qtbot.addWidget(tab)
    assert not tab.cluster_button.isEnabled()
    assert "face detection for 'b'" in tab.blocked_label.text()


def test_stored_unidentified_tracklets_are_shown_without_running(qtbot, experiment):
    experiment.identities.set_data(
        pd.DataFrame(
            [("a", 1, None)],
            columns=["video_id", "track_id", "participant_id"],
        )
    )
    tab = ParticipantIdentificationTab(experiment)
    qtbot.addWidget(tab)
    assert overlay_labels(tab) == ["Unidentified"]
    assert overlay_colors(tab) == {"#808080"}
    tab.video_selector.setCurrentIndex(1)
    assert overlay_labels(tab) == ["Unidentified"]
    assert overlay_colors(tab) == {"#808080"}


def test_failure_releases_busy_state_and_preserves_previous_identities(
    qtbot, experiment, monkeypatch
):
    experiment.identities.set_data(
        pd.DataFrame(
            [("a", 1, "b")],
            columns=["video_id", "track_id", "participant_id"],
        )
    )

    def fail(_experiment):
        raise RuntimeError("clustering failed")

    monkeypatch.setattr(tab_module, "cluster_experiment_tracklets", fail)
    dialogs = []
    monkeypatch.setattr(
        QMessageBox, "exec", lambda dialog: dialogs.append(dialog.text())
    )
    tab = ParticipantIdentificationTab(experiment)
    qtbot.addWidget(tab)
    tab.cluster_button.click()
    qtbot.waitUntil(lambda: bool(dialogs), timeout=10000)
    assert not tab.is_busy()
    assert tab.cluster_button.isEnabled()
    assert tab.settings_group.isEnabled()
    assert dialogs == ["clustering failed"]
    assert experiment.identities.participants == ["b"]


def test_switching_experiments_clears_previous_video(qtbot, experiment):
    tab = ParticipantIdentificationTab(experiment)
    qtbot.addWidget(tab)
    tab.settings_form._widgets["face_distance_threshold"].setValue(0.4)
    tab.cluster_button.click()
    qtbot.waitUntil(lambda: not tab.is_busy(), timeout=10000)
    tab.set_experiment(Experiment(ExperimentConfig()))
    assert tab.settings_form._widgets["face_distance_threshold"].value() == 0.6
    assert tab.video_selector.count() == 0
    assert tab.video_viewer.video is None
    assert overlay_labels(tab) == []
    assert not tab.cluster_button.isEnabled()
