import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.config import (
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.postprocess import cluster_experiment_tracklets
from body_eye_sync.pipeline.face_detection import FaceBox, FaceFrameResult
from body_eye_sync.pipeline.object_tracking import BoundingBox


def _add_faces(video, people, embeddings_per_track=1):
    rows = [(frame, tid) for tid, person in people for frame in range(4)]
    tracks = pd.DataFrame(rows, columns=["frame", "track_id"])
    for column, value in {
        "x1": 0.0,
        "y1": 0.0,
        "x2": 1.0,
        "y2": 1.0,
        "conf": 0.9,
    }.items():
        tracks[column] = value
    video.set_data(tracks)
    video.begin_face_detection(embeddings_per_track=embeddings_per_track)
    for tid, person in people:
        if person is None:
            continue
        for frame in range(4):
            face = FaceBox(
                BoundingBox(0.0, 0.0, 1.0, 1.0, tid),
                0.9,
                landmarks=[(0.0, 0.0)] * 5,
                embedding=np.eye(3)[person],
            )
            video.add_face_detection_frame(FaceFrameResult(frame, [face]))
    video.finish_face_detection()


def _experiment(tmp_path):
    experiment = Experiment(
        ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id=video_id, path=f"{video_id}.mp4", gaze_path=f"{video_id}.tsv"
                )
                for video_id in ["a", "b", "c"]
            ],
            fixed_videos=[FixedVideoInput(id="room", path="room.mp4")],
        ),
        tmp_path,
    )
    experiment.pipeline.cluster_post_processing.min_face_frames = 4
    _add_faces(experiment.glasses_videos[0], [(1, 1), (2, 2), (9, None)])
    _add_faces(experiment.glasses_videos[1], [(1, 0), (2, 2)])
    _add_faces(experiment.glasses_videos[2], [(1, 0), (2, 1)])
    _add_faces(experiment.fixed_videos[0], [(1, 0)])
    return experiment


def test_clustering_associates_wearers_and_persists_tracklet_assignments(tmp_path):
    experiment = _experiment(tmp_path)

    result = cluster_experiment_tracklets(experiment)

    assert result.person_face_frame_counts.columns.tolist() == ["a", "b", "c"]
    assert all(video_id != "room" for video_id, _ in result.tracklet_id_to_person_id)
    person_b = result.tracklet_id_to_person_id[("a", 1)]
    assert result.person_face_frame_counts.loc[person_b].tolist() == [4, 0, 4]
    expected = pd.DataFrame(
        [
            ("a", 1, "b"),
            ("a", 2, "c"),
            ("a", 9, None),
            ("b", 1, "a"),
            ("b", 2, "c"),
            ("c", 1, "a"),
            ("c", 2, "b"),
        ],
        columns=["video_id", "track_id", "participant_id"],
    )
    pd.testing.assert_frame_equal(experiment.identities.data, expected)
    # Each tracklet stores only one embedding, but contributes four face frames.
    assert len(experiment.glasses_videos[0].face_embeddings) == 2
    experiment.save()
    pd.testing.assert_frame_equal(Experiment.load(tmp_path).identities.data, expected)


def test_insufficient_visibility_leaves_all_tracklets_unidentified(tmp_path):
    experiment = _experiment(tmp_path)
    experiment.pipeline.cluster_post_processing.min_face_frames = 5
    cluster_experiment_tracklets(experiment)
    assert experiment.identities.data["participant_id"].isna().all()


def test_unprocessed_glasses_video_cannot_be_treated_as_zero_visibility(tmp_path):
    experiment = _experiment(tmp_path)
    experiment.glasses_videos[1].clear()
    with pytest.raises(ValueError, match="face detection for 'b'"):
        cluster_experiment_tracklets(experiment)
    assert not experiment.identities.has_data()


def test_no_videos_produces_completed_empty_identities(tmp_path):
    experiment = Experiment(ExperimentConfig(), tmp_path)
    result = cluster_experiment_tracklets(experiment)
    assert experiment.identities.has_data()
    assert experiment.identities.data.empty
    assert result.person_id_to_glasses_video_id == {}


def test_disabled_embedding_collection_cannot_supply_evidence_of_absence(tmp_path):
    experiment = _experiment(tmp_path)
    _add_faces(experiment.glasses_videos[1], [(1, 0), (2, 2)], embeddings_per_track=0)
    with pytest.raises(ValueError, match="recognition embeddings for 'b'"):
        cluster_experiment_tracklets(experiment)


def test_processed_glasses_video_without_faces_remains_a_matrix_column(tmp_path):
    experiment = _experiment(tmp_path)
    _add_faces(experiment.glasses_videos[1], [(1, None)])
    result = cluster_experiment_tracklets(experiment)
    assert (result.person_face_frame_counts["b"] == 0).all()
    # Person B is visible in A and C, and absent only from B.
    assert (
        experiment.identities.for_video("a")
        .set_index("track_id")
        .loc[1, "participant_id"]
        == "b"
    )


def test_fixed_videos_have_no_effect_on_clustering(tmp_path):
    experiment = _experiment(tmp_path)
    expected = cluster_experiment_tracklets(experiment)
    identities = experiment.identities.data.copy()
    # Conflicting recognition evidence in the fixed recording is ignored.
    _add_faces(experiment.fixed_videos[0], [(1, 2), (2, 1), (3, 0)])
    actual = cluster_experiment_tracklets(experiment)
    assert actual.tracklet_id_to_person_id == expected.tracklet_id_to_person_id
    pd.testing.assert_frame_equal(
        actual.person_face_frame_counts, expected.person_face_frame_counts
    )
    pd.testing.assert_frame_equal(experiment.identities.data, identities)
