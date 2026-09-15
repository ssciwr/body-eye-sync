import numpy as np
import pandas as pd
import pytest

from body_eye_sync.postprocessing.tracklets_clustering import (
    ClusteringResult,
    identify_glasses_wearers,
)


def _faces(*rows):
    return pd.DataFrame(rows, columns=["frame", "track_id", "face_score"])


def test_face_counts_include_all_frames_and_deduplicate_overlapping_tracklets():
    mapping = {
        ("a", 1): 1,
        ("a", 2): 1,
        ("b", 1): 1,
        ("b", 2): 2,
        ("c", 1): 2,
        ("a", 9): 3,
    }
    clustering = ClusteringResult(
        tracklet_id_to_person_id=mapping,
        tracklet_face_embedding={
            key: np.array([1.0, 0.0]) for key in mapping if key != ("a", 9)
        },
    )
    frames = {
        "a": _faces(
            (0, 1, 0.9),
            (1, 1, 0.9),
            (2, 1, 0.9),
            (1, 2, 0.9),
            (2, 2, 0.9),
            (3, 2, 0.9),
            (4, 1, None),
            (0, 9, 0.9),
        ),
        "b": _faces(*[(frame, tid, 0.9) for frame in range(4) for tid in [1, 2]]),
        "c": _faces(*[(frame, 1, 0.9) for frame in range(4)], (0, 99, 0.9)),
    }

    identify_glasses_wearers(clustering, frames, min_face_frames=4)

    expected = pd.DataFrame(
        [[4, 4, 0], [0, 4, 4], [0, 0, 0]],
        index=pd.Index([1, 2, 3], name="person_id"),
        columns=pd.Index(["a", "b", "c"], name="video_id"),
        dtype=np.int64,
    )
    pd.testing.assert_frame_equal(clustering.person_face_frame_counts, expected)
    assert clustering.person_id_to_glasses_video_id == {1: "c", 2: "a", 3: None}


@pytest.mark.parametrize(
    "visible_counts,expected",
    [
        ([0, 4, 4], "a"),
        ([0, 3, 4], None),
        ([0, 0, 4], None),
        ([1, 4, 4], None),
        ([0, 0, 0], None),
        ([0, 4], "a"),
        ([0], None),
    ],
)
def test_only_one_zero_with_sufficient_evidence_identifies_a_wearer(
    visible_counts, expected
):
    video_ids = list("abc")[: len(visible_counts)]
    clustering = ClusteringResult(
        tracklet_id_to_person_id={(video_id, 1): 1 for video_id in video_ids},
        tracklet_face_embedding={
            (video_id, 1): np.array([1.0, 0.0]) for video_id in video_ids
        },
    )
    frames = {
        video_id: _faces(*[(frame, 1, 0.9) for frame in range(count)])
        for video_id, count in zip(video_ids, visible_counts)
    }

    identify_glasses_wearers(clustering, frames, min_face_frames=4)

    assert clustering.person_id_to_glasses_video_id == {1: expected}


@pytest.mark.parametrize("embedding", [None, np.zeros(2), np.array([np.nan, 1.0])])
def test_faces_without_valid_recognition_do_not_supply_identity_evidence(embedding):
    clustering = ClusteringResult(
        tracklet_id_to_person_id={("a", 1): 1},
        tracklet_face_embedding={("a", 1): embedding},
    )
    identify_glasses_wearers(
        clustering, {"a": _faces((0, 1, 0.9)), "b": _faces()}, min_face_frames=1
    )
    assert clustering.person_face_frame_counts.loc[1].tolist() == [0, 0]
    assert clustering.person_id_to_glasses_video_id == {1: None}


def test_empty_clustering_has_no_assignments_but_keeps_all_video_columns():
    clustering = ClusteringResult()
    identify_glasses_wearers(clustering, {"a": _faces(), "b": _faces()})
    assert clustering.person_face_frame_counts.columns.tolist() == ["a", "b"]
    assert clustering.person_face_frame_counts.empty
    assert clustering.person_id_to_glasses_video_id == {}
