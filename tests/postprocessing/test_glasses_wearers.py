import numpy as np
import pandas as pd
import pytest

from body_eye_sync.postprocessing.tracklets_clustering import (
    _ClusteringState,
    identify_glasses_wearers,
)


def _faces(*rows):
    return pd.DataFrame(rows, columns=["frame", "track_id", "face_score"])


def test_face_visibility_identifies_wearers_and_ignores_unrecognized_tracklets():
    mapping = {
        ("a", 1): 1,
        ("a", 2): 1,
        ("b", 1): 1,
        ("b", 2): 2,
        ("c", 1): 2,
        ("a", 9): 3,
    }
    clustering = _ClusteringState(
        person_by_tracklet=mapping,
        face_embedding_by_tracklet={
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

    associations = identify_glasses_wearers(clustering, frames)

    assert associations == {1: "c", 2: "a", 3: None}


@pytest.mark.parametrize(
    "visible_counts,expected",
    [
        ([0, 4, 4], "a"),
        ([0, 1, 4], "a"),
        ([0, 0, 4], None),
        ([1, 4, 4], None),
        ([0, 0, 0], None),
        ([0, 4], "a"),
        ([0], None),
    ],
)
def test_only_one_zero_with_nonzero_visibility_elsewhere_identifies_a_wearer(
    visible_counts, expected
):
    video_ids = list("abc")[: len(visible_counts)]
    clustering = _ClusteringState(
        person_by_tracklet={(video_id, 1): 1 for video_id in video_ids},
        face_embedding_by_tracklet={
            (video_id, 1): np.array([1.0, 0.0]) for video_id in video_ids
        },
    )
    frames = {
        video_id: _faces(*[(frame, 1, 0.9) for frame in range(count)])
        for video_id, count in zip(video_ids, visible_counts)
    }

    associations = identify_glasses_wearers(clustering, frames)

    assert associations == {1: expected}


@pytest.mark.parametrize("embedding", [None, np.zeros(2), np.array([np.nan, 1.0])])
def test_faces_without_valid_recognition_do_not_supply_identity_evidence(embedding):
    clustering = _ClusteringState(
        person_by_tracklet={("a", 1): 1},
        face_embedding_by_tracklet={("a", 1): embedding},
    )
    associations = identify_glasses_wearers(
        clustering, {"a": _faces((0, 1, 0.9)), "b": _faces()}
    )
    assert associations == {1: None}


def test_empty_clustering_has_no_assignments():
    associations = identify_glasses_wearers(
        _ClusteringState({}, {}), {"a": _faces(), "b": _faces()}
    )
    assert associations == {}
