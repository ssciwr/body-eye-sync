from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.postprocessing import tracklets_clustering as clustering
from body_eye_sync.experiment.video import GlassesVideo
from body_eye_sync.pipeline.face_detection import FaceBox, FaceFrameResult
from body_eye_sync.pipeline.object_tracking import BoundingBox


def _embeddings_df(
    *rows: tuple[int, np.ndarray | None], video_id: str = "video1"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": [video_id] * len(rows),
            "frame_idx": [i for i in range(len(rows))],
            "track_id": [track_id for track_id, _ in rows],
            "embedding": [embedding for _, embedding in rows],
        }
    )


def _video(video_id, track_ids, face_embeddings=None, body_embeddings=None):
    video = GlassesVideo(id=video_id)
    video.begin_object_tracking(embeddings_per_track=32)
    video.add_object_tracking_frame(
        SimpleNamespace(
            frame_idx=1,
            tracks=np.array([[0.0, 0.0, 1.0, 1.0, tid, 0.9] for tid in track_ids]),
        )
    )
    if body_embeddings is not None:
        for index, row in enumerate(body_embeddings.itertuples(index=False)):
            video.add_object_tracking_frame(
                SimpleNamespace(
                    frame_idx=index + 1,
                    tracks=np.array([[0.0, 0.0, 1.0, 1.0, row.track_id, 0.9]]),
                    embeddings=np.array([row.embedding]),
                )
            )
    video.finish_object_tracking()
    video.begin_face_detection(embeddings_per_track=32)
    if face_embeddings is not None:
        for row in face_embeddings.itertuples(index=False):
            if row.embedding is not None:
                video.add_face_detection_frame(
                    FaceFrameResult(
                        0,
                        [
                            FaceBox(
                                BoundingBox(0.0, 0.0, 1.0, 1.0, row.track_id),
                                0.9,
                                landmarks=[(0.0, 0.0)] * 5,
                                embedding=row.embedding,
                            )
                        ],
                    )
                )
    video.finish_face_detection()
    return video


def test_aggregate_embeddings_empty_input_or_no_emb_column():
    # empty input
    embeddings = _embeddings_df()

    aggregated = clustering._aggregate_embeddings(embeddings)

    assert aggregated == {}

    # no embedding column
    embeddings = pd.DataFrame(
        {
            "frame_idx": [1, 2, 3],
            "track_id": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        clustering._aggregate_embeddings(embeddings)


def test_aggregate_embeddings_averages_per_tracklet_and_skips_missing_rows():
    embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (1, np.array([1.0, 0.0])),
        (2, np.array([0.0, 2.0])),
        (3, None),
    )

    aggregated = clustering._aggregate_embeddings(embeddings)

    assert set(aggregated) == {("video1", 1), ("video1", 2)}
    assert np.allclose(aggregated[("video1", 1)], np.array([1.0, 0.0]))
    assert np.allclose(aggregated[("video1", 2)], np.array([0.0, 1.0]))


def test_aggregate_embeddings_averages_per_tracklet_with_video_id():
    embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (1, np.array([1.0, 0.0])),
        (2, np.array([0.0, 2.0])),
        (3, None),
    )
    embeddings["video_id"] = ["video1", "video1", "video2", "video2"]

    aggregated = clustering._aggregate_embeddings(embeddings)

    assert set(aggregated) == {("video1", 1), ("video2", 2)}
    assert np.allclose(aggregated[("video1", 1)], np.array([1.0, 0.0]))
    assert np.allclose(aggregated[("video2", 2)], np.array([0.0, 1.0]))


def test_aggregate_embeddings_none_embs_are_skipped():
    embeddings = _embeddings_df(
        (1, None),
        (1, None),
        (2, None),
    )

    aggregated = clustering._aggregate_embeddings(embeddings)

    assert aggregated == {}


def test_cosine_distance_matrix_computes_pairwise_cosine_distance():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    distance_matrix = clustering._cosine_distance_matrix(embeddings)

    assert distance_matrix.shape == (3, 3)
    assert np.allclose(
        distance_matrix,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
    )


def test_cosine_distance_matrix_returns_empty_for_empty_input():
    distance_matrix = clustering._cosine_distance_matrix(np.empty((0, 2)))

    assert distance_matrix.shape == (0, 0)


def test_cosine_distance_matrix_with_nan():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [np.nan, np.nan],
            [0.5, 0.5],
        ]
    )

    distance_matrix = clustering._cosine_distance_matrix(embeddings)

    assert distance_matrix.shape == (3, 3)

    assert np.allclose(
        distance_matrix,
        np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],  # nan was replaced by 0.5
            ]
        ),
    )


def test_cosine_distance_matrix_with_all_nan():
    embeddings = np.array(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    distance_matrix = clustering._cosine_distance_matrix(embeddings)

    assert distance_matrix.shape == (2, 2)
    assert np.allclose(
        distance_matrix,
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )


def test_cluster_embeddings_groups_vectors_within_threshold():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    labels = clustering._cluster_embeddings(embeddings, distance_threshold=0.1)

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]

    labels_loose = clustering._cluster_embeddings(embeddings, distance_threshold=1.5)

    assert labels_loose[0] == labels_loose[1] == labels_loose[2]


def test_build_identity_mappings_orders_person_ids_deterministically():
    # single video case
    mappings = clustering._build_identity_mappings(
        [("video1", 10), ("video1", 11), ("video1", 12)],
        np.array([0, 0, 1]),
    )

    assert mappings == {1: {("video1", 10), ("video1", 11)}, 2: {("video1", 12)}}

    # multi-video case
    mappings_multi = clustering._build_identity_mappings(
        [("video1", 10), ("video1", 11), ("video2", 12)],
        np.array([0, 0, 1]),
    )

    assert mappings_multi == {1: {("video1", 10), ("video1", 11)}, 2: {("video2", 12)}}

    # Numeric labels and input order are arbitrary; cluster membership defines
    # the canonical person ID.
    reordered = clustering._build_identity_mappings(
        [("video2", 12), ("video1", 11), ("video1", 10)],
        np.array([42, 7, 7]),
    )

    assert reordered == mappings_multi


def test_jaccard_similarity_uses_intersection_over_union():
    assert clustering._jaccard_similarity(
        {("video1", 1), ("video1", 2)}, {("video1", 2), ("video1", 3)}
    ) == pytest.approx(1 / 3)
    assert clustering._jaccard_similarity(set(), set()) == 0.0


def test_merge_identity_mappings_same_priority_no_source_return_target():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video1", 3), ("video1", 4)}}
    source = {}

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.5
    )

    assert target == {
        1: {("video1", 1), ("video1", 2)},
        2: {("video1", 3), ("video1", 4)},
    }


def test_merge_identity_mappings_same_priority_merges_overlapping_clusters_and_preserves_new_ones():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video1", 3), ("video1", 4)}}
    source = {
        11: {("video1", 2), ("video1", 5)},
        12: {("video1", 3), ("video1", 6)},
        13: {("video1", 10)},
    }

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.0
    )

    assert target == {
        1: {("video1", 1), ("video1", 2), ("video1", 5)},
        2: {("video1", 3), ("video1", 4), ("video1", 6)},
        3: {("video1", 10)},
    }


def test_merge_identity_mappings_same_priority_chooses_the_lowest_person_id_on_a_tie():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video1", 3), ("video1", 4)}}
    source = {9: {("video1", 2), ("video1", 4)}}

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.3
    )

    assert target == {1: {("video1", 1), ("video1", 2), ("video1", 3), ("video1", 4)}}


def test_merge_identity_mappings_same_priority_merges_multiple_overlapping_clusters():
    target = {
        12: {("video1", 1), ("video1", 2), ("video1", 5)},
        11: {("video1", 3), ("video1", 4)},
    }
    source = {9: {("video1", 2), ("video1", 3), ("video1", 5)}, 10: {("video1", 4)}}

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.5
    )

    assert target == {
        12: {("video1", 1), ("video1", 2), ("video1", 3), ("video1", 5)},
        11: {("video1", 4)},
    }


def test_merge_identity_mappings_prioritize_target_no_source_only_ignore_source():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video1", 3), ("video1", 4)}}
    source = {11: {("video1", 1)}, 12: {("video1", 2)}}

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2)},
        2: {("video1", 3), ("video1", 4)},
    }


def test_merge_identity_mappings_prioritize_target_one_id_overlapping_update_target():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video1", 3), ("video1", 4)}}
    source = {
        11: {("video1", 2), ("video1", 5)},
        12: {("video1", 3), ("video1", 6)},
        13: {("video1", 10)},
    }

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2), ("video1", 5)},
        2: {("video1", 3), ("video1", 4), ("video1", 6)},
        3: {("video1", 10)},
    }


def test_merge_identity_mappings_prioritize_target_no_overlapping_add_new_to_target():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video1", 3), ("video1", 4)}}
    source = {11: {("video1", 5)}, 12: {("video1", 6)}}

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2)},
        2: {("video1", 3), ("video1", 4)},
        3: {("video1", 5)},
        4: {("video1", 6)},
    }


def test_merge_identity_mappings_prioritize_target_multiple_overlapping_ignore_source():
    target = {
        1: {("video1", 1), ("video1", 2), ("video1", 5)},
        2: {("video1", 3), ("video1", 4)},
    }
    source = {
        9: {("video1", 2), ("video1", 3), ("video1", 5), ("video1", 6)}
    }  # track 6 is ignored as source has multiple overlapping clusters with target

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2), ("video1", 5)},
        2: {("video1", 3), ("video1", 4)},
    }


def test_merge_identity_mappings_prioritize_target_multi_video_no_source_only_ignore_source():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video2", 3), ("video2", 4)}}
    source = {11: {("video1", 1)}, 12: {("video1", 2)}}

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2)},
        2: {("video2", 3), ("video2", 4)},
    }


def test_merge_identity_mappings_prioritize_target_multi_video_one_id_overlapping_update_target():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video2", 3), ("video2", 4)}}
    source = {
        11: {("video1", 2), ("video1", 5)},
        12: {("video2", 3), ("video2", 6)},
        13: {("video2", 10)},
    }

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2), ("video1", 5)},
        2: {("video2", 3), ("video2", 4), ("video2", 6)},
        3: {("video2", 10)},
    }


def test_merge_identity_mappings_prioritize_target_multi_video_no_overlapping_add_new_to_target():
    target = {1: {("video1", 1), ("video1", 2)}, 2: {("video2", 3), ("video2", 4)}}
    source = {11: {("video1", 5)}, 12: {("video2", 6)}}

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2)},
        2: {("video2", 3), ("video2", 4)},
        3: {("video1", 5)},
        4: {("video2", 6)},
    }


def test_merge_identity_mappings_prioritize_target_multi_video_multiple_overlapping_ignore_source():
    target = {
        1: {("video1", 1), ("video1", 2), ("video1", 5)},
        2: {("video2", 3), ("video2", 4)},
    }
    source = {
        9: {("video1", 2), ("video2", 3), ("video1", 5), ("video2", 6)}
    }  # track 6 is ignored as source has multiple overlapping clusters with target

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {
        1: {("video1", 1), ("video1", 2), ("video1", 5)},
        2: {("video2", 3), ("video2", 4)},
    }


def test_cluster_tracklets_single_video_uses_face_then_body_then_unique_ids(capsys):
    face_embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    body_embeddings = _embeddings_df(
        (1, np.array([0.0, 1.0])),
        (2, np.array([0.0, 1.0])),
        (3, np.array([0.0, 1.0])),
        (4, np.array([1.0, 0.0])),
    )

    result_no_debug = clustering._cluster_tracklets(
        tracklet_ids=[("video1", tid) for tid in [1, 2, 3, 4, 5]],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=False,  # no debug info printed out
    )

    assert result_no_debug.person_by_tracklet == {
        ("video1", 1): 1,
        ("video1", 2): 1,
        ("video1", 3): 1,
        ("video1", 4): 2,
        ("video1", 5): 3,
    }
    assert set(result_no_debug.face_embedding_by_tracklet) == {
        ("video1", 1),
        ("video1", 2),
    }

    # assert debug information
    printout_no_debug = capsys.readouterr().out

    assert printout_no_debug == ""

    clustering._cluster_tracklets(
        tracklet_ids=[("video1", tid) for tid in [1, 2, 3, 4, 5]],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=True,  # print debug info
    )

    printout_with_debug = capsys.readouterr().out

    assert "Face clusters: 1 (tracklets: 2)" in printout_with_debug
    assert "Body clusters: 2 (tracklets: 4)" in printout_with_debug
    assert "Final identities: 3" in printout_with_debug
    assert f"{'Person ID':<10} {'Tracklet IDs'}" in printout_with_debug
    assert "-" * 40 in printout_with_debug
    assert (
        "1          [('video1', 1), ('video1', 2), ('video1', 3)]"
        in printout_with_debug
    )
    assert "2          [('video1', 4)]" in printout_with_debug
    assert "3          [('video1', 5)]" in printout_with_debug


def test_cluster_tracklets_single_video_returns_an_empty_result_for_empty_input():
    result = clustering._cluster_tracklets(tracklet_ids=[])

    assert result.person_by_tracklet == {}
    assert result.face_embedding_by_tracklet == {}


def test_cluster_tracklets_multi_video_uses_face_then_body_then_unique_ids():
    face_embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    body_embeddings = _embeddings_df(
        (1, np.array([0.0, 1.0])),
        (2, np.array([0.0, 1.0])),
        (3, np.array([0.0, 1.0])),
        (4, np.array([1.0, 0.0])),
    )

    face_embeddings["video_id"] = ["video1", "video1"]
    body_embeddings["video_id"] = ["video1", "video1", "video2", "video2"]

    result = clustering._cluster_tracklets(
        tracklet_ids=[("video1", 1), ("video1", 2), ("video2", 3), ("video2", 4)],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=False,
    )

    assert result.person_by_tracklet == {
        ("video1", 1): 1,
        ("video1", 2): 1,
        ("video2", 3): 1,
        ("video2", 4): 2,
    }
    assert set(result.face_embedding_by_tracklet) == {("video1", 1), ("video1", 2)}

    reordered = clustering._cluster_tracklets(
        tracklet_ids=[("video2", 4), ("video2", 3), ("video1", 2), ("video1", 1)],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=False,
    )

    assert reordered.person_by_tracklet == result.person_by_tracklet


def test_cluster_tracklets_multi_video_returns_an_empty_result_for_empty_input():
    result = clustering._cluster_tracklets(tracklet_ids=[])

    assert result.person_by_tracklet == {}
    assert result.face_embedding_by_tracklet == {}


def test_cluster_tracklets_single_video_returns_identity_table():
    face_embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    video = _video("video1", [1, 2], face_embeddings=face_embeddings)

    result = clustering.cluster_tracklets(
        [video],
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )
    assert result.columns.tolist() == ["video_id", "track_id", "participant_id"]
    assert result["video_id"].tolist() == ["video1", "video1"]
    assert result["track_id"].tolist() == [1, 2]
    assert result["participant_id"].isna().all()
    assert result["video_id"].cat.categories.tolist() == ["video1"]
    assert result["participant_id"].cat.categories.tolist() == ["video1"]
    assert result["track_id"].dtype == np.dtype("int64")


def test_cluster_tracklets_multi_video_returns_participant_assignments():
    face_embeddings1 = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    video1 = _video("video1", [1, 2], face_embeddings=face_embeddings1)

    face_embeddings2 = _embeddings_df(
        (3, np.array([0.0, 1.0])),
        (4, np.array([1.0, 0.0])),
    )
    body_embeddings2 = _embeddings_df(
        (3, np.array([0.0, 1.0])),
        (4, np.array([1.0, 0.0])),
    )
    video2 = _video("video2", [3, 4], face_embeddings2, body_embeddings2)

    result = clustering.cluster_tracklets(
        [video1, video2],
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )

    assert result[["video_id", "track_id"]].values.tolist() == [
        ["video1", 1],
        ["video1", 2],
        ["video2", 3],
        ["video2", 4],
    ]
    assert result["participant_id"].iloc[:2].isna().all()
    assert result.loc[2, "participant_id"] == "video1"
    assert pd.isna(result.loc[3, "participant_id"])
    assert result["video_id"].cat.categories.tolist() == ["video1", "video2"]
    assert result["participant_id"].cat.categories.tolist() == ["video1", "video2"]

    reordered = clustering.cluster_tracklets(
        [video2, video1],
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )
    pd.testing.assert_frame_equal(reordered, result)


def test_cluster_tracklets_returns_an_empty_result_for_empty_input():
    result = clustering.cluster_tracklets(
        glasses_videos=[],
    )

    assert result.empty
    assert result.columns.tolist() == ["video_id", "track_id", "participant_id"]
    assert isinstance(result["video_id"].dtype, pd.CategoricalDtype)
    assert result["track_id"].dtype == np.dtype("int64")
    assert isinstance(result["participant_id"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("video_id", ["", None, "video1"])
def test_cluster_tracklets_requires_nonempty_unique_video_ids(video_id):
    face_embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    video1 = _video("video1", [1, 2], face_embeddings)

    face_embeddings2 = _embeddings_df(
        (3, np.array([0.0, 1.0])),
        (4, np.array([1.0, 0.0])),
    )
    video2 = _video(video_id, [3, 4], face_embeddings2)

    with pytest.raises(ValueError):
        clustering.cluster_tracklets(
            [video1, video2],
            face_distance_threshold=0.1,
            body_distance_threshold=0.1,
            min_face_detections=1,
            min_body_detections=2,
        )
