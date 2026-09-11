import numpy as np
import pandas as pd
import pytest

from body_eye_sync.postprocessing import tracklets_clustering as clustering


def _embeddings_df(*rows: tuple[int, np.ndarray | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_idx": [i for i in range(len(rows))],
            "track_id": [track_id for track_id, _ in rows],
            "embedding": [embedding for _, embedding in rows],
        }
    )


def test_aggregate_embeddings_empty_input_or_no_emb_column():
    # empty input
    embeddings = _embeddings_df()

    aggregated = clustering._aggregate_embeddings(embeddings, video_id=None)

    assert aggregated == {}

    # no embedding column
    embeddings = pd.DataFrame(
        {
            "frame_idx": [1, 2, 3],
            "track_id": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        clustering._aggregate_embeddings(embeddings, video_id=None)


def test_aggregate_embeddings_averages_per_tracklet_and_skips_missing_rows():
    embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (1, np.array([1.0, 0.0])),
        (2, np.array([0.0, 2.0])),
        (3, None),
    )

    aggregated_none_videoid = clustering._aggregate_embeddings(
        embeddings, video_id=None
    )

    assert set(aggregated_none_videoid) == {1, 2}
    assert np.allclose(aggregated_none_videoid[1], np.array([1.0, 0.0]))
    assert np.allclose(
        aggregated_none_videoid[2], np.array([0.0, 1.0])
    )  # L2-normalized

    aggregated_with_videoid = clustering._aggregate_embeddings(
        embeddings, video_id="video1"
    )

    assert set(aggregated_with_videoid) == {("video1", 1), ("video1", 2)}
    assert np.allclose(aggregated_with_videoid[("video1", 1)], np.array([1.0, 0.0]))
    assert np.allclose(aggregated_with_videoid[("video1", 2)], np.array([0.0, 1.0]))


def test_aggregate_embeddings_averages_per_tracklet_with_video_id():
    embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (1, np.array([1.0, 0.0])),
        (2, np.array([0.0, 2.0])),
        (3, None),
    )
    embeddings["video_id"] = ["video1", "video1", "video2", "video2"]

    aggregated_with_videoid = clustering._aggregate_embeddings(
        embeddings, video_id="video2"
    )

    assert set(aggregated_with_videoid) == {("video2", 2)}
    assert np.allclose(aggregated_with_videoid[("video2", 2)], np.array([0.0, 1.0]))


def test_aggregate_embeddings_none_embs_are_skipped():
    embeddings = _embeddings_df(
        (1, None),
        (1, None),
        (2, None),
    )

    aggregated = clustering._aggregate_embeddings(embeddings, video_id=None)

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
        [10, 11, 12],
        np.array([0, 0, 1]),
    )

    assert mappings == {1: {10, 11}, 2: {12}}

    # multi-video case
    mappings_multi = clustering._build_identity_mappings(
        [("video1", 10), ("video1", 11), ("video2", 12)],
        np.array([0, 0, 1]),
    )

    assert mappings_multi == {1: {("video1", 10), ("video1", 11)}, 2: {("video2", 12)}}


def test_jaccard_similarity_uses_intersection_over_union():
    assert clustering._jaccard_similarity({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert clustering._jaccard_similarity(set(), set()) == 0.0


def test_merge_identity_mappings_same_priority_no_source_return_target():
    target = {1: {1, 2}, 2: {3, 4}}
    source = {}

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.5
    )

    assert target == {1: {1, 2}, 2: {3, 4}}


def test_merge_identity_mappings_same_priority_merges_overlapping_clusters_and_preserves_new_ones():
    target = {1: {1, 2}, 2: {3, 4}}
    source = {11: {2, 5}, 12: {3, 6}, 13: {10}}

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.0
    )

    assert target == {1: {1, 2, 5}, 2: {3, 4, 6}, 3: {10}}


def test_merge_identity_mappings_same_priority_chooses_the_lowest_person_id_on_a_tie():
    target = {1: {1, 2}, 2: {3, 4}}
    source = {9: {2, 4}}

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.3
    )

    assert target == {1: {1, 2, 3, 4}}


def test_merge_identity_mappings_same_priority_merges_multiple_overlapping_clusters():
    target = {12: {1, 2, 5}, 11: {3, 4}}
    source = {9: {2, 3, 5}, 10: {4}}

    clustering._merge_identity_mappings_same_priority(
        target, source, merge_jaccard_threshold=0.5
    )

    assert target == {12: {1, 2, 3, 5}, 11: {4}}


def test_merge_identity_mappings_prioritize_target_no_source_only_ignore_source():
    target = {1: {1, 2}, 2: {3, 4}}
    source = {11: {1}, 12: {2}}

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {1: {1, 2}, 2: {3, 4}}


def test_merge_identity_mappings_prioritize_target_one_id_overlapping_update_target():
    target = {1: {1, 2}, 2: {3, 4}}
    source = {11: {2, 5}, 12: {3, 6}, 13: {10}}

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {1: {1, 2, 5}, 2: {3, 4, 6}, 3: {10}}


def test_merge_identity_mappings_prioritize_target_no_overlapping_add_new_to_target():
    target = {1: {1, 2}, 2: {3, 4}}
    source = {11: {5}, 12: {6}}

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {1: {1, 2}, 2: {3, 4}, 3: {5}, 4: {6}}


def test_merge_identity_mappings_prioritize_target_multiple_overlapping_ignore_source():
    target = {1: {1, 2, 5}, 2: {3, 4}}
    source = {
        9: {2, 3, 5, 6}
    }  # track 6 is ignored as source has multiple overlapping clusters with target

    clustering._merge_identity_mappings_prioritize_target(target, source)

    assert target == {1: {1, 2, 5}, 2: {3, 4}}


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
        tracklet_ids=[1, 2, 3, 4, 5],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=False,  # no debug info printed out
        video_id=None,
    )

    assert result_no_debug.tracklet_id_to_person_id == {1: 1, 2: 1, 3: 1, 4: 2, 5: 3}
    assert result_no_debug.person_id_to_tracklet_ids == {1: [1, 2, 3], 2: [4], 3: [5]}
    assert set(result_no_debug.tracklet_face_embedding) == {1, 2}
    assert set(result_no_debug.tracklet_body_embedding) == {1, 2, 3, 4}

    # assert debug information
    printout_no_debug = capsys.readouterr().out

    assert printout_no_debug == ""

    result_with_debug = clustering._cluster_tracklets(
        tracklet_ids=[1, 2, 3, 4, 5],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=True,  # print debug info
        video_id=None,
    )

    printout_with_debug = capsys.readouterr().out

    assert "Face clusters: 1 (tracklets: 2)" in printout_with_debug
    assert "Body clusters: 2 (tracklets: 4)" in printout_with_debug
    assert "Final identities: 3" in printout_with_debug
    assert f"{'Person ID':<10} {'Tracklet IDs'}" in printout_with_debug
    assert "-" * 40 in printout_with_debug
    for pid, tlids in result_with_debug.person_id_to_tracklet_ids.items():
        assert f"{pid:<10} {tlids}" in printout_with_debug


def test_cluster_tracklets_single_video_returns_an_empty_result_for_empty_input():
    result = clustering._cluster_tracklets(tracklet_ids=[])

    assert result.tracklet_id_to_person_id == {}
    assert result.person_id_to_tracklet_ids == {}
    assert result.tracklet_face_embedding == {}
    assert result.tracklet_body_embedding == {}


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
        video_id=None,
    )

    assert result.tracklet_id_to_person_id == {
        ("video1", 1): 1,
        ("video1", 2): 1,
        ("video2", 3): 1,
        ("video2", 4): 2,
    }
    assert result.person_id_to_tracklet_ids == {
        1: [("video1", 1), ("video1", 2), ("video2", 3)],
        2: [("video2", 4)],
    }
    assert set(result.tracklet_face_embedding) == {("video1", 1), ("video1", 2)}
    assert set(result.tracklet_body_embedding) == {
        ("video1", 1),
        ("video1", 2),
        ("video2", 3),
        ("video2", 4),
    }

    # filter video case
    filtered_result = clustering._cluster_tracklets(
        tracklet_ids=[("video1", 1), ("video1", 2), ("video2", 3), ("video2", 4)],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=False,
        video_id="video2",  # cluster only video2 tracklets
    )

    assert filtered_result.tracklet_id_to_person_id == {
        ("video2", 3): 1,
        ("video2", 4): 2,
    }


def test_cluster_tracklets_multi_video_returns_an_empty_result_for_empty_input():
    result = clustering._cluster_tracklets(tracklet_ids=[])

    assert result.tracklet_id_to_person_id == {}
    assert result.person_id_to_tracklet_ids == {}
    assert result.tracklet_face_embedding == {}
    assert result.tracklet_body_embedding == {}


def test_cluster_tracklets_from_input_single_video_delegates_to_cluster_tracklets():
    face_embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    input = clustering.TrackletClusteringInput(
        track_ids=[1, 2],
        face_embeddings=face_embeddings,
    )

    result_from_input = clustering.cluster_tracklets_from_input(
        [input],
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )
    direct_result = clustering._cluster_tracklets(
        tracklet_ids=[1, 2],
        face_embeddings=face_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )

    assert (
        result_from_input.tracklet_id_to_person_id
        == direct_result.tracklet_id_to_person_id
    )
    assert (
        result_from_input.person_id_to_tracklet_ids
        == direct_result.person_id_to_tracklet_ids
    )
    assert (
        result_from_input.tracklet_body_embedding
        == direct_result.tracklet_body_embedding
    )
    assert (
        result_from_input.tracklet_face_embedding.keys()
        == direct_result.tracklet_face_embedding.keys()
    )
    for tracklet_id in result_from_input.tracklet_face_embedding:
        assert np.allclose(
            result_from_input.tracklet_face_embedding[tracklet_id],
            direct_result.tracklet_face_embedding[tracklet_id],
        )


def test_cluster_tracklets_from_input_multi_video_delegates_to_cluster_tracklets():
    face_embeddings1 = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    input1 = clustering.TrackletClusteringInput(
        track_ids=[1, 2], face_embeddings=face_embeddings1, video_id="video1"
    )

    face_embeddings2 = _embeddings_df(
        (3, np.array([0.0, 1.0])),
        (4, np.array([1.0, 0.0])),
    )
    body_embeddings2 = _embeddings_df(
        (3, np.array([0.0, 1.0])),
        (4, np.array([1.0, 0.0])),
    )
    input2 = clustering.TrackletClusteringInput(
        track_ids=[3, 4],
        face_embeddings=face_embeddings2,
        body_embeddings=body_embeddings2,
        video_id="video2",
    )

    result = clustering.cluster_tracklets_from_input(
        [input1, input2],
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )

    assert result.tracklet_id_to_person_id == {
        ("video1", 1): 1,
        ("video1", 2): 1,
        ("video2", 3): 2,
        ("video2", 4): 1,
    }

    assert result.person_id_to_tracklet_ids == {
        1: [("video1", 1), ("video1", 2), ("video2", 4)],
        2: [("video2", 3)],
    }

    assert set(result.tracklet_face_embedding) == {
        ("video1", 1),
        ("video1", 2),
        ("video2", 3),
        ("video2", 4),
    }
    assert set(result.tracklet_body_embedding) == {("video2", 3), ("video2", 4)}


def test_cluster_tracklets_from_input_returns_an_empty_result_for_empty_input():
    result = clustering.cluster_tracklets_from_input(
        input_data=[],
    )

    assert result.tracklet_id_to_person_id == {}
    assert result.person_id_to_tracklet_ids == {}
    assert result.tracklet_face_embedding == {}
    assert result.tracklet_body_embedding == {}
