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


def test_aggregate_embeddings_averages_per_tracklet_and_skips_missing_rows():
    embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (1, np.array([1.0, 0.0])),
        (2, np.array([0.0, 2.0])),
        (3, None),
    )

    aggregated = clustering._aggregate_embeddings(embeddings)

    assert set(aggregated) == {1, 2}
    assert np.allclose(aggregated[1], np.array([1.0, 0.0]))
    assert np.allclose(aggregated[2], np.array([0.0, 1.0]))  # L2-normalized


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
    mappings = clustering._build_identity_mappings(
        [10, 11, 12],
        np.array([0, 0, 1]),
    )

    assert mappings == {1: {10, 11}, 2: {12}}


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


def test_cluster_tracklets_uses_face_then_body_then_unique_ids(capsys):
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

    result_no_debug = clustering.cluster_tracklets(
        track_ids=[1, 2, 3, 4, 5],
        face_embeddings=face_embeddings,
        body_embeddings=body_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
        debug=False,  # no debug info printed out
    )

    assert result_no_debug.track_id_to_person_id == {1: 1, 2: 1, 3: 1, 4: 2, 5: 3}
    assert result_no_debug.person_id_to_track_ids == {1: [1, 2, 3], 2: [4], 3: [5]}
    assert set(result_no_debug.tracklet_face_embedding) == {1, 2}
    assert set(result_no_debug.tracklet_body_embedding) == {1, 2, 3, 4}

    # assert debug information
    printout_no_debug = capsys.readouterr().out

    assert printout_no_debug == ""

    result_with_debug = clustering.cluster_tracklets(
        track_ids=[1, 2, 3, 4, 5],
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
    assert f"{'Person ID':<10} {'Track IDs'}" in printout_with_debug
    assert "-" * 40 in printout_with_debug
    for pid, tids in result_with_debug.person_id_to_track_ids.items():
        assert f"{pid:<10} {tids}" in printout_with_debug


def test_cluster_tracklets_returns_an_empty_result_for_empty_input():
    result = clustering.cluster_tracklets(track_ids=[])

    assert result.track_id_to_person_id == {}
    assert result.person_id_to_track_ids == {}
    assert result.tracklet_face_embedding == {}
    assert result.tracklet_body_embedding == {}


def test_cluster_tracklets_from_input_delegates_to_cluster_tracklets():
    face_embeddings = _embeddings_df(
        (1, np.array([1.0, 0.0])),
        (2, np.array([1.0, 0.0])),
    )
    inputs = clustering.TrackletClusteringInput(
        track_ids=[1, 2],
        face_embeddings=face_embeddings,
    )

    result_from_input = clustering.cluster_tracklets_from_input(
        inputs,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )
    direct_result = clustering.cluster_tracklets(
        track_ids=[1, 2],
        face_embeddings=face_embeddings,
        face_distance_threshold=0.1,
        body_distance_threshold=0.1,
        min_face_detections=1,
        min_body_detections=2,
    )

    assert (
        result_from_input.track_id_to_person_id == direct_result.track_id_to_person_id
    )
    assert (
        result_from_input.person_id_to_track_ids == direct_result.person_id_to_track_ids
    )
    assert (
        result_from_input.tracklet_body_embedding
        == direct_result.tracklet_body_embedding
    )
    assert (
        result_from_input.tracklet_face_embedding.keys()
        == direct_result.tracklet_face_embedding.keys()
    )
    for track_id in result_from_input.tracklet_face_embedding:
        assert np.allclose(
            result_from_input.tracklet_face_embedding[track_id],
            direct_result.tracklet_face_embedding[track_id],
        )
