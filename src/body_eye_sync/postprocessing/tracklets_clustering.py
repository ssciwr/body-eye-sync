"""Cluster tracklets from single or multiple videos
into persistent person IDs using face and body embeddings.

Clustering strategy
--------------------------
1. **Face embeddings as the primary identity signal**
    - After running an experiment with face detection enabled,
    each ``Video`` instance carries top K face embeddings per tracklet, via
    ``video.face_embeddings``.
    - The K embeddings are first aggregated into a single
    representative embedding per tracklet (mean of L2-normalised embeddings).
    - Two tracklets are considered the same person if their representative embeddings
    are within the ``face_distance_threshold`` in cosine distance.


2. **Body embeddings as fallback** to associate tracklets
    without-a-face embedding with an existing face identity
    - Tracklets that yield no valid face embedding (e.g. the person's face was
    never visible) are clustered using body embeddings.
    - The top K body embeddings are also carried per tracklet, via ``video.body_embeddings``.
    - The K embeddings are aggregated into a single representative embedding per tracklet.
    - Similarly, two tracklets are considered the same person if their representative embeddings
    are within the ``body_distance_threshold`` in cosine distance.
    - However, body embeddings are never allowed to merge two distinct face identities,
    assuming that face embeddings are more reliable than body embeddings.
        - If a body cluster anchors with one face cluster,
        the body tracklets are merged into that face identity.
        - If no face identity anchors with a body cluster,
        the body tracklets are assigned a new face identity.
        - If a body cluster anchors with multiple face clusters,
        the body evidence is ambiguous and ignored.

Outputs
-------
A dict mapping ``track_id -> person_id`` (1-indexed).  Tracklets that carry
neither a face nor a body embedding are each assigned a unique person ID of
their own so they are never spuriously merged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# A local BoxMOT id for single-video cases, or a video-qualified id for
# cross-video cases.  The video component prevents equal numeric track ids from
# different recordings from being treated as the same tracklet.
TrackletId = int | tuple[str, int]


@dataclass
class ClusteringResult:
    """Container for the outputs of :func:`cluster_tracklets`.

    For a single-video case, ``track_id_to_person_id`` uses integer BoxMOT
    track ids.  For a multi-video case, its keys are ``(video_id, track_id)``
    pairs so equal numeric ids from different recordings remain distinct.

    Attributes
    ----------
    tracklet_id_to_person_id:
        Maps each tracklet to a 1-indexed ``person_id``.  Every tracklet that
        appeared in an input is present.
    person_id_to_tracklet_ids:
        Inverse mapping: ``person_id`` -> ordered list of tracklet ids merged
        into that identity.
    tracklet_face_embedding:
        Representative (mean, L2-normalised) face embedding for each tracklet,
        or ``None`` when no face was ever detected for that tracklet.
    tracklet_body_embedding:
        Representative mean body embedding for each tracklet, or ``None``
        when no valid body embeddings were ever detected for that tracklet.
    """

    tracklet_id_to_person_id: dict[TrackletId, int] = field(default_factory=dict)
    person_id_to_tracklet_ids: dict[int, list[TrackletId]] = field(default_factory=dict)
    tracklet_face_embedding: dict[TrackletId, np.ndarray | None] = field(
        default_factory=dict
    )
    tracklet_body_embedding: dict[TrackletId, np.ndarray | None] = field(
        default_factory=dict
    )


@dataclass
class TrackletClusteringInput:
    """Convenience container that bundles inputs for one video.

    Attributes
    ----------
    track_ids:
        Ordered set of all BoxMOT track IDs that appear in the video.
    face_embeddings:
        Top-K face embeddings per tracklet, as returned by
        :attr:`~body_eye_sync.experiment.video.Video.face_embeddings`.
    body_embeddings:
        Top-K body embeddings per tracklet, as returned by
        :attr:`~body_eye_sync.experiment.video.Video.body_embeddings`.
    video_id:
        Stable id of the video.  It is optional for the single-video
        helper and required when clustering more than one video together.
    """

    track_ids: Sequence[int]
    face_embeddings: pd.DataFrame | None = None
    body_embeddings: pd.DataFrame | None = None
    video_id: str = ""


def _aggregate_embeddings(
    embeddings: pd.DataFrame | None,
    video_id: str | None = None,
) -> dict[TrackletId, np.ndarray]:
    """Return one mean L2-normalised embedding per tracklet.

    For each tracklet, the top-K embeddings are aggregated into a single
    representative embedding by taking the mean of the L2-normalised
    embeddings.  When ``video_id`` is supplied, returned keys are qualified by
    that id; otherwise the legacy integer keys are retained.

    ``video_id`` optionally scopes the input to a single video.
        - When the embedding table contains a ``video_id`` column and
        no specific video is requested, keys are ``(video_id, track_id)`` pairs.

        - When video_id is supplied, the returned keys are ``(video_id, track_id)`` pairs

        - When no video_id is supplied, the returned keys are just ``track_id`` integers.

    Tracklets that never yielded a valid embedding are absent from the returned
    dict.

    Parameters
    ----------
    embeddings:
        DataFrame of embeddings per tracklet.  The embedding column may be
        named ``embedding`` (the storage format) or ``face_embedding`` /
        ``body_embedding``.
    video_id:
        Optional video scope.  If the frame already has a ``video_id`` column,
        only rows for this scope are consumed.

    Returns
    -------
    dict[TrackletId, np.ndarray]
        Tracklet id -> L2-normalised representative embedding.
    """
    if embeddings is None:
        return {}

    embedding_column = next(
        (
            column
            for column in ("embedding", "face_embedding", "body_embedding")
            if column in embeddings.columns
        ),  # get the correct embedding column name from the embeddings DataFrame
        None,
    )
    if embedding_column is None:
        raise ValueError(
            "embeddings table must contain an 'embedding', 'face_embedding', "
            "or 'body_embedding' column"
        )

    table = embeddings

    if video_id is not None and "video_id" in table.columns:
        # only consider row for the requested video
        table = table[table["video_id"].astype(str) == str(video_id)]

    has_video_id_column = "video_id" in table.columns

    if has_video_id_column:
        dropped_nan_cols = [embedding_column, "video_id"]
    else:
        dropped_nan_cols = [embedding_column]

    # drop rows with no embedding
    table = table.dropna(subset=dropped_nan_cols)
    if table.empty:
        return {}

    if video_id is None and has_video_id_column:
        group_columns = ["video_id", "track_id"]
    else:
        group_columns = "track_id"

    result: dict[TrackletId, np.ndarray] = {}

    groups = table.groupby(group_columns, sort=False)

    for group_key, group in groups:
        emb_array = np.asarray(group[embedding_column].to_list(), dtype=np.float64)
        if emb_array.size == 0:
            continue

        mean_emb = emb_array.mean(axis=0)

        norm_emb = np.linalg.norm(mean_emb)

        # define the key for the result dict
        if isinstance(group_key, tuple):
            group_video_id, track_id = group_key
            key: TrackletId = (str(group_video_id), int(track_id))
        else:
            track_id = group_key
            key: TrackletId = (
                (str(video_id), int(track_id))
                if video_id is not None
                else int(track_id)
            )

        result[key] = mean_emb / norm_emb if norm_emb > 0 else np.zeros_like(mean_emb)

    return result


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance matrix for a ``(n, d)`` embedding matrix.

    Because the embeddings are L2-normalised, ``cosine distance = 1 - dot``.
    NaNs (from failed aggregations) are replaced with the maximum observed
    finite distance so that NaN rows are treated as maximally dissimilar and
    receive their own singleton clusters.
    """
    dot_sim = embeddings @ embeddings.T  # (n, n) cosine similarities
    dist = 1.0 - dot_sim
    nans = np.isnan(dist)
    np.fill_diagonal(dist, 0.0)  # self-distance is zero
    remaining_nans = np.isnan(dist)
    if nans.any():
        max_finite = (
            float(dist[~nans].max()) if (~nans).any() else 1.0
        )  # 1.0 here represents undefined/missing
        dist[remaining_nans] = max_finite
    return np.clip(dist, 0.0, 2.0)


def _cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """Agglomerative clustering on cosine distance.

    Parameters
    ----------
    embeddings:
        ``(n, d)`` L2-normalised feature matrix (one row per tracklet).
    distance_threshold:
        Cosine-distance cutoff above which two tracklets are considered
        different people, or
        Average linkage distance threshold between two clusters.

    Returns
    -------
    np.ndarray
        Integer cluster labels of shape ``(n,)``, values in ``[0, n_clusters)``.
    """
    dist = _cosine_distance_matrix(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return clustering.fit_predict(dist)


def _build_identity_mappings(
    tracklet_ids: Sequence[TrackletId],
    cluster_labels: np.ndarray,
) -> dict[int, set[TrackletId]]:
    """Convert cluster labels to a ``person_id -> {tracklet_ids}`` mapping.

    Parameters
    ----------
    tracklet_ids:
        Ordered list of tracklet ids corresponding to the rows of
        *cluster_labels*.
    cluster_labels:
        Integer cluster label for each tracklet.

    Returns
    -------
    dict[int, set[TrackletId]]
        ``person_id`` (1-indexed, derived from sorted unique labels) ->
        set of tracklet ids in that identity cluster.
    """
    clusters: dict[int, set[TrackletId]] = defaultdict(set)
    for tracklet_id, label in zip(tracklet_ids, cluster_labels):
        clusters[int(label)].add(tracklet_id)
    # convert to 1-indexed person IDs with deterministic ordering
    # here we don't use cluters.keys as the keys can be arbitrary integers
    return {pid + 1: tlids for pid, tlids in enumerate(sorted(clusters.values()))}


def _jaccard_similarity(set_a: set[int], set_b: set[int]) -> float:
    """Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _merge_identity_mappings_same_priority(
    target: dict[int, set[TrackletId]],
    source: dict[int, set[TrackletId]],
    merge_jaccard_threshold: float = 0.0,
) -> None:
    """Merge *source* into *target*, deduplicating overlapping identity groups.

    With *target* and *source* having the same priority, when two identity clusters
    share at least one tracklet they are most likely to belong to the same cluster.

    (case1) If one identity anchors with *source* tracklets:
    update the *target* cluster with the new tracklets from *source*.

    (case2) If no identity anchors with *source* tracklets:
    the tracklets are added as a new cluster under in *target* with a fresh ID.

    (case3) multiple identities anchor with *source* tracklets:

        - the overlapping clusters are merged into the canonical cluster in *target*,
        and the rest are removed.

        - the canonical cluster is chosen as:
            1. The cluster in *target* with the highest jaccard similarity to the *source* cluster.
            2. If there is a tie, the cluster with the lowest ID is chosen

        - for other overlapping clusters, if the jaccard similarity between two clusters
        is below the threshold, they are not merged. By default, any overlap
        (jaccard similarity > 0) is sufficient to trigger a merge.

    Parameters
    ----------
    target:
        Existing identity mapping; mutated in place.
        person_id -> set of tracklet IDs.
    source:
        New identity mapping whose clusters might be merged into *target*.
    merge_jaccard_threshold:
        Jaccard similarity threshold for merging overlapping clusters.  When two
        clusters share tracklets, they are considered the same cluster and merged
        if their jaccard similarity is above this threshold.
        Default is ``0.0``, i.e. any overlap is sufficient to merge.
    """
    if not source:
        return

    # build tracklet -> person_id index for the current target state
    tracklet_to_pid: dict[TrackletId, int] = {}
    for pid, tlids in target.items():
        for tlid in tlids:
            tracklet_to_pid[tlid] = pid

    # find the next available person ID
    next_pid = max(target, default=0) + 1

    for source_tlids in source.values():
        overlapping_pids = {
            tracklet_to_pid[t] for t in source_tlids if t in tracklet_to_pid
        }

        if not overlapping_pids:
            # no overlap - add as a new cluster under a fresh person ID
            target[next_pid] = set(source_tlids)
            for tlid in source_tlids:
                tracklet_to_pid[tlid] = next_pid

            next_pid += 1
            continue

        # calculate jaccard similarity for each overlapping cluster in target
        # with the source cluster
        similarities = {
            pid: _jaccard_similarity(target[pid], source_tlids)
            for pid in overlapping_pids
        }

        # pick the target cluster that best matches the source cluster
        canonical_pid = max(
            similarities.keys(),
            key=lambda pid: (similarities[pid], -pid),
        )

        # update the canonical cluster with the new tracklets from source
        target[canonical_pid].update(source_tlids)

        # point all source tracklets to the canonical cluster
        for tlid in source_tlids:
            tracklet_to_pid[tlid] = canonical_pid

        # all person IDs in overlapping_pids are considered the same person,
        # if their jaccard similarity is above the threshold,
        # so we need to merge them into the canonical cluster
        # and remove the duplicates in target
        dup_pids = {
            p
            for p, sim in similarities.items()
            if p != canonical_pid and sim >= merge_jaccard_threshold
        }

        for dup_pid in dup_pids:
            merged_tlids = target.pop(dup_pid, set())
            target[canonical_pid].update(merged_tlids)

            # point all merged tracklets to the canonical cluster
            for tlid in merged_tlids:
                tracklet_to_pid[tlid] = canonical_pid

        # remove source-overlapping tracklets from clusters that weren't merged
        remaining_pids = overlapping_pids - {canonical_pid} - dup_pids

        for remaining_pid in remaining_pids:
            target[remaining_pid].difference_update(source_tlids)


def _merge_identity_mappings_prioritize_target(
    target: dict[int, set[TrackletId]],
    source: dict[int, set[TrackletId]],
) -> None:
    """Merge *source* into *target*, deduplicating overlapping identity groups.

    Tracklets that already have an identity in *target* are NOT allowed to be merged
    with other clusters in *target*, based on overlapping tracklets in *source*.

    (case1) If one identity anchors with *source* tracklets:
    update the *target* cluster with the new tracklets from *source*.

    (case2) If no identity anchors with *source* tracklets:
    the tracklets are added as a new cluster under in *target* with a fresh ID.

    (case3) If multiple identities anchor with *source* tracklets:
    the *source* evidence is ambiguous and ignored.

    Parameters
    ----------
    target:
        Existing identity mapping; mutated in place.
        person_id -> set of tracklet IDs.
    source:
        New identity mapping whose clusters might be merged into *target*.
    """
    if not source:
        return

    # build tracklet -> person_id index for the current target state
    tracklet_to_pid: dict[TrackletId, int] = {}
    for pid, tlids in target.items():
        for tlid in tlids:
            tracklet_to_pid[tlid] = pid

    # find the next available person ID
    next_pid = max(target, default=0) + 1

    for source_tlids in source.values():
        source_only_tlids = source_tlids - set(tracklet_to_pid)

        if not source_only_tlids:
            # all source tracklets already have an identity in target
            # ignore this source cluster
            continue

        # find all target clusters that overlap with the current source cluster
        overlapping_pids = {
            tracklet_to_pid[t] for t in source_tlids if t in tracklet_to_pid
        }

        # case 1
        if len(overlapping_pids) == 1:
            canonical_pid = next(iter(overlapping_pids))
            target[canonical_pid].update(source_only_tlids)

            for tlid in source_only_tlids:
                tracklet_to_pid[tlid] = canonical_pid

            continue

        # case 2
        if not overlapping_pids:
            # no overlap - add as a new cluster under a fresh identity ID
            target[next_pid] = set(source_tlids)

            for tlid in source_tlids:
                tracklet_to_pid[tlid] = next_pid

            next_pid += 1
            continue

        # case 3
        if len(overlapping_pids) > 1:
            # ambiguous source cluster - ignore it
            continue


def _cluster_tracklets(
    tracklet_ids: Sequence[TrackletId],
    face_embeddings: pd.DataFrame | None = None,
    body_embeddings: pd.DataFrame | None = None,
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_body_detections: int = 3,
    debug: bool = False,
    video_id: str | None = None,
) -> ClusteringResult:
    """Group single/cross-video tracklets into person identities.

    Face embeddings are the primary identity signal.
    Body embeddings are used as a fallback to associate tracklets
    without a face embedding with an existing face identity.

    Body evidence is never allowed to merge two distinct face identities,
    given that face embeddings are more reliable than body embeddings.

    In case of clustering across videos, video_id is expected to be present
    in the face_embeddings and body_embeddings DataFrames, and the returned
    mappings will use ``(video_id, track_id)`` pairs as keys.

    Parameters
    ----------
    tracklet_ids:
        A sequence of tracklet IDs to be clustered.
        Each ID is either ``track_id`` of a single video or a ``(video_id, track_id)``
    face_embeddings:
        Per-tracklet face embeddings.  Pass ``None`` to skip face-based
        clustering and use only body embeddings.
        ``video_id`` is in ``face_embeddings`` if the embeddings are from multiple videos.
    body_embeddings:
        Per-tracklet body-pose embeddings.  Used for tracklets that lack a
        face embedding and as a fallback when *face_embeddings* is ``None``.
        ``video_id`` is in ``body_embeddings`` if the embeddings are from multiple videos.
    face_distance_threshold:
        Cosine-distance cutoff for face embeddings (default ``0.6``, consistent
        with common ArcFace verification thresholds).
    body_distance_threshold:
        Cosine-distance cutoff for body embeddings (default ``0.15``; body
        features are noisier so a tighter threshold is appropriate).
    min_face_detections:
        Minimum number of tracklets that must carry a valid face embedding for a
        tracklet to be clustered via face features (default ``1``).
    min_body_detections:
        Minimum number of tracklets that must carry a valid body pose for a
        tracklet to be clustered via body features (default ``3``).
    debug:
        When ``True``, print per-tracklet embedding statistics and the final
        identity assignment table.
    video_id:
        The ID of a specific video to be processed.
        If ``None``, ``video_id`` in embedding dataframes is used, if any.

    Returns
    -------
    ClusteringResult
        See :class:`ClusteringResult` for field descriptions.
    """
    if not tracklet_ids:
        return ClusteringResult(
            tracklet_id_to_person_id={},
            person_id_to_tracklet_ids={},
            tracklet_face_embedding={},
            tracklet_body_embedding={},
        )

    # ------------------------------------------------------------------ faces
    face_emb_map: dict[TrackletId, np.ndarray] = {}
    if face_embeddings is not None:
        face_emb_map = _aggregate_embeddings(face_embeddings, video_id=video_id)

    face_clusters: dict[int, set[TrackletId]] = {}
    face_tracklet_ids = [
        tlid
        for tlid in tracklet_ids
        if tlid in face_emb_map and len(face_emb_map[tlid].flatten()) > 0
    ]
    if len(face_tracklet_ids) > 1 and len(face_tracklet_ids) >= min_face_detections:
        face_mat = np.stack([face_emb_map[tlid] for tlid in face_tracklet_ids])
        face_labels = _cluster_embeddings(face_mat, face_distance_threshold)
        face_clusters = _build_identity_mappings(face_tracklet_ids, face_labels)

    # ---------------------------------------------------------------- bodies
    body_emb_map: dict[TrackletId, np.ndarray] = {}
    if body_embeddings is not None:
        body_emb_map = _aggregate_embeddings(body_embeddings, video_id=video_id)

    # cluster ALL body-capable tracklets, including those that already have
    # face identities. Those face tracklets act as anchors that allow us to
    # associate face-less tracklets with the correct face identity.
    body_tracklet_ids = [
        tlid
        for tlid in tracklet_ids
        if tlid in body_emb_map and len(body_emb_map[tlid].flatten()) > 0
    ]
    body_clusters: dict[int, set[TrackletId]] = {}
    if len(body_tracklet_ids) > 1 and len(body_tracklet_ids) >= min_body_detections:
        body_mat = np.stack([body_emb_map[tlid] for tlid in body_tracklet_ids])
        body_labels = _cluster_embeddings(body_mat, body_distance_threshold)
        body_clusters = _build_identity_mappings(body_tracklet_ids, body_labels)

    # --------------------------------------------------------------- assemble

    # face identities are authoritative. Start with them unchanged
    identity_map: dict[int, set[TrackletId]] = {
        pid: set(tlids) for pid, tlids in face_clusters.items()
    }

    # body clusters are used only to associate otherwise-unidentified
    # tracklets with existing face identities
    # merge body clusters into the face identity map with
    # face identities prioritized over body clusters
    _merge_identity_mappings_prioritize_target(identity_map, body_clusters)

    # assign deterministic 1-indexed person IDs
    sorted_pids = sorted(identity_map.keys())
    pid_to_tlids: dict[int, list[TrackletId]] = {}
    for pid in sorted_pids:
        pid_to_tlids[pid] = sorted(identity_map[pid])

    # final flat mapping of tracklet_id -> person_id
    tracklet_id_to_person_id: dict[TrackletId, int] = {}
    for pid, tids in pid_to_tlids.items():
        for tid in tids:
            tracklet_id_to_person_id[tid] = pid

    # tracklets with no embedding at all -> unique person ID each
    unassigned = [
        tlid
        for tlid in tracklet_ids
        if tlid not in tracklet_id_to_person_id
        and (video_id is None or not isinstance(tlid, tuple) or tlid[0] == video_id)
    ]
    next_pid = max(pid_to_tlids.keys(), default=0) + 1

    for _, tlid in enumerate(sorted(unassigned)):
        tracklet_id_to_person_id[tlid] = next_pid
        pid_to_tlids[next_pid] = [tlid]
        next_pid += 1

    if debug:
        print(
            f"Face clusters: {len(face_clusters)} (tracklets: {sum(len(v) for v in face_clusters.values())})"
        )
        print(
            f"Body clusters: {len(body_clusters)} (tracklets: {sum(len(v) for v in body_clusters.values())})"
        )
        print(f"Final identities: {len(pid_to_tlids)}")
        print(f"{'Person ID':<10} {'Tracklet IDs'}")
        print("-" * 40)
        for pid in sorted(pid_to_tlids):
            print(f"{pid:<10} {pid_to_tlids[pid]}")

    return ClusteringResult(
        tracklet_id_to_person_id=tracklet_id_to_person_id,
        person_id_to_tracklet_ids=dict(pid_to_tlids),
        tracklet_face_embedding=dict(face_emb_map),
        tracklet_body_embedding=dict(body_emb_map),
    )


def cluster_tracklets_from_input(
    input_data: Sequence[TrackletClusteringInput],
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_body_detections: int = 3,
    debug: bool = False,
):
    """Cluster tracklets from one or more videos into persistent person IDs.

    This function accepts a sequence of :class:`TrackletClusteringInput` instances,
    one per video.  It merges the results into a single identity mapping across all
    videos.

    Parameters
    ----------
    input_data:
        Sequence of :class:`TrackletClusteringInput` instances, one per video.
    face_distance_threshold:
        Cosine-distance cutoff for face embeddings (default ``0.6``, consistent
        with common ArcFace verification thresholds).
    body_distance_threshold:
        Cosine-distance cutoff for body embeddings (default ``0.15``, consistent
        with common ArcFace verification thresholds).
    min_face_detections:
        Minimum number of face detections required for a tracklet to be considered
        for face-based clustering (default ``1``).
    min_body_detections:
        Minimum number of body detections required for a tracklet to be considered
        for body-based clustering (default ``3``).
    debug:
        If ``True``, print debug information (default ``False``).
    """
    # prepare merged inputs for clustering

    if not input_data:
        return ClusteringResult(
            tracklet_id_to_person_id={},
            person_id_to_tracklet_ids={},
            tracklet_face_embedding={},
            tracklet_body_embedding={},
        )

    if len(input_data) == 1:
        # single video case
        single_input = input_data[0]
        return _cluster_tracklets(
            video_id=single_input.video_id if single_input.video_id else None,
            tracklet_ids=(
                [(single_input.video_id, int(tid)) for tid in single_input.track_ids]
                if single_input.video_id
                else single_input.track_ids
            ),
            face_embeddings=single_input.face_embeddings,
            body_embeddings=single_input.body_embeddings,
            face_distance_threshold=face_distance_threshold,
            body_distance_threshold=body_distance_threshold,
            min_face_detections=min_face_detections,
            min_body_detections=min_body_detections,
            debug=debug,
        )

    # multi-video case
    all_tracklet_ids: set[TrackletId] = set()
    all_face_embeddings: pd.DataFrame = pd.DataFrame()
    all_body_embeddings: pd.DataFrame = pd.DataFrame()

    for video_input in input_data:
        video_id = video_input.video_id
        if video_id is None:
            raise ValueError("video_id must be provided for multi-video clustering")

        all_tracklet_ids.add((str(video_id), int(tid)) for tid in video_input.track_ids)

        if video_input.face_embeddings is not None:
            face_df = video_input.face_embeddings.copy()
            face_df["video_id"] = str(video_id)
            all_face_embeddings = pd.concat(
                [all_face_embeddings, face_df], ignore_index=True
            )

        if video_input.body_embeddings is not None:
            body_df = video_input.body_embeddings.copy()
            body_df["video_id"] = str(video_id)
            all_body_embeddings = pd.concat(
                [all_body_embeddings, body_df], ignore_index=True
            )

    return _cluster_tracklets(
        video_id=None,
        tracklet_ids=list(all_tracklet_ids),
        face_embeddings=all_face_embeddings if not all_face_embeddings.empty else None,
        body_embeddings=all_body_embeddings if not all_body_embeddings.empty else None,
        face_distance_threshold=face_distance_threshold,
        body_distance_threshold=body_distance_threshold,
        min_face_detections=min_face_detections,
        min_body_detections=min_body_detections,
        debug=debug,
    )
