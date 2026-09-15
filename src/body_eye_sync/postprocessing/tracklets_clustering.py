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
A :class:`ClusteringResult` maps ``(video_id, track_id)`` to 1-indexed person IDs
and nullable glasses wearer IDs. Tracklets that carry
neither a face nor a body embedding are each assigned a unique person ID of
their own so they are never spuriously merged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

if TYPE_CHECKING:
    from body_eye_sync.experiment.video import GlassesVideo

# The video component keeps local track IDs distinct across recordings.
TrackletId = tuple[str, int]


@dataclass
class ClusteringResult:
    """Container for the outputs of :func:`cluster_tracklets`.

    The public entry point uses ``(video_id, track_id)`` pairs as tracklet keys,
    so equal numeric IDs from different glasses recordings remain distinct.

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
    person_face_frame_counts:
        Face visibility matrix populated by :func:`identify_glasses_wearers`,
        with person IDs as rows and glasses video IDs as columns.
    person_id_to_glasses_video_id:
        Wearer associations populated by :func:`identify_glasses_wearers`,
        or ``None`` for people without sufficient absence/visibility evidence.
    """

    tracklet_id_to_person_id: dict[TrackletId, int] = field(default_factory=dict)
    person_id_to_tracklet_ids: dict[int, list[TrackletId]] = field(default_factory=dict)
    tracklet_face_embedding: dict[TrackletId, np.ndarray | None] = field(
        default_factory=dict
    )
    tracklet_body_embedding: dict[TrackletId, np.ndarray | None] = field(
        default_factory=dict
    )
    person_face_frame_counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    person_id_to_glasses_video_id: dict[int, str | None] = field(default_factory=dict)


def identify_glasses_wearers(
    clustering: ClusteringResult,
    face_frames_by_video: Mapping[str, pd.DataFrame],
    min_face_frames: int = 30,
) -> None:
    """Associate clustered people with their glasses IDs.

    For each person, count how many frames there are in each glasses video
    where their face is detected. If a person has zero faces in one video,
    and at least min_face_frames in all other glasses videos, assign them
    to that video.
    """
    if min_face_frames < 1:
        raise ValueError("min_face_frames must be at least 1")

    person_ids = sorted(set(clustering.tracklet_id_to_person_id.values()))
    video_ids = sorted(face_frames_by_video)
    counts = pd.DataFrame(
        0,
        index=pd.Index(person_ids, name="person_id"),
        columns=pd.Index(video_ids, name="video_id"),
        dtype=np.int64,
    )
    for video_id, frames in face_frames_by_video.items():
        recognized = {}
        for tracklet_id, person_id in clustering.tracklet_id_to_person_id.items():
            embedding = clustering.tracklet_face_embedding.get(tracklet_id)
            if (
                tracklet_id[0] == video_id
                and embedding is not None
                and np.isfinite(embedding).all()
                and np.linalg.norm(embedding) > 0
            ):
                recognized[tracklet_id[1]] = person_id
        faces = frames.loc[frames["face_score"].notna(), ["frame", "track_id"]].copy()
        faces["person_id"] = faces["track_id"].map(recognized)
        observed = (
            faces.dropna(subset=["person_id"]).groupby("person_id")["frame"].nunique()
        )
        counts.loc[observed.index, video_id] = observed.to_numpy()

    associations: dict[int, str | None] = {person_id: None for person_id in person_ids}
    if len(video_ids) >= 2:
        for person_id, row in counts.iterrows():
            absent = row.index[row == 0]
            if len(absent) == 1 and (row.drop(absent) >= min_face_frames).all():
                associations[person_id] = absent[0]
    clustering.person_face_frame_counts = counts
    clustering.person_id_to_glasses_video_id = associations


def _aggregate_embeddings(
    embeddings: pd.DataFrame | None,
) -> dict[TrackletId, np.ndarray]:
    """Return one mean L2-normalised embedding per video-qualified tracklet.

    The embedding column can be named ``embedding``, ``face_embedding``, or
    ``body_embedding``. Rows without an embedding are skipped.
    """
    if embeddings is None or embeddings.empty:
        return {}

    embedding_column = next(
        (
            column
            for column in ("embedding", "face_embedding", "body_embedding")
            if column in embeddings.columns
        ),
        None,
    )
    if embedding_column is None:
        raise ValueError(
            "embeddings table must contain an 'embedding', 'face_embedding', "
            "or 'body_embedding' column"
        )

    table = embeddings.dropna(subset=[embedding_column, "video_id"])
    result: dict[TrackletId, np.ndarray] = {}
    for (video_id, track_id), group in table.groupby(
        ["video_id", "track_id"], sort=False
    ):
        emb_array = np.asarray(group[embedding_column].to_list(), dtype=np.float64)
        mean_emb = emb_array.mean(axis=0)
        norm_emb = np.linalg.norm(mean_emb)
        result[(str(video_id), int(track_id))] = (
            mean_emb / norm_emb if norm_emb > 0 else np.zeros_like(mean_emb)
        )
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
) -> ClusteringResult:
    """Group single/cross-video tracklets into person identities.

    Face embeddings are the primary identity signal.
    Body embeddings are used as a fallback to associate tracklets
    without a face embedding with an existing face identity.

    Body evidence is never allowed to merge two distinct face identities,
    given that face embeddings are more reliable than body embeddings.

    Embedding tables and tracklet keys always include the source video ID.

    Parameters
    ----------
    tracklet_ids:
        A sequence of tracklet IDs to be clustered.
        Each ID is a ``(video_id, track_id)`` pair.
    face_embeddings:
        Per-tracklet face embeddings.  Pass ``None`` to skip face-based
        clustering and use only body embeddings.
        The table includes ``video_id`` for every row.
    body_embeddings:
        Per-tracklet body-pose embeddings.  Used for tracklets that lack a
        face embedding and as a fallback when *face_embeddings* is ``None``.
        The table includes ``video_id`` for every row.
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
        face_emb_map = _aggregate_embeddings(face_embeddings)

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
        body_emb_map = _aggregate_embeddings(body_embeddings)

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
    unassigned = [tlid for tlid in tracklet_ids if tlid not in tracklet_id_to_person_id]
    next_pid = max(pid_to_tlids.keys(), default=0) + 1

    for tlid in sorted(unassigned):
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


def cluster_tracklets(
    glasses_videos: Sequence[GlassesVideo],
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_body_detections: int = 3,
    debug: bool = False,
    *,
    min_face_frames: int = 30,
) -> ClusteringResult:
    """Cluster glasses videos and associate people with their glasses IDs.

    Read tracks, embeddings, and face observations directly from each video.
    Tracklet keys are always ``(video_id, track_id)``, including single-video
    results. Fixed videos are not used. Every glasses video must have completed
    tracking and face detection, with recognition embeddings retained for faces.
    The result includes the face-frame matrix and nullable wearer associations.
    """
    video_ids = [video.id for video in glasses_videos]
    if any(not video_id for video_id in video_ids) or len(set(video_ids)) != len(
        video_ids
    ):
        raise ValueError("glasses videos must have nonempty, unique IDs")
    for video in glasses_videos:
        if video.data is None or "face_score" not in video.data.columns:
            raise ValueError(f"run tracking and face detection for {video.id!r} first")
        if video.data["face_score"].notna().any() and video.face_embeddings is None:
            raise ValueError(
                f"collect face recognition embeddings for {video.id!r} first"
            )

    tracklet_ids = sorted(
        (video.id, int(track_id))
        for video in glasses_videos
        for track_id in video.data["track_id"].unique()
    )
    face_tables = [
        video.face_embeddings.assign(video_id=video.id)
        for video in glasses_videos
        if video.face_embeddings is not None
    ]
    body_tables = [
        video.body_embeddings.assign(video_id=video.id)
        for video in glasses_videos
        if video.body_embeddings is not None
    ]
    clustering = _cluster_tracklets(
        tracklet_ids=tracklet_ids,
        face_embeddings=pd.concat(face_tables, ignore_index=True)
        if face_tables
        else None,
        body_embeddings=pd.concat(body_tables, ignore_index=True)
        if body_tables
        else None,
        face_distance_threshold=face_distance_threshold,
        body_distance_threshold=body_distance_threshold,
        min_face_detections=min_face_detections,
        min_body_detections=min_body_detections,
        debug=debug,
    )
    identify_glasses_wearers(
        clustering,
        {video.id: video.data for video in glasses_videos},
        min_face_frames=min_face_frames,
    )
    return clustering
