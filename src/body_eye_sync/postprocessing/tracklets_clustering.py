"""Cluster tracklets of the same person using face and body embeddings.

Clustering strategy
--------------------------
1. **Face embeddings first** – after running an experiment with face detection enabled,
    each Video instance carries top K face embeddings per tracklet, via
    video.face_embeddings. The K embeddings are first aggregated into a single
    representative embedding per tracklet (mean of L2-normalised embeddings).
    Two tracklets are considered the same person if their representative embeddings
    are within the *face_distance_threshold* in cosine distance.


2. **Body embeddings as fallback** – tracklets that yield no valid face
   embedding (e.g. the person's face was never visible) or can't be matched to a face
   embedding are clustered using body embeddings. The top K body embeddings are also
   carried per tracklet, via video.body_embeddings. The K embeddings are aggregated into
   a single representative embedding per tracklet. Similarly, two tracklets are considered
   the same person if their representative embeddings are within the *body_distance_threshold*
   in cosine distance.

Outputs
-------
A dict mapping ``track_id -> person_id`` (1-indexed).  Tracklets that carry
neither a face nor a body embedding are each assigned a unique person ID of
their own so they are never spuriously merged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


@dataclass
class ClusteringResult:
    """Container for the outputs of :func:`cluster_tracklets`.

    Attributes
    ----------
    track_id_to_person_id:
        Maps each BoxMOT ``track_id`` to a 1-indexed ``person_id``.  Every
        tracklet that appeared in *tracks* is present.
    person_id_to_track_ids:
        Inverse mapping: ``person_id`` → ordered list of ``track_id`` s that
        were merged into that identity.
    tracklet_face_embedding:
        Representative (mean, L2-normalised) face embedding for each tracklet,
        or ``None`` when no face was ever detected for that tracklet.
    tracklet_body_embedding:
        Representative mean body embedding for each tracklet, or ``None``
        when no valid body embeddings were ever detected for that tracklet.
    """

    track_id_to_person_id: dict[int, int] = field(default_factory=dict)
    person_id_to_track_ids: dict[int, list[int]] = field(default_factory=dict)
    tracklet_face_embedding: dict[int, np.ndarray | None] = field(default_factory=dict)
    tracklet_body_embedding: dict[int, np.ndarray | None] = field(default_factory=dict)


@dataclass
class TrackletClusteringInput:
    """Convenience container that bundles all inputs needed for clustering.

    Attributes
    ----------
    track_ids:
        Ordered set of all BoxMOT track IDs that appear in a video.
    face_embeddings:
        TopK face embeddings per tracklet,
        as returned by :func:`~body_eye_sync.experiment.video.Video.face_embeddings`.
        columns: ``track_id, frame_idx, face_score, face_embedding``.
    body_embeddings:
        TopK body embeddings per tracklet,
        as returned by :func:`~body_eye_sync.experiment.video.Video.body_embeddings`.
        columns: ``track_id, frame_idx, body_score, body_embedding``.

    """

    track_ids: Sequence[int]
    face_embeddings: pd.DataFrame | None = None
    body_embeddings: pd.DataFrame | None = None


def _aggregate_embeddings(
    embeddings: pd.DataFrame | None,
) -> dict[int, np.ndarray]:
    """Mean L2-normalised embedding per tracklet across all frames.

    For each tracklet, the top-K embeddings are aggregated into a single representative embedding
    by taking the mean of the L2-normalised embeddings.

    Tracklets that never yielded a valid face embedding are absent from the returned dict.

    Parameters
    ----------
    embeddings:
        DataFrame of embeddings per tracklet.

    Returns
    -------
    dict[(int, int), np.ndarray]
        ``track_id`` → L2-normalised representative embedding, or absent when
        no target was ever detected for that tracklet.
    """
    # group embeddings by track_id
    group_embs = (
        embeddings.dropna(subset=["embedding"])
        .groupby("track_id")["embedding"]
        .apply(lambda row: [np.asarray(emb, dtype=np.float64) for emb in row])
    )

    # compute mean L2-normalised embedding per tracklet
    result: dict[int, np.ndarray] = {}
    for track_id, emb_list in group_embs.items():
        if not emb_list:
            continue
        mean_emb = np.mean(emb_list, axis=0)
        norm = np.linalg.norm(mean_emb)
        result[track_id] = mean_emb / norm if norm > 0 else mean_emb

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
    np.fill_diagonal(dist, 0.0)  # self-distance is zero
    nans = np.isnan(dist)
    if nans.any():
        max_finite = float(dist[~nans].max()) if (~nans).any() else 1.0
        dist[nans] = max_finite
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
    tracklet_ids: Sequence[int],
    cluster_labels: np.ndarray,
) -> dict[int, set[int]]:
    """Convert cluster labels to a ``person_id -> {tracklet_ids}`` mapping.

    Parameters
    ----------
    tracklet_ids:
        Ordered list of track IDs corresponding to the rows of
        *cluster_labels*.
    cluster_labels:
        Integer cluster label for each tracklet.

    Returns
    -------
    dict[int, set[int]]
        ``person_id`` (1-indexed, derived from sorted unique labels) →
        set of track IDs in that identity cluster.
    """
    clusters: dict[int, set[int]] = defaultdict(set)
    for track_id, label in zip(tracklet_ids, cluster_labels):
        clusters[int(label)].add(track_id)
    # convert to 1-indexed person IDs with deterministic ordering
    # here we don't use cluters.keys as the keys can be arbitrary integers
    return {
        pid + 1: track_ids for pid, track_ids in enumerate(sorted(clusters.values()))
    }


def _jaccard_similarity(set_a: set[int], set_b: set[int]) -> float:
    """Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _merge_identity_mappings(
    target: dict[int, set[int]],
    source: dict[int, set[int]],
) -> None:
    """Merge *source* into *target*, deduplicating overlapping identity groups.

    When two identity clusters share at least one tracklet they are most likely
    to belong to the same person.

    If tracklets from *source* overlap with multiple clusters in *target*,
    the overlapping clusters are merged into the canonicalcluster in *target*,
    and the rest are removed.

    The canonical cluster is chosen as:
    1. The cluster in *target* with the highest jaccard similarity to the *source* cluster.
    2. If there is a tie, the cluster with the lowest person ID is chosen

    Parameters
    ----------
    target:
        Existing identity mapping; mutated in place.
        person_id → set of tracklet IDs.
    source:
        New identity mapping whose clusters are merged into *target*.
    """
    if not source:
        return

    # build tracklet -> person_id index for the current target state
    tracklet_to_pid: dict[int, int] = {}
    for pid, tids in target.items():
        for tid in tids:
            tracklet_to_pid[tid] = pid

    for pid, tids in source.items():
        overlapping_pids = {tracklet_to_pid[t] for t in tids if t in tracklet_to_pid}
        if not overlapping_pids:
            # no overlap — add as a new cluster under a fresh person ID
            next_pid = max(target.keys(), default=0) + 1
            target[next_pid] = set(tids)
            for tid in tids:
                tracklet_to_pid[tid] = next_pid
        else:
            # merge into the canonical cluster in target
            canonical_pid = max(
                overlapping_pids,
                key=lambda p: (_jaccard_similarity(target[p], tids), -p),
            )
            target[canonical_pid].update(tids)

            # update the tracklet -> person_id index for the merged tracklets
            # so the next iteration sees the updated mapping
            for tid in tids:
                tracklet_to_pid[tid] = canonical_pid

            # all person IDs in overlapping_pids are considered the same person,
            # merge them into the canonical cluster and remove the duplicates in target
            # TODO: maybe put a threshold on the jaccard_similarity
            # to avoid merging clusters that only share a single tracklet?
            for dup_pid in overlapping_pids - {canonical_pid}:
                merged_tids = target.pop(dup_pid, set())
                target[canonical_pid].update(merged_tids)
                for tid in merged_tids:
                    tracklet_to_pid[tid] = canonical_pid


def cluster_tracklets(
    track_ids: Sequence[int],
    face_embeddings: pd.DataFrame | None = None,
    body_embeddings: pd.DataFrame | None = None,
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_body_detections: int = 3,
    debug: bool = False,
) -> ClusteringResult:
    """Group BoxMOT tracklets into person identities.

    Face embeddings are used as the primary re-identification signal.  Tracklets
    that carry no valid face embedding fall back to body-pose-derived features.

    Parameters
    ----------
    track_ids:
        A sequence of track IDs to be clustered.
    face_embeddings:
        Per-frame face embeddings.  Pass ``None`` to skip face-based
        clustering and use only body embeddings.
    body_embeddings:
        Per-frame body-pose embeddings.  Used for tracklets that lack a
        face embedding and as a fallback when *face_embeddings* is ``None``.
    face_distance_threshold:
        Cosine-distance cutoff for face embeddings (default ``0.6``, consistent
        with common ArcFace verification thresholds).
    body_distance_threshold:
        Cosine-distance cutoff for body embeddings (default ``0.15``; body
        features are noisier so a tighter threshold is appropriate).
    min_face_detections:
        Minimum number of frames that must carry a valid face embedding for a
        tracklet to be clustered via face features (default ``1``).
    min_body_detections:
        Minimum number of frames that must carry a valid body pose for a
        tracklet to be clustered via body features (default ``3``).
    debug:
        When ``True``, print per-tracklet embedding statistics and the final
        identity assignment table.

    Returns
    -------
    ClusteringResult
        See :class:`ClusteringResult` for field descriptions.
    """
    if not track_ids:
        return ClusteringResult(
            track_id_to_person_id={},
            person_id_to_track_ids={},
            tracklet_face_embedding={},
            tracklet_body_embedding={},
        )

    # ------------------------------------------------------------------ faces
    face_emb_map: dict[int, np.ndarray] = {}
    if face_embeddings is not None:
        face_emb_map = _aggregate_embeddings(face_embeddings)

    face_clusters: dict[int, set[int]] = {}
    face_track_ids = [
        tid
        for tid in track_ids
        if tid in face_emb_map and len(face_emb_map[tid].flatten()) > 0
    ]
    if len(face_track_ids) > 1 and len(face_track_ids) >= min_face_detections:
        face_mat = np.stack([face_emb_map[tid] for tid in face_track_ids])
        face_labels = _cluster_embeddings(face_mat, face_distance_threshold)
        face_clusters = _build_identity_mappings(face_track_ids, face_labels)

    # ---------------------------------------------------------------- bodies
    body_emb_map: dict[int, np.ndarray] = {}
    if body_embeddings is not None:
        body_emb_map = _aggregate_embeddings(body_embeddings)

    # tracklets not already assigned by face embeddings.
    body_candidate_tids = [
        tid
        for tid in track_ids
        if tid not in {t for tids in face_clusters.values() for t in tids}
        and tid in body_emb_map
        and len(body_emb_map[tid].flatten()) > 0
    ]
    body_clusters: dict[int, set[int]] = {}
    if len(body_candidate_tids) > 1 and len(body_candidate_tids) >= min_body_detections:
        body_mat = np.stack([body_emb_map[tid] for tid in body_candidate_tids])
        body_labels = _cluster_embeddings(body_mat, body_distance_threshold)
        body_clusters = _build_identity_mappings(body_candidate_tids, body_labels)

    # --------------------------------------------------------------- assemble
    identity_map: dict[int, set[int]] = {}
    _merge_identity_mappings(identity_map, face_clusters)
    _merge_identity_mappings(identity_map, body_clusters)

    # assign deterministic 1-indexed person IDs
    sorted_pids = sorted(identity_map.keys())
    pid_to_tids: dict[int, list[int]] = {}
    for pid in sorted_pids:
        pid_to_tids[pid] = sorted(identity_map[pid])

    # final flat mapping of track_id -> person_id
    track_id_to_person_id: dict[int, int] = {}
    for pid, tids in pid_to_tids.items():
        for tid in tids:
            track_id_to_person_id[tid] = pid

    # tracklets with no embedding at all → unique person ID each
    unassigned = [tid for tid in track_ids if tid not in track_id_to_person_id]
    next_pid = max(pid_to_tids.keys(), default=0)
    for _, tid in enumerate(sorted(unassigned)):
        next_pid += 1
        track_id_to_person_id[tid] = next_pid
        pid_to_tids[next_pid] = [tid]

    if debug:
        print(
            f"Face clusters   : {len(face_clusters)} (tracklets: {sum(len(v) for v in face_clusters.values())})"
        )
        print(
            f"Body  clusters  : {len(body_clusters)} (tracklets: {sum(len(v) for v in body_clusters.values())})"
        )
        print(f"Final identities: {len(pid_to_tids)}")
        print(f"{'Person ID':<10} {'Track IDs'}")
        print("-" * 40)
        for pid in sorted(pid_to_tids):
            print(f"{pid:<10} {pid_to_tids[pid]}")

    return ClusteringResult(
        track_id_to_person_id=track_id_to_person_id,
        person_id_to_track_ids=dict(pid_to_tids),
        tracklet_face_embedding=dict(face_emb_map),
        tracklet_body_embedding=dict(body_emb_map),
    )


def cluster_tracklets_from_input(
    inputs: TrackletClusteringInput,
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_body_detections: int = 3,
    debug: bool = False,
) -> ClusteringResult:
    """Convenience wrapper around :func:`cluster_tracklets`.

    Accepts a :class:`TrackletClusteringInput` that bundles the track ids
    with (optional) face and body embeddings, then delegates to
    :func:`cluster_tracklets`.
    """
    return cluster_tracklets(
        track_ids=inputs.track_ids,
        face_embeddings=inputs.face_embeddings,
        body_embeddings=inputs.body_embeddings,
        face_distance_threshold=face_distance_threshold,
        body_distance_threshold=body_distance_threshold,
        min_face_detections=min_face_detections,
        min_body_detections=min_body_detections,
        debug=debug,
    )
