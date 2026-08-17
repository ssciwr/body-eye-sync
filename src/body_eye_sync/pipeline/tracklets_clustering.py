"""Cluster tracklets of the same person using face and body embeddings.

Re-identification strategy
--------------------------
1. **Face embeddings first** – every :class:`~body_eye_sync.pipeline.face_detection.FaceBox`
   produced by InsightFace carries an L2-normalised ArcFace recognition vector
   (``normed_embedding``).  These are the gold-standard identity features.  For
   each tracklet the per-frame embeddings are L2-normalised, then averaged to
   produce a single representative face vector.  Tracklets whose representative
   vectors are within ``face_distance_threshold`` (cosine distance) are grouped
   into the same identity cluster.

2. **Body embeddings as fallback** – tracklets that yield no valid face
   embedding (e.g. the person's face was never visible) are clustered using a
   lightweight body representation derived from COCO keypoints.  For every
   frame that carries a valid :class:`~body_eye_sync.pipeline.body_pose.BodyPose`
   the 17 ``(x, y)`` keypoints are divided by the person-box width and height,
   giving a scale-invariant 34-dimensional feature vector; missing keypoints
   are zeroed out.  Per-tracklet averages are then clustered with the same
   agglomerative/cosine-distance machinery, using the typically lower
   ``body_distance_threshold``.

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

from body_eye_sync.pipeline.body_pose import (
    KEYPOINT_NAMES,
    PoseFrameResult,
)
from body_eye_sync.pipeline.face_detection import FaceFrameResult
from body_eye_sync.pipeline.object_tracking import BoundingBox, tracks_to_dataframe


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


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
        when no valid body pose was ever detected for that tracklet.
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
    tracks:
        DataFrame produced by :func:`~body_eye_sync.pipeline.object_tracking.tracks_to_dataframe`
        (columns ``frame, track_id, x1, y1, x2, y2, conf``).
    face_frames:
        Per-frame face detection results from
        :func:`~body_eye_sync.pipeline.face_detection.detect_faces`.
        Pass ``None`` when face detection was not run.
    pose_frames:
        Per-frame body-pose detection results from
        :func:`~body_eye_sync.pipeline.body_pose.detect_body_poses`.
        Pass ``None`` when pose detection was not run.
    """

    tracks: any  # pandas.DataFrame — kept generic to avoid hard dependency at type-check time
    face_frames: Sequence[FaceFrameResult] | None = None
    pose_frames: Sequence[PoseFrameResult] | None = None


# ---------------------------------------------------------------------------
# Embedding aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_face_embeddings(
    face_frames: Sequence[FaceFrameResult],
) -> dict[int, np.ndarray]:
    """Mean L2-normalised ArcFace embedding per tracklet across all frames.

    For each tracklet the per-frame embeddings are first individually
    L2-normalised (they arrive from InsightFace already normalised, but the
    extra normalisation guards against numerical drift), then averaged.  The
    result is L2-normalised again to give the representative identity vector.

    Parameters
    ----------
    face_frames:
        Per-frame face detection results.

    Returns
    -------
    dict[int, np.ndarray]
        ``track_id`` → L2-normalised representative embedding, or absent when
        no face was ever detected for that tracklet.
    """
    embeddings: dict[int, list[np.ndarray]] = defaultdict(list)

    for frame_result in face_frames:
        for face in frame_result.faces:
            emb = face.embedding
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float64)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings[face.box.track_id].append(emb)

    result: dict[int, np.ndarray] = {}
    for track_id, emb_list in embeddings.items():
        mean_emb = np.mean(emb_list, axis=0)
        norm = np.linalg.norm(mean_emb)
        result[track_id] = mean_emb / norm if norm > 0 else mean_emb

    return result


def _aggregate_body_embeddings(
    pose_frames: Sequence[PoseFrameResult],
) -> dict[int, np.ndarray]:
    """Mean scale-invariant keypoint feature vector per tracklet.

    For every detected pose the 17 COCO ``(x, y)`` keypoints are divided by
    the person-box width and height respectively, yielding a 34-dimensional
    feature vector that is invariant to the person's absolute position and
    size within the frame.  Keypoints with a confidence below 0.1 are treated
    as missing and zeroed out.  Per-tracklet means are then L2-normalised.

    Parameters
    ----------
    pose_frames:
        Per-frame body-pose detection results.

    Returns
    -------
    dict[int, np.ndarray]
        ``track_id`` → L2-normalised body feature vector, or absent when no
        valid pose was ever detected for that tracklet.
    """
    n_kpts = len(KEYPOINT_NAMES)
    features: dict[int, list[np.ndarray]] = defaultdict(list)

    for frame_result in pose_frames:
        for pose in frame_result.poses:
            box: BoundingBox = pose.box
            w = box.x2 - box.x1
            h = box.y2 - box.y1
            if w <= 0 or h <= 0:
                continue

            kp_arr = np.asarray(pose.keypoints, dtype=np.float64)  # (17, 3)
            if kp_arr.shape[0] != n_kpts:
                continue

            # Mask keypoints with low confidence before normalising.
            valid = kp_arr[:, 2] >= 0.1
            normed = np.zeros((n_kpts, 2), dtype=np.float64)
            normed[valid, 0] = (kp_arr[valid, 0] - box.x1) / w
            normed[valid, 1] = (kp_arr[valid, 1] - box.y1) / h

            feat = normed.ravel()  # (34,)
            feat_norm = np.linalg.norm(feat)
            if feat_norm > 0:
                feat = feat / feat_norm
            features[pose.box.track_id].append(feat)

    result: dict[int, np.ndarray] = {}
    for track_id, feat_list in features.items():
        mean_feat = np.mean(feat_list, axis=0)
        norm = np.linalg.norm(mean_feat)
        result[track_id] = mean_feat / norm if norm > 0 else mean_feat

    return result


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance matrix for a ``(n, d)`` embedding matrix.

    Because the embeddings are L2-normalised, ``cosine distance = 1 - dot``.
    NaNs (from failed aggregations) are replaced with the maximum observed
    finite distance so that NaN rows are treated as maximally dissimilar and
    receive their own singleton clusters.
    """
    dot_sim = embeddings @ embeddings.T  # (n, n) cosine similarities
    dist = 1.0 - dot_sim
    np.fill_diagonal(dist, 0.0)
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
        different people.

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
    # Convert to 1-indexed person IDs with deterministic ordering.
    return {
        pid + 1: track_ids for pid, track_ids in enumerate(sorted(clusters.values()))
    }


def _merge_identity_mappings(
    target: dict[int, set[int]],
    source: dict[int, set[int]],
) -> None:
    """Merge *source* into *target*, deduplicating overlapping identity groups.

    When two identity clusters share at least one tracklet they are the same
    person; their tracklet sets are unified under a single new person ID.

    Parameters
    ----------
    target:
        Existing identity mapping; mutated in place.
    source:
        New identity mapping whose clusters are merged into *target*.
    """
    if not source:
        return

    # Build tracklet -> person_id index for the current target state.
    tracklet_to_pid: dict[int, int] = {}
    for pid, tids in target.items():
        for tid in tids:
            tracklet_to_pid[tid] = pid

    for pid, tids in source.items():
        overlapping_pids = {tracklet_to_pid[t] for t in tids if t in tracklet_to_pid}
        if not overlapping_pids:
            # No overlap — add as a new cluster under a fresh person ID.
            target[pid] = set(tids)
            for tid in tids:
                tracklet_to_pid[tid] = pid
        else:
            # Merge into the first overlapping cluster and remove the rest.
            canonical_pid = min(overlapping_pids)
            target[canonical_pid].update(tids)
            for tid in tids:
                tracklet_to_pid[tid] = canonical_pid
            for dup_pid in overlapping_pids - {canonical_pid}:
                merged_tids = target.pop(dup_pid, set())
                target[canonical_pid].update(merged_tids)
                for tid in merged_tids:
                    tracklet_to_pid[tid] = canonical_pid


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def cluster_tracklets(
    tracks: any,  # pandas.DataFrame — kept generic to avoid hard dep at type-check time
    face_frames: Sequence[FaceFrameResult] | None = None,
    pose_frames: Sequence[PoseFrameResult] | None = None,
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_pose_detections: int = 3,
    debug: bool = False,
) -> ClusteringResult:
    """Group BoxMOT tracklets into person identities.

    Face embeddings are used as the primary re-identification signal.  Tracklets
    that carry no valid face embedding fall back to body-pose-derived features.

    Parameters
    ----------
    tracks:
        DataFrame with columns ``frame, track_id, x1, y1, x2, y2, conf``, as
        returned by :func:`~body_eye_sync.pipeline.object_tracking.tracks_to_dataframe`.
    face_frames:
        Per-frame face detection results.  Pass ``None`` to skip face-based
        clustering and use only body embeddings.
    pose_frames:
        Per-frame body-pose detection results.  Used for tracklets that lack a
        face embedding and as a fallback when *face_frames* is ``None``.
    face_distance_threshold:
        Cosine-distance cutoff for face embeddings (default ``0.6``, consistent
        with common ArcFace verification thresholds).
    body_distance_threshold:
        Cosine-distance cutoff for body embeddings (default ``0.15``; body
        features are noisier so a tighter threshold is appropriate).
    min_face_detections:
        Minimum number of frames that must carry a valid face embedding for a
        tracklet to be clustered via face features (default ``1``).
    min_pose_detections:
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
    df = tracks if hasattr(tracks, "columns") else tracks_to_dataframe(tracks)
    track_ids = sorted(df["track_id"].unique().astype(int).tolist())

    # ------------------------------------------------------------------ faces
    face_emb_map: dict[int, np.ndarray] = {}
    if face_frames is not None:
        face_emb_map = _aggregate_face_embeddings(face_frames)

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
    if pose_frames is not None:
        body_emb_map = _aggregate_body_embeddings(pose_frames)

    # Tracklets not already assigned by face embeddings.
    body_candidate_ids = [
        tid
        for tid in track_ids
        if tid not in face_clusters
        and tid not in {t for tids in face_clusters.values() for t in tids}
        and tid in body_emb_map
        and len(body_emb_map[tid].flatten()) > 0
    ]
    body_clusters: dict[int, set[int]] = {}
    if len(body_candidate_ids) > 1 and len(body_candidate_ids) >= min_pose_detections:
        body_mat = np.stack([body_emb_map[tid] for tid in body_candidate_ids])
        body_labels = _cluster_embeddings(body_mat, body_distance_threshold)
        body_clusters = _build_identity_mappings(body_candidate_ids, body_labels)

    # --------------------------------------------------------------- assemble
    identity_map: dict[int, set[int]] = {}
    _merge_identity_mappings(identity_map, face_clusters)
    _merge_identity_mappings(identity_map, body_clusters)

    # Assign deterministic 1-indexed person IDs.
    sorted_pids = sorted(identity_map.keys())
    pid_to_tids: dict[int, list[int]] = {}
    for pid in sorted_pids:
        pid_to_tids[pid] = sorted(identity_map[pid])

    # Final flat mapping.
    track_id_to_person_id: dict[int, int] = {}
    for pid, tids in pid_to_tids.items():
        for tid in tids:
            track_id_to_person_id[tid] = pid

    # Tracklets with no embedding at all → unique person ID each.
    unassigned = [tid for tid in track_ids if tid not in track_id_to_person_id]
    next_pid = max(pid_to_tids.keys(), default=0)
    for i, tid in enumerate(sorted(unassigned)):
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
    min_pose_detections: int = 3,
    debug: bool = False,
) -> ClusteringResult:
    """Convenience wrapper around :func:`cluster_tracklets`.

    Accepts a :class:`TrackletClusteringInput` that bundles the tracks
    DataFrame with (optional) face and pose frame results, then delegates to
    :func:`cluster_tracklets`.
    """
    return cluster_tracklets(
        tracks=inputs.tracks,
        face_frames=inputs.face_frames,
        pose_frames=inputs.pose_frames,
        face_distance_threshold=face_distance_threshold,
        body_distance_threshold=body_distance_threshold,
        min_face_detections=min_face_detections,
        min_pose_detections=min_pose_detections,
        debug=debug,
    )
