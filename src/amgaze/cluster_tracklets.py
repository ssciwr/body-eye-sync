#!/usr/bin/env python
"""Cluster BoxMoT tracklets into per-person identities.

Pipeline
--------
1. Parse a BoxMoT/MOT ``tracks.txt`` (``frame,id,x,y,w,h,conf,cls,det_idx``).
2. For every tracklet, sample a handful of high-quality detections spread across
   its lifetime and crop them from the *original* (un-annotated) video.
3. Embed each crop with two models:
     * body  -> Torchreid OSNet x1.0 (MSMT17) appearance feature (512-d)
     * face  -> InsightFace buffalo_l ArcFace feature (512-d), when a face is found
4. Average embeddings per tracklet, cluster face-bearing tracklets by face,
   then attach faceless tracklets to the nearest face cluster by body appearance.
   The number of people is discovered automatically (best silhouette over a
   range of k, or a fixed cosine distance threshold).
5. Write the person id per tracklet, an annotated tracks file, and one montage
   image per discovered cluster so the result can be eyeballed.

Example
-------
    uv run python -m amgaze.cluster_tracklets \
        --tracks runs/track/quartet2/tracks.txt \
        --video  Archiv/quartet.mp4 \
        --out-dir runs/track/quartet2/clustering
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# tracks.txt parsing & sampling
# --------------------------------------------------------------------------- #
def parse_tracks(path: Path):
    """Return {track_id: [(frame, x, y, w, h, conf), ...]} sorted by frame."""
    tracks: dict[int, list] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            conf = float(p[6]) if len(p) > 6 else 1.0
            tracks[tid].append((frame, x, y, w, h, conf))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return dict(tracks)


def select_samples(tracks, max_samples, min_conf, min_w, min_h):
    """Pick up to ``max_samples`` good detections per track, evenly in time.

    Returns (per_track_samples, frame_index) where frame_index maps a
    *video frame number* (the MOT frame) to a list of (track_id, bbox).
    """
    per_track: dict[int, list] = {}
    frame_index: dict[int, list] = defaultdict(list)
    for tid, dets in tracks.items():
        good = [d for d in dets if d[5] >= min_conf and d[3] >= min_w and d[4] >= min_h]
        if not good:
            # fall back to the largest boxes regardless of conf so the track is
            # never silently dropped
            good = sorted(dets, key=lambda d: d[3] * d[4], reverse=True)[:max_samples]
        if len(good) > max_samples:
            idx = np.linspace(0, len(good) - 1, max_samples).round().astype(int)
            good = [good[i] for i in sorted(set(idx))]
        per_track[tid] = good
        for frame, x, y, w, h, conf in good:
            frame_index[frame].append((tid, (x, y, w, h)))
    return per_track, frame_index


# --------------------------------------------------------------------------- #
# embedding extraction (single pass over the video)
# --------------------------------------------------------------------------- #
def clamp_box(x, y, w, h, W, H):
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + w)))
    y1 = min(H, int(round(y + h)))
    return x0, y0, x1, y1


def extract_embeddings(args, frame_index):
    """One sequential pass over the video; returns per-track features & thumbs."""
    import torch
    import onnxruntime as ort

    # Make the CUDA execution provider find torch's bundled cuDNN/cuBLAS.
    try:
        ort.preload_dlls()
    except Exception as exc:  # pragma: no cover - older onnxruntime
        print(f"[warn] ort.preload_dlls() failed ({exc}); face model may run on CPU")

    from torchreid.reid.utils import FeatureExtractor
    from insightface.app import FaceAnalysis

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    body = FeatureExtractor(
        model_name=args.reid_model,
        model_path=args.reid_weights,
        device=device,
    )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    face_app = FaceAnalysis(name=args.face_model, providers=providers)
    face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

    body_feats: dict[int, list] = defaultdict(list)
    face_feats: dict[int, list] = defaultdict(list)
    thumbs: dict[int, list] = defaultdict(list)
    face_thumbs: dict[int, list] = defaultdict(list)

    # batching buffers for the body model
    crop_buf: list = []
    owner_buf: list = []

    def flush():
        if not crop_buf:
            return
        feats = body(crop_buf).cpu().numpy()  # (B, 512)
        for tid, f in zip(owner_buf, feats):
            n = np.linalg.norm(f)
            if n > 0:
                body_feats[tid].append(f / n)
        crop_buf.clear()
        owner_buf.clear()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] video {W}x{H}, {n_frames} frames")

    # MOT frames are 1-indexed; map to 0-indexed video positions.
    wanted = sorted(frame_index.keys())
    targets = {f + args.frame_offset: f for f in wanted}
    todo = sorted(targets)
    n_faces_found = 0

    from tqdm import tqdm

    pbar = tqdm(total=len(todo), desc="sampling frames", unit="frame")
    idx = -1
    ptr = 0
    while ptr < len(todo):
        ret = cap.grab()
        idx += 1
        if not ret:
            break
        if idx != todo[ptr]:
            continue
        ret, frame = cap.retrieve()
        ptr += 1
        pbar.update(1)
        if not ret:
            continue
        mot_frame = targets[idx]
        for tid, (x, y, w, h) in frame_index[mot_frame]:
            x0, y0, x1, y1 = clamp_box(x, y, w, h, W, H)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = frame[y0:y1, x0:x1]  # BGR

            # ---- face embedding (InsightFace expects BGR) -------------------
            if face_app is not None:
                faces = face_app.get(crop)
                if faces:
                    best = max(
                        faces,
                        key=lambda fa: (
                            float(fa.det_score)
                            * (fa.bbox[2] - fa.bbox[0])
                            * (fa.bbox[3] - fa.bbox[1])
                        ),
                    )
                    if float(best.det_score) >= args.face_det_thresh:
                        emb = best.normed_embedding.astype(np.float32)
                        face_feats[tid].append(emb)
                        n_faces_found += 1
                        if len(face_thumbs[tid]) < args.montage_per_track:
                            fx0, fy0, fx1, fy1 = best.bbox.astype(int)
                            pad = int(0.3 * (fy1 - fy0))
                            fx0 = max(0, fx0 - pad)
                            fy0 = max(0, fy0 - pad)
                            fx1 = min(crop.shape[1], fx1 + pad)
                            fy1 = min(crop.shape[0], fy1 + pad)
                            fc = crop[fy0:fy1, fx0:fx1]
                            if fc.size:
                                face_thumbs[tid].append(cv2.resize(fc, (96, 96)))

            # ---- body embedding (Torchreid expects RGB) --------------------
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_buf.append(rgb)
            owner_buf.append(tid)
            if len(crop_buf) >= args.batch_size:
                flush()

            # ---- keep a few thumbnails for the montage ---------------------
            if len(thumbs[tid]) < args.montage_per_track:
                tw = 96
                th = max(1, int(round((y1 - y0) * tw / (x1 - x0))))
                thumbs[tid].append(cv2.resize(crop, (tw, min(th, 192))))
    flush()
    pbar.close()
    cap.release()
    print(f"[info] face embeddings collected: {n_faces_found}")
    return body_feats, face_feats, thumbs, face_thumbs


# --------------------------------------------------------------------------- #
# aggregation, fusion & clustering
# --------------------------------------------------------------------------- #
def aggregate(track_ids, body_feats, face_feats):
    """Mean + renormalize per track. Returns arrays aligned with track_ids."""
    body = np.zeros((len(track_ids), 512), np.float32)
    face = np.zeros((len(track_ids), 512), np.float32)
    has_body = np.zeros(len(track_ids), bool)
    has_face = np.zeros(len(track_ids), bool)
    for i, tid in enumerate(track_ids):
        if body_feats.get(tid):
            v = np.mean(body_feats[tid], axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                body[i] = v / n
                has_body[i] = True
        if face_feats.get(tid):
            v = np.mean(face_feats[tid], axis=0)
            n = np.linalg.norm(v)
            if n > 0:
                face[i] = v / n
                has_face[i] = True
    return body, face, has_body, has_face


def layout_conflict_matrix(tracks, track_ids, args):
    """Return left/right consistency penalties for a fixed seating layout.

    conflict[i, j] is high when tracklets i and j disagree about whether they
    sit to the left or right of the same third-party tracklets. overlap[i, j]
    marks tracklets visible in the same frame, which cannot be the same person.
    """
    n = len(track_ids)
    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    frame_dets: dict[int, list] = defaultdict(list)
    for tid in track_ids:
        for frame, x, y, w, h, conf in tracks[tid]:
            frame_dets[frame].append((tid_to_idx[tid], x + 0.5 * w))

    relation_sum = np.zeros((n, n), np.float32)
    relation_count = np.zeros((n, n), np.int32)
    copresent = np.zeros((n, n), bool)

    for dets in frame_dets.values():
        if len(dets) < 2:
            continue
        for a_pos in range(len(dets) - 1):
            ia, xa = dets[a_pos]
            for ib, xb in dets[a_pos + 1 :]:
                copresent[ia, ib] = True
                copresent[ib, ia] = True
                dx = xa - xb
                if abs(dx) < args.layout_min_x_gap:
                    continue
                # +1 means row tracklet is to the right of column tracklet.
                rel = 1.0 if dx > 0 else -1.0
                relation_sum[ia, ib] += rel
                relation_sum[ib, ia] -= rel
                relation_count[ia, ib] += 1
                relation_count[ib, ia] += 1

    relation = np.zeros((n, n), np.float32)
    known = relation_count > 0
    relation[known] = relation_sum[known] / relation_count[known]

    conflict = np.zeros((n, n), np.float32)
    for i in range(n - 1):
        for j in range(i + 1, n):
            anchors = known[i] & known[j]
            anchors[i] = False
            anchors[j] = False
            if int(anchors.sum()) < args.layout_min_shared_anchors:
                continue
            disagreement = np.abs(relation[i, anchors] - relation[j, anchors]) / 2.0
            val = float(np.clip(disagreement.mean(), 0.0, 1.0))
            conflict[i, j] = val
            conflict[j, i] = val

    np.fill_diagonal(conflict, 0.0)
    np.fill_diagonal(copresent, False)
    info = {
        "weight": float(args.layout_weight),
        "min_x_gap": float(args.layout_min_x_gap),
        "min_shared_anchors": int(args.layout_min_shared_anchors),
        "n_copresent_pairs": int(np.triu(copresent, 1).sum()),
        "n_conflict_pairs": int(np.triu(conflict > 0, 1).sum()),
        "max_simultaneous_tracklets": max(
            (len(detections) for detections in frame_dets.values()), default=0
        ),
    }
    return conflict, copresent, info


def apply_layout_constraints(D, layout_conflict, args):
    """Increase distances for fixed-layout contradictions."""
    D = D.copy()
    penalized = layout_conflict > 0
    D[penalized] = D[penalized] + args.layout_weight * layout_conflict[penalized]
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    return (D + D.T) / 2.0


def constrained_agglomerative(
    D, cannot_link, *, n_clusters=None, distance_threshold=None, snapshot_counts=()
):
    """Average-linkage clustering with hard pairwise cannot-link constraints.

    A merge is legal only if no original cannot-link pair would end up in the
    merged cluster. ``snapshot_counts`` returns intermediate flat clusterings
    from the same hierarchy, avoiding repeated fits during silhouette search.
    """
    n = D.shape[0]
    if n == 0:
        raise ValueError("cannot cluster an empty distance matrix")
    if cannot_link.shape != (n, n):
        raise ValueError("cannot-link matrix shape does not match distances")
    if n_clusters is not None and not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be between 1 and {n}, got {n_clusters}")

    capacity = 2 * n - 1
    distances = np.full((capacity, capacity), np.inf, np.float64)
    distances[:n, :n] = D
    forbidden = np.zeros((capacity, capacity), bool)
    forbidden[:n, :n] = cannot_link
    distances[forbidden] = np.inf
    np.fill_diagonal(distances, np.inf)

    active = np.zeros(capacity, bool)
    active[:n] = True
    sizes = np.zeros(capacity, np.int64)
    sizes[:n] = 1
    members = [None] * capacity
    for i in range(n):
        members[i] = [i]

    requested_snapshots = set(snapshot_counts)
    snapshots = {}

    def labels_now():
        labels = np.full(n, -1, int)
        for label, cluster_idx in enumerate(np.flatnonzero(active)):
            labels[members[cluster_idx]] = label
        return labels

    active_count = n
    if active_count in requested_snapshots:
        snapshots[active_count] = labels_now()

    next_idx = n
    while active_count > 1:
        flat = int(np.argmin(distances))
        best_distance = float(distances.flat[flat])
        if not np.isfinite(best_distance):
            break
        if n_clusters is not None and active_count <= n_clusters:
            break
        if distance_threshold is not None and best_distance > distance_threshold:
            break

        left, right = np.unravel_index(flat, distances.shape)
        if left == right or not active[left] or not active[right]:
            raise RuntimeError("invalid constrained clustering state")

        others = np.flatnonzero(active)
        others = others[(others != left) & (others != right)]
        total_size = sizes[left] + sizes[right]
        merged_distances = (
            sizes[left] * distances[left, others]
            + sizes[right] * distances[right, others]
        ) / total_size
        merged_forbidden = forbidden[left, others] | forbidden[right, others]

        active[left] = active[right] = False
        active[next_idx] = True
        sizes[next_idx] = total_size
        members[next_idx] = members[left] + members[right]
        forbidden[next_idx, others] = merged_forbidden
        forbidden[others, next_idx] = merged_forbidden
        merged_distances[merged_forbidden] = np.inf
        distances[next_idx, others] = merged_distances
        distances[others, next_idx] = merged_distances
        distances[left, :] = distances[:, left] = np.inf
        distances[right, :] = distances[:, right] = np.inf
        distances[next_idx, next_idx] = np.inf

        next_idx += 1
        active_count -= 1
        if active_count in requested_snapshots:
            snapshots[active_count] = labels_now()

    if n_clusters is not None and active_count > n_clusters:
        raise ValueError(
            f"cannot produce {n_clusters} clusters without assigning "
            "co-present tracklets together; the constrained hierarchy stopped "
            f"at {active_count} clusters"
        )
    return labels_now(), snapshots


def cluster_face_primary(
    body, face, has_face, args, layout_conflict=None, copresent=None
):
    """Cluster the face-bearing tracklets on FACE distance, then attach the
    faceless ones to the nearest face-cluster by BODY distance.

    Designed for egocentric / dim footage where body crops are unreliable but
    faces, when visible, give true identity.
    """
    from sklearn.metrics import silhouette_score

    n = len(has_face)
    fidx = np.where(has_face)[0]
    F = face[fidx]
    Df = np.clip(1.0 - F @ F.T, 0.0, 2.0)
    np.fill_diagonal(Df, 0.0)
    Df = (Df + Df.T) / 2.0
    if layout_conflict is not None:
        Df = apply_layout_constraints(Df, layout_conflict[np.ix_(fidx, fidx)], args)
    face_cannot_link = (
        copresent[np.ix_(fidx, fidx)]
        if copresent is not None
        else np.zeros_like(Df, dtype=bool)
    )

    if args.n_clusters:
        flabels, _ = constrained_agglomerative(
            Df, face_cannot_link, n_clusters=args.n_clusters
        )
        chosen = args.n_clusters
        scores = {}
    elif args.distance_threshold is not None:
        flabels, _ = constrained_agglomerative(
            Df, face_cannot_link, distance_threshold=args.distance_threshold
        )
        chosen, scores = int(flabels.max() + 1), {}
    else:
        kmax = min(args.kmax, len(fidx) - 1)
        _, candidates = constrained_agglomerative(
            Df, face_cannot_link, snapshot_counts=range(2, kmax + 1)
        )
        scores, best_s, chosen, flabels = {}, -1.0, None, None
        for k, lab in sorted(candidates.items()):
            s = silhouette_score(Df, lab, metric="precomputed")
            scores[k] = float(s)
            if s > best_s:
                best_s, chosen, flabels = s, k, lab
        if flabels is None:
            raise ValueError(
                f"no feasible constrained clustering at or below kmax={kmax}; "
                "increase the automatic k maximum"
            )
        print("[info] face-only silhouette by k:")
        for k, s in scores.items():
            print(
                f"        k={k:2d}  silhouette={s:.4f}"
                + ("  <- best" if k == chosen else "")
            )

    # body centroid per face-cluster, to attach faceless tracklets
    labels = np.full(n, -1, int)
    labels[fidx] = flabels
    n_clusters = int(flabels.max() + 1)
    cluster_members = []
    body_sums = []
    body_centroids = []
    for c in range(n_clusters):
        members = fidx[flabels == c]
        cluster_members.append(list(members))
        v = body[members].sum(axis=0)
        body_sums.append(v.copy())
        nrm = np.linalg.norm(v)
        body_centroids.append(v / nrm if nrm > 0 else v)

    faceless = np.where(~has_face)[0]
    n_attached = 0
    n_new_clusters = 0
    for i in faceless:
        if np.linalg.norm(body[i]) == 0:
            continue
        d = 1.0 - np.stack(body_centroids) @ body[i]
        for c, members_list in enumerate(cluster_members):
            members = np.asarray(members_list, dtype=int)
            if copresent is not None and bool(copresent[i, members].any()):
                d[c] = np.inf
                continue
            if layout_conflict is not None:
                conflicts = layout_conflict[i, members]
                if conflicts.size:
                    d[c] += args.layout_weight * float(conflicts.mean())
        if np.isfinite(d).any():
            chosen_cluster = int(np.argmin(d))
            labels[i] = chosen_cluster
            cluster_members[chosen_cluster].append(i)
            body_sums[chosen_cluster] += body[i]
            nrm = np.linalg.norm(body_sums[chosen_cluster])
            body_centroids[chosen_cluster] = (
                body_sums[chosen_cluster] / nrm
                if nrm > 0
                else body_sums[chosen_cluster]
            )
        else:
            labels[i] = len(cluster_members)
            cluster_members.append([i])
            body_sums.append(body[i].copy())
            body_centroids.append(body[i].copy())
            n_new_clusters += 1
        n_attached += 1
    n_clusters = len(cluster_members)
    if copresent is not None:
        duplicate_identity = (
            copresent & (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
        )
        if bool(np.triu(duplicate_identity, 1).any()):
            raise RuntimeError("hard co-presence constraint was violated")
    info = {
        "method": "face-primary",
        "n_clusters": n_clusters,
        "n_face_tracklets": int(len(fidx)),
        "n_attached_by_body": int(n_attached),
        "n_clusters_added_for_copresence": int(n_new_clusters),
        "hard_copresence_constraint": True,
        "scores": scores,
    }
    return labels, info


def read_person_assignments(path: Path):
    """Read track_id -> person_id from a result directory or assignment file."""
    if path.is_dir():
        path = path / "track_to_person.csv"
    if path.name == "track_to_person.csv":
        out = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                out[int(row["track_id"])] = int(row["person_id"])
        return out

    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            out[int(p[1])] = int(p[-1])
    return out


def align_labels_to_reference(track_ids, labels, reference_path):
    """Remap cluster ids to match a reference assignment where possible."""
    ref = read_person_assignments(reference_path)
    overlaps: dict[tuple[int, int], int] = defaultdict(int)
    current_labels = sorted({int(lab) for lab in labels if int(lab) >= 0})
    ref_labels = sorted({int(lab) for lab in ref.values() if int(lab) >= 0})

    for tid, lab in zip(track_ids, labels):
        lab = int(lab)
        ref_lab = ref.get(tid, -1)
        if lab >= 0 and ref_lab >= 0:
            overlaps[(lab, ref_lab)] += 1

    remap = {}
    used_current = set()
    used_ref = set()
    pairs = sorted(
        ((count, lab, ref_lab) for (lab, ref_lab), count in overlaps.items()),
        reverse=True,
    )
    for count, lab, ref_lab in pairs:
        if count <= 0 or lab in used_current or ref_lab in used_ref:
            continue
        remap[lab] = ref_lab
        used_current.add(lab)
        used_ref.add(ref_lab)

    next_label = 0
    used_labels = set(ref_labels)
    for lab in current_labels:
        if lab in remap:
            continue
        while next_label in used_labels:
            next_label += 1
        remap[lab] = next_label
        used_labels.add(next_label)

    aligned = np.array([remap.get(int(lab), int(lab)) for lab in labels], int)
    info = {
        "reference": str(reference_path),
        "mapping": {str(k): int(v) for k, v in sorted(remap.items())},
        "matched_clusters": int(len(used_current)),
    }
    return aligned, info


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_outputs(
    args,
    tracks,
    track_ids,
    labels,
    has_body,
    has_face,
    body_feats,
    face_feats,
    thumbs,
    info,
):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tid2person = {tid: int(lab) for tid, lab in zip(track_ids, labels)}

    # 1) per-track table
    with open(out / "track_to_person.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "track_id",
                "person_id",
                "n_det",
                "n_body_samples",
                "n_face_samples",
                "has_face",
            ]
        )
        for tid in track_ids:
            wr.writerow(
                [
                    tid,
                    tid2person[tid],
                    len(tracks[tid]),
                    len(body_feats.get(tid, [])),
                    len(face_feats.get(tid, [])),
                    int(tid in face_feats and len(face_feats[tid]) > 0),
                ]
            )

    # 2) annotated MOT file: original columns + appended person_id
    with open(args.tracks) as fin, open(out / "tracks_with_person.txt", "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            tid = int(line.split(",")[1])
            fout.write(f"{line},{tid2person.get(tid, -1)}\n")

    # 3) summary json
    persons = defaultdict(list)
    for tid in track_ids:
        persons[tid2person[tid]].append(tid)
    summary = {
        "n_tracklets": len(track_ids),
        "n_persons": len(persons),
        "clustering": info,
        "persons": {str(p): sorted(t) for p, t in sorted(persons.items())},
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # 4) montage per cluster
    make_montages(out, persons, thumbs)

    print(f"\n[done] {len(track_ids)} tracklets -> {len(persons)} persons")
    for p, ts in sorted(persons.items()):
        print(f"   person {p}: {len(ts):3d} tracklets  e.g. {sorted(ts)[:12]}")
    print(f"[done] outputs written to {out}/")


def make_montages(out, persons, thumbs, cols=12, cell=(100, 200)):
    cw, ch = cell
    mdir = out / "montages"
    mdir.mkdir(exist_ok=True)
    for old_montage in mdir.glob("person_*.png"):
        old_montage.unlink()
    for p, ts in sorted(persons.items()):
        cells = []
        for tid in sorted(ts):
            for t in thumbs.get(tid, []):
                canvas = np.zeros((ch, cw, 3), np.uint8)
                th, tw = t.shape[:2]
                scale = min(cw / tw, ch / th)
                rt = cv2.resize(t, (max(1, int(tw * scale)), max(1, int(th * scale))))
                yy, xx = (ch - rt.shape[0]) // 2, (cw - rt.shape[1]) // 2
                canvas[yy : yy + rt.shape[0], xx : xx + rt.shape[1]] = rt
                cv2.putText(
                    canvas,
                    str(tid),
                    (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cells.append(canvas)
        if not cells:
            continue
        rows = []
        for i in range(0, len(cells), cols):
            row = cells[i : i + cols]
            while len(row) < cols:
                row.append(np.zeros((ch, cw, 3), np.uint8))
            rows.append(np.hstack(row))
        cv2.imwrite(str(mdir / f"person_{p:02d}.png"), np.vstack(rows))


# distinct, high-contrast BGR colors for person ids (cycled if more are needed)
PALETTE = [
    (60, 60, 255),
    (60, 220, 60),
    (255, 160, 30),
    (60, 230, 255),
    (255, 70, 220),
    (230, 230, 60),
    (140, 80, 255),
    (40, 160, 255),
    (180, 255, 120),
    (255, 120, 120),
]


def person_color(pid):
    if pid < 0:
        return (160, 160, 160)  # gray for unlabelled / noise
    return PALETTE[pid % len(PALETTE)]


def render_video(args, tracks, tid2person):
    """Write a copy of the video with each detection's box + a large colored
    person id drawn at the centre of the box, for visual verification."""
    from tqdm import tqdm

    # frame (MOT, 1-indexed) -> list of (x, y, w, h, track_id)
    frame_dets: dict[int, list] = defaultdict(list)
    for tid, dets in tracks.items():
        for frame, x, y, w, h, conf in dets:
            frame_dets[frame].append((x, y, w, h, tid))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = args.render_scale
    out_w, out_h = int(W * s), int(H * s)

    out_path = Path(args.out_dir) / args.render_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        sys.exit(f"could not open VideoWriter for {out_path}")

    print(f"[info] rendering labelled video -> {out_path}")
    for i in tqdm(range(n_frames), desc="rendering", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        mot_frame = i - args.frame_offset  # video index -> MOT frame
        for x, y, w, h, tid in frame_dets.get(mot_frame, []):
            pid = tid2person.get(tid, -1)
            col = person_color(pid)
            x0, y0, x1, y1 = clamp_box(x, y, w, h, W, H)
            cv2.rectangle(frame, (x0, y0), (x1, y1), col, 3)
            # small track-id tag in the corner of the box
            cv2.putText(
                frame,
                f"t{tid}",
                (x0 + 3, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                col,
                2,
                cv2.LINE_AA,
            )
            # large person-id at the box centre, with a black outline
            label = str(pid) if pid >= 0 else "?"
            fs = max(1.5, (y1 - y0) / 130.0)
            th = max(3, int(fs * 2))
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cx = (x0 + x1) // 2 - tw // 2
            cy = (y0 + y1) // 2 + tht // 2
            cv2.putText(
                frame,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (0, 0, 0),
                th + 3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                col,
                th,
                cv2.LINE_AA,
            )
        if s != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))
        writer.write(frame)
    writer.release()
    cap.release()
    print(f"[done] labelled video written to {out_path}")


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tracks", type=Path, default="runs/track/quartet2/tracks.txt")
    ap.add_argument("--video", type=Path, default="Archiv/quartet.mp4")
    ap.add_argument("--out-dir", type=Path, default="runs/track/quartet2/clustering")
    ap.add_argument("--reid-weights", default="weights/osnet_x1_0_msmt17.pt")
    ap.add_argument("--reid-model", default="osnet_x1_0")
    ap.add_argument("--face-model", default="buffalo_l")

    ap.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="max detections sampled per tracklet",
    )
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--min-w", type=float, default=40)
    ap.add_argument("--min-h", type=float, default=80)
    ap.add_argument(
        "--frame-offset",
        type=int,
        default=-1,
        help="video_index = mot_frame + offset (MOT is 1-indexed)",
    )
    ap.add_argument("--batch-size", type=int, default=256)

    ap.add_argument("--face-det-thresh", type=float, default=0.55)

    # clustering: default fully-automatic (silhouette). Override with either flag.
    ap.add_argument(
        "--n-clusters", type=int, default=None, help="force a fixed number of people"
    )
    ap.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help="cosine distance cut instead of silhouette auto-selection",
    )
    ap.add_argument(
        "--kmax",
        type=int,
        default=10,
        help="max number of people to consider in silhouette search",
    )
    ap.add_argument(
        "--align-labels-to",
        type=Path,
        default=None,
        help="remap output person ids to best match a reference "
        "result directory, track_to_person.csv, or "
        "tracks_with_person.txt",
    )

    ap.add_argument(
        "--fixed-layout",
        action="store_true",
        help="use fixed left/right seating constraints to reduce "
        "identity swaps between tracklets",
    )
    ap.add_argument(
        "--layout-weight",
        type=float,
        default=0.35,
        help="extra distance added for contradictory fixed-layout left/right evidence",
    )
    ap.add_argument(
        "--layout-min-x-gap",
        type=float,
        default=20.0,
        help="ignore left/right evidence when tracklet centers are "
        "closer than this many pixels",
    )
    ap.add_argument(
        "--layout-min-shared-anchors",
        type=int,
        default=1,
        help="minimum shared third-party tracklets needed before "
        "adding a left/right contradiction penalty",
    )
    ap.add_argument(
        "--montage-source",
        choices=["body", "face"],
        default="body",
        help="which thumbnails to use in the montages",
    )
    ap.add_argument("--montage-per-track", type=int, default=4)
    ap.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="load embeddings.pkl from out-dir instead of re-extracting",
    )
    ap.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="path to embeddings cache; defaults to out-dir/embeddings.pkl",
    )

    ap.add_argument(
        "--render-video",
        action="store_true",
        help="also write a copy of the video with person ids drawn on it",
    )
    ap.add_argument(
        "--render-name",
        default="labelled.mp4",
        help="filename (inside out-dir) for the rendered video",
    )
    ap.add_argument(
        "--render-scale",
        type=float,
        default=1.0,
        help="scale factor for the rendered video (e.g. 0.5 for half size)",
    )
    return ap.parse_args()


def preprocess_embeddings(args):
    """Extract reusable body/face embeddings and thumbnails into the cache."""
    print(f"[info] parsing {args.tracks}")
    tracks = parse_tracks(args.tracks)
    print(f"[info] {len(tracks)} raw tracklets")

    cache = args.embedding_cache or (Path(args.out_dir) / "embeddings.pkl")
    per_track, frame_index = select_samples(
        tracks, args.max_samples, args.min_conf, args.min_w, args.min_h
    )
    n_samp = sum(len(v) for v in per_track.values())
    print(f"[info] sampling {n_samp} detections across {len(frame_index)} frames")
    body_feats, face_feats, thumbs, face_thumbs = extract_embeddings(args, frame_index)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(
            {
                "body": dict(body_feats),
                "face": dict(face_feats),
                "thumbs": dict(thumbs),
                "face_thumbs": dict(face_thumbs),
            },
            fh,
        )
    print(f"[info] cached embeddings to {cache}")
    return cache


def cluster_cached_embeddings(args):
    """Cluster an existing embedding cache without running model inference."""
    print(f"[info] parsing {args.tracks}")
    tracks = parse_tracks(args.tracks)
    print(f"[info] {len(tracks)} raw tracklets")

    cache = args.embedding_cache or (Path(args.out_dir) / "embeddings.pkl")
    if not cache.exists():
        raise FileNotFoundError(f"embedding cache does not exist: {cache}")
    print(f"[info] reusing cached embeddings from {cache}")
    with open(cache, "rb") as fh:
        d = pickle.load(fh)
    body_feats, face_feats = d["body"], d["face"]
    thumbs, face_thumbs = d["thumbs"], d.get("face_thumbs", {})

    montage_thumbs = face_thumbs if args.montage_source == "face" else thumbs

    track_ids = sorted(tid for tid in tracks if body_feats.get(tid))
    dropped = [tid for tid in tracks if not body_feats.get(tid)]
    if dropped:
        print(
            f"[warn] {len(dropped)} tracklets had no usable crop, labelled -1: {dropped}"
        )

    body, face, has_body, has_face = aggregate(track_ids, body_feats, face_feats)
    print(f"[info] {has_face.sum()}/{len(track_ids)} tracklets have a face embedding")

    layout_conflict, copresent, layout_info = layout_conflict_matrix(
        tracks, track_ids, args
    )
    print(
        "[info] hard co-presence constraints: "
        f"{layout_info['n_copresent_pairs']} cannot-link pairs, "
        f"at least {layout_info['max_simultaneous_tracklets']} clusters required"
    )
    if (
        args.n_clusters is not None
        and args.n_clusters < layout_info["max_simultaneous_tracklets"]
    ):
        raise ValueError(
            f"cannot produce {args.n_clusters} clusters: the tracking data has "
            f"{layout_info['max_simultaneous_tracklets']} tracklets visible in "
            "the same frame"
        )
    active_layout_conflict = layout_conflict if args.fixed_layout else None
    if args.fixed_layout:
        print(
            "[info] fixed-layout constraints: "
            f"{layout_info['n_conflict_pairs']} left/right conflict pairs"
        )

    labels, info = cluster_face_primary(
        body, face, has_face, args, active_layout_conflict, copresent
    )
    info["copresence"] = {
        "hard_constraint": True,
        "n_cannot_link_pairs": layout_info["n_copresent_pairs"],
        "max_simultaneous_tracklets": layout_info["max_simultaneous_tracklets"],
    }
    if args.fixed_layout:
        info["fixed_layout"] = layout_info

    if args.align_labels_to is not None:
        labels, alignment_info = align_labels_to_reference(
            track_ids, labels, args.align_labels_to
        )
        info["label_alignment"] = alignment_info
        print(
            "[info] aligned person ids to "
            f"{args.align_labels_to}: {alignment_info['mapping']}"
        )

    # re-attach dropped tracklets as their own label -1
    all_ids = track_ids + dropped
    all_labels = list(labels) + [-1] * len(dropped)

    write_outputs(
        args,
        tracks,
        all_ids,
        all_labels,
        has_body,
        has_face,
        body_feats,
        face_feats,
        montage_thumbs,
        info,
    )

    if args.render_video:
        tid2person = {tid: int(lab) for tid, lab in zip(all_ids, all_labels)}
        render_video(args, tracks, tid2person)


def main():
    args = parse_args()
    cache = args.embedding_cache or (Path(args.out_dir) / "embeddings.pkl")
    if not args.reuse_embeddings or not cache.exists():
        preprocess_embeddings(args)
    cluster_cached_embeddings(args)


if __name__ == "__main__":
    main()
