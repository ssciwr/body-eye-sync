#!/usr/bin/env python
"""Extract InsightFace facial landmarks for tracked faces and write CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2


LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]


def parse_tracks(path: Path, stride: int, min_w: float, min_h: float):
    """Return MOT frame -> detections from a tracks file.

    The input may be either a raw tracks file or tracks_with_person.txt. If a
    person id is present as the final column, it is preserved in the output.
    """
    frame_index: dict[int, list] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame = int(p[0])
            if stride > 1 and frame % stride != 0:
                continue
            tid = int(p[1])
            x, y, w, h = map(float, p[2:6])
            if w < min_w or h < min_h:
                continue
            person_id = int(p[-1]) if len(p) > 9 else -1
            frame_index[frame].append((tid, person_id, x, y, w, h))
    return frame_index


def clamp_box(x, y, w, h, W, H):
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + w)))
    y1 = min(H, int(round(y + h)))
    return x0, y0, x1, y1


def best_face(faces):
    return max(
        faces,
        key=lambda fa: (
            float(fa.det_score) * (fa.bbox[2] - fa.bbox[0]) * (fa.bbox[3] - fa.bbox[1])
        ),
    )


def output_columns():
    cols = [
        "frame",
        "video_frame",
        "track_id",
        "person_id",
        "track_x",
        "track_y",
        "track_w",
        "track_h",
        "face_score",
        "face_x1",
        "face_y1",
        "face_x2",
        "face_y2",
    ]
    for name in LANDMARK_NAMES:
        cols.extend([f"{name}_x", f"{name}_y"])
    return cols


def extract_landmarks(args):
    import torch
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from tqdm import tqdm

    try:
        ort.preload_dlls()
    except Exception as exc:  # pragma: no cover - older onnxruntime
        print(f"[warn] ort.preload_dlls() failed ({exc}); face model may run on CPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    print(f"[info] device = {device}")
    face_app = FaceAnalysis(name=args.face_model, providers=providers)
    face_app.prepare(
        ctx_id=0 if device == "cuda" else -1, det_size=(args.det_size, args.det_size)
    )

    frame_index = parse_tracks(args.tracks, args.stride, args.min_w, args.min_h)
    wanted = sorted(frame_index)
    targets = {f + args.frame_offset: f for f in wanted}
    todo = sorted(targets)
    print(f"[info] extracting landmarks from {len(wanted)} frames")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_faces = 0
    n_crops = 0
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=output_columns())
        wr.writeheader()

        idx = -1
        ptr = 0
        pbar = tqdm(total=len(todo), desc="landmark frames", unit="frame")
        while ptr < len(todo):
            ret = cap.grab()
            idx += 1
            if not ret:
                break
            if idx != todo[ptr]:
                continue
            ret, frame = cap.retrieve()
            mot_frame = targets[idx]
            ptr += 1
            pbar.update(1)
            if not ret:
                continue

            for tid, person_id, x, y, w, h in frame_index[mot_frame]:
                x0, y0, x1, y1 = clamp_box(x, y, w, h, W, H)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                crop = frame[y0:y1, x0:x1]
                n_crops += 1
                faces = face_app.get(crop)
                if not faces:
                    continue
                face = best_face(faces)
                if float(face.det_score) < args.face_det_thresh:
                    continue

                fx1, fy1, fx2, fy2 = [float(v) for v in face.bbox]
                row = {
                    "frame": mot_frame,
                    "video_frame": idx,
                    "track_id": tid,
                    "person_id": person_id,
                    "track_x": x,
                    "track_y": y,
                    "track_w": w,
                    "track_h": h,
                    "face_score": float(face.det_score),
                    "face_x1": x0 + fx1,
                    "face_y1": y0 + fy1,
                    "face_x2": x0 + fx2,
                    "face_y2": y0 + fy2,
                }
                for name, (px, py) in zip(LANDMARK_NAMES, face.kps):
                    row[f"{name}_x"] = x0 + float(px)
                    row[f"{name}_y"] = y0 + float(py)
                wr.writerow(row)
                n_faces += 1
        pbar.close()

    cap.release()
    print(f"[done] {n_faces} faces with landmarks from {n_crops} tracklet crops")
    print(f"[done] wrote {args.out}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=Path("Archiv/quartet.mp4"))
    ap.add_argument(
        "--tracks",
        type=Path,
        default=Path(
            "runs/track/quartet2/clustering_facep_fixed_layout/tracks_with_person.txt"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/track/quartet2/clustering_facep_fixed_layout/face_landmarks.csv"
        ),
    )
    ap.add_argument("--face-model", default="buffalo_l")
    ap.add_argument("--face-det-thresh", type=float, default=0.55)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument(
        "--frame-offset", type=int, default=-1, help="video_index = mot_frame + offset"
    )
    ap.add_argument("--stride", type=int, default=1, help="process every Nth MOT frame")
    ap.add_argument("--min-w", type=float, default=40)
    ap.add_argument("--min-h", type=float, default=80)
    return ap.parse_args()


def main():
    extract_landmarks(parse_args())


if __name__ == "__main__":
    main()
