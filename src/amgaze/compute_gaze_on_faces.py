#!/usr/bin/env python
"""Assign each aligned gaze sample to a detected face box, if any."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


FEATURE_NAMES = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_int(value):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def read_landmarks(path: Path):
    by_video_frame: dict[int, list] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            video_frame = int(row["video_frame"])
            points = {
                name: (float(row[f"{name}_x"]), float(row[f"{name}_y"]))
                for name in FEATURE_NAMES
            }
            by_video_frame[video_frame].append(
                {
                    "frame": int(row["frame"]),
                    "video_frame": video_frame,
                    "track_id": int(row["track_id"]),
                    "person_id": int(row["person_id"]),
                    "face_score": float(row["face_score"]),
                    "bbox": (
                        float(row["face_x1"]),
                        float(row["face_y1"]),
                        float(row["face_x2"]),
                        float(row["face_y2"]),
                    ),
                    "points": points,
                }
            )
    return by_video_frame


def candidate_faces(by_video_frame, video_frame, tolerance):
    faces = list(by_video_frame.get(video_frame, []))
    if faces or tolerance <= 0:
        return faces
    for delta in range(1, tolerance + 1):
        faces.extend(by_video_frame.get(video_frame - delta, []))
        faces.extend(by_video_frame.get(video_frame + delta, []))
        if faces:
            return faces
    return faces


def point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def nearest_feature(x, y, points):
    best_name = ""
    best_dist = math.inf
    best_point = (math.nan, math.nan)
    for name, (px, py) in points.items():
        dist = math.hypot(x - px, y - py)
        if dist < best_dist:
            best_name = name
            best_dist = dist
            best_point = (px, py)
    return best_name, best_point[0], best_point[1], best_dist


def choose_face(x, y, faces):
    containing = [face for face in faces if point_in_bbox(x, y, face["bbox"])]
    if not containing:
        return None

    def score(face):
        x1, y1, x2, y2 = face["bbox"]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return math.hypot(x - cx, y - cy)

    return min(containing, key=score)


def output_fields(input_fields):
    fields = list(input_fields)
    for field in [
        "gaze_on_face",
        "gaze_person_id",
        "gaze_track_id",
        "face_score",
        "face_x1",
        "face_y1",
        "face_x2",
        "face_y2",
        "nearest_face_feature",
        "nearest_feature_x",
        "nearest_feature_y",
        "nearest_feature_distance",
    ]:
        if field not in fields:
            fields.append(field)
    return fields


def empty_assignment(row):
    row.update(
        {
            "gaze_on_face": "0",
            "gaze_person_id": "",
            "gaze_track_id": "",
            "face_score": "",
            "face_x1": "",
            "face_y1": "",
            "face_x2": "",
            "face_y2": "",
            "nearest_face_feature": "",
            "nearest_feature_x": "",
            "nearest_feature_y": "",
            "nearest_feature_distance": "",
        }
    )


def assign_gaze(args):
    landmarks = read_landmarks(args.landmarks)
    if not landmarks:
        sys.exit(f"no landmarks found in {args.landmarks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_valid = 0
    n_on_face = 0
    person_counts: dict[int, int] = defaultdict(int)
    feature_counts: dict[str, int] = defaultdict(int)

    with open(args.gaze, newline="") as fin, open(args.out, "w", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        if reader.fieldnames is None:
            sys.exit(f"empty gaze file: {args.gaze}")
        writer = csv.DictWriter(
            fout, fieldnames=output_fields(reader.fieldnames), delimiter="\t"
        )
        writer.writeheader()

        for row in reader:
            n_rows += 1
            out = dict(row)
            empty_assignment(out)
            gx = to_float(row.get("gaze_x"))
            gy = to_float(row.get("gaze_y"))
            video_frame = to_int(row.get("video_frame"))
            if not math.isfinite(gx) or not math.isfinite(gy) or video_frame is None:
                writer.writerow(out)
                continue
            n_valid += 1
            face = choose_face(
                gx,
                gy,
                candidate_faces(landmarks, video_frame, args.frame_tolerance),
            )
            if face is None:
                writer.writerow(out)
                continue

            nearest_name, px, py, dist = nearest_feature(gx, gy, face["points"])
            x1, y1, x2, y2 = face["bbox"]
            out.update(
                {
                    "gaze_on_face": "1",
                    "gaze_person_id": str(face["person_id"]),
                    "gaze_track_id": str(face["track_id"]),
                    "face_score": f"{face['face_score']:.6f}",
                    "face_x1": f"{x1:.3f}",
                    "face_y1": f"{y1:.3f}",
                    "face_x2": f"{x2:.3f}",
                    "face_y2": f"{y2:.3f}",
                    "nearest_face_feature": nearest_name,
                    "nearest_feature_x": f"{px:.3f}",
                    "nearest_feature_y": f"{py:.3f}",
                    "nearest_feature_distance": f"{dist:.3f}",
                }
            )
            n_on_face += 1
            person_counts[face["person_id"]] += 1
            feature_counts[nearest_name] += 1
            writer.writerow(out)

    summary = {
        "gaze": str(args.gaze),
        "landmarks": str(args.landmarks),
        "out": str(args.out),
        "frame_tolerance": args.frame_tolerance,
        "n_rows": n_rows,
        "n_valid_gaze_rows": n_valid,
        "n_gaze_on_face_rows": n_on_face,
        "person_counts": {str(k): v for k, v in sorted(person_counts.items())},
        "nearest_feature_counts": dict(sorted(feature_counts.items())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[done] {n_on_face}/{n_valid} valid gaze samples are inside a face box")
    print(f"[done] wrote {args.out}")
    print(f"[done] wrote {args.summary}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gaze", type=Path, default=Path("runs/track/quartet2/gaze_aligned.tsv")
    )
    ap.add_argument(
        "--landmarks",
        type=Path,
        default=Path(
            "runs/track/quartet2/clustering_facep_fixed_layout/face_landmarks.csv"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/track/quartet2/clustering_facep_fixed_layout/gaze_on_faces.tsv"
        ),
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "runs/track/quartet2/clustering_facep_fixed_layout/"
            "gaze_on_faces_summary.json"
        ),
    )
    ap.add_argument(
        "--frame-tolerance",
        type=int,
        default=0,
        help="use face boxes from a nearby video frame if the exact "
        "frame has no landmarks",
    )
    return ap.parse_args()


def main():
    assign_gaze(parse_args())


if __name__ == "__main__":
    main()
