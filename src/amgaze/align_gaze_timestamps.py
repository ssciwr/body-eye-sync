#!/usr/bin/env python
"""Align gaze TSV timestamps to a video by matching the overlaid gaze cursor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_gaze_tsv(path: Path, time_col: str):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        if not rows:
            sys.exit(f"empty gaze TSV: {path}")
        for col in [time_col, "gaze_x", "gaze_y"]:
            if col not in rows[0]:
                sys.exit(f"missing required column {col!r} in {path}")

    times = np.array([to_float(r[time_col]) for r in rows], np.float64)
    xs = np.array([to_float(r["gaze_x"]) for r in rows], np.float64)
    ys = np.array([to_float(r["gaze_y"]) for r in rows], np.float64)
    valid = np.isfinite(times) & np.isfinite(xs) & np.isfinite(ys)
    return rows, times[valid], xs[valid], ys[valid]


def smooth_series(values, times, smooth_ms):
    if smooth_ms <= 0 or len(values) < 3:
        return values
    dt = np.median(np.diff(times))
    if not np.isfinite(dt) or dt <= 0:
        return values
    win = max(1, int(round(smooth_ms / dt)))
    if win <= 1:
        return values
    kernel = np.ones(win, np.float64) / win
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_marker(frame, args):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([args.hue_min, args.sat_min, args.val_min], np.uint8)
    hi = np.array([args.hue_max, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    if args.morph_kernel > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.morph_kernel, args.morph_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < args.min_area or area > args.max_area:
            continue
        x, y, w, h = [int(v) for v in stats[i, :4]]
        if w < args.min_diameter or h < args.min_diameter:
            continue
        if w > args.max_diameter or h > args.max_diameter:
            continue
        ratio = w / max(1, h)
        if ratio < args.min_aspect or ratio > args.max_aspect:
            continue
        candidates.append((area, float(cents[i][0]), float(cents[i][1]), x, y, w, h))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])


def extract_markers(args):
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    markers = []
    from tqdm import tqdm

    for frame_idx in tqdm(
        range(n_frames), desc="extracting video gaze marker", unit="frame"
    ):
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % args.marker_stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        marker = find_marker(frame, args)
        if marker is None:
            continue
        area, x, y, bx, by, bw, bh = marker
        markers.append(
            {
                "frame": frame_idx,
                "video_time_ms": frame_idx * 1000.0 / fps,
                "marker_x": x,
                "marker_y": y,
                "area": area,
                "bbox_x": bx,
                "bbox_y": by,
                "bbox_w": bw,
                "bbox_h": bh,
            }
        )
    cap.release()
    return markers, fps, n_frames


def write_marker_csv(path: Path, markers):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "frame",
        "video_time_ms",
        "marker_x",
        "marker_y",
        "area",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(markers)


def read_marker_csv(path: Path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [
        {
            "frame": int(r["frame"]),
            "video_time_ms": float(r["video_time_ms"]),
            "marker_x": float(r["marker_x"]),
            "marker_y": float(r["marker_y"]),
        }
        for r in rows
    ]


def loss_for_offset(offset_ms, marker_t, marker_x, marker_y, gaze_t, gaze_x, gaze_y):
    sample_t = marker_t - offset_ms
    valid = (sample_t >= gaze_t[0]) & (sample_t <= gaze_t[-1])
    if int(valid.sum()) < 10:
        return np.inf, 0
    pred_x = np.interp(sample_t[valid], gaze_t, gaze_x)
    pred_y = np.interp(sample_t[valid], gaze_t, gaze_y)
    dist = np.hypot(pred_x - marker_x[valid], pred_y - marker_y[valid])
    return float(np.median(dist)), int(valid.sum())


def estimate_offset(markers, gaze_t, gaze_x, gaze_y, args):
    marker_t = np.array([m["video_time_ms"] for m in markers], np.float64)
    marker_x = np.array([m["marker_x"] for m in markers], np.float64)
    marker_y = np.array([m["marker_y"] for m in markers], np.float64)

    if args.align_marker_stride > 1:
        marker_t = marker_t[:: args.align_marker_stride]
        marker_x = marker_x[:: args.align_marker_stride]
        marker_y = marker_y[:: args.align_marker_stride]

    coarse_offsets = np.arange(
        args.offset_min_ms,
        args.offset_max_ms + args.coarse_step_ms,
        args.coarse_step_ms,
        dtype=np.float64,
    )
    best = (np.inf, None, 0)
    for off in coarse_offsets:
        loss, n = loss_for_offset(
            off, marker_t, marker_x, marker_y, gaze_t, gaze_x, gaze_y
        )
        if loss < best[0]:
            best = (loss, off, n)

    if best[1] is None:
        sys.exit("could not estimate offset: no overlapping marker/TSV samples")

    fine_min = best[1] - args.coarse_step_ms
    fine_max = best[1] + args.coarse_step_ms
    fine_offsets = np.arange(
        fine_min, fine_max + args.fine_step_ms, args.fine_step_ms, dtype=np.float64
    )
    for off in fine_offsets:
        loss, n = loss_for_offset(
            off, marker_t, marker_x, marker_y, gaze_t, gaze_x, gaze_y
        )
        if loss < best[0]:
            best = (loss, off, n)
    return float(best[1]), float(best[0]), int(best[2])


def write_shifted_tsv(path: Path, rows, time_col: str, offset_ms: float, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for col in ["video_time_ms", "video_frame", "alignment_offset_ms"]:
        if col not in fieldnames:
            fieldnames.append(col)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        wr.writeheader()
        for row in rows:
            out = dict(row)
            shifted = float(row[time_col]) + offset_ms
            out["video_time_ms"] = f"{shifted:.3f}"
            out["video_frame"] = str(int(round(shifted * fps / 1000.0)))
            out["alignment_offset_ms"] = f"{offset_ms:.3f}"
            wr.writerow(out)


def align_gaze(args):
    """Align a gaze TSV to the video and write the configured outputs."""
    rows, gaze_t, gaze_x, gaze_y = read_gaze_tsv(args.gaze_tsv, args.time_col)
    gaze_x = smooth_series(gaze_x, gaze_t, args.tsv_smooth_ms)
    gaze_y = smooth_series(gaze_y, gaze_t, args.tsv_smooth_ms)

    if args.reuse_marker_csv and args.marker_csv.exists():
        markers = read_marker_csv(args.marker_csv)
        cap = cv2.VideoCapture(str(args.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"[info] reusing video gaze markers from {args.marker_csv}")
    else:
        markers, fps, n_frames = extract_markers(args)
        write_marker_csv(args.marker_csv, markers)
        print(f"[info] wrote video gaze markers to {args.marker_csv}")
    if len(markers) < 10:
        sys.exit(f"too few video gaze markers detected: {len(markers)}")

    offset_ms, median_error_px, n_aligned = estimate_offset(
        markers, gaze_t, gaze_x, gaze_y, args
    )
    write_shifted_tsv(args.out, rows, args.time_col, offset_ms, fps)
    summary = {
        "gaze_tsv": str(args.gaze_tsv),
        "video": str(args.video),
        "time_col": args.time_col,
        "offset_ms": offset_ms,
        "definition": f"video_time_ms = {args.time_col} + offset_ms",
        "median_error_px": median_error_px,
        "n_aligned_marker_samples": n_aligned,
        "n_video_markers": len(markers),
        "video_fps": fps,
        "video_frames": n_frames,
        "tsv_smooth_ms": args.tsv_smooth_ms,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] offset_ms = {offset_ms:.3f}")
    print(
        f"[done] median alignment error = {median_error_px:.2f}px over {n_aligned} samples"
    )
    print(f"[done] wrote shifted gaze TSV to {args.out}")
    print(f"[done] wrote summary to {args.summary}")
    return summary


def main():
    align_gaze(parse_args())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=Path("Archiv/quartet.mp4"))
    ap.add_argument("--gaze-tsv", type=Path, default=Path("Archiv/G3_1401mp4.tsv"))
    ap.add_argument(
        "--out", type=Path, default=Path("runs/track/quartet2/gaze_aligned.tsv")
    )
    ap.add_argument(
        "--marker-csv",
        type=Path,
        default=Path("runs/track/quartet2/video_gaze_marker.csv"),
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/track/quartet2/gaze_alignment_summary.json"),
    )
    ap.add_argument(
        "--time-col",
        default="gaze_video_time",
        help="TSV timestamp column to shift, in milliseconds",
    )
    ap.add_argument("--reuse-marker-csv", action="store_true")
    ap.add_argument(
        "--marker-stride",
        type=int,
        default=1,
        help="extract video marker every Nth frame",
    )
    ap.add_argument(
        "--align-marker-stride",
        type=int,
        default=5,
        help="use every Nth detected marker during offset search",
    )
    ap.add_argument("--offset-min-ms", type=float, default=-2_000_000)
    ap.add_argument("--offset-max-ms", type=float, default=200_000)
    ap.add_argument("--coarse-step-ms", type=float, default=1000)
    ap.add_argument("--fine-step-ms", type=float, default=20)
    ap.add_argument("--tsv-smooth-ms", type=float, default=200)

    ap.add_argument("--hue-min", type=int, default=130)
    ap.add_argument("--hue-max", type=int, default=179)
    ap.add_argument("--sat-min", type=int, default=40)
    ap.add_argument("--val-min", type=int, default=60)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--max-area", type=int, default=2500)
    ap.add_argument("--min-diameter", type=int, default=25)
    ap.add_argument("--max-diameter", type=int, default=90)
    ap.add_argument("--min-aspect", type=float, default=0.6)
    ap.add_argument("--max-aspect", type=float, default=1.6)
    ap.add_argument("--morph-kernel", type=int, default=3)
    return ap.parse_args()


if __name__ == "__main__":
    main()
