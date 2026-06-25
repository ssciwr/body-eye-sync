"""Backend stages and visualization helpers for the AMGAZE desktop GUI."""

from __future__ import annotations

import csv
import shutil
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


@dataclass(frozen=True)
class PipelinePaths:
    workspace: Path

    @property
    def tracks(self) -> Path:
        return self.workspace / "tracks.txt"

    @property
    def tracked_video(self) -> Path:
        return self.workspace / "tracks.mp4"

    @property
    def aligned_gaze(self) -> Path:
        return self.workspace / "gaze_aligned.tsv"

    @property
    def marker_csv(self) -> Path:
        return self.workspace / "video_gaze_marker.csv"

    @property
    def alignment_summary(self) -> Path:
        return self.workspace / "gaze_alignment_summary.json"

    @property
    def embeddings(self) -> Path:
        return self.workspace / "embeddings.pkl"

    @property
    def raw_landmarks(self) -> Path:
        return self.workspace / "face_landmarks_unclustered.csv"

    @property
    def clustering_dir(self) -> Path:
        return self.workspace / "clustering"

    @property
    def clustered_tracks(self) -> Path:
        return self.clustering_dir / "tracks_with_person.txt"

    @property
    def clustered_landmarks(self) -> Path:
        return self.clustering_dir / "face_landmarks.csv"

    @property
    def gaze_on_faces(self) -> Path:
        return self.clustering_dir / "gaze_on_faces.tsv"

    @property
    def gaze_on_faces_summary(self) -> Path:
        return self.clustering_dir / "gaze_on_faces_summary.json"


@dataclass
class TrackingConfig:
    video: Path
    workspace: Path
    detector: str = "yolo26l"
    reid: str = "osnet_x0_25_msmt17"
    tracker: str = "botsort"
    device: str = "0"


@dataclass
class AlignmentConfig:
    video: Path
    gaze_tsv: Path
    workspace: Path
    time_col: str = "gaze_video_time"
    reuse_markers: bool = True


@dataclass
class FaceConfig:
    video: Path
    workspace: Path
    reid_model: str = "osnet_x1_0"
    reid_weights: Path = Path("weights/osnet_x1_0_msmt17.pt")
    face_model: str = "buffalo_l"
    face_threshold: float = 0.55
    max_samples: int = 30
    landmark_stride: int = 1
    frame_offset: int = -1


@dataclass
class ClusteringConfig:
    video: Path
    workspace: Path
    fixed_layout: bool = False
    n_clusters: int | None = None
    distance_threshold: float | None = None
    kmax: int = 10
    layout_weight: float = 0.35
    layout_min_x_gap: float = 20.0
    layout_min_shared_anchors: int = 1
    frame_offset: int = -1


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _copy_result(source: Path | None, destination: Path, label: str) -> None:
    if source is None:
        raise RuntimeError(f"BoxMOT did not return a {label} path")
    source = Path(source)
    _require(source, f"BoxMOT {label}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def run_tracking(config: TrackingConfig) -> PipelinePaths:
    """Run BoxMOT through its Python API and normalize its two outputs."""
    _require(config.video, "video")
    paths = PipelinePaths(config.workspace)
    paths.workspace.mkdir(parents=True, exist_ok=True)

    from boxmot import Boxmot

    print("[stage] tracking")
    runner = Boxmot(
        detector=config.detector,
        reid=config.reid,
        tracker=config.tracker,
        classes=[0],
        project=paths.workspace / "boxmot",
    )
    result = runner.track(
        source=str(config.video),
        device=config.device,
        save=True,
        save_txt=True,
        show=False,
        verbose=True,
    )
    _copy_result(result.text_path, paths.tracks, "tracks file")
    _copy_result(result.video_path, paths.tracked_video, "annotated video")
    _unlink(paths.embeddings)
    _unlink(paths.raw_landmarks)
    if paths.clustering_dir.exists():
        shutil.rmtree(paths.clustering_dir)
    print(f"[done] tracks: {paths.tracks}")
    print(f"[done] preview: {paths.tracked_video}")
    return paths


def run_alignment(config: AlignmentConfig) -> PipelinePaths:
    from .align_gaze_timestamps import align_gaze

    _require(config.video, "video")
    _require(config.gaze_tsv, "gaze TSV")
    paths = PipelinePaths(config.workspace)
    paths.workspace.mkdir(parents=True, exist_ok=True)
    args = Namespace(
        video=config.video,
        gaze_tsv=config.gaze_tsv,
        out=paths.aligned_gaze,
        marker_csv=paths.marker_csv,
        summary=paths.alignment_summary,
        time_col=config.time_col,
        reuse_marker_csv=config.reuse_markers,
        marker_stride=1,
        align_marker_stride=5,
        offset_min_ms=-2_000_000,
        offset_max_ms=200_000,
        coarse_step_ms=1000,
        fine_step_ms=20,
        tsv_smooth_ms=200,
        hue_min=130,
        hue_max=179,
        sat_min=40,
        val_min=60,
        min_area=200,
        max_area=2500,
        min_diameter=25,
        max_diameter=90,
        min_aspect=0.6,
        max_aspect=1.6,
        morph_kernel=3,
    )
    print("[stage] gaze timestamp alignment")
    align_gaze(args)
    _unlink(paths.gaze_on_faces)
    _unlink(paths.gaze_on_faces_summary)
    return paths


def run_face_preprocessing(config: FaceConfig) -> PipelinePaths:
    """Extract embeddings and unclustered facial landmarks."""
    from .cluster_tracklets import preprocess_embeddings
    from .extract_face_landmarks import extract_landmarks

    _require(config.video, "video")
    paths = PipelinePaths(config.workspace)
    _require(paths.tracks, "tracks file")
    paths.workspace.mkdir(parents=True, exist_ok=True)

    embedding_args = Namespace(
        tracks=paths.tracks,
        video=config.video,
        out_dir=paths.workspace,
        embedding_cache=paths.embeddings,
        max_samples=config.max_samples,
        min_conf=0.5,
        min_w=40.0,
        min_h=80.0,
        frame_offset=config.frame_offset,
        batch_size=256,
        reid_model=config.reid_model,
        reid_weights=str(config.reid_weights),
        face_model=config.face_model,
        face_det_thresh=config.face_threshold,
        montage_per_track=4,
    )
    print("[stage] face/body embedding extraction")
    preprocess_embeddings(embedding_args)

    landmark_args = Namespace(
        video=config.video,
        tracks=paths.tracks,
        out=paths.raw_landmarks,
        face_model=config.face_model,
        face_det_thresh=config.face_threshold,
        det_size=640,
        frame_offset=config.frame_offset,
        stride=config.landmark_stride,
        min_w=40.0,
        min_h=80.0,
    )
    print("[stage] facial landmark extraction")
    extract_landmarks(landmark_args)
    if paths.clustering_dir.exists():
        shutil.rmtree(paths.clustering_dir)
    return paths


def _attach_person_ids(raw_landmarks: Path, assignments: Path, output: Path) -> None:
    with open(assignments, newline="") as fh:
        person_for_track = {
            int(row["track_id"]): int(row["person_id"]) for row in csv.DictReader(fh)
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_landmarks, newline="") as fin, open(output, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise RuntimeError(f"empty landmark file: {raw_landmarks}")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["person_id"] = str(person_for_track.get(int(row["track_id"]), -1))
            writer.writerow(row)


def run_clustering(config: ClusteringConfig) -> PipelinePaths:
    """Run only cached clustering and inexpensive post-processing."""
    from .cluster_tracklets import cluster_cached_embeddings
    from .compute_gaze_on_faces import assign_gaze

    paths = PipelinePaths(config.workspace)
    _require(paths.tracks, "tracks file")
    _require(paths.embeddings, "embedding cache")
    _require(paths.raw_landmarks, "unclustered landmarks")
    paths.clustering_dir.mkdir(parents=True, exist_ok=True)

    args = Namespace(
        tracks=paths.tracks,
        video=config.video,
        out_dir=paths.clustering_dir,
        embedding_cache=paths.embeddings,
        montage_source="body",
        fixed_layout=config.fixed_layout,
        n_clusters=config.n_clusters,
        distance_threshold=config.distance_threshold,
        kmax=config.kmax,
        layout_weight=config.layout_weight,
        layout_min_x_gap=config.layout_min_x_gap,
        layout_min_shared_anchors=config.layout_min_shared_anchors,
        align_labels_to=None,
        render_video=False,
        render_name="labelled.mp4",
        render_scale=1.0,
        frame_offset=config.frame_offset,
    )
    print("[stage] cached clustering")
    cluster_cached_embeddings(args)
    _attach_person_ids(
        paths.raw_landmarks,
        paths.clustering_dir / "track_to_person.csv",
        paths.clustered_landmarks,
    )

    _unlink(paths.gaze_on_faces)
    _unlink(paths.gaze_on_faces_summary)
    if paths.aligned_gaze.exists():
        print("[stage] gaze-on-face assignment")
        assign_gaze(
            Namespace(
                gaze=paths.aligned_gaze,
                landmarks=paths.clustered_landmarks,
                out=paths.gaze_on_faces,
                summary=paths.gaze_on_faces_summary,
                frame_tolerance=0,
            )
        )
    else:
        print("[warn] aligned gaze is missing; skipped gaze-on-face assignment")
    return paths


def parse_tracks(path: Path | None, person_labels: bool = False):
    by_frame: dict[int, list] = defaultdict(list)
    if path is None or not path.exists():
        return by_frame
    with open(path) as fh:
        for line in fh:
            fields = line.strip().split(",")
            if len(fields) < 6:
                continue
            frame, track_id = int(fields[0]), int(fields[1])
            x, y, w, h = map(float, fields[2:6])
            label = int(fields[-1]) if person_labels else track_id
            by_frame[frame].append((x, y, w, h, track_id, label, person_labels))
    return by_frame


def parse_landmarks(path: Path | None):
    by_frame: dict[int, list] = defaultdict(list)
    if path is None or not path.exists():
        return by_frame
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            points = [
                (float(row[f"{name}_x"]), float(row[f"{name}_y"]))
                for name in [
                    "left_eye",
                    "right_eye",
                    "nose",
                    "mouth_left",
                    "mouth_right",
                ]
            ]
            by_frame[int(row["frame"])].append(
                (
                    int(row["person_id"]),
                    tuple(
                        float(row[k])
                        for k in ["face_x1", "face_y1", "face_x2", "face_y2"]
                    ),
                    points,
                )
            )
    return by_frame


def parse_gaze(path: Path | None):
    if path is None or not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    values = []
    for row in rows:
        try:
            values.append(
                (
                    float(row["video_time_ms"]),
                    float(row["gaze_x"]),
                    float(row["gaze_y"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    values.sort()
    return tuple(np.asarray(v, np.float64) for v in zip(*values))


def parse_gaze_on_faces(path: Path | None):
    by_frame: dict[int, list] = defaultdict(list)
    if path is None or not path.exists():
        return by_frame
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("gaze_on_face") != "1":
                continue
            by_frame[int(row["video_frame"])].append(
                (
                    int(row["gaze_person_id"]),
                    tuple(
                        float(row[k])
                        for k in ["face_x1", "face_y1", "face_x2", "face_y2"]
                    ),
                )
            )
    return by_frame


def gaze_at_time(gaze, video_time_ms: float):
    if gaze is None or video_time_ms < gaze[0][0] or video_time_ms > gaze[0][-1]:
        return None
    return (
        float(np.interp(video_time_ms, gaze[0], gaze[1])),
        float(np.interp(video_time_ms, gaze[0], gaze[2])),
    )


def draw_preview_frame(
    frame,
    frame_index: int,
    fps: float,
    frame_offset: int,
    tracks=None,
    landmarks=None,
    gaze=None,
    gaze_on_faces=None,
):
    """Draw one GUI preview frame from already parsed stage artifacts."""
    out = frame.copy()
    mot_frame = frame_index - frame_offset
    gazed_faces = (gaze_on_faces or {}).get(frame_index, [])
    if gazed_faces:
        fill = out.copy()
        for person_id, bbox in gazed_faces:
            color = (
                PALETTE[person_id % len(PALETTE)] if person_id >= 0 else (160, 160, 160)
            )
            x1, y1, x2, y2 = map(round, bbox)
            cv2.rectangle(fill, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(fill, 0.5, out, 0.5, 0.0, out)
    for x, y, w, h, track_id, label, person_label in (tracks or {}).get(mot_frame, []):
        color_index = label if person_label and label >= 0 else track_id
        color = (
            PALETTE[color_index % len(PALETTE)] if color_index >= 0 else (160, 160, 160)
        )
        x0, y0 = max(0, round(x)), max(0, round(y))
        x1, y1 = min(out.shape[1], round(x + w)), min(out.shape[0], round(y + h))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        text = (str(label) if label >= 0 else "?") if person_label else f"t{track_id}"
        cv2.putText(
            out,
            text,
            (x0 + 4, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    for person_id, bbox, points in (landmarks or {}).get(mot_frame, []):
        color = PALETTE[person_id % len(PALETTE)] if person_id >= 0 else (255, 255, 255)
        x1, y1, x2, y2 = map(round, bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        for x, y in points:
            cv2.circle(out, (round(x), round(y)), 3, color, -1, cv2.LINE_AA)
    point = gaze_at_time(gaze, frame_index * 1000.0 / fps) if fps else None
    if point is not None:
        x, y = map(round, point)
        cv2.line(out, (x - 18, y), (x + 18, y), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(out, (x, y - 18), (x, y + 18), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(out, (x - 18, y), (x + 18, y), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(out, (x, y - 18), (x, y + 18), (0, 0, 255), 3, cv2.LINE_AA)
    return out
