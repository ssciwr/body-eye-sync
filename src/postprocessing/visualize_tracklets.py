#!/usr/bin/env python3
"""Track two videos, cluster their tracklets, and display person-ID overlays.

Pipeline results are saved and reloaded before clustering. Annotated videos are
written to the configured output directory while they are displayed.

Usage::

    python src/postprocessing/visualize_tracklets.py VIDEO_1 VIDEO_2
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in {None, ""}:
    # src/postprocessing is not a package, so add src when run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body_eye_sync.experiment.video import Video  # noqa: E402
from body_eye_sync.pipeline.face_detection import detect_faces  # noqa: E402
from body_eye_sync.pipeline.object_tracking import (  # noqa: E402
    BoundingBox,
    detect_tracklets,
)
from body_eye_sync.postprocessing.tracklets_clustering import (  # noqa: E402
    ClusteringResult,
    TrackletClusteringInput,
    cluster_tracklets_from_input,
)

logger = logging.getLogger(__name__)

# MP4 sample entry for H.264; the default writer uses libx264 for portability.
CHROME_COMPATIBLE_CODEC = "avc1"


def run_video_pipeline(
    video_path: str | Path,
    video_id: str,
    *,
    detector: str = "yolo26m",
    reid: str = "osnet_x1_0_msmt17",
    tracker: str = "botsort",
    face_model: str = "antelopev2",
    face_det_size: int = 640,
    face_det_thresh: float = 0.5,
    embeddings_per_track: int = 32,
    device: str | None = None,
) -> Video:
    """Run object tracking and face detection, storing both on ``video``."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")

    video = Video(id=video_id, path=path)

    logger.info("Tracking objects in %s", path)
    video.begin_object_tracking(embeddings_per_track=embeddings_per_track)
    tracked_frames = 0
    for frame in detect_tracklets(
        path,
        detector=detector,
        reid=reid,
        tracker=tracker,
        device=device,
    ):
        video.add_object_tracking_frame(frame)
        tracked_frames += 1
        if tracked_frames % 30 == 0:
            logger.info("Tracked %d frames in %s", tracked_frames, path.name)
    video.finish_object_tracking()
    if video.data is None or video.data.empty:
        raise RuntimeError(f"object tracking found no people in {path}")

    logger.info("Detecting faces in %s", path)
    video.begin_face_detection(embeddings_per_track=embeddings_per_track)
    face_frames = 0
    for result in detect_faces(
        path,
        video.all_boxes_by_frame(),
        model_name=face_model,
        det_size=face_det_size,
        det_thresh=face_det_thresh,
    ):
        video.add_face_detection_frame(result)
        face_frames += 1
        if face_frames % 30 == 0:
            logger.info("Checked %d frames for faces in %s", face_frames, path.name)
    video.finish_face_detection()
    logger.info(
        "Prepared %s: %d tracklets",
        video_id,
        video.data["track_id"].nunique(),
    )
    return video


def save_video_results(video: Video, output_dir: str | Path) -> Path:
    """Save one completed video pipeline and return its results directory."""
    directory = Path(output_dir) / "results" / video.id
    video.save(directory)
    logger.info("Saved pipeline results for %s to %s", video.id, directory)
    return directory


def load_video_results(
    video_id: str,
    video_path: str | Path,
    results_dir: str | Path,
) -> Video:
    """Load a persisted video pipeline from ``results_dir``."""
    directory = Path(results_dir)
    results_path = directory / "results.parquet"
    if not results_path.is_file():
        raise FileNotFoundError(f"saved video results not found: {results_path}")

    video = Video(id=video_id, path=video_path)
    video.load(directory)
    if video.data is None or video.data.empty:
        raise RuntimeError(f"saved video results contain no tracklets: {directory}")
    logger.info("Loaded pipeline results for %s from %s", video_id, directory)
    return video


def cluster_videos(
    videos: Sequence[Video],
    *,
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_body_detections: int = 3,
    debug: bool = False,
) -> ClusteringResult:
    """Cluster all tracklets from ``videos`` into shared person IDs."""
    if len(videos) != 2:
        raise ValueError(f"expected exactly two videos, got {len(videos)}")

    inputs = [
        TrackletClusteringInput(
            track_ids=(
                [int(track_id) for track_id in video.data["track_id"].drop_duplicates()]
                if video.data is not None
                else []
            ),
            face_embeddings=video.face_embeddings,
            body_embeddings=video.body_embeddings,
            video_id=video.id,
        )
        for video in videos
    ]
    return cluster_tracklets_from_input(
        inputs,
        face_distance_threshold=face_distance_threshold,
        body_distance_threshold=body_distance_threshold,
        min_face_detections=min_face_detections,
        min_body_detections=min_body_detections,
        debug=debug,
    )


def visualize_saved_videos(
    video_paths: Sequence[str | Path],
    results_dirs: Sequence[str | Path],
    output_dir: str | Path = Path("postprocessing_output/videos"),
    *,
    face_distance_threshold: float = 0.6,
    body_distance_threshold: float = 0.15,
    min_face_detections: int = 1,
    min_body_detections: int = 3,
    codec: str = CHROME_COMPATIBLE_CODEC,
    debug: bool = False,
) -> list[Path]:
    """Load, cluster, display, and save two previously processed videos.

    ``video_paths`` and ``results_dirs`` must have the same order. Each results
    directory must contain the ``results.parquet`` file written by ``Video.save()``.
    Annotated videos default to ``postprocessing_output/videos``.
    """
    if len(video_paths) != 2:
        raise ValueError(f"expected exactly two video paths, got {len(video_paths)}")
    if len(results_dirs) != 2:
        raise ValueError(
            f"expected exactly two results directories, got {len(results_dirs)}"
        )
    if Path(video_paths[0]).resolve() == Path(video_paths[1]).resolve():
        raise ValueError("the two input video paths must be different")
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters")

    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    loaded_videos = [
        load_video_results(
            Path(results_dir).name,
            video_path,
            results_dir,
        )
        for video_path, results_dir in zip(video_paths, results_dirs)
    ]
    clustering = cluster_videos(
        loaded_videos,
        face_distance_threshold=face_distance_threshold,
        body_distance_threshold=body_distance_threshold,
        min_face_detections=min_face_detections,
        min_body_detections=min_body_detections,
        debug=debug,
    )

    output_paths: list[Path] = []
    try:
        for video_path, video in zip(video_paths, loaded_videos):
            output_path = destination_dir / f"{video.id}_annotated.mp4"
            interrupted = not display_video(
                video_path,
                video,
                clustering.tracklet_id_to_person_id,
                output_path=output_path,
                codec=codec,
            )
            if interrupted:
                logger.info("Playback interrupted")
                break
            output_paths.append(output_path)
    finally:
        cv2.destroyAllWindows()
    return output_paths


@lru_cache(maxsize=None)
def _person_color(person_id: int) -> tuple[int, int, int]:
    """Return a stable, high-contrast BGR color for a person ID."""
    hue = (person_id * 47) % 180
    sample = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    return tuple(int(value) for value in cv2.cvtColor(sample, cv2.COLOR_HSV2BGR)[0, 0])


def _person_id_for_track(
    person_ids: Mapping[tuple[str, int], int],
    video_id: str,
    track_id: int,
) -> int:
    key = (video_id, track_id)
    try:
        return person_ids[key]
    except KeyError as error:
        raise RuntimeError(
            f"no person ID for {video_id!r} track_id {track_id}"
        ) from error


class _ChromeCompatibleWriter:
    """Write H.264 video in an MP4 container using PyAV."""

    def __init__(
        self,
        path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        import av

        width, height = frame_size
        # H.264's broadly supported yuv420p format requires even dimensions.
        self._width = width + width % 2
        self._height = height + height % 2
        fps_fraction = Fraction.from_float(fps).limit_denominator(65_535)
        self._time_base = Fraction(1, 1) / fps_fraction
        self._frame_index = 0
        self._closed = False

        self._container = av.open(
            str(path),
            "w",
            options={"movflags": "+faststart"},
        )
        try:
            self._stream = self._container.add_stream(
                "libx264",
                rate=fps_fraction,
                options={"preset": "veryfast", "crf": "23"},
            )
            self._stream.width = self._width
            self._stream.height = self._height
            self._stream.pix_fmt = "yuv420p"
            self._stream.time_base = self._time_base
        except BaseException:
            self._container.close()
            raise

    def write(self, frame: np.ndarray) -> None:
        import av

        if self._closed:
            raise RuntimeError("cannot write to a closed video writer")

        frame_height, frame_width = frame.shape[:2]
        if (frame_width, frame_height) != (self._width, self._height):
            padded = np.zeros(
                (self._height, self._width, frame.shape[2]),
                dtype=frame.dtype,
            )
            padded[:frame_height, :frame_width] = frame
            frame = padded

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = self._frame_index
        video_frame.time_base = self._time_base
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)
        self._frame_index += 1

    def release(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        finally:
            self._container.close()


def annotate_frame(
    frame: np.ndarray,
    boxes: Sequence[BoundingBox],
    person_ids: Mapping[tuple[str, int], int],
    video_id: str,
) -> np.ndarray:
    """Draw object boxes labeled with their clustered person IDs."""
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    font_scale = max(0.45, min(frame_height, frame_width) / 900.0)
    thickness = max(1, round(min(frame_height, frame_width) / 600.0))

    for box in boxes:
        person_id = _person_id_for_track(person_ids, video_id, int(box.track_id))
        x1 = max(0, min(frame_width - 1, int(round(box.x1))))
        y1 = max(0, min(frame_height - 1, int(round(box.y1))))
        x2 = max(0, min(frame_width - 1, int(round(box.x2))))
        y2 = max(0, min(frame_height - 1, int(round(box.y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        color = _person_color(person_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness + 1)

        label = f"Person {person_id}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_top = y1 - label_height - baseline - 4
        if label_top < 0:
            label_top = y2 + 4
        label_bottom = min(
            frame_height - 1,
            label_top + label_height + baseline + 4,
        )
        if x1 + label_width + 6 <= frame_width:
            label_left = x1
        else:
            label_left = max(0, x2 - label_width - 6)
        label_right = min(frame_width - 1, label_left + label_width + 6)
        cv2.rectangle(
            annotated,
            (label_left, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_left + 3, label_top + label_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def display_video(
    video_path: str | Path,
    video: Video,
    person_ids: Mapping[tuple[str, int], int],
    output_path: str | Path | None = None,
    codec: str = CHROME_COMPATIBLE_CODEC,
) -> bool:
    """Display ``video`` with person-ID overlays and optionally save it.

    Returns ``False`` when playback is interrupted by ``q``, ``Esc``, or closing
    the window; otherwise returns ``True`` after reaching the last frame.
    """
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters")

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video for display: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    delay = max(1, round(1000 / fps))
    window_name = f"Person IDs: {path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    destination = Path(output_path) if output_path is not None else None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            boxes = video.boxes_for_frame(frame_index)
            if boxes:
                frame = annotate_frame(frame, boxes, person_ids, video.id)
            if destination is not None and writer is None:
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                # OpenCV's mp4v commonly emits MPEG-4 Part 2, which Chrome rejects.
                if codec.lower() == CHROME_COMPATIBLE_CODEC:
                    try:
                        writer = _ChromeCompatibleWriter(
                            destination,
                            fps,
                            (frame_width, frame_height),
                        )
                    except Exception as error:
                        raise OSError(
                            f"could not create output video: {destination}"
                        ) from error
                else:
                    writer = cv2.VideoWriter(
                        str(destination),
                        cv2.VideoWriter_fourcc(*codec),
                        fps,
                        (frame_width, frame_height),
                    )
                    if not writer.isOpened():
                        raise OSError(f"could not create output video: {destination}")
            if writer is not None:
                writer.write(frame)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(delay) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                return False
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyWindow(window_name)

    if destination is not None:
        logger.info("Saved %d annotated frames to %s", frame_index, destination)
    else:
        logger.info("Displayed %d frames from %s", frame_index, path.name)
    return True


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run object/face detection on two videos, cluster their tracklets, "
            "and display and save object boxes labeled with person IDs."
        )
    )
    parser.add_argument("video_paths", nargs=2, type=Path, metavar="VIDEO")
    parser.add_argument(
        "--processed-dirs",
        nargs=2,
        type=Path,
        metavar="PROCESSED_DIR",
        help=(
            "optional directories containing previously processed video results "
            "(otherwise the videos will be processed again)"
        ),
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("postprocessing_output"),
        help="directory for saved pipeline results and annotated videos",
    )
    parser.add_argument(
        "--codec",
        default=CHROME_COMPATIBLE_CODEC,
        help=(
            "four-character codec selector; avc1 is encoded as H.264 "
            "for browser playback (default: avc1)"
        ),
    )
    parser.add_argument("--detector", default="yolo26m")
    parser.add_argument("--reid", default="osnet_x1_0_msmt17")
    parser.add_argument("--tracker", default="botsort")
    parser.add_argument("--face-model", default="antelopev2")
    parser.add_argument("--face-det-size", type=int, default=640)
    parser.add_argument("--face-det-thresh", type=float, default=0.5)
    parser.add_argument("--embeddings-per-track", type=int, default=32)
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. 0, cpu, or mps",
    )
    parser.add_argument("--face-distance-threshold", type=float, default=0.6)
    parser.add_argument("--body-distance-threshold", type=float, default=0.15)
    parser.add_argument("--min-face-detections", type=int, default=1)
    parser.add_argument("--min-body-detections", type=int, default=3)
    parser.add_argument("--debug-clustering", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    video_paths = [path.resolve() for path in args.video_paths]
    if video_paths[0] == video_paths[1]:
        raise ValueError("the two input video paths must be different")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    if args.processed_dirs is not None:
        if len(args.processed_dirs) != 2:
            raise ValueError("exactly two processed directories must be provided")

    output_dir = args.output_dir.resolve()
    videos: list[Video] = []
    results_dirs: list[Path] = []

    if args.processed_dirs is None:
        for index, path in enumerate(video_paths, start=1):
            video = run_video_pipeline(
                path,
                f"video_{index}",
                detector=args.detector,
                reid=args.reid,
                tracker=args.tracker,
                face_model=args.face_model,
                face_det_size=args.face_det_size,
                face_det_thresh=args.face_det_thresh,
                embeddings_per_track=args.embeddings_per_track,
                device=args.device,
            )
            videos.append(video)
            results_dirs.append(save_video_results(video, output_dir))
    else:
        results_dirs = [dir.resolve() for dir in args.processed_dirs]

    visualize_saved_videos(
        video_paths,
        results_dirs,
        output_dir / "videos",
        face_distance_threshold=args.face_distance_threshold,
        body_distance_threshold=args.body_distance_threshold,
        min_face_detections=args.min_face_detections,
        min_body_detections=args.min_body_detections,
        codec=args.codec,
        debug=args.debug_clustering,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
