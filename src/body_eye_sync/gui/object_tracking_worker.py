from __future__ import annotations

from typing import Iterator

from body_eye_sync.experiment.config import ObjectTrackingStep
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.base_worker import BaseWorker


class ObjectTrackingWorker(BaseWorker):
    """Runs :func:`detect_tracklets` off the GUI thread, into a :class:`Video`.

    Each tracked frame is appended to the :class:`Video` as it is computed and
    emitted via ``new_frame`` so the GUI can draw it live; the results are folded
    into the video once the run finishes, or discarded if it is cancelled/fails.
    The tracking arguments come from ``step``.
    """

    operation_name = "Object tracking"

    def __init__(self, video: Video, step: ObjectTrackingStep) -> None:
        super().__init__(video)
        self._step = step

    def _items(self) -> Iterator:
        # lazy import to avoid making GUI startup slow due to module loading
        from body_eye_sync.pipeline.object_tracking import detect_tracklets

        # embeddings_per_track drives the post-pass reduction in Video, not the
        # detector call, so it is not forwarded to detect_tracklets.
        return detect_tracklets(
            self._video.video_path,
            **self._step.model_dump(exclude={"embeddings_per_track"}),
        )

    def _accumulate(self, frame) -> None:
        self._video.add_object_tracking_frame(frame)

    def _finalise(self) -> None:
        self._video.finish_object_tracking()

    def _discard(self) -> None:
        self._video.discard_object_tracking()
