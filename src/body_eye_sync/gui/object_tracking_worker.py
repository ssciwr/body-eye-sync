from __future__ import annotations

from typing import Iterator

from body_eye_sync.gui.base_worker import BaseWorker


class ObjectTrackingWorker(BaseWorker):
    """Runs :func:`detect_tracklets` off the GUI thread, into a :class:`Video`.

    Each tracked frame is appended to the :class:`Video` as it is computed and
    emitted via ``new_frame`` so the GUI can draw it live; the results are folded
    into the video once the run finishes, or discarded if it is cancelled/fails.
    """

    operation_name = "Object tracking"

    def _items(self) -> Iterator:
        # lazy import to avoid making GUI startup slow due to module loading
        from body_eye_sync.pipeline.object_tracking import detect_tracklets

        return detect_tracklets(self._video.video_path)

    def _accumulate(self, frame) -> None:
        self._video.add_object_tracking_frame(frame)

    def _finalise(self) -> None:
        self._video.finish_object_tracking()

    def _discard(self) -> None:
        self._video.discard_object_tracking()
