from __future__ import annotations

import threading

from qtpy.QtCore import QObject, Signal, Slot

from body_eye_sync.experiment.video import Video


class TrackingWorker(QObject):
    """Runs :func:`detect_tracklets` off the GUI thread, into a :class:`Video`.

    Each tracked frame is appended to the :class:`Video` as it is computed and
    emitted via ``new_frame`` so the GUI can draw it live. ``finished`` (after
    the results are folded into the video) or ``cancelled`` fires once the run
    ends.
    """

    new_frame = Signal(object)
    finished = Signal()
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, video: Video) -> None:
        super().__init__()
        self._video = video
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        try:
            # lazy import to avoid making GUI startup slow due to module loading
            from body_eye_sync.pipeline.tracking import detect_tracklets

            for frame in detect_tracklets(self._video.video_path):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                self._video.add_frame(frame)
                self.new_frame.emit(frame)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        if self._cancel.is_set():
            self.cancelled.emit()
        else:
            self._video.finish_tracking()
            self.finished.emit()
