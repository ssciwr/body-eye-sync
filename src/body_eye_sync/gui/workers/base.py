from __future__ import annotations

import threading
import traceback
from typing import Iterator

from qtpy.QtCore import QObject, Signal, Slot

from body_eye_sync.experiment.video import Video


class BaseWorker(QObject):
    """Runs a video pipeline off the GUI thread, into a :class:`Video`.

    Subclasses supply the per-run work: :meth:`_items` yields each computed
    frame/result, :meth:`_accumulate` stores one into the video, :meth:`_finalise`
    folds the accumulated results once the run completes, and :meth:`_discard`
    rolls the video back if the run is cancelled or fails. Each item is emitted
    via ``new_frame`` so the GUI can draw it live; ``finished`` (after
    :meth:`_finalise`) or ``cancelled`` (after :meth:`_discard`) fires once the
    run ends, and any exception is reported via ``failed`` with a traceback (also
    after :meth:`_discard`). ``operation_name`` labels the run for the GUI.
    """

    #: Human-readable name of the operation, for the GUI's status/error messages.
    operation_name: str = ""

    new_frame = Signal(object)
    finished = Signal()
    failed = Signal(str, str)
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
            for item in self._items():
                if self._cancel.is_set():
                    self._discard()
                    self.cancelled.emit()
                    return
                self._accumulate(item)
                self.new_frame.emit(item)
        except Exception as exc:
            self._discard()
            self.failed.emit(str(exc), traceback.format_exc())
            return
        if self._cancel.is_set():
            self._discard()
            self.cancelled.emit()
        else:
            self._finalise()
            self.finished.emit()

    def _items(self) -> Iterator:
        """Yield each computed frame/result. Lazy-import the pipeline here."""
        raise NotImplementedError

    def _accumulate(self, item) -> None:
        """Store one computed item into the video."""
        raise NotImplementedError

    def _finalise(self) -> None:
        """Fold the accumulated items into the video's stored data."""
        raise NotImplementedError

    def _discard(self) -> None:
        """Roll the video back when the run is cancelled or fails."""
        raise NotImplementedError
