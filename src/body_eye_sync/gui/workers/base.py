from __future__ import annotations

import threading
import time
import traceback
from typing import Iterator

from qtpy.QtCore import QObject, Signal, Slot


class BaseWorker(QObject):
    """Runs one pipeline step off the GUI thread, into the results it computes.

    ``target`` is what the step writes into: a
    :class:`~body_eye_sync.experiment.video.Video` for the video stages, a
    :class:`~body_eye_sync.experiment.speech.Speech` for the speech ones.
    Subclasses supply the per-run work: :meth:`_items` yields each computed
    frame/result, :meth:`_accumulate` stores one into the target, :meth:`_finalise`
    folds the accumulated results once the run completes, and :meth:`_discard`
    rolls the target back if the run is cancelled or fails. Items are emitted
    (at most one every :data:`LIVE_FRAME_INTERVAL_SECS` seconds) via ``new_frame``
    so the GUI can show them live, and a step that can say how far through it is
    reports that as a fraction via ``progress``; ``finished`` (after :meth:`_finalise`)
    or ``cancelled`` (after :meth:`_discard`) fires once the run ends,
    and any exception is reported via ``failed`` with a traceback (also after :meth:`_discard`).
    ``operation_name`` labels the run for the GUI.
    """

    LIVE_FRAME_INTERVAL_SECS = 1 / 30

    #: Human-readable name of the operation, for the GUI's status/error messages.
    operation_name: str = ""

    new_frame = Signal(object)
    progress = Signal(float)
    finished = Signal()
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, target) -> None:
        super().__init__()
        self._target = target
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        item = None
        emitted = True
        last_emit = -float("inf")
        try:
            for item in self._items():
                if self._cancel.is_set():
                    self._discard()
                    self.cancelled.emit()
                    return
                self._accumulate(item)
                now = time.monotonic()
                emitted = now - last_emit >= self.LIVE_FRAME_INTERVAL_SECS
                if emitted:
                    last_emit = now
                    self.new_frame.emit(item)
        except Exception as exc:
            self._discard()
            self.failed.emit(str(exc), traceback.format_exc())
            return
        if self._cancel.is_set():
            self._discard()
            self.cancelled.emit()
        else:
            if not emitted:
                self.new_frame.emit(item)
            self._finalise()
            self.finished.emit()

    def _items(self) -> Iterator:
        """Yield each computed frame/result. Lazy-import the pipeline here."""
        raise NotImplementedError

    def _accumulate(self, item) -> None:
        """Store one computed item into the target."""
        raise NotImplementedError

    def _finalise(self) -> None:
        """Fold the accumulated items into the target's stored results."""
        raise NotImplementedError

    def _discard(self) -> None:
        """Roll the target back when the run is cancelled or fails."""
        raise NotImplementedError
