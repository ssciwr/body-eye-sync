from __future__ import annotations

from pathlib import Path
from typing import Iterator

from body_eye_sync.experiment.config import TranscriptionStep
from body_eye_sync.experiment.speech import Speech
from body_eye_sync.gui.workers.base import BaseWorker


class TranscriptionWorker(BaseWorker):
    """Runs :func:`transcribe` off the GUI thread, into a :class:`Speech`.

    Whisper decodes the recording front to back, so each segment is emitted via
    ``new_frame`` as it arrives and ``progress`` follows how far into the
    recording it has reached. The text is attributed to the diarized speech turns
    once the run finishes; a cancelled/failed pass leaves those turns as they
    were, without a transcript. The transcription arguments come from ``step``.
    """

    operation_name = "Transcription"

    def __init__(
        self, speech: Speech, media_path: Path, step: TranscriptionStep
    ) -> None:
        super().__init__(speech)
        self._media_path = media_path
        self._step = step

    def _items(self) -> Iterator:
        # lazy import to avoid making GUI startup slow due to module loading
        from body_eye_sync.media import media_duration
        from body_eye_sync.pipeline.transcription import transcribe

        # Whisper reports no progress of its own, so how far the last segment
        # reached into the recording stands in for it. Some containers do not
        # say how long they run, and then there is no progress to report.
        duration = media_duration(self._media_path) or 0.0
        for segment in transcribe(self._media_path, **self._step.model_dump()):
            if duration > 0:
                self.progress.emit(min(segment.end / duration, 1.0))
            yield segment

    def _accumulate(self, segment) -> None:
        self._target.add_transcription_segment(segment)

    def _finalise(self) -> None:
        self._target.finish_transcription()

    def _discard(self) -> None:
        # Drops the partial transcript and keeps the diarized speech turns it
        # was being attributed to.
        self._target.begin_transcription()
