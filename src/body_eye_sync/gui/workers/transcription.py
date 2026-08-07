from __future__ import annotations

from pathlib import Path
from typing import Iterator

from body_eye_sync.experiment.config import TranscriptionStep
from body_eye_sync.experiment.speech import Speech
from body_eye_sync.gui.workers.base import BaseWorker


class TranscriptionWorker(BaseWorker):
    """Runs :func:`transcribe` off the GUI thread, into a :class:`Speech`."""

    operation_name = "Transcription"

    def __init__(
        self, speech: Speech, media_path: Path, step: TranscriptionStep
    ) -> None:
        super().__init__(speech)
        self._media_path = media_path
        self._step = step

    def _items(self) -> Iterator:
        from body_eye_sync.preprocessing.timing_correction import media_duration
        from body_eye_sync.pipeline.transcription import transcribe

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
        self._target.begin_transcription()
