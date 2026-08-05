from __future__ import annotations

from pathlib import Path
from typing import Iterator

from body_eye_sync.experiment.config import DiarizationStep
from body_eye_sync.experiment.speech import Speech
from body_eye_sync.gui.workers.base import BaseWorker
from body_eye_sync.pipeline.diarization import SpeakerEmbedding

#: How much of the reported progress the diarization pass itself accounts for
#: when voice embeddings are collected after it.
_DIARIZATION_SHARE = 0.5


class DiarizationWorker(BaseWorker):
    """Runs :func:`diarize` off the GUI thread, into a :class:`Speech`.

    sherpa-onnx works through the whole recording before it says who spoke when,
    so the speech turns all arrive at the end of the pass, with ``progress``
    reporting how far through it is meanwhile. If the step asks for voice
    embeddings, a second pass embeds each turn once diarization has settled what
    the turns are; both passes are reported as one run, and both yield their
    results through :meth:`_items` so a cancellation lands between two items and
    rolls the whole run back. The diarization arguments come from ``step``.
    """

    operation_name = "Diarization"

    def __init__(self, speech: Speech, media_path: Path, step: DiarizationStep) -> None:
        super().__init__(speech)
        self._media_path = media_path
        self._step = step
        self._wants_embeddings = step.embeddings_per_speaker > 0

    def _items(self) -> Iterator:
        # lazy import to avoid making GUI startup slow due to module loading
        from body_eye_sync.pipeline.diarization import (
            diarize,
            extract_speaker_embeddings,
        )

        # embeddings_per_speaker drives the reduction in Speech, not the
        # diarizer call, so it is not forwarded to diarize.
        yield from diarize(
            self._media_path,
            **self._step.model_dump(exclude={"embeddings_per_speaker"}),
            progress=self._report,
        )
        if not self._wants_embeddings or self._cancel.is_set():
            return

        # A turn can only be embedded once diarization has decided what the
        # turns are, so they are folded in here rather than in _finalise, and
        # _finalise only has the embeddings left to fold in.
        self._target.finish_diarization()
        segments = self._target.segments
        for embedding in extract_speaker_embeddings(
            self._media_path,
            segments,
            embedding_model=self._step.embedding_model,
        ):
            self.progress.emit(
                _DIARIZATION_SHARE
                + (1.0 - _DIARIZATION_SHARE)
                * (embedding.segment_id + 1)
                / max(len(segments), 1)
            )
            yield embedding

    def _report(self, fraction: float) -> bool:
        """Report how far the diarization pass is; False stops it."""
        share = _DIARIZATION_SHARE if self._wants_embeddings else 1.0
        self.progress.emit(fraction * share)
        return not self._cancel.is_set()

    def _accumulate(self, item) -> None:
        # The pass yields speech turns first, then one embedding per turn.
        if isinstance(item, SpeakerEmbedding):
            self._target.add_speaker_embedding(item)
        else:
            self._target.add_diarization_segment(item)

    def _finalise(self) -> None:
        if self._wants_embeddings:
            self._target.finish_speaker_embeddings()
        else:
            self._target.finish_diarization()

    def _discard(self) -> None:
        # Diarization is the base pass, so a cancelled/failed run leaves the
        # recording with no speech results at all.
        self._target.clear()
