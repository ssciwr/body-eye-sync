"""Speech transcription using Whisper-family models.

This says what was said and when, on the recording's own clock. Who said it is
not something one recording can answer, and is worked out later by comparing the
experiment's recordings against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from body_eye_sync.pipeline.model_cache import model_cache_dir
from body_eye_sync.pipeline.whisper_model import (
    download_whisper_model,
    resolve_whisper_model,
)

#: Columns of the transcribed segments table, one row per stretch of speech.
SEGMENT_COLUMNS = ["segment_id", "start", "end", "text"]

#: Columns of the companion per-word table.
WORD_COLUMNS = ["segment_id", "word_index", "start", "end", "word", "score"]


@dataclass
class Word:
    """One transcribed word, timed on the recording's own clock."""

    start: float
    end: float
    word: str
    score: float


@dataclass
class TranscriptSegment:
    """One stretch of speech as Whisper segmented it.

    Whisper's segmentation follows sentence and pause structure, so a segment is
    a natural unit of speech but not necessarily one person's turn.
    """

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def _is_crisper_whisper(model_name: str) -> bool:
    return model_name.startswith("nyralabs/CrisperWhisper2.0_")


def _transcribe_crisper(
    audio_path: str | Path,
    model_name: str,
    language: str | None,
    device: str,
    compute_type: str,
) -> Iterator[TranscriptSegment]:
    """Run a CrisperWhisper 2 model in its verbatim mode."""
    if language is None:
        raise ValueError("CrisperWhisper requires an explicit language")

    from crisperwhisper import CrisperWhisperModel

    source = download_whisper_model(model_name)
    if compute_type == "default":
        import torch

        compute_type = (
            "float16"
            if device == "cuda" or (device == "auto" and torch.cuda.is_available())
            else "float32"
        )
    options = {
        "backend": "transformers",
        "device": device,
        "cache_dir": model_cache_dir(),
        "compute_type": compute_type,
    }
    model = CrisperWhisperModel(str(source), **options)
    result = model.transcribe(
        str(audio_path),
        language=language,
        mode="verbatim",
        word_timestamps=True,
    )
    text = result.text.strip()
    if not text:
        return

    words = [
        Word(float(word.start), float(word.end), word.word, float("nan"))
        for word in (result.words or [])
    ]
    start = words[0].start if words else 0.0
    end = words[-1].end if words else float(result.duration)
    yield TranscriptSegment(start, end, text, words)


def transcribe(
    audio_path: str | Path,
    model_name: str,
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    device: str = "auto",
    compute_type: str = "default",
) -> Iterator[TranscriptSegment]:
    """Transcribe a whole recording, yielding a result per Whisper segment.

    ``language`` is an ISO 639-1 code such as ``"de"``; ``None`` detects it from
    the first 30 seconds. ``vad_filter`` skips silent stretches, which both
    speeds the pass up and suppresses the text Whisper otherwise invents to fill
    silence. Segments are yielded as they are decoded, so callers can show
    progress; stop iterating to cancel.

    ``device`` is ``"auto"``, ``"cpu"`` or ``"cuda"``; ``"auto"`` uses a GPU when
    the machine has one. CTranslate2 loads its CUDA libraries by name the first
    time it uses a GPU, so a machine whose CUDA does not match the one it was
    built against fails here rather than transcribing more slowly on the CPU.
    """
    if _is_crisper_whisper(model_name):
        yield from _transcribe_crisper(
            audio_path, model_name, language, device, compute_type
        )
        return

    from faster_whisper import WhisperModel

    model = WhisperModel(
        resolve_whisper_model(model_name),
        device=device,
        compute_type=compute_type,
        download_root=str(model_cache_dir()),
    )
    segments, _ = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=True,
    )
    for segment in segments:
        words = [
            Word(float(w.start), float(w.end), w.word, float(w.probability))
            for w in (segment.words or [])
        ]
        yield TranscriptSegment(
            float(segment.start), float(segment.end), segment.text.strip(), words
        )


def transcript_to_dataframes(
    transcript: Iterable[TranscriptSegment],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack a transcript into a segments table and a per-word table.

    The segments are numbered by ``segment_id`` in the order they were spoken,
    with :data:`SEGMENT_COLUMNS`; each word carries the ``segment_id`` it came
    from and its position within it, with :data:`WORD_COLUMNS`.
    """
    ordered = sorted(transcript, key=lambda s: (s.start, s.end))
    segment_rows = [(index, s.start, s.end, s.text) for index, s in enumerate(ordered)]
    word_rows = [
        (index, position, w.start, w.end, w.word.strip(), w.score)
        for index, segment in enumerate(ordered)
        for position, w in enumerate(segment.words)
    ]

    segments = pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS)
    words = pd.DataFrame(word_rows, columns=WORD_COLUMNS)
    return (
        segments.astype({"segment_id": int, "start": float, "end": float, "text": str}),
        words.astype(
            {
                "segment_id": int,
                "word_index": int,
                "start": float,
                "end": float,
                "word": str,
                "score": float,
            }
        ),
    )
