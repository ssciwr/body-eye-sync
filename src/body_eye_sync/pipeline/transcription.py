"""Speech transcription using faster-whisper, attributed to diarized speakers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from body_eye_sync.pipeline.diarization import SpeakerSegment
from body_eye_sync.pipeline.model_cache import model_cache_dir

#: The column transcription adds to the diarized speech turns.
TRANSCRIPT_COLUMNS = ["text"]

#: Columns of the companion per-word table.
WORD_COLUMNS = ["segment_id", "word_index", "start", "end", "word", "score", "speaker"]

#: ``segment_id`` given to words that fall outside every diarized speech turn.
UNASSIGNED = -1


@dataclass
class Word:
    """One transcribed word, timed on the recording's own clock."""

    start: float
    end: float
    word: str
    score: float


@dataclass
class TranscriptSegment:
    """One stretch of speech as Whisper segmented it, before speaker attribution.

    Whisper's own segmentation follows sentence and pause structure and does not
    line up with diarized speech turns, so these are an intermediate result:
    :func:`assign_words` redistributes their :attr:`words` onto the turns.
    """

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


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
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name,
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


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    """Seconds the two intervals share; zero if they are disjoint."""
    return max(0.0, min(end, other_end) - max(start, other_start))


def _best_segment(word: Word, segments: Sequence[SpeakerSegment]) -> int:
    """Index of the speech turn a word belongs to, or :data:`UNASSIGNED`.

    A word goes to the turn it overlaps most in time. Words that overlap nothing
    -- Whisper text over a stretch diarization heard no speech in -- are left
    unassigned rather than forced onto whichever turn happens to be nearest.
    """
    best_index, best_overlap = UNASSIGNED, 0.0
    for index, segment in enumerate(segments):
        shared = _overlap(word.start, word.end, segment.start, segment.end)
        if shared > best_overlap:
            best_index, best_overlap = index, shared
    return best_index


def assign_words(
    transcript: Iterable[TranscriptSegment], segments: Sequence[SpeakerSegment]
) -> tuple[list[str], pd.DataFrame]:
    """Attribute transcribed words to diarized speech turns.

    ``segments`` must be in the ``segment_id`` order
    :func:`~body_eye_sync.pipeline.diarization.segments_to_dataframe` assigned,
    i.e. sorted by start time. Returns the joined text for each turn -- empty
    for turns no word landed in -- along with the companion per-word table with
    :data:`WORD_COLUMNS`. Words matching no turn are kept in that table with
    ``segment_id`` and ``speaker`` set to :data:`UNASSIGNED`, so the transcript
    never silently loses text.
    """
    words = [word for segment in transcript for word in segment.words]
    words.sort(key=lambda w: (w.start, w.end))

    texts: list[list[str]] = [[] for _ in segments]
    rows = []
    for word in words:
        index = _best_segment(word, segments)
        text = word.word.strip()
        if index == UNASSIGNED:
            word_index, speaker = UNASSIGNED, UNASSIGNED
        else:
            word_index, speaker = len(texts[index]), segments[index].speaker
            texts[index].append(text)
        rows.append(
            (index, word_index, word.start, word.end, text, word.score, speaker)
        )

    table = pd.DataFrame(rows, columns=WORD_COLUMNS)
    if table.empty:
        table = table.astype(
            {
                "segment_id": int,
                "word_index": int,
                "start": float,
                "end": float,
                "word": str,
                "score": float,
                "speaker": int,
            }
        )
    return [" ".join(parts) for parts in texts], table


def transcript_to_dataframe(
    transcript: Iterable[TranscriptSegment], segments: Sequence[SpeakerSegment]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Text keyed on ``segment_id``, plus the companion per-word table.

    The first frame has columns ``segment_id`` and :data:`TRANSCRIPT_COLUMNS`,
    ready to left-merge onto the diarized turns.
    """
    texts, words = assign_words(transcript, segments)
    text_frame = pd.DataFrame(
        {"segment_id": np.arange(len(segments), dtype=int), "text": texts}
    )
    return text_frame, words
