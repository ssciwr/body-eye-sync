"""Speech transcription using faster-whisper.

This says what was said and when, on the recording's own clock. Who said it is
not something one recording can answer, and is worked out later by comparing the
experiment's recordings against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Iterator

import pandas as pd

from body_eye_sync.pipeline.model_cache import model_cache_dir

#: Columns of the transcribed segments table, one row per stretch of speech.
SEGMENT_COLUMNS = ["segment_id", "start", "end", "text"]

#: Columns of the companion per-word table.
WORD_COLUMNS = ["segment_id", "word_index", "start", "end", "word", "score"]

# These repositories publish standard Transformers checkpoints rather than the
# CTranslate2 weights faster-whisper loads. Convert the official checkpoints on
# first use instead of depending on an unverified third-party conversion.
_TRANSFORMERS_CHECKPOINTS = frozenset(
    {
        "primeline/whisper-large-v3-turbo-german",
        "primeline/whisper-large-v3-german",
    }
)
_CONVERTED_MODEL_FILES = ["preprocessor_config.json"]
_TRANSFORMERS_MODEL_FILES = ["*.json", "*.safetensors", "*.txt"]
_FASTER_WHISPER_TOKENIZER_REPO = "Systran/faster-whisper-large-v3"


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


def _converted_model_path(model_name: str) -> Path:
    """Return a cached CTranslate2 conversion of a Transformers checkpoint.

    Conversion writes into a temporary sibling and renames the finished model
    into place. An interrupted conversion therefore never looks like a usable
    cached model, and concurrent processes can safely share the winner.
    """
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import hf_hub_download, snapshot_download

    models_dir = model_cache_dir()
    cache_dir = models_dir / "ctranslate2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / model_name.replace("/", "--")
    if (target / "model.bin").is_file() and (target / "tokenizer.json").is_file():
        return target

    # Only this function owns these directories. A target without model.bin is
    # debris from an older interrupted conversion and is safe to replace.
    if target.exists():
        shutil.rmtree(target)

    temporary_root = Path(tempfile.mkdtemp(prefix=".convert-", dir=cache_dir))
    converted = temporary_root / "model"
    try:
        source = snapshot_download(
            repo_id=model_name,
            cache_dir=models_dir / "transformers",
            allow_patterns=_TRANSFORMERS_MODEL_FILES,
        )
        converter = TransformersConverter(
            source,
            copy_files=_CONVERTED_MODEL_FILES,
            load_as_float16=True,
            low_cpu_mem_usage=True,
        )
        converter.convert(str(converted), quantization="float16")
        # The primeLine repositories publish the tokenizer as vocab.json plus
        # merges.txt, while faster-whisper loads a tokenizer.json. They use the
        # unchanged multilingual Whisper vocabulary, so copy the canonical
        # faster-whisper tokenizer into the converted model.
        tokenizer = hf_hub_download(
            repo_id=_FASTER_WHISPER_TOKENIZER_REPO,
            filename="tokenizer.json",
            cache_dir=models_dir / "transformers",
        )
        shutil.copyfile(tokenizer, converted / "tokenizer.json")
        try:
            converted.rename(target)
        except FileExistsError:
            # Another process completed the same conversion first.
            if not (
                (target / "model.bin").is_file()
                and (target / "tokenizer.json").is_file()
            ):
                raise
        return target
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _faster_whisper_model(model_name: str) -> str:
    """Resolve supported Transformers checkpoints for faster-whisper."""
    if model_name in _TRANSFORMERS_CHECKPOINTS:
        return str(_converted_model_path(model_name))
    return model_name


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
    from faster_whisper import WhisperModel

    model = WhisperModel(
        _faster_whisper_model(model_name),
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
