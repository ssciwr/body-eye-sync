"""Speaker diarization -- who spoke when -- using sherpa-onnx."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from body_eye_sync.media import SAMPLE_RATE, load_audio
from body_eye_sync.pipeline.model_cache import model_cache_dir

# Hugging Face repo holding all the sherpa-onnx speaker embedding models
_EMBEDDING_REPO = "csukuangfj/speaker-embedding-models"

# Segmentation models get a repo each, every one holding a ``model.onnx``.
_SEGMENTATION_REPO = "csukuangfj/{name}"

# The file every segmentation repo stores its full-precision model under.
_SEGMENTATION_FILE = "model.onnx"

# The columns diarization contributes: one row per speech turn.
SEGMENT_COLUMNS = ["segment_id", "start", "end", "speaker"]

# Speech turns shorter than this have too little voice in them to embed.
MIN_EMBEDDING_DURATION = 0.2


@dataclass
class SpeakerEmbedding:
    """A voice embedding computed from one speech turn."""

    segment_id: int
    speaker: int
    duration: float
    embedding: np.ndarray


@dataclass
class SpeakerSegment:
    """One speech turn: ``[start, end)`` in seconds, attributed to ``speaker``."""

    start: float
    end: float
    speaker: int


def segments_to_dataframe(segments: Iterable[SpeakerSegment]) -> pd.DataFrame:
    """Stack speech turns into a DataFrame with :data:`SEGMENT_COLUMNS`."""
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.speaker))
    rows = [(index, s.start, s.end, s.speaker) for index, s in enumerate(ordered)]
    data = (
        np.asarray(rows, dtype=float) if rows else np.empty((0, len(SEGMENT_COLUMNS)))
    )
    frame = pd.DataFrame(data, columns=SEGMENT_COLUMNS)
    return frame.astype({"segment_id": int, "speaker": int})


def _download(repo_id: str, filename: str) -> str:
    """Fetch a model file from the Hugging Face Hub into the shared model cache."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, cache_dir=str(model_cache_dir()))


def _segmentation_model_path(model_name: str | Path) -> str:
    """Resolve a segmentation model name to a local ``.onnx`` file."""
    path = Path(model_name)
    if path.is_file():
        return str(path)
    if path.is_dir():
        return str(path / _SEGMENTATION_FILE)
    return _download(_SEGMENTATION_REPO.format(name=path.name), _SEGMENTATION_FILE)


def _embedding_model_path(model_name: str | Path) -> str:
    """Resolve an embedding model name to a local ``.onnx`` file."""
    path = Path(model_name)
    if path.is_file():
        return str(path)
    return _download(_EMBEDDING_REPO, path.name)


def diarize(
    audio_path: str | Path,
    segmentation_model: str | Path,
    embedding_model: str | Path,
    num_speakers: int = -1,
    threshold: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    num_threads: int = 1,
    progress: Callable[[float], bool] | None = None,
) -> list[SpeakerSegment]:
    """Split a recording into speech turns, one per speaker per stretch of speech."""
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=_segmentation_model_path(segmentation_model)
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=_embedding_model_path(embedding_model),
            num_threads=num_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold
        ),
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    if not config.validate():
        raise ValueError("invalid diarization configuration; check the model names")

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    samples = load_audio(audio_path, diarizer.sample_rate)
    if samples.size == 0:
        return []

    aborted = False

    def on_progress(processed: int, total: int) -> int:
        nonlocal aborted
        if progress is None or total <= 0:
            return 0
        if progress(min(processed / total, 1.0)) is False:
            aborted = True
            return 1  # non-zero aborts the sherpa-onnx run
        return 0

    result = diarizer.process(samples, callback=on_progress)
    if aborted:
        return []
    return [
        SpeakerSegment(float(s.start), float(s.end), int(s.speaker))
        for s in result.sort_by_start_time()
    ]


def extract_speaker_embeddings(
    audio_path: str | Path,
    segments: Iterable[SpeakerSegment],
    embedding_model: str | Path,
    num_threads: int = 1,
) -> Iterator[SpeakerEmbedding]:
    """Compute one voice embedding per speech turn, in the order given."""
    import sherpa_onnx

    segments = list(segments)
    if not segments:
        return

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=_embedding_model_path(embedding_model), num_threads=num_threads
        )
    )
    samples = load_audio(audio_path, SAMPLE_RATE)
    if samples.size == 0:
        return

    for segment_id, segment in enumerate(segments):
        duration = segment.end - segment.start
        if duration < MIN_EMBEDDING_DURATION:
            continue
        start = max(0, int(segment.start * SAMPLE_RATE))
        end = min(samples.size, int(segment.end * SAMPLE_RATE))
        turn = samples[start:end]
        if turn.size == 0:
            continue
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=turn)
        stream.input_finished()
        if not extractor.is_ready(stream):
            continue
        yield SpeakerEmbedding(
            segment_id=segment_id,
            speaker=segment.speaker,
            duration=float(duration),
            embedding=np.asarray(extractor.compute(stream), dtype=np.float32),
        )
