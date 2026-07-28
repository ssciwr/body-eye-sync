"""Speaker diarization -- who spoke when -- using sherpa-onnx.

Diarization is to an audio input what object tracking is to a video: the first
pass, which finds the things later passes describe. It splits the recording into
speech turns and labels each with a ``speaker`` id, the audio counterpart of a
video ``track_id``. Like track ids, speaker ids are only meaningful within one
recording; relating them across inputs is a separate identity-clustering step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from body_eye_sync.pipeline.model_cache import model_cache_dir

#: Hugging Face repo holding all the sherpa-onnx speaker embedding models, one
#: ``.onnx`` file each. sherpa-onnx has no downloader of its own -- its configs
#: take plain file paths -- so models come from the Hub mirror through
#: ``huggingface_hub``, the same way faster-whisper fetches its own weights,
#: rather than from the tarball release assets upstream documents.
_EMBEDDING_REPO = "csukuangfj/speaker-embedding-models"

#: Segmentation models get a repo each, every one holding a ``model.onnx``.
_SEGMENTATION_REPO = "csukuangfj/{name}"

#: The file every segmentation repo stores its full-precision model under.
_SEGMENTATION_FILE = "model.onnx"

#: Default speaker segmentation model: pyannote's ``segmentation-3.0`` exported
#: to ONNX. Small (7 MB) and MIT licensed, so it ships without a token.
DEFAULT_SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0"

#: Default speaker embedding model, used to cluster turns into speakers.
DEFAULT_EMBEDDING_MODEL = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"

#: The sample rate the segmentation and embedding models expect.
SAMPLE_RATE = 16000

#: The columns diarization contributes: one row per speech turn.
SEGMENT_COLUMNS = ["segment_id", "start", "end", "speaker"]


@dataclass
class SpeakerSegment:
    """One speech turn: ``[start, end)`` in seconds, attributed to ``speaker``.

    Times are on the recording's own clock; the input's ``time_offset`` places
    them on the shared experiment timeline. Turns by different speakers may
    overlap, which is how simultaneous speech is represented.
    """

    start: float
    end: float
    speaker: int


def segments_to_dataframe(segments: Iterable[SpeakerSegment]) -> pd.DataFrame:
    """Stack speech turns into a DataFrame with :data:`SEGMENT_COLUMNS`.

    Turns are sorted by start time and numbered from zero, so ``segment_id``
    keys the rows that transcription later merges its text onto.
    """
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.speaker))
    rows = [(index, s.start, s.end, s.speaker) for index, s in enumerate(ordered)]
    data = (
        np.asarray(rows, dtype=float) if rows else np.empty((0, len(SEGMENT_COLUMNS)))
    )
    frame = pd.DataFrame(data, columns=SEGMENT_COLUMNS)
    return frame.astype({"segment_id": int, "speaker": int})


def load_audio(audio_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a recording to mono ``float32`` samples at ``sample_rate``.

    Uses the PyAV-based decoder that comes with faster-whisper, so every format
    FFmpeg understands (including the Opus files these recordings often use) is
    read without requiring an ``ffmpeg`` binary on the user's PATH.
    """
    from faster_whisper.audio import decode_audio

    return decode_audio(str(audio_path), sampling_rate=sample_rate)


def _download(repo_id: str, filename: str) -> str:
    """Fetch a model file from the Hugging Face Hub into the shared model cache."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, cache_dir=str(model_cache_dir()))


def _segmentation_model_path(model_name: str | Path) -> str:
    """Resolve a segmentation model name to a local ``.onnx`` file.

    A bare name is the Hugging Face repo holding that model; a path to an
    existing file is used as given, and a directory is taken to be an already
    unpacked model.
    """
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
    segmentation_model: str | Path = DEFAULT_SEGMENTATION_MODEL,
    embedding_model: str | Path = DEFAULT_EMBEDDING_MODEL,
    num_speakers: int = -1,
    threshold: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
    num_threads: int = 1,
    progress: Callable[[float], bool] | None = None,
) -> list[SpeakerSegment]:
    """Split a recording into speech turns, one per speaker per stretch of speech.

    ``num_speakers`` switches sherpa-onnx to clustering into a fixed number of
    speakers instead of thresholding. It is an upper bound rather than a
    guarantee -- asking for more speakers than the embeddings support yields
    fewer -- and on clearly separated material it tends to be *less* accurate
    than the default ``-1``, which clusters by ``threshold`` instead. Prefer
    tuning ``threshold``, and reach for ``num_speakers`` only to stop a
    recording being split into obviously too many speakers.

    ``threshold`` is the distance below which two turns are treated as the same
    speaker, so lower values split the recording into more speakers.

    ``min_duration_on`` drops speech turns shorter than that many seconds, and
    ``min_duration_off`` bridges pauses shorter than it rather than splitting the
    turn in two.

    ``progress`` is called with a fraction in ``[0, 1]`` as the pass runs;
    returning ``False`` from it aborts, in which case the turns found so far are
    discarded and an empty list is returned. Unlike the video stages this cannot
    stream its results -- sherpa-onnx clusters over the whole recording at once,
    so no turn is final until all of them are.
    """
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
