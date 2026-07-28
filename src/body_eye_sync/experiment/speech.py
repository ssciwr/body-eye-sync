"""Speech model outputs: who spoke when, and what they said."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from body_eye_sync.experiment.embeddings import (
    TopK,
    read_embeddings,
    write_embeddings,
)
from body_eye_sync.pipeline.diarization import (
    SpeakerEmbedding,
    SpeakerSegment,
    segments_to_dataframe,
)
from body_eye_sync.pipeline.transcription import (
    TRANSCRIPT_COLUMNS,
    TranscriptSegment,
    transcript_to_dataframe,
)

# The speech turns table, one row per turn, with speaker id and start/end times.
TURNS_FILENAME = "speaker_turns.parquet"

# Per-word timings from transcription.
WORDS_FILENAME = "speaker_words.parquet"

# Voice embeddings, kept per speaker.
EMBEDDINGS_FILENAME = "speaker_embeddings.parquet"

# Column layout of the speaker embeddings table, the audio counterpart of a
# video's. Embeddings are kept per ``speaker``, the identity a video keeps them
# per ``track_id`` for, and ``segment_id`` records the turn each came from.
_SPEAKER_EMBEDDING_COLUMNS = ["speaker", "segment_id", "duration", "embedding"]


def _write_table(path: Path, table: pd.DataFrame) -> None:
    """Write a results table straight to Parquet."""
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), str(path))


class Speech:
    """The speech results computed from one recording's audio."""

    def __init__(self) -> None:
        # persistent results
        self._data: pd.DataFrame | None = None
        self._words: pd.DataFrame | None = None
        self._embeddings: pd.DataFrame | None = None
        # temporary data while running a pass
        self._tmp_segments: list[SpeakerSegment] = []
        self._tmp_transcript: list[TranscriptSegment] = []
        self._tmp_speaker_topk = TopK(0, _SPEAKER_EMBEDDING_COLUMNS)

    def begin_diarization(self, embeddings_per_speaker: int = 0) -> None:
        """Drop any previous model outputs."""
        self.clear()
        self._tmp_speaker_topk = TopK(
            embeddings_per_speaker, _SPEAKER_EMBEDDING_COLUMNS
        )

    def add_diarization_segment(self, segment: SpeakerSegment) -> None:
        """Accumulate one diarized speech turn for the final table."""
        self._tmp_segments.append(segment)

    def finish_diarization(self) -> None:
        """Collapse the accumulated speech turns into :attr:`data`."""
        self.set_data(segments_to_dataframe(self._tmp_segments))

    def discard_diarization(self) -> None:
        """Drop a cancelled or failed run; its partial output is unusable."""
        self.clear()

    def add_speaker_embedding(self, embedding: SpeakerEmbedding) -> None:
        """Accumulate one turn's voice embedding, keeping the best per speaker."""
        self._tmp_speaker_topk.add(
            embedding.speaker,
            embedding.segment_id,
            embedding.duration,
            embedding.embedding,
        )

    def finish_speaker_embeddings(self) -> None:
        """Collapse the kept voice embeddings into :attr:`embeddings`."""
        self._embeddings = self._tmp_speaker_topk.to_frame()

    def begin_transcription(self) -> None:
        """Drop any previous transcript so a fresh pass starts clean."""
        if self._data is not None:
            present = [c for c in TRANSCRIPT_COLUMNS if c in self._data.columns]
            if present:
                self.set_data(self._data.drop(columns=present))
        self._tmp_transcript = []
        self._words = None

    def add_transcription_segment(self, segment: TranscriptSegment) -> None:
        """Accumulate one transcribed segment for the final merge."""
        self._tmp_transcript.append(segment)

    def finish_transcription(self) -> None:
        """Attribute the transcript to speakers and merge it onto :attr:`data`."""
        if self._data is None:
            return
        text, words = transcript_to_dataframe(self._tmp_transcript, self.segments)
        self.set_data(self._data.merge(text, on="segment_id", how="left"))
        self._words = words
        self._tmp_transcript = []

    def discard_transcription(self) -> None:
        """Drop a cancelled or failed pass; the speech turns are left intact."""
        self._tmp_transcript = []
        self._words = None

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace the speech turns with a complete data DataFrame."""
        if "segment_id" not in data.columns:
            raise ValueError("speech results table has no 'segment_id' column")
        self._data = data
        self._tmp_segments = []

    @property
    def data(self) -> pd.DataFrame | None:
        """The speech turns, or ``None`` until diarization has run."""
        return self._data

    @property
    def words(self) -> pd.DataFrame | None:
        """Per-word timings from transcription, or ``None`` if it has not run."""
        return self._words

    @property
    def embeddings(self) -> pd.DataFrame | None:
        """Voice embeddings per speaker, or ``None`` if none were collected."""
        return self._embeddings

    @property
    def segments(self) -> list[SpeakerSegment]:
        """The diarized speech turns, in ``segment_id`` order."""
        if self._data is None:
            return []
        return [
            SpeakerSegment(float(r.start), float(r.end), int(r.speaker))
            for r in self._data.itertuples(index=False)
        ]

    @property
    def speakers(self) -> list[int]:
        """The distinct speaker ids found in this recording, in order."""
        if self._data is None:
            return []
        return sorted(self._data["speaker"].unique().tolist())

    def clear(self) -> None:
        self._data = None
        self._words = None
        self._embeddings = None
        self._tmp_segments = []
        self._tmp_transcript = []
        self._tmp_speaker_topk = TopK(0, _SPEAKER_EMBEDDING_COLUMNS)

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``, one file per kind."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for filename, table, write in (
            (TURNS_FILENAME, self._data, _write_table),
            (WORDS_FILENAME, self._words, _write_table),
            (EMBEDDINGS_FILENAME, self._embeddings, write_embeddings),
        ):
            path = directory / filename
            if table is None:
                path.unlink(missing_ok=True)
            else:
                write(path, table)

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any."""
        directory = Path(directory)
        self.clear()
        turns_path = directory / TURNS_FILENAME
        if not turns_path.exists():
            return
        self.set_data(pd.read_parquet(turns_path))
        words_path = directory / WORDS_FILENAME
        if words_path.exists():
            self._words = pd.read_parquet(words_path)
        embeddings_path = directory / EMBEDDINGS_FILENAME
        if embeddings_path.exists():
            self._embeddings = read_embeddings(embeddings_path)
