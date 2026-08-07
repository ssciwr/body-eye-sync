"""What was said in one recording, as Whisper heard it.

These are the results of a per-input pass, so they say what was said and when,
but nothing about who said it: a recording on its own cannot tell one voice from
another. Speakers are worked out later, by comparing every recording of the
experiment against the others.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from body_eye_sync.pipeline.transcription import (
    SEGMENT_COLUMNS,
    WORD_COLUMNS,
    TranscriptSegment,
    transcript_to_dataframes,
)

# The transcribed segments, one row per stretch of speech Whisper returned.
SEGMENTS_FILENAME = "transcript_segments.parquet"

# Per-word timings within those segments.
WORDS_FILENAME = "transcript_words.parquet"


def _write_table(path: Path, table: pd.DataFrame) -> None:
    """Write a results table straight to Parquet."""
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), str(path))


class Speech:
    """The transcript computed from one recording's audio.

    Times are on the recording's own clock, as every per-input result is; the
    experiment timeline is applied when the recordings are related to each other.
    """

    def __init__(self) -> None:
        # persistent results
        self._data: pd.DataFrame | None = None
        self._words: pd.DataFrame | None = None
        # temporary data while running a pass
        self._tmp_transcript: list[TranscriptSegment] = []

    def begin_transcription(self) -> None:
        """Drop any previous transcript so a fresh pass starts clean."""
        self.clear()

    def add_transcription_segment(self, segment: TranscriptSegment) -> None:
        """Accumulate one transcribed segment for the final table."""
        self._tmp_transcript.append(segment)

    def finish_transcription(self) -> None:
        """Collapse the accumulated segments into :attr:`data` and :attr:`words`."""
        self._data, self._words = transcript_to_dataframes(self._tmp_transcript)
        self._tmp_transcript = []

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace the transcribed segments with a complete data DataFrame."""
        if "segment_id" not in data.columns:
            raise ValueError("transcript table has no 'segment_id' column")
        self._data = data
        self._tmp_transcript = []

    @property
    def data(self) -> pd.DataFrame | None:
        """The transcribed segments, or ``None`` until transcription has run."""
        return self._data

    @property
    def words(self) -> pd.DataFrame | None:
        """Per-word timings, or ``None`` until transcription has run."""
        return self._words

    @property
    def text(self) -> str:
        """The whole transcript as one string, in the order it was spoken."""
        if self._data is None:
            return ""
        return " ".join(self._data["text"].astype(str)).strip()

    def clear(self) -> None:
        self._data = None
        self._words = None
        self._tmp_transcript = []

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``, one file per kind."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for filename, table in (
            (SEGMENTS_FILENAME, self._data),
            (WORDS_FILENAME, self._words),
        ):
            path = directory / filename
            if table is None:
                path.unlink(missing_ok=True)
            else:
                _write_table(path, table)

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any."""
        directory = Path(directory)
        self.clear()
        segments_path = directory / SEGMENTS_FILENAME
        if not segments_path.exists():
            return
        self.set_data(pd.read_parquet(segments_path))
        words_path = directory / WORDS_FILENAME
        if words_path.exists():
            self._words = pd.read_parquet(words_path)


__all__ = [
    "SEGMENTS_FILENAME",
    "SEGMENT_COLUMNS",
    "WORDS_FILENAME",
    "WORD_COLUMNS",
    "Speech",
]
