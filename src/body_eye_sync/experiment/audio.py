"""Model outputs for a separately recorded audio input."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from body_eye_sync.pipeline.diarization import (
    SpeakerSegment,
    segments_to_dataframe,
)
from body_eye_sync.pipeline.transcription import (
    TRANSCRIPT_COLUMNS,
    TranscriptSegment,
    transcript_to_dataframe,
)

if TYPE_CHECKING:
    from body_eye_sync.experiment.video import GlassesVideo


def _words_path(path: str | Path) -> Path:
    """Companion per-word file beside a main output, e.g. ``mic1.words.parquet``."""
    path = Path(path)
    return path.with_name(f"{path.stem}.words.parquet")


class Audio:
    """An audio input: its settings and the model outputs computed from it.

    Audio recorded on its own device, such as a directional microphone; the
    video inputs carry their own audio separately.

    ``id`` names the input and its output files, and ``time_offset`` is the
    seconds to add to this recording's own clock to reach experiment time.
    ``glasses_video`` is the glasses video worn by the participant this
    recording captures, when it is aimed at one.

    Completed results live in a single :attr:`data` DataFrame of speech turns,
    one row per turn. Diarization produces those rows, playing the part object
    tracking plays for a video; transcription then runs as a later pass over the
    whole recording and folds its text onto the matching rows, the way face
    detection folds its columns onto tracked boxes. Per-word timings are kept
    alongside in :attr:`words`.
    """

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        time_offset: float = 0.0,
        glasses_video: GlassesVideo | None = None,
    ) -> None:
        self.id = id
        self.audio_path = Path(path) if path is not None else None
        self.time_offset = time_offset
        self.glasses_video = glasses_video
        # Persistent results.
        self._data: pd.DataFrame | None = None
        self._words: pd.DataFrame | None = None
        # Per-pass scratch: accumulated while a pass runs, then collapsed into
        # the results above and reset. ``_tmp_`` marks them as transient.
        self._tmp_segments: list[SpeakerSegment] = []
        self._tmp_transcript: list[TranscriptSegment] = []

    def set_audio(self, path: str | Path) -> None:
        """Set the current recording, invalidating any previous model outputs."""
        self.clear()
        self.audio_path = Path(path)

    def begin_diarization(self) -> None:
        """Drop any previous model outputs."""
        self.clear()

    def add_diarization_segment(self, segment: SpeakerSegment) -> None:
        """Accumulate one diarized speech turn for the final table."""
        self._tmp_segments.append(segment)

    def finish_diarization(self) -> None:
        """Collapse the accumulated speech turns into :attr:`data`."""
        self.set_data(segments_to_dataframe(self._tmp_segments))

    def discard_diarization(self) -> None:
        """Drop a cancelled or failed run; its partial output is unusable."""
        self.clear()

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
        """Attribute the transcript to speakers and merge it onto :attr:`data`.

        Whisper segments the recording on its own terms, so its words are
        redistributed onto the diarized speech turns by time overlap before the
        text is merged on ``segment_id``.
        """
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
        """Replace all results with a complete data DataFrame."""
        self._data = data
        self._tmp_segments = []

    @property
    def data(self) -> pd.DataFrame | None:
        """All results for this recording, or ``None`` until there are any."""
        return self._data

    @property
    def words(self) -> pd.DataFrame | None:
        """Per-word timings from transcription, or ``None`` if it has not run."""
        return self._words

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
        self._tmp_segments = []
        self._tmp_transcript = []

    def to_parquet(self, path: str | Path) -> None:
        """Write the completed :attr:`data` to a Parquet file.

        Any per-word timings are written to a companion file beside ``path``
        (``<stem>.words.parquet``). Raises :class:`ValueError` if there is no
        completed data to write.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._data is None:
            raise ValueError("no data to write; run the pipeline first")
        table = pa.Table.from_pandas(self._data, preserve_index=False)
        pq.write_table(table, str(path))
        if self._words is not None:
            words = pa.Table.from_pandas(self._words, preserve_index=False)
            pq.write_table(words, str(_words_path(path)))

    def load_parquet(self, path: str | Path) -> None:
        """Load results written by :meth:`to_parquet` into this recording.

        Replaces any current results with the table at ``path`` and its
        companion ``<stem>.words.parquet``, if present.
        """
        self.clear()
        self.set_data(pd.read_parquet(path))
        words_path = _words_path(path)
        if words_path.exists():
            self._words = pd.read_parquet(words_path)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "Audio":
        """A new :class:`Audio` loaded from a Parquet file (see :meth:`load_parquet`)."""
        audio = cls()
        audio.load_parquet(path)
        return audio
