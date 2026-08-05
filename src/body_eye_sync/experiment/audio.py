"""Model outputs for a separately recorded audio input."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from body_eye_sync.experiment.video import GlassesVideo


class Audio:
    """An audio input: its settings and the model outputs computed from it.

    Audio recorded on its own device, such as a directional microphone.  The inputs carry their own audio separately -
    meaning embedded audio in video input files is played with the video itself, for example during the Alignment stage.

    ``id`` names the input and its output directory, and ``time_offset`` is the
    seconds to add to this recording's own clock to reach experiment time.
    ``glasses_video`` is the glasses video worn by the participant this
    recording captures, when it is aimed at one.
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
        self._data: pd.DataFrame | None = None

    def set_audio(self, path: str | Path) -> None:
        """Set the current recording, invalidating any previous model outputs."""
        self.clear()
        self.audio_path = Path(path)

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace all results with a complete data DataFrame."""
        self._data = data

    @property
    def data(self) -> pd.DataFrame | None:
        """All results for this recording, or ``None`` until there are any."""
        return self._data

    def clear(self) -> None:
        self._data = None

    def to_parquet(self, path: str | Path) -> None:
        """Write the completed :attr:`data` to a Parquet file.

        Raises :class:`ValueError` if there is no completed data to write.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._data is None:
            raise ValueError("no data to write; run the pipeline first")
        table = pa.Table.from_pandas(self._data, preserve_index=False)
        pq.write_table(table, str(path))

    def load_parquet(self, path: str | Path) -> None:
        """Load results written by :meth:`to_parquet` into this recording."""
        self.clear()
        self.set_data(pd.read_parquet(path))

    @classmethod
    def from_parquet(cls, path: str | Path) -> "Audio":
        """A new :class:`Audio` loaded from a Parquet file (see :meth:`load_parquet`)."""
        audio = cls()
        audio.load_parquet(path)
        return audio
