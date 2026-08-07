"""The experiment's speech turns: who spoke when, and what they said.

Unlike every other result, this one belongs to the experiment rather than to any
one input: it can only be worked out by comparing the recordings against each
other, and its speakers name the inputs they were attributed to. So it is stored
beside the per-input directories rather than inside one of them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from body_eye_sync.preprocessing.attribution import TURN_COLUMNS

#: Where the turns are written, inside the experiment's output directory.
TURNS_FILENAME = "speech_turns.parquet"


class SpeechTurns:
    """Every speech turn of an experiment, on the shared experiment clock.

    Turns may overlap: two people talking at once are two turns covering the
    same stretch of time, each attributed to the input whose microphone heard
    that speaker loudest.
    """

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self._data: pd.DataFrame | None = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace the turns with a complete table."""
        missing = [column for column in TURN_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"speech turns table is missing columns: {missing}")
        self._data = data

    @property
    def data(self) -> pd.DataFrame | None:
        """The speech turns, or ``None`` until they have been worked out."""
        return self._data

    @property
    def speakers(self) -> list[str]:
        """The inputs speech was attributed to, in the order they are named."""
        if self._data is None:
            return []
        return sorted(self._data["speaker"].unique().tolist())

    def for_speaker(self, speaker: str) -> pd.DataFrame:
        """One speaker's turns, in the order they spoke them."""
        if self._data is None:
            return pd.DataFrame(columns=TURN_COLUMNS)
        return self._data[self._data["speaker"] == speaker]

    def has_data(self) -> bool:
        return self._data is not None

    def clear(self) -> None:
        self._data = None

    def save(self, directory: str | Path) -> None:
        """Write the turns into ``directory``, or remove them if there are none."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / TURNS_FILENAME
        if self._data is None:
            path.unlink(missing_ok=True)
            return
        pq.write_table(
            pa.Table.from_pandas(self._data, preserve_index=False), str(path)
        )

    def load(self, directory: str | Path) -> None:
        """Load turns written by :meth:`save`, if ``directory`` holds any."""
        self.clear()
        path = Path(directory) / TURNS_FILENAME
        if path.exists():
            self.set_data(pd.read_parquet(path))
