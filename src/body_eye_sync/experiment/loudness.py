"""How loud one recording is over time, computed from its audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from body_eye_sync.pipeline.loudness import LOUDNESS_COLUMNS, measure_loudness

LOUDNESS_FILENAME = "loudness.parquet"


class Loudness:
    """The loudness measured from one recording's audio."""

    def __init__(self) -> None:
        self._data: pd.DataFrame | None = None

    def measure(self, path: str | Path) -> None:
        """Measure a recording, replacing whatever was measured before."""
        self._data = measure_loudness(path)

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace the measured loudness with a complete data DataFrame."""
        missing = [column for column in LOUDNESS_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"loudness table has no {missing[0]!r} column")
        self._data = data

    @property
    def data(self) -> pd.DataFrame | None:
        """The measured loudness, or ``None`` until the recording is measured."""
        return self._data

    @property
    def levels(self) -> np.ndarray:
        """The measured levels in dB, or nothing when this is unmeasured."""
        return self._values("level_db")

    @property
    def times(self) -> np.ndarray:
        """When each level was measured, on the recording's own clock."""
        return self._values("time")

    def _values(self, column: str) -> np.ndarray:
        if self._data is None:
            return np.empty(0)
        return self._data[column].to_numpy(dtype=float)

    def clear(self) -> None:
        self._data = None

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LOUDNESS_FILENAME
        if self._data is None:
            path.unlink(missing_ok=True)
        else:
            pq.write_table(
                pa.Table.from_pandas(self._data, preserve_index=False), str(path)
            )

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any."""
        self.clear()
        path = Path(directory) / LOUDNESS_FILENAME
        if path.exists():
            self.set_data(pd.read_parquet(path))


__all__ = [
    "LOUDNESS_COLUMNS",
    "LOUDNESS_FILENAME",
    "Loudness",
]
