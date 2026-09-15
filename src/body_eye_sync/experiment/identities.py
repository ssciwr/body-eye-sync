"""People identified across an experiment's video tracklets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

IDENTITIES_FILENAME = "identities.parquet"
IDENTITY_COLUMNS = ["video_id", "track_id", "participant_id"]


class Identities:
    """An experiment-wide mapping from video tracklets to people.

    Each row identifies a tracklet in a glasses recording by
    ``(video_id, track_id)``. ``participant_id`` is the glasses video ID of the
    person seen in that tracklet, or null when the person is unidentified.
    """

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self._data: pd.DataFrame | None = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace identities, requiring the schema and unique video tracklets."""
        missing = [column for column in IDENTITY_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"identities table is missing columns: {missing}")
        if data.duplicated(["video_id", "track_id"]).any():
            raise ValueError("identities table has duplicate video tracklets")
        self._data = data.copy()

    @property
    def data(self) -> pd.DataFrame | None:
        """The identity table, or ``None`` until identities have been determined."""
        return self._data

    @property
    def participants(self) -> list[str]:
        """The glasses IDs associated with identified people."""
        if self._data is None:
            return []
        return sorted(self._data["participant_id"].dropna().unique().tolist())

    def for_video(self, video_id: str) -> pd.DataFrame:
        """The tracklet identities observed in one video."""
        if self._data is None:
            return pd.DataFrame(columns=IDENTITY_COLUMNS)
        return self._data[self._data["video_id"] == video_id].copy()

    def rename_video(self, old_id: str, new_id: str) -> None:
        """Update recording and wearer references when an experiment input moves."""
        if self._data is None:
            return
        renamed = self._data.copy()
        for column in ("video_id", "participant_id"):
            renamed.loc[renamed[column] == old_id, column] = new_id
        self.set_data(renamed)

    def has_data(self) -> bool:
        return self._data is not None

    def clear(self) -> None:
        self._data = None

    def save(self, directory: str | Path) -> None:
        """Write identities, or remove a stale file if identities were cleared."""
        directory = Path(directory)
        path = directory / IDENTITIES_FILENAME
        if self._data is None:
            path.unlink(missing_ok=True)
            return
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(self._data, preserve_index=False), str(path)
        )

    def load(self, directory: str | Path) -> None:
        """Load a stored identity table, if present."""
        self.clear()
        path = Path(directory) / IDENTITIES_FILENAME
        if path.exists():
            self.set_data(pd.read_parquet(path))
