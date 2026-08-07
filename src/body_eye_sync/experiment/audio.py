"""Model outputs for a separately recorded audio input."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from body_eye_sync.experiment.timeline import Timeline

if TYPE_CHECKING:
    from body_eye_sync.experiment.video import GlassesVideo


class Audio(Timeline):
    """An audio input: its settings and the model outputs computed from it.

    Audio recorded on its own device, such as a directional microphone; the
    video inputs carry their own audio separately.

    ``id`` names the input and its output directory, and ``time_offset`` is the
    seconds to add to this recording's own clock to reach experiment time.
    ``glasses_video`` is the glasses video worn by the participant this
    recording captures, when it is aimed at one.
    """

    #: This recording's results, its only output.
    _RESULTS_FILENAME: ClassVar[str] = "results.parquet"

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        glasses_video: GlassesVideo | None = None,
        **timeline,
    ) -> None:
        # ``timeline`` is where this input sits on the experiment clock; see
        # :class:`~body_eye_sync.experiment.timeline.Timeline`.
        super().__init__(**timeline)
        self.id = id
        self.audio_path = Path(path) if path is not None else None
        self.glasses_video = glasses_video
        self._data: pd.DataFrame | None = None

    @classmethod
    def from_config(
        cls, spec, resolve, glasses_video: GlassesVideo | None = None
    ) -> "Audio":
        """This recording as its stored form describes it.

        ``glasses_video`` is the input ``spec.glasses_video`` names, which only
        the experiment holding the other inputs can look up.
        """
        return cls(
            spec.id,
            resolve(spec.path),
            glasses_video,
            **cls.timeline_kwargs(spec),
        )

    @property
    def path(self) -> Path | None:
        return self.audio_path

    def set_data(self, data: pd.DataFrame) -> None:
        """Replace all results with a complete data DataFrame."""
        self._data = data

    @property
    def data(self) -> pd.DataFrame | None:
        """All results for this recording, or ``None`` until there are any."""
        return self._data

    def clear(self) -> None:
        self._data = None

    def has_data(self) -> bool:
        """Whether this recording has completed results in memory."""
        return self._data is not None

    def has_results(self, directory: str | Path) -> bool:
        """Whether ``directory`` already holds results for a recording."""
        return (Path(directory) / self._RESULTS_FILENAME).exists()

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``, a file per kind of result.

        Raises :class:`ValueError` if there is no completed data to write.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._data is None:
            raise ValueError("no data to write; run the pipeline first")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(self._data, preserve_index=False)
        pq.write_table(table, str(directory / self._RESULTS_FILENAME))

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any.

        Replaces any current results. A directory with nothing in it leaves
        this recording empty rather than failing.
        """
        self.clear()
        results_path = Path(directory) / self._RESULTS_FILENAME
        if results_path.exists():
            self.set_data(pd.read_parquet(results_path))

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Audio":
        """A new :class:`Audio` loaded from an output directory (see :meth:`load`)."""
        audio = cls()
        audio.load(directory)
        return audio
