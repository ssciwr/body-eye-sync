"""Model outputs for a separately recorded audio input."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from body_eye_sync.experiment.speech import SEGMENTS_FILENAME, Speech
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
        self.speech = Speech()

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

    def clear(self) -> None:
        self.speech.clear()

    def has_data(self) -> bool:
        """Whether this recording has a transcript in memory."""
        return self.speech.data is not None

    def has_results(self, directory: str | Path) -> bool:
        """Whether ``directory`` already holds results for a recording."""
        return (Path(directory) / SEGMENTS_FILENAME).exists()

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``, a file per kind of result."""
        if self.speech.data is None:
            raise ValueError("no data to write; run the pipeline first")
        self.speech.save(directory)

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any.

        Replaces any current results. A directory with nothing in it leaves
        this recording empty rather than failing.
        """
        self.speech.load(directory)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Audio":
        """A new :class:`Audio` loaded from an output directory (see :meth:`load`)."""
        audio = cls()
        audio.load(directory)
        return audio
