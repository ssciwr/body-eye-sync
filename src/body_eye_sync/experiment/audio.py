"""Model outputs for a separately recorded audio input."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from body_eye_sync.experiment.loudness import Loudness
from body_eye_sync.experiment.speech import SEGMENTS_FILENAME, Speech
from body_eye_sync.experiment.timeline import Timeline

if TYPE_CHECKING:
    from body_eye_sync.experiment.video import GlassesVideo


class Audio:
    """An audio input: its settings and the model outputs computed from it.

    Audio recorded on its own device, such as a directional microphone.  The inputs carry their own audio separately -
    meaning embedded audio in video input files is played with the video itself, for example during the Alignment stage.

    ``id`` names the input and its output directory, and ``timeline`` places
    the recording's own clock on the experiment clock.
    ``glasses_video`` is the glasses video worn by the participant this
    recording captures, when it is aimed at one.
    """

    def __init__(
        self,
        id: str = "",
        path: str | Path | None = None,
        glasses_video: GlassesVideo | None = None,
        timeline: Timeline | None = None,
    ) -> None:
        self.id = id
        self.audio_path = Path(path) if path is not None else None
        self.glasses_video = glasses_video
        self.timeline = timeline if timeline is not None else Timeline()
        self.speech = Speech()
        self.loudness = Loudness()

    @property
    def path(self) -> Path | None:
        return self.audio_path

    def has_audio_track(self) -> bool:
        """Whether this recording carries sound"""
        return self.audio_path is not None

    def clear(self) -> None:
        self.speech.clear()
        self.loudness.clear()

    def has_data(self) -> bool:
        """Whether this recording has any audio processing results in memory."""
        return self.speech.data is not None or self.loudness.data is not None

    def has_results(self, directory: str | Path) -> bool:
        """Whether ``directory`` already holds results for a recording."""
        return (Path(directory) / SEGMENTS_FILENAME).exists()

    def save(self, directory: str | Path) -> None:
        """Write these results into ``directory``, a file per kind of result."""
        if not self.has_data():
            raise ValueError("no data to write; run the pipeline first")
        self.speech.save(directory)
        self.loudness.save(directory)

    def load(self, directory: str | Path) -> None:
        """Load results written by :meth:`save`, if ``directory`` holds any.

        Replaces any current results. A directory with nothing in it leaves
        this recording empty rather than failing.
        """
        self.speech.load(directory)
        self.loudness.load(directory)
