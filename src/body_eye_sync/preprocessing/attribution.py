"""Who was speaking, decided by whose microphone heard them loudest.

Each glasses recording belongs to one wearer, and a head-mounted microphone
hears its wearer far louder than it hears anyone else in the room. So speech can
be attributed by comparing the recordings against each other on the shared
experiment clock, rather than by telling voices apart within any one of them --
and the speaker's identity comes free, since it is simply whose recording won.

The comparison is made on loudness alone: each recording is measured against its
own quiet baseline, which absorbs the differences in microphone gain between
devices, and a recording is *live* while it is well above that baseline. Whole
transcribed segments are attributed rather than single words: a sentence goes to
whoever won most of it, so a momentary flip in the levels cannot take a word out
of the middle of it, and the text of any one sentence always comes from a single
recording's transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from body_eye_sync.media import SAMPLE_RATE, load_audio
from body_eye_sync.preprocessing.alignment import TimelineFit, to_experiment_times

#: Seconds of audio each loudness measurement covers. Shorter than the shortest
#: word, so a segment's span always holds several.
HOP_SECONDS = 0.05

#: A recording's own quiet baseline, as a percentile of its own levels. Robust
#: to how much of the recording its wearer spends talking.
FLOOR_PERCENTILE = 10.0

#: How far above its own baseline a recording counts as live, in dB.
LIVE_ABOVE_FLOOR_DB = 12.0

#: How much of a segment a recording must win to be given the whole of it.
OWNERSHIP_SHARE = 0.5

#: How much two recordings must say the same thing, over the same stretch of
#: time, to be judged one voice heard twice rather than two people talking at
#: once. Above this they agree too well to be different speakers.
BLEED_AGREEMENT = 0.5

#: Columns of the experiment-wide speech turns table.
TURN_COLUMNS = [
    "turn_id",
    "start",
    "end",
    "speaker",
    "source",
    "source_segment_id",
    "text",
]


class AttributionCancelled(Exception):
    """Raised when a caller's ``progress`` callback asks for the pass to stop."""


def _continue(progress: Callable[[float], bool] | None, value: float) -> None:
    if progress is not None and progress(value) is False:
        raise AttributionCancelled


def envelope(path: str | Path, hop: float = HOP_SECONDS) -> np.ndarray:
    """How loud a recording is over time, in dB, one value per ``hop`` seconds."""
    samples = load_audio(path, SAMPLE_RATE)
    frame = int(hop * SAMPLE_RATE)
    if frame <= 0 or samples.size < frame:
        return np.empty(0)
    frames = samples[: samples.size // frame * frame].reshape(-1, frame)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    # A silent frame has no logarithm, so the floor of the scale stands in.
    return 20 * np.log10(np.maximum(rms, 1e-8))


@dataclass
class Levels:
    """How loud every recording is, over one shared stretch of experiment time.

    :attr:`above_floor` is each recording's level relative to its own quiet
    baseline, so the recordings are comparable despite their microphones being
    set to different gains. Rows follow :attr:`ids`, columns follow
    :attr:`times`.
    """

    ids: list[str]
    times: np.ndarray
    above_floor: np.ndarray

    def live(self, threshold: float = LIVE_ABOVE_FLOOR_DB) -> np.ndarray:
        """Which recordings are carrying speech at each moment."""
        return self.above_floor > threshold

    def loudest(self, threshold: float = LIVE_ABOVE_FLOOR_DB) -> np.ndarray:
        """The row of the loudest live recording at each moment, or ``-1``.

        ``-1`` means nobody was speaking loudly enough to be attributed, which
        is most of a recording: silence, and the room noise between turns.
        """
        if not self.ids or self.times.size == 0:
            return np.empty(0, dtype=int)
        live = self.live(threshold).any(axis=0)
        return np.where(live, self.above_floor.argmax(axis=0), -1)

    def share(
        self,
        name: str,
        start: float,
        end: float,
        threshold: float = LIVE_ABOVE_FLOOR_DB,
    ) -> float:
        """How much of ``[start, end)`` one recording is the loudest for.

        Measured over the moments somebody was speaking, so a segment that
        spans a pause is judged on the speech in it rather than the silence.
        """
        if name not in self.ids:
            return 0.0
        window = (self.times >= start) & (self.times < max(end, start + HOP_SECONDS))
        winners = self.loudest(threshold)[window]
        winners = winners[winners >= 0]
        if winners.size == 0:
            return 0.0
        return float((winners == self.ids.index(name)).mean())

    def live_share(
        self,
        name: str,
        start: float,
        end: float,
        threshold: float = LIVE_ABOVE_FLOOR_DB,
    ) -> float:
        """How much of ``[start, end)`` one recording is carrying speech for.

        Unlike :meth:`share` this asks nothing about the other recordings: a
        microphone can be live while another is louder, which is what two people
        talking at once looks like.
        """
        if name not in self.ids:
            return 0.0
        window = (self.times >= start) & (self.times < max(end, start + HOP_SECONDS))
        if not window.any():
            return 0.0
        return float(self.live(threshold)[self.ids.index(name)][window].mean())


def measure_levels(
    paths: dict[str, Path],
    fits: dict[str, TimelineFit],
    hop: float = HOP_SECONDS,
    floor_percentile: float = FLOOR_PERCENTILE,
    progress: Callable[[float], bool] | None = None,
) -> Levels:
    """Measure every recording's loudness and put it on the experiment clock.

    Only the stretch of experiment time that *every* recording covers is kept:
    outside it the recordings cannot be compared, so nothing there can be
    attributed to anyone. ``progress`` is called with how far along this is, and
    stops the pass with :class:`AttributionCancelled` if it returns ``False``.
    """
    measured: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, (name, path) in enumerate(paths.items()):
        # Decoding the audio is the slow part, so this is what there is to report.
        _continue(progress, index / max(len(paths), 1))
        levels = envelope(path, hop)
        if levels.size == 0:
            continue
        fit = fits.get(name, TimelineFit(0.0, 1.0, []))
        local = np.arange(levels.size) * hop + hop / 2
        measured[name] = (
            to_experiment_times(local, fit.offset, fit.shifts, fit.scale),
            levels,
        )
    if not measured:
        return Levels([], np.empty(0), np.empty((0, 0)))

    start = max(times[0] for times, _ in measured.values())
    end = min(times[-1] for times, _ in measured.values())
    if end <= start:
        # The recordings do not overlap, so there is nothing to compare.
        return Levels([], np.empty(0), np.empty((0, 0)))

    grid = np.arange(start, end, hop)
    ids = sorted(measured)
    rows = [np.interp(grid, measured[name][0], measured[name][1]) for name in ids]
    matrix = np.vstack(rows)
    floors = np.percentile(matrix, floor_percentile, axis=1, keepdims=True)
    return Levels(ids, grid, matrix - floors)


@dataclass
class _Segment:
    """One transcribed segment, placed on the experiment clock."""

    name: str
    segment_id: int
    start: float
    end: float
    text: str
    #: How much of it this recording was the loudest for.
    share: float
    #: How much of it this recording was carrying speech at all.
    live: float


def _on_experiment_clock(
    table: pd.DataFrame, fit: TimelineFit, columns: tuple[str, str] = ("start", "end")
) -> np.ndarray:
    """A table's start/end columns, moved onto the experiment clock."""
    return to_experiment_times(
        table[list(columns)].to_numpy().ravel(), fit.offset, fit.shifts, fit.scale
    ).reshape(-1, 2)


def _spoken_words(
    words: dict[str, pd.DataFrame], fits: dict[str, TimelineFit]
) -> dict[str, pd.DataFrame]:
    """Every recording's words, timed on the experiment clock by their middles."""
    spoken = {}
    for name, table in words.items():
        if table is None or table.empty:
            continue
        bounds = _on_experiment_clock(table, fits.get(name, TimelineFit(0.0, 1.0, [])))
        spoken[name] = pd.DataFrame(
            {
                "at": bounds.mean(axis=1),
                "word": table["word"].astype(str).str.lower(),
            }
        )
    return spoken


def _said_between(
    spoken: dict[str, pd.DataFrame], name: str, start: float, end: float
) -> list[str]:
    """The words one recording puts inside a stretch of experiment time."""
    table = spoken.get(name)
    if table is None:
        return []
    inside = table[(table["at"] >= start) & (table["at"] < end)]
    return inside["word"].tolist()


def _agreement(first: list[str], second: list[str]) -> float:
    """How much two stretches of transcript say the same thing, 0 to 1."""
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


def _is_bleed(
    candidate: _Segment,
    accepted: list[_Segment],
    spoken: dict[str, pd.DataFrame],
    agreement: float,
) -> bool:
    """Whether a segment is another speaker's voice reaching this microphone.

    Bleed is the same words over again, quieter; a second person talking is
    different words. The two transcripts are compared over the stretch of time
    they share rather than segment against segment, because Whisper splits each
    recording differently and the same speech lands in different segments.
    """
    for turn in accepted:
        start = max(candidate.start, turn.start)
        end = min(candidate.end, turn.end)
        if end <= start:
            continue
        theirs = (
            _said_between(spoken, turn.name, start, end) or turn.text.lower().split()
        )
        mine = _said_between(spoken, candidate.name, start, end)
        if not mine:
            # No words of this recording's own fall in the shared stretch, so
            # compare what each segment says as a whole instead.
            mine = candidate.text.lower().split()
        if _agreement(mine, theirs) > agreement:
            return True
    return False


def _accept(
    segments: list[_Segment],
    spoken: dict[str, pd.DataFrame],
    agreement: float,
) -> list[_Segment]:
    """Keep every turn that is not a copy of one already kept, strongest first.

    One utterance can reach several microphones and be transcribed by all of
    them, so the same speech turns up more than once whether or not anybody won
    it outright: the recordings are cut into segments differently, and the
    loudest one can change partway through. Judging the turns in order of how
    well each recording heard them, and comparing every turn against everything
    kept so far rather than only against the outright winners, leaves one copy
    of each utterance -- the copy from the microphone that heard it best.
    """
    kept: list[_Segment] = []
    for turn in sorted(
        segments, key=lambda s: (-s.share, -s.live, -(s.end - s.start), s.name)
    ):
        if not _is_bleed(turn, kept, spoken, agreement):
            kept.append(turn)
    return kept


def attribute_segments(
    transcripts: dict[str, pd.DataFrame],
    levels: Levels,
    fits: dict[str, TimelineFit],
    words: dict[str, pd.DataFrame] | None = None,
    threshold: float = LIVE_ABOVE_FLOOR_DB,
    ownership: float = OWNERSHIP_SHARE,
    agreement: float = BLEED_AGREEMENT,
) -> pd.DataFrame:
    """Give each transcribed segment to the wearer whose microphone won it.

    ``transcripts`` maps a recording id to its transcript, timed on that
    recording's own clock, and ``words`` to the per-word table beside it. The
    returned table has :data:`TURN_COLUMNS`, timed on the experiment clock and
    sorted by start time.

    A segment goes to whoever was loudest for most of it. A segment somebody
    else was louder for is kept as well if this recording was live through it
    *and* says something different -- that is two people talking at once, which
    the loudest microphone alone cannot represent. If it says the same thing, it
    is that other person's voice reaching this microphone, and is dropped. Two
    segments that both won their own version of one utterance are reduced to the
    copy that won the most of it.
    """
    spoken = _spoken_words(words or {}, fits)

    segments: list[_Segment] = []
    for name, transcript in transcripts.items():
        if transcript is None or transcript.empty or name not in levels.ids:
            continue
        bounds = _on_experiment_clock(
            transcript, fits.get(name, TimelineFit(0.0, 1.0, []))
        )
        for (start, end), segment_id, text in zip(
            bounds, transcript["segment_id"], transcript["text"]
        ):
            segments.append(
                _Segment(
                    name,
                    int(segment_id),
                    start,
                    end,
                    str(text),
                    levels.share(name, start, end, threshold),
                    levels.live_share(name, start, end, threshold),
                )
            )

    # A microphone that heard nothing of its own cannot hold anybody's turn,
    # whatever its transcript says: that text is somebody else's voice, or
    # invented over silence.
    spoke = [
        segment
        for segment in segments
        if segment.share > ownership or segment.live > ownership
    ]
    rows = sorted(
        (
            (
                turn.start,
                turn.end,
                turn.name,
                turn.name,
                turn.segment_id,
                turn.text,
            )
            for turn in _accept(spoke, spoken, agreement)
        ),
        key=lambda row: (row[0], row[1], row[2]),
    )
    table = pd.DataFrame(
        [(index, *row) for index, row in enumerate(rows)], columns=TURN_COLUMNS
    )
    return table.astype(
        {
            "turn_id": int,
            "start": float,
            "end": float,
            "speaker": str,
            "source": str,
            "source_segment_id": int,
            "text": str,
        }
    )
