"""Who was speaking, decided by whose glasses video microphone heard them loudest."""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass
from functools import cached_property
from operator import attrgetter
import re
from typing import Callable
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from body_eye_sync.experiment.config import SpeechPostProcessingSettings
from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.pipeline.loudness import HOP_SECONDS as _HOP_SECONDS

Progress = Callable[[float], bool]

_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")
_COMMA_END = re.compile(r",[\"')\]]*$")

TURN_COLUMNS = [
    "turn_id",
    "start",
    "end",
    "speaker",
    "source_segment_id",
    "text",
]


class AttributionCancelled(Exception):
    """Raised when a caller's ``progress`` callback asks for the pass to stop."""


@dataclass
class Levels:
    """How loud every recording is compared to its own quiet baseline."""

    ids: list[str]
    times: np.ndarray
    above_floor: np.ndarray
    live_above_floor_db: float

    @cached_property
    def live(self) -> np.ndarray:
        """Which recordings are carrying speech at each moment."""
        return self.above_floor > self.live_above_floor_db

    @cached_property
    def loudest(self) -> np.ndarray:
        """The row of the loudest live recording at each moment, or ``-1`` if unclear."""
        if not self.ids or self.times.size == 0:
            return np.empty(0, dtype=int)
        return np.where(self.live.any(axis=0), self.above_floor.argmax(axis=0), -1)

    def _window(self, start: float, end: float) -> slice:
        """The columns covering ``[start, end)``, on the evenly spaced time grid."""
        stop = max(end, start + _HOP_SECONDS)
        return slice(
            int(np.searchsorted(self.times, start, side="left")),
            int(np.searchsorted(self.times, stop, side="left")),
        )

    def share(
        self,
        name: str,
        start: float,
        end: float,
    ) -> float:
        """How much of ``[start, end)`` one recording is the loudest for, ignoring silent stretches."""
        if name not in self.ids:
            return 0.0
        winners = self.loudest[self._window(start, end)]
        winners = winners[winners >= 0]
        if winners.size == 0:
            return 0.0
        return float((winners == self.ids.index(name)).mean())

    def live_share(
        self,
        name: str,
        start: float,
        end: float,
    ) -> float:
        """How much of ``[start, end)`` one recording is live for.

        Unlike :meth:`share` this doesn't depend on the other recordings: a
        microphone can be live while another is louder, which is what two people
        talking at once looks like.
        """
        if name not in self.ids:
            return 0.0
        window = self._window(start, end)
        if window.start >= window.stop:
            return 0.0
        return float(self.live[self.ids.index(name)][window].mean())


def measure_levels(
    loudness: dict[str, pd.DataFrame],
    timelines: dict[str, Timeline],
    settings: SpeechPostProcessingSettings,
) -> Levels:
    """Put every recording's measured loudness on the experiment clock."""
    measured: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, table in loudness.items():
        if table is None or table.empty:
            continue
        levels = table["level_db"].to_numpy(dtype=float)
        timeline = timelines.get(name, Timeline())
        measured[name] = (
            timeline.to_experiment_times(table["time"].to_numpy(dtype=float)),
            levels,
        )
    if not measured:
        return Levels(
            [],
            np.empty(0),
            np.empty((0, 0)),
            settings.live_above_floor_db,
        )

    start = max(times[0] for times, _ in measured.values())
    end = min(times[-1] for times, _ in measured.values())
    if end <= start:
        # the recordings do not overlap, so there is nothing to compare
        return Levels(
            [],
            np.empty(0),
            np.empty((0, 0)),
            settings.live_above_floor_db,
        )

    grid = np.arange(start, end, _HOP_SECONDS)
    ids = sorted(measured)
    rows = [np.interp(grid, measured[name][0], measured[name][1]) for name in ids]
    matrix = np.vstack(rows)
    floors = np.percentile(matrix, settings.floor_percentile, axis=1, keepdims=True)
    return Levels(
        ids,
        grid,
        matrix - floors,
        settings.live_above_floor_db,
    )


@dataclass
class _Segment:
    """One transcribed segment, placed on the experiment clock."""

    name: str
    segment_id: int
    start: float
    end: float
    text: str
    # how much of the segment this recording was the loudest for.
    share: float
    # how much of the segment this recording was live (but not necessarily the loudest) for.
    live: float


def _on_experiment_clock(
    table: pd.DataFrame,
    timeline: Timeline,
    columns: tuple[str, str] = ("start", "end"),
) -> np.ndarray:
    """A table's start/end columns, moved onto the experiment clock."""
    return timeline.to_experiment_times(
        table[list(columns)].to_numpy().ravel()
    ).reshape(-1, 2)


def _words_by_segment(words: pd.DataFrame) -> dict[int, list[object]]:
    """Every segment's words, in the order they were spoken.

    Grouping the table once costs what a single scan of it costs; filtering it
    again for every segment is what an hour of conversation cannot afford.
    """
    usable = words.dropna(subset=["start", "end", "word"])
    usable = usable.sort_values(["segment_id", "word_index", "start"])
    grouped: dict[int, list[object]] = {}
    for word in usable.itertuples(index=False):
        grouped.setdefault(int(word.segment_id), []).append(word)
    return grouped


def _split_attribution_segments(
    transcript: pd.DataFrame,
    words: pd.DataFrame | None,
    settings: SpeechPostProcessingSettings,
) -> pd.DataFrame:
    """Split multi-utterance Whisper segments at punctuation and word gaps."""
    required = {"segment_id", "word_index", "start", "end", "word"}
    if words is None or words.empty or not required.issubset(words.columns):
        return transcript

    split_rows: list[dict[str, object]] = []

    def join_words(group: list[object]) -> str:
        text = ""
        for word in group:
            token = str(word.word).strip()
            if not token:
                continue
            if not text or token[0] in ".,!?;:%)]}–—-'’":
                text += token
            else:
                text += " " + token
        return text

    by_segment = _words_by_segment(words)
    for segment in transcript.itertuples(index=False):
        ordered_words = by_segment.get(int(segment.segment_id), [])
        if not ordered_words:
            split_rows.append(
                {
                    "segment_id": int(segment.segment_id),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": str(segment.text),
                }
            )
            continue

        gap_breaks = {
            index - 1
            for index in range(1, len(ordered_words))
            if float(ordered_words[index].start) - float(ordered_words[index - 1].end)
            >= settings.split_gap_seconds
        }
        sentence_breaks = {
            index
            for index, word in enumerate(ordered_words)
            if _SENTENCE_END.search(str(word.word).strip())
        }

        # Only split at a comma when both adjacent clauses satisfy minimum_clause_words and minimum_clause_seconds.
        comma_breaks: set[int] = set()
        if settings.split_on_comma:
            clause_limits = (
                gap_breaks
                | sentence_breaks
                | {
                    index
                    for index, word in enumerate(ordered_words)
                    if _COMMA_END.search(str(word.word).strip())
                }
            )
            limits = sorted(clause_limits | {len(ordered_words) - 1})
            for position, region_end in enumerate(limits):
                word = ordered_words[region_end]
                if _COMMA_END.search(str(word.word).strip()):
                    left_start = limits[position - 1] + 1 if position else 0
                    right_end = (
                        limits[position + 1]
                        if position + 1 < len(limits)
                        else len(ordered_words) - 1
                    )
                    left_words = ordered_words[left_start : region_end + 1]
                    right_words = ordered_words[region_end + 1 : right_end + 1]
                    left_seconds = (
                        float(left_words[-1].end) - float(left_words[0].start)
                        if left_words
                        else 0.0
                    )
                    right_seconds = (
                        float(right_words[-1].end) - float(right_words[0].start)
                        if right_words
                        else 0.0
                    )
                    if (
                        len(left_words) >= settings.minimum_clause_words
                        and len(right_words) >= settings.minimum_clause_words
                        and left_seconds >= settings.minimum_clause_seconds
                        and right_seconds >= settings.minimum_clause_seconds
                    ):
                        comma_breaks.add(region_end)

        groups: list[list[object]] = []
        current: list[object] = []
        for index, word in enumerate(ordered_words):
            if current and index - 1 in gap_breaks:
                groups.append(current)
                current = []
            current.append(word)
            if (
                settings.split_on_sentence_end and index in sentence_breaks
            ) or index in comma_breaks:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        if len(groups) <= 1:
            split_rows.append(
                {
                    "segment_id": int(segment.segment_id),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": str(segment.text),
                }
            )
            continue

        for group in groups:
            start = max(float(segment.start), float(group[0].start))
            end = min(float(segment.end), float(group[-1].end))
            if end <= start:
                end = min(float(segment.end), start + _HOP_SECONDS)
            if end <= start:
                continue
            text = join_words(group)
            if text:
                split_rows.append(
                    {
                        "segment_id": int(segment.segment_id),
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                )

    return pd.DataFrame(split_rows, columns=["segment_id", "start", "end", "text"])


@dataclass
class _Spoken:
    """One recording's words, in the order they were spoken."""

    #: When each word was said, on the experiment clock, ascending.
    at: np.ndarray
    words: list[str]


def _spoken_words(
    words: dict[str, pd.DataFrame | None], timelines: dict[str, Timeline]
) -> dict[str, _Spoken]:
    """Every recording's words, timed on the experiment clock by their middles."""
    spoken = {}
    for name, table in words.items():
        if table is None or table.empty:
            continue
        bounds = _on_experiment_clock(table, timelines.get(name, Timeline()))
        at = bounds.mean(axis=1)
        # In time order, so that a stretch of it is a slice of the words.
        order = np.argsort(at, kind="stable")
        spoken[name] = _Spoken(
            at[order],
            table["word"].astype(str).str.lower().to_numpy()[order].tolist(),
        )
    return spoken


def _said_between(
    spoken: dict[str, _Spoken], name: str, start: float, end: float
) -> list[str]:
    """The words one recording puts inside a stretch of experiment time."""
    said = spoken.get(name)
    if said is None:
        return []
    return said.words[
        int(np.searchsorted(said.at, start, side="left")) : int(
            np.searchsorted(said.at, end, side="left")
        )
    ]


def _normalised_characters(text: str) -> str:
    """Case-fold text and remove separators which Whisper tokenizes inconsistently."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


def _fuzzy_agreement(first: str, second: str) -> float:
    """Normalized character similarity between two transcript fragments."""
    first = _normalised_characters(first)
    second = _normalised_characters(second)
    if not first or not second:
        return 0.0
    return fuzz.ratio(first, second) / 100.0


_START = attrgetter("start")


def _is_bleed(
    candidate: _Segment,
    accepted: list[_Segment],
    spoken: dict[str, _Spoken],
    settings: SpeechPostProcessingSettings,
) -> bool:
    """Whether a segment is another speaker's voice reaching this microphone.

    Bleed is the same words over again, quieter; a second person talking is
    different words. Compare both the complete overlapping segments and the
    words timed inside their shared stretch, because Whisper can split the same
    speech differently between recordings.
    """
    for turn in accepted:
        start = max(candidate.start, turn.start)
        end = min(candidate.end, turn.end)
        if end <= start:
            continue
        if _fuzzy_agreement(candidate.text, turn.text) > settings.fuzzy_agreement:
            return True
        theirs = _said_between(spoken, turn.name, start, end)
        mine = _said_between(spoken, candidate.name, start, end)
        if (
            _fuzzy_agreement(" ".join(mine), " ".join(theirs))
            > settings.fuzzy_agreement
        ):
            return True
    return False


def _accept(
    segments: list[_Segment],
    spoken: dict[str, _Spoken],
    settings: SpeechPostProcessingSettings,
    progress: Progress | None = None,
) -> list[_Segment]:
    """Keep every turn that is not a copy of one already kept, strongest first.

    The turns kept so far are held in time order, so that only the few whose
    windows can overlap a candidate are compared against it.
    """
    kept: list[_Segment] = []
    longest = 0.0
    candidates = sorted(
        segments, key=lambda s: (-s.share, -s.live, -(s.end - s.start), s.name)
    )
    for index, turn in enumerate(candidates):
        # report progress every 100 segments
        if index % 100 == 0 and progress is not None:
            if progress(index / len(candidates)) is False:
                raise AttributionCancelled
        # A kept turn can only overlap this one if it starts before this one
        # ends, and no earlier than the longest turn kept so far before it.
        first = bisect_left(kept, turn.start - longest, key=_START)
        last = bisect_left(kept, turn.end, key=_START)
        if not _is_bleed(turn, kept[first:last], spoken, settings):
            insort(kept, turn, key=_START)
            longest = max(longest, turn.end - turn.start)
    return kept


def attribute_segments(
    transcripts: dict[str, pd.DataFrame],
    levels: Levels,
    timelines: dict[str, Timeline],
    settings: SpeechPostProcessingSettings,
    words: dict[str, pd.DataFrame | None] | None = None,
    *,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Give each transcribed segment to the wearer whose microphone won it."""
    words = words or {}
    spoken = _spoken_words(words, timelines)

    segments: list[_Segment] = []
    for name, transcript in transcripts.items():
        if transcript.empty or name not in levels.ids:
            continue
        transcript = _split_attribution_segments(
            transcript,
            words.get(name),
            settings,
        )
        bounds = _on_experiment_clock(transcript, timelines.get(name, Timeline()))
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
                    levels.share(name, start, end),
                    levels.live_share(name, start, end),
                )
            )

    # only include segments which were the loudest or live for more than the ownership share threshold
    spoke = [
        segment
        for segment in segments
        if segment.share > settings.ownership_share
        or segment.live > settings.ownership_share
    ]
    rows = sorted(
        (
            (
                turn.start,
                turn.end,
                turn.name,
                turn.segment_id,
                turn.text,
            )
            for turn in _accept(spoke, spoken, settings, progress)
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
            "source_segment_id": int,
            "text": str,
        }
    )
