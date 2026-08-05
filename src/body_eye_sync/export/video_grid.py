"""Render a synchronized grid video from an experiment's recordings."""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.timeline import Timeline
from body_eye_sync.experiment.video import Video

logger = logging.getLogger(__name__)

OUTPUT_FPS = 25
_AUDIO_SAMPLE_RATE = 48_000
_MERGED_AUDIO_TRACK = "Merged audio"


class VideoGridError(RuntimeError):
    """The synchronized video could not be constructed."""


class VideoGridCancelled(VideoGridError):
    """Construction was cancelled through its progress callback."""


@dataclass(frozen=True)
class ObservedSpan:
    """One uninterrupted piece of a recording on both of its clocks."""

    local_start: float
    local_end: float
    experiment_start: float
    experiment_end: float


@dataclass(frozen=True)
class _StreamInfo:
    """The first usable stream of one media kind in an input file."""

    index: int
    duration: float


@dataclass(frozen=True)
class _MediaInfo:
    """The video and audio streams relevant to one experiment input."""

    path: Path
    video: _StreamInfo | None
    audio: _StreamInfo | None


@dataclass(frozen=True)
class VideoGridResult:
    """Description of a completed synchronized grid video."""

    path: Path
    experiment_start: float
    experiment_end: float
    columns: int
    rows: int
    audio_tracks: tuple[str, ...]


@dataclass(frozen=True)
class _Source:
    data: Video | Audio
    input_index: int
    media: _MediaInfo


@dataclass(frozen=True)
class _LabelOverlay:
    """The non-transparent part of a pre-rendered input label."""

    x: int
    y: int
    colors: np.ndarray
    alpha: np.ndarray


def _validate_timeline(timeline: Timeline) -> None:
    """Reject runtime timeline edits that cannot describe media timing."""
    if not math.isfinite(timeline.time_offset):
        raise ValueError(f"input {timeline.id!r} has a non-finite time offset")
    if not math.isfinite(timeline.time_scale) or timeline.time_scale <= 0:
        raise ValueError(f"input {timeline.id!r} has an invalid time scale")
    for shift in timeline.time_shifts:
        if not math.isfinite(shift.at):
            raise ValueError(f"input {timeline.id!r} has a non-finite shift time")
        if not math.isfinite(shift.seconds) or shift.seconds <= 0:
            raise ValueError(f"input {timeline.id!r} has an invalid shift duration")


def _observed_spans(timeline: Timeline, duration: float) -> list[ObservedSpan]:
    """Split ``[0, duration)`` wherever the recording lost content."""
    if not math.isfinite(duration) or duration <= 0:
        return []

    cursor = 0.0
    spans: list[ObservedSpan] = []
    shifts = sorted(timeline.time_shifts, key=lambda shift: shift.at)
    for shift in shifts:
        at = float(shift.at)
        if not 0.0 < at < duration:
            continue
        experiment_start = timeline.to_experiment_time(cursor)
        spans.append(
            ObservedSpan(
                local_start=cursor,
                local_end=at,
                experiment_start=experiment_start,
                experiment_end=(experiment_start + timeline.time_scale * (at - cursor)),
            )
        )
        cursor = at

    experiment_start = timeline.to_experiment_time(cursor)
    spans.append(
        ObservedSpan(
            local_start=cursor,
            local_end=duration,
            experiment_start=experiment_start,
            experiment_end=(
                experiment_start + timeline.time_scale * (duration - cursor)
            ),
        )
    )
    return spans


def _stream_duration(container, stream) -> float | None:
    """Duration of one stream, falling back to the whole container."""
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    if container.duration is not None:
        duration = float(container.duration / av.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    return None


def _probe_media(path: Path) -> _MediaInfo:
    """Read the first video and audio stream metadata from ``path``."""
    if not path.exists():
        raise FileNotFoundError(f"input media not found: {path}")
    try:
        with av.open(str(path)) as container:
            video_stream = next(iter(container.streams.video), None)
            audio_stream = next(iter(container.streams.audio), None)
            video_duration = (
                _stream_duration(container, video_stream)
                if video_stream is not None
                else None
            )
            audio_duration = (
                _stream_duration(container, audio_stream)
                if audio_stream is not None
                else None
            )
            return _MediaInfo(
                path=path,
                video=(
                    _StreamInfo(video_stream.index, video_duration)
                    if video_stream is not None and video_duration is not None
                    else None
                ),
                audio=(
                    _StreamInfo(audio_stream.index, audio_duration)
                    if audio_stream is not None and audio_duration is not None
                    else None
                ),
            )
    except (OSError, ValueError, av.FFmpegError) as exc:
        raise VideoGridError(f"could not read input media {path}: {exc}") from exc


def _number(value: float) -> str:
    """A stable decimal representation suitable for an FFmpeg filter option."""
    return f"{value:.12g}"


def _tempo_factors(scale: float) -> list[float]:
    """Pitch-preserving tempo factors whose product is ``1 / scale``."""
    tempo = 1.0 / scale
    factors: list[float] = []
    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5
    while tempo > 100.0:
        factors.append(100.0)
        tempo /= 100.0
    factors.append(tempo)
    return factors


def _link_chain(filters: list) -> None:
    """Connect adjacent PyAV filter contexts."""
    for source, destination in zip(filters, filters[1:]):
        source.link_to(destination)


class _VideoSampler:
    """Decode one video sequentially and sample it on experiment time."""

    def __init__(
        self,
        source: _Source,
        spans: list[ObservedSpan],
        cell_size: tuple[int, int],
    ) -> None:
        assert source.media.video is not None
        self.source = source
        self.spans = spans
        self.width, self.height = cell_size
        self.container = av.open(str(source.media.path))
        self.stream = self.container.streams[source.media.video.index]
        self._frames = iter(self.container.decode(self.stream))
        self._origin: float | None = None
        self._previous: tuple[float, av.VideoFrame] | None = None
        try:
            self._current = self._next_frame()
        except BaseException:
            self.container.close()
            raise
        self._span_index = 0
        self._cached_frame: av.VideoFrame | None = None
        self._cached_cell: np.ndarray | None = None

    def close(self) -> None:
        self.container.close()

    def _next_frame(self) -> tuple[float, av.VideoFrame] | None:
        for frame in self._frames:
            if frame.pts is None or frame.time_base is None:
                continue
            absolute_time = float(frame.pts * frame.time_base)
            if self._origin is None:
                self._origin = absolute_time
            return absolute_time - self._origin, frame
        return None

    def _local_time(self, experiment_time: float) -> float | None:
        while (
            self._span_index < len(self.spans)
            and experiment_time >= self.spans[self._span_index].experiment_end
        ):
            self._span_index += 1
        if self._span_index >= len(self.spans):
            return None
        span = self.spans[self._span_index]
        if experiment_time < span.experiment_start:
            return None
        return (
            span.local_start
            + (experiment_time - span.experiment_start) / self.source.data.time_scale
        )

    def _frame_at(self, local_time: float) -> av.VideoFrame | None:
        while self._current is not None and self._current[0] < local_time:
            self._previous = self._current
            self._current = self._next_frame()
        candidates = [
            value for value in (self._previous, self._current) if value is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda value: abs(value[0] - local_time))[1]

    def cell_at(self, experiment_time: float) -> np.ndarray | None:
        local_time = self._local_time(experiment_time)
        if local_time is None:
            return None
        frame = self._frame_at(local_time)
        if frame is None:
            return None
        if frame is self._cached_frame:
            return self._cached_cell

        scale = min(self.width / frame.width, self.height / frame.height)
        width = max(1, min(self.width, round(frame.width * scale)))
        height = max(1, min(self.height, round(frame.height * scale)))
        image = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
        cell = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        x = (self.width - width) // 2
        y = (self.height - height) // 2
        cell[y : y + height, x : x + width] = image
        self._cached_frame = frame
        self._cached_cell = cell
        return cell


def _label_overlay(text: str, size: tuple[int, int]) -> _LabelOverlay:
    """Render the label once; it is blended onto every cell frame."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    text = "".join(" " if ord(character) < 32 else character for character in text)
    left, top, right, bottom = draw.textbbox((10, 10), text, font=font)
    border = 5
    draw.rectangle(
        (
            max(0, left - border),
            max(0, top - border),
            min(width, right + border),
            min(height, bottom + border),
        ),
        fill=(0, 0, 0, 153),
    )
    draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))
    bounds = image.getbbox()
    if bounds is None:
        return _LabelOverlay(
            0,
            0,
            np.empty((0, 0, 3), dtype=np.float32),
            np.empty((0, 0, 1), dtype=np.float32),
        )
    x, y, right, bottom = bounds
    pixels = np.asarray(image.crop(bounds)).astype(np.float32)
    return _LabelOverlay(x, y, pixels[:, :, :3], pixels[:, :, 3:4] / 255.0)


def _blend_label(cell: np.ndarray, overlay: _LabelOverlay) -> None:
    height, width = overlay.alpha.shape[:2]
    target = cell[overlay.y : overlay.y + height, overlay.x : overlay.x + width]
    target[:] = (
        target.astype(np.float32) * (1.0 - overlay.alpha)
        + overlay.colors * overlay.alpha
    ).astype(np.uint8)


class _SynchronizedAudio:
    """Stream one source through a PyAV audio graph onto the output clock."""

    def __init__(
        self,
        source: _Source,
        spans: list[ObservedSpan],
        output_start: float,
        output_duration: float,
    ) -> None:
        assert source.media.audio is not None
        self.container = av.open(str(source.media.path))
        self.stream = self.container.streams[source.media.audio.index]
        self._frames = self._filtered_frames(
            source, spans, output_start, output_duration
        )
        self._fifo = av.AudioFifo()
        self._finished = False

    def close(self) -> None:
        self._frames.close()
        self.container.close()

    def _filtered_frames(
        self,
        source: _Source,
        spans: list[ObservedSpan],
        output_start: float,
        output_duration: float,
    ) -> Iterator[av.AudioFrame]:
        graph = av.filter.Graph()
        input_filter = graph.add_abuffer(template=self.stream)
        if len(spans) == 1:
            branches = [input_filter]
        else:
            split = graph.add("asplit", str(len(spans)))
            input_filter.link_to(split)
            branches = [split] * len(spans)

        segments = []
        for index, (branch, span) in enumerate(zip(branches, spans)):
            trim = graph.add(
                "atrim",
                f"start={_number(span.local_start)}:end={_number(span.local_end)}",
            )
            if len(spans) == 1:
                branch.link_to(trim)
            else:
                branch.link_to(trim, output_idx=index)
            chain = [trim, graph.add("asetpts", "PTS-STARTPTS")]
            chain.extend(
                graph.add("atempo", _number(factor))
                for factor in _tempo_factors(source.data.time_scale)
            )
            delay_samples = max(
                0,
                round((span.experiment_start - output_start) * _AUDIO_SAMPLE_RATE),
            )
            chain.extend(
                [
                    graph.add("aresample", str(_AUDIO_SAMPLE_RATE)),
                    graph.add(
                        "aformat",
                        "sample_fmts=fltp:sample_rates="
                        f"{_AUDIO_SAMPLE_RATE}:channel_layouts=stereo",
                    ),
                    graph.add("adelay", f"delays={delay_samples}S:all=1"),
                ]
            )
            _link_chain(chain)
            segments.append(chain[-1])

        silence = graph.add("anullsrc", f"r={_AUDIO_SAMPLE_RATE}:cl=stereo")
        silence_trim = graph.add("atrim", f"duration={_number(output_duration)}")
        silence_pts = graph.add("asetpts", "PTS-STARTPTS")
        _link_chain([silence, silence_trim, silence_pts])
        mix = graph.add(
            "amix",
            f"inputs={len(segments) + 1}:duration=first:"
            "normalize=0:dropout_transition=0",
        )
        silence_pts.link_to(mix, input_idx=0)
        for index, segment in enumerate(segments, start=1):
            segment.link_to(mix, input_idx=index)
        output_trim = graph.add("atrim", f"duration={_number(output_duration)}")
        output_pts = graph.add("asetpts", "PTS-STARTPTS")
        sink = graph.add("abuffersink")
        _link_chain([mix, output_trim, output_pts, sink])
        graph.configure()

        origin: float | None = None
        ended = False
        for frame in self.container.decode(self.stream):
            if frame.pts is not None and frame.time_base is not None:
                absolute_time = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = absolute_time
                frame.pts = round((absolute_time - origin) / float(frame.time_base))
            input_filter.push(frame)
            while True:
                try:
                    yield sink.pull()
                except av.error.BlockingIOError:
                    break
                except av.error.EOFError:
                    ended = True
                    break
            if ended:
                return

        input_filter.push(None)
        while True:
            try:
                yield sink.pull()
            except av.error.BlockingIOError:
                continue
            except av.error.EOFError:
                return

    def read(self, samples: int) -> np.ndarray:
        while self._fifo.samples < samples and not self._finished:
            try:
                self._fifo.write(next(self._frames))
            except StopIteration:
                self._finished = True
        frame = self._fifo.read(samples, partial=True)
        if frame is None:
            data = np.empty((2, 0), dtype=np.float32)
        else:
            data = frame.to_ndarray().astype(np.float32, copy=False)
        if data.shape[1] < samples:
            data = np.pad(data, ((0, 0), (0, samples - data.shape[1])))
        return data


def _compose_frame(
    samplers: list[_VideoSampler],
    labels: list[_LabelOverlay] | None,
    experiment_time: float,
    columns: int,
    rows: int,
    cell_size: tuple[int, int],
) -> av.VideoFrame:
    width, height = cell_size
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, sampler in enumerate(samplers):
        cell = sampler.cell_at(experiment_time)
        if cell is None:
            cell = np.zeros((height, width, 3), dtype=np.uint8)
        elif labels is not None:
            cell = cell.copy()
        if labels is not None:
            _blend_label(cell, labels[index])
        x = index % columns * width
        y = index // columns * height
        grid[y : y + height, x : x + width] = cell
    return av.VideoFrame.from_ndarray(grid, format="rgb24")


def _encode_frame(container, stream, frame) -> None:
    for packet in stream.encode(frame):
        container.mux(packet)


def _render(
    output_path: Path,
    video_sources: list[_Source],
    audio_sources: list[_Source],
    video_spans: list[list[ObservedSpan]],
    audio_spans: list[list[ObservedSpan]],
    output_start: float,
    output_end: float,
    columns: int,
    rows: int,
    cell_size: tuple[int, int],
    show_labels: bool,
    include_merged_audio: bool,
    progress: Callable[[float], bool] | None,
) -> None:
    """Decode, synchronize, compose, encode, and mux entirely through PyAV."""
    duration = output_end - output_start
    samplers: list[_VideoSampler] = []
    audio_readers: list[_SynchronizedAudio] = []
    try:
        for source, spans in zip(video_sources, video_spans):
            samplers.append(_VideoSampler(source, spans, cell_size))
        for source, spans in zip(audio_sources, audio_spans):
            audio_readers.append(
                _SynchronizedAudio(source, spans, output_start, duration)
            )
        labels = (
            [_label_overlay(source.data.id, cell_size) for source in video_sources]
            if show_labels
            else None
        )

        with av.open(str(output_path), "w", options={"movflags": "+faststart"}) as out:
            grid_width = columns * cell_size[0]
            grid_height = rows * cell_size[1]
            video_stream = out.add_stream(
                "libx264",
                rate=OUTPUT_FPS,
                options={"preset": "medium", "crf": "18"},
            )
            video_stream.width = grid_width
            video_stream.height = grid_height
            video_stream.pix_fmt = "yuv420p"

            audio_track_names = [source.data.id for source in audio_sources]
            if include_merged_audio and audio_sources:
                audio_track_names.append(_MERGED_AUDIO_TRACK)
            audio_streams = []
            default_index = len(audio_track_names) - 1 if include_merged_audio else 0
            for index, name in enumerate(audio_track_names):
                stream = out.add_stream("aac", rate=_AUDIO_SAMPLE_RATE)
                stream.bit_rate = 192_000
                stream.layout = "stereo"
                stream.metadata["title"] = name
                stream.metadata["handler_name"] = name
                stream.disposition = (
                    av.stream.Disposition.default if index == default_index else 0
                )
                audio_streams.append(stream)

            video_frames = math.ceil(duration * OUTPUT_FPS - 1e-9)
            audio_samples = round(duration * _AUDIO_SAMPLE_RATE) if audio_streams else 0
            video_index = 0
            audio_index = 0
            last_reported = -1.0
            while video_index < video_frames or audio_index < audio_samples:
                video_time = video_index / OUTPUT_FPS
                audio_time = audio_index / _AUDIO_SAMPLE_RATE
                if video_index < video_frames and (
                    audio_index >= audio_samples or video_time <= audio_time
                ):
                    frame = _compose_frame(
                        samplers,
                        labels,
                        output_start + video_time,
                        columns,
                        rows,
                        cell_size,
                    )
                    frame.pts = video_index
                    frame.time_base = Fraction(1, OUTPUT_FPS)
                    _encode_frame(out, video_stream, frame)
                    video_index += 1
                    elapsed = video_time
                else:
                    count = min(1024, audio_samples - audio_index)
                    tracks = [reader.read(count) for reader in audio_readers]
                    if include_merged_audio and tracks:
                        tracks.append(np.mean(tracks, axis=0, dtype=np.float32))
                    for stream, data in zip(audio_streams, tracks):
                        frame = av.AudioFrame.from_ndarray(
                            data, format="fltp", layout="stereo"
                        )
                        frame.sample_rate = _AUDIO_SAMPLE_RATE
                        frame.pts = audio_index
                        frame.time_base = Fraction(1, _AUDIO_SAMPLE_RATE)
                        _encode_frame(out, stream, frame)
                    audio_index += count
                    elapsed = audio_time

                fraction = min(1.0, elapsed / duration)
                if progress is not None and fraction - last_reported >= 0.001:
                    last_reported = fraction
                    if not progress(fraction):
                        raise VideoGridCancelled(
                            "video-grid construction was cancelled"
                        )

            _encode_frame(out, video_stream, None)
            for stream in audio_streams:
                _encode_frame(out, stream, None)
        if progress is not None:
            progress(1.0)
    finally:
        for sampler in samplers:
            sampler.close()
        for reader in audio_readers:
            reader.close()


def construct_video_grid(
    experiment: Experiment,
    output_path: str | Path,
    *,
    columns: int | None = None,
    cell_size: tuple[int, int] = (640, 360),
    show_labels: bool = True,
    input_ids: Iterable[str] | None = None,
    include_merged_audio: bool = False,
    overwrite: bool = False,
    progress: Callable[[float], bool] | None = None,
) -> VideoGridResult:
    """Write selected experiment videos as a synchronized 25 fps grid.

    The output interval is the union of all selected media on the experiment
    clock. ``input_ids`` defaults to every experiment input. Every selected
    input that carries audio contributes a separately selectable, full-duration
    audio track named after its input id; selected audio-only inputs contribute
    a track but no grid cell. ``include_merged_audio`` appends a track mixing all
    of those synchronized source tracks. Before and after a recording, and
    wherever it lost content, its cell is black and its audio track is silent.

    ``progress`` receives fractions between zero and one. Returning ``False``
    cancels construction and leaves no partial output behind.
    """
    all_inputs = {data.id: data for data in experiment.inputs}
    if input_ids is None:
        selected_ids = set(all_inputs)
    else:
        if isinstance(input_ids, (str, bytes)):
            raise TypeError("input_ids must be an iterable of input ids, not a string")
        selected_ids = set(input_ids)
        unknown_ids = selected_ids - set(all_inputs)
        if unknown_ids:
            raise ValueError(f"unknown input ids: {sorted(unknown_ids)}")

    selected_inputs = [data for data in experiment.inputs if data.id in selected_ids]
    videos = [
        video
        for video in [*experiment.glasses_videos, *experiment.fixed_videos]
        if video.id in selected_ids
    ]
    if not videos:
        raise ValueError("no video inputs selected")
    if columns is not None and columns <= 0:
        raise ValueError("columns must be positive")
    if len(cell_size) != 2 or any(value <= 0 or value % 2 for value in cell_size):
        raise ValueError("cell dimensions must be positive even integers")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("video-grid output path must have an .mp4 extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources: list[_Source] = []
    sources_by_data: dict[int, _Source] = {}
    for data in selected_inputs:
        _validate_timeline(data)
        if data.path is None:
            raise FileNotFoundError(f"input {data.id!r} has no media path")
        source = _Source(data, len(sources), _probe_media(data.path))
        sources.append(source)
        sources_by_data[id(data)] = source

    video_sources = [sources_by_data[id(video)] for video in videos]
    for source in video_sources:
        if source.media.video is None:
            raise VideoGridError(f"input {source.data.id!r} has no usable video stream")

    video_spans = [
        _observed_spans(source.data, source.media.video.duration)
        for source in video_sources
    ]
    audio_sources = [source for source in sources if source.media.audio is not None]
    audio_spans = [
        _observed_spans(source.data, source.media.audio.duration)
        for source in audio_sources
    ]
    media_spans = [*video_spans, *audio_spans]
    output_start = min(spans[0].experiment_start for spans in media_spans)
    output_end = max(spans[-1].experiment_end for spans in media_spans)

    column_count = columns or math.ceil(math.sqrt(len(videos)))
    column_count = min(column_count, len(videos))
    row_count = math.ceil(len(videos) / column_count)
    temporary_output = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}{output_path.suffix}"
    )
    try:
        try:
            _render(
                temporary_output,
                video_sources,
                audio_sources,
                video_spans,
                audio_spans,
                output_start,
                output_end,
                column_count,
                row_count,
                cell_size,
                show_labels,
                include_merged_audio,
                progress,
            )
        except (OSError, ValueError, av.FFmpegError) as exc:
            raise VideoGridError(f"could not construct video grid: {exc}") from exc
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output_path}")
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    logger.info("wrote synchronized video grid to %s", output_path)
    return VideoGridResult(
        path=output_path,
        experiment_start=output_start,
        experiment_end=output_end,
        columns=column_count,
        rows=row_count,
        audio_tracks=(
            tuple(source.data.id for source in audio_sources)
            + ((_MERGED_AUDIO_TRACK,) if include_merged_audio and audio_sources else ())
        ),
    )
