"""Render a synchronized grid video from an experiment's recordings."""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

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

    # Several losses may have been detected at the same local instant. Combining
    # them avoids creating empty spans while preserving their total missing time.
    shifts_at: dict[float, float] = {}
    for shift in timeline.time_shifts:
        shifts_at[float(shift.at)] = shifts_at.get(float(shift.at), 0.0) + float(
            shift.seconds
        )

    missing = sum(seconds for at, seconds in shifts_at.items() if at <= 0.0)
    cursor = 0.0
    spans: list[ObservedSpan] = []
    for at in sorted(at for at in shifts_at if 0.0 < at < duration):
        spans.append(
            ObservedSpan(
                local_start=cursor,
                local_end=at,
                experiment_start=(
                    timeline.time_scale * cursor + timeline.time_offset + missing
                ),
                experiment_end=(
                    timeline.time_scale * at + timeline.time_offset + missing
                ),
            )
        )
        missing += shifts_at[at]
        cursor = at

    spans.append(
        ObservedSpan(
            local_start=cursor,
            local_end=duration,
            experiment_start=(
                timeline.time_scale * cursor + timeline.time_offset + missing
            ),
            experiment_end=(
                timeline.time_scale * duration + timeline.time_offset + missing
            ),
        )
    )
    return spans


def _clip_spans(
    spans: list[ObservedSpan], start: float, end: float, scale: float
) -> list[ObservedSpan]:
    """Clip observed spans to the requested experiment interval."""
    clipped: list[ObservedSpan] = []
    for span in spans:
        experiment_start = max(span.experiment_start, start)
        experiment_end = min(span.experiment_end, end)
        if experiment_end <= experiment_start:
            continue
        clipped.append(
            ObservedSpan(
                local_start=(
                    span.local_start
                    + (experiment_start - span.experiment_start) / scale
                ),
                local_end=(
                    span.local_end - (span.experiment_end - experiment_end) / scale
                ),
                experiment_start=experiment_start,
                experiment_end=experiment_end,
            )
        )
    return clipped


def _stream_duration(container, stream) -> float | None:
    """Duration of one stream, falling back to the whole container."""
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    if container.duration is not None:
        import av

        duration = float(container.duration / av.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    return None


def _probe_media(path: Path) -> _MediaInfo:
    """Read the first video and audio stream metadata from ``path``."""
    import av

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


def _find_ffmpeg(explicit: str | Path | None = None) -> Path:
    """Locate FFmpeg, including its fixed location in a conda environment."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"FFmpeg executable not found: {path}")
        return path

    candidates = (
        [Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"]
        if os.name == "nt"
        else [Path(sys.prefix) / "bin" / "ffmpeg"]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("ffmpeg")
    if found is None:
        raise VideoGridError(
            "FFmpeg is required to construct a video grid but was not found"
        )
    return Path(found)


def _filter_graph_option(ffmpeg: Path) -> str:
    """Use the non-deprecated graph-file option when FFmpeg supports it."""
    try:
        result = subprocess.run(
            [str(ffmpeg), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "-filter_complex_script"
    match = re.search(r"^ffmpeg version\s+(\d+)", result.stdout)
    if match and int(match.group(1)) >= 7:
        return "-/filter_complex"
    return "-filter_complex_script"


def _number(value: float) -> str:
    """A stable decimal representation suitable for an FFmpeg filter option."""
    return f"{value:.12g}"


def _filter_text(value: str) -> str:
    """Escape an input id for a single-quoted FFmpeg filter option."""
    value = re.sub(r"[\r\n\x00-\x1f]", " ", value)
    return (
        value.replace("\\", "\\\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
        .replace("%", r"\%")
    )


def _tempo_filters(scale: float) -> list[str]:
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
    return [f"atempo={_number(factor)}" for factor in factors]


def _split_source(
    lines: list[str], source: str, count: int, prefix: str, *, audio: bool = False
) -> list[str]:
    """Return one filter label per consumer of an input stream."""
    if count == 1:
        return [source]
    labels = [f"{prefix}{index}" for index in range(count)]
    split = "asplit" if audio else "split"
    lines.append(f"{source}{split}={count}" + "".join(f"[{v}]" for v in labels))
    return [f"[{label}]" for label in labels]


def _video_cell(
    lines: list[str],
    source: _Source,
    spans: list[ObservedSpan],
    output_start: float,
    output_duration: float,
    cell_width: int,
    cell_height: int,
    cell_index: int,
    show_labels: bool,
) -> str:
    """Add one full-duration, synchronized grid cell to a filter graph."""
    base = f"vbase{cell_index}"
    lines.append(
        f"color=c=black:s={cell_width}x{cell_height}:r={OUTPUT_FPS}:"
        f"d={_number(output_duration)},setsar=1[{base}]"
    )
    if not spans:
        current = f"[{base}]"
    else:
        stream = f"[{source.input_index}:{source.media.video.index}]"
        inputs = _split_source(lines, stream, len(spans), f"vsplit{cell_index}_")
        current = f"[{base}]"
        for span_index, (input_label, span) in enumerate(zip(inputs, spans)):
            segment = f"vseg{cell_index}_{span_index}"
            overlaid = f"vover{cell_index}_{span_index}"
            delay = span.experiment_start - output_start
            lines.append(
                f"{input_label}trim=start={_number(span.local_start)}:"
                f"end={_number(span.local_end)},"
                f"setpts=(PTS-STARTPTS)*{_number(source.data.time_scale)},"
                f"fps={OUTPUT_FPS}:round=near,"
                f"scale={cell_width}:{cell_height}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1,setpts=PTS+{_number(delay)}/TB[{segment}]"
            )
            lines.append(
                f"{current}[{segment}]overlay=eof_action=pass:shortest=0[{overlaid}]"
            )
            current = f"[{overlaid}]"

    output = f"cell{cell_index}"
    filters = [f"fps={OUTPUT_FPS}", f"trim=duration={_number(output_duration)}"]
    if show_labels:
        filters.append(
            "drawtext="
            f"text='{_filter_text(source.data.id)}':"
            "expansion=none:x=10:y=10:fontsize=24:fontcolor=white:"
            "box=1:boxcolor=black@0.6:boxborderw=5"
        )
    lines.append(f"{current}" + ",".join(filters) + f"[{output}]")
    return f"[{output}]"


def _audio_track(
    lines: list[str],
    source: _Source,
    spans: list[ObservedSpan],
    output_start: float,
    output_duration: float,
    track_index: int,
) -> str:
    """Add one padded, synchronized audio track to a filter graph."""
    base = f"abase{track_index}"
    lines.append(
        f"anullsrc=r={_AUDIO_SAMPLE_RATE}:cl=stereo,"
        f"atrim=duration={_number(output_duration)},asetpts=PTS-STARTPTS[{base}]"
    )
    if not spans:
        output = f"aout{track_index}"
        lines.append(f"[{base}]anull[{output}]")
        return f"[{output}]"

    stream = f"[{source.input_index}:{source.media.audio.index}]"
    inputs = _split_source(
        lines, stream, len(spans), f"asplit{track_index}_", audio=True
    )
    segments: list[str] = []
    for span_index, (input_label, span) in enumerate(zip(inputs, spans)):
        segment = f"aseg{track_index}_{span_index}"
        delay_samples = round(
            (span.experiment_start - output_start) * _AUDIO_SAMPLE_RATE
        )
        filters = [
            f"atrim=start={_number(span.local_start)}:end={_number(span.local_end)}",
            "asetpts=PTS-STARTPTS",
            *_tempo_filters(source.data.time_scale),
            f"aresample={_AUDIO_SAMPLE_RATE}",
            (
                "aformat=sample_fmts=fltp:"
                f"sample_rates={_AUDIO_SAMPLE_RATE}:channel_layouts=stereo"
            ),
            f"adelay=delays={delay_samples}S:all=1",
        ]
        lines.append(f"{input_label}" + ",".join(filters) + f"[{segment}]")
        segments.append(f"[{segment}]")

    output = f"aout{track_index}"
    inputs_to_mix = f"[{base}]" + "".join(segments)
    lines.append(
        f"{inputs_to_mix}amix=inputs={len(segments) + 1}:duration=first:"
        "normalize=0:dropout_transition=0,"
        f"atrim=duration={_number(output_duration)},asetpts=PTS-STARTPTS[{output}]"
    )
    return f"[{output}]"


def _filter_graph(
    video_sources: list[_Source],
    audio_sources: list[_Source],
    video_spans: list[list[ObservedSpan]],
    audio_spans: list[list[ObservedSpan]],
    output_start: float,
    output_end: float,
    columns: int,
    cell_size: tuple[int, int],
    show_labels: bool,
    include_merged_audio: bool,
) -> tuple[str, list[str]]:
    """Return an FFmpeg graph and its output audio labels."""
    lines: list[str] = []
    duration = output_end - output_start
    cell_width, cell_height = cell_size
    cells = [
        _video_cell(
            lines,
            source,
            spans,
            output_start,
            duration,
            cell_width,
            cell_height,
            index,
            show_labels,
        )
        for index, (source, spans) in enumerate(zip(video_sources, video_spans))
    ]

    rows = math.ceil(len(video_sources) / columns)
    while len(cells) < rows * columns:
        index = len(cells)
        label = f"empty{index}"
        lines.append(
            f"color=c=black:s={cell_width}x{cell_height}:r={OUTPUT_FPS}:"
            f"d={_number(duration)},setsar=1[{label}]"
        )
        cells.append(f"[{label}]")

    if len(cells) == 1:
        lines.append(f"{cells[0]}null,format=yuv420p[vout]")
    else:
        layout = "|".join(
            f"{index % columns * cell_width}_{index // columns * cell_height}"
            for index in range(len(cells))
        )
        lines.append(
            "".join(cells) + f"xstack=inputs={len(cells)}:layout={layout}:fill=black,"
            "format=yuv420p[vout]"
        )

    audio_labels = [
        _audio_track(
            lines,
            source,
            spans,
            output_start,
            duration,
            index,
        )
        for index, (source, spans) in enumerate(zip(audio_sources, audio_spans))
    ]
    if include_merged_audio and audio_labels:
        separate_labels: list[str] = []
        merged_inputs: list[str] = []
        for index, label in enumerate(audio_labels):
            separate, merged = _split_source(
                lines, label, 2, f"amixsplit{index}_", audio=True
            )
            separate_labels.append(separate)
            merged_inputs.append(merged)
        lines.append(
            "".join(merged_inputs) + f"amix=inputs={len(merged_inputs)}:duration=first:"
            "normalize=1:dropout_transition=0,"
            f"atrim=duration={_number(duration)},"
            "asetpts=PTS-STARTPTS[amerged]"
        )
        audio_labels = [*separate_labels, "[amerged]"]
    return ";\n".join(lines), audio_labels


def _run_ffmpeg(
    command: list[str],
    duration: float,
    progress: Callable[[float], bool] | None,
) -> None:
    """Run FFmpeg, reporting progress and preserving its error output."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        cancelled = False
        try:
            for line in process.stdout:
                key, separator, value = line.strip().partition("=")
                if separator and key in ("out_time_us", "out_time_ms"):
                    # Despite its historical name, out_time_ms is expressed in
                    # microseconds too.
                    try:
                        elapsed = float(value) / 1_000_000
                    except ValueError:  # N/A before the first output packet.
                        continue
                    fraction = min(1.0, max(0.0, elapsed / duration))
                    if progress is not None and not progress(fraction):
                        cancelled = True
                        process.terminate()
                        break
            try:
                return_code = process.wait(timeout=10 if cancelled else None)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if cancelled:
            raise VideoGridCancelled("video-grid construction was cancelled")
        if return_code != 0:
            errors.seek(0)
            detail = errors.read().strip()
            raise VideoGridError(
                f"FFmpeg failed with exit code {return_code}"
                + (f":\n{detail}" if detail else "")
            )
    if progress is not None:
        progress(1.0)


def construct_video_grid(
    experiment: Experiment,
    output_path: str | Path,
    *,
    columns: int | None = None,
    cell_size: tuple[int, int] = (640, 360),
    start_time: float | None = None,
    end_time: float | None = None,
    show_labels: bool = True,
    input_ids: Iterable[str] | None = None,
    include_merged_audio: bool = False,
    overwrite: bool = False,
    progress: Callable[[float], bool] | None = None,
    ffmpeg_path: str | Path | None = None,
) -> VideoGridResult:
    """Write selected experiment videos as a synchronized 25 fps grid.

    The output interval defaults to the union of the video recordings on the
    experiment clock. ``input_ids`` defaults to every experiment input. Every
    selected input that carries audio contributes a separately selectable,
    full-duration audio track named after its input id; selected audio-only
    inputs contribute a track but no grid cell. ``include_merged_audio`` appends
    a track mixing all of those synchronized source tracks. Before and after a
    recording, and wherever it lost content, its cell is black and its audio
    track is silent.

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

    all_video_spans = [
        _observed_spans(source.data, source.media.video.duration)
        for source in video_sources
    ]
    union_start = min(spans[0].experiment_start for spans in all_video_spans)
    union_end = max(spans[-1].experiment_end for spans in all_video_spans)
    output_start = union_start if start_time is None else float(start_time)
    output_end = union_end if end_time is None else float(end_time)
    if not math.isfinite(output_start) or not math.isfinite(output_end):
        raise ValueError("output times must be finite")
    if output_end <= output_start:
        raise ValueError("end_time must be later than start_time")

    video_spans = [
        _clip_spans(spans, output_start, output_end, source.data.time_scale)
        for source, spans in zip(video_sources, all_video_spans)
    ]
    audio_sources = [source for source in sources if source.media.audio is not None]
    audio_spans = [
        _clip_spans(
            _observed_spans(source.data, source.media.audio.duration),
            output_start,
            output_end,
            source.data.time_scale,
        )
        for source in audio_sources
    ]

    column_count = columns or math.ceil(math.sqrt(len(videos)))
    column_count = min(column_count, len(videos))
    row_count = math.ceil(len(videos) / column_count)
    graph, audio_labels = _filter_graph(
        video_sources,
        audio_sources,
        video_spans,
        audio_spans,
        output_start,
        output_end,
        column_count,
        cell_size,
        show_labels,
        include_merged_audio,
    )

    ffmpeg = _find_ffmpeg(ffmpeg_path)
    temporary_output = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}{output_path.suffix}"
    )
    duration = output_end - output_start
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ffgraph", encoding="utf-8", delete=False
        ) as graph_file:
            graph_file.write(graph)
            graph_path = Path(graph_file.name)
        try:
            command = [str(ffmpeg), "-hide_banner", "-loglevel", "warning", "-y"]
            for source in sources:
                command.extend(["-i", str(source.media.path)])
            command.extend(
                [
                    _filter_graph_option(ffmpeg),
                    str(graph_path),
                    "-map",
                    "[vout]",
                ]
            )
            for label in audio_labels:
                command.extend(["-map", label])
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(OUTPUT_FPS),
                ]
            )
            if audio_labels:
                command.extend(
                    [
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-ar",
                        str(_AUDIO_SAMPLE_RATE),
                        "-ac",
                        "2",
                    ]
                )
                audio_track_names = [source.data.id for source in audio_sources]
                if include_merged_audio:
                    audio_track_names.append(_MERGED_AUDIO_TRACK)
                default_index = (
                    len(audio_track_names) - 1 if include_merged_audio else 0
                )
                for index, name in enumerate(audio_track_names):
                    command.extend(
                        [
                            f"-metadata:s:a:{index}",
                            f"title={name}",
                            f"-metadata:s:a:{index}",
                            f"handler_name={name}",
                            f"-disposition:a:{index}",
                            "default" if index == default_index else "0",
                        ]
                    )
            command.extend(
                [
                    "-t",
                    _number(duration),
                    "-movflags",
                    "+faststart",
                    "-progress",
                    "pipe:1",
                    "-nostats",
                    str(temporary_output),
                ]
            )
            _run_ffmpeg(command, duration, progress)
        finally:
            graph_path.unlink(missing_ok=True)
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
