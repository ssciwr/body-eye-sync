"""Estimate glasses-video ``time_offset`` proposals from embedded audio."""

from __future__ import annotations

import subprocess

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from body_eye_sync.experiment.video import GlassesVideo


# Estimate offsets with only one signal: a low-rate log-energy audio envelope.
"""
That is effectively taking many samples ("low" for audio, high in reality) and comparing patterns across the clips
energy meaning how the samples change and envelope rate being
"""


def assign_automatic_estimated_offset(
    *videos: GlassesVideo,
    reference: GlassesVideo | None = None,
    sample_rate: int = 8_000,
    envelope_rate: int = 20,
    max_lag_seconds: float = 300.0,
    min_score: float = 3.0,
    middle_fraction: float = 0.8,
) -> dict[GlassesVideo, float]:
    offsets = {video: 0.0 for video in videos}
    usable = [video for video in videos if video.video_path is not None]
    if len(usable) < 2 or (reference is not None and reference not in usable):
        return offsets
    if sample_rate <= 0 or envelope_rate <= 0 or not 0.0 < middle_fraction <= 1.0:
        return offsets

    ffmpeg = get_ffmpeg_exe()
    hop = max(1, round(sample_rate / envelope_rate))
    seconds_per_bin = hop / sample_rate
    prepared: dict[GlassesVideo, tuple[np.ndarray, float]] = {}
    # todo: document this command
    for video in usable:
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video.video_path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                check=False,
                capture_output=True,
            )
        except OSError:
            continue
        if result.returncode != 0:
            # can happen e.g. my broken mp4 file.
            continue

        data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
        trim = round(data.size * (1.0 - middle_fraction) / 2.0)
        if trim:
            data = data[trim : data.size - trim]
        data = data[: data.size // hop * hop]
        if data.size == 0:
            continue

        # The most important line of code: this helps to try and match the audio
        # overlap of videos by characterising them.
        values = np.log1p(np.mean(np.square(data.reshape(-1, hop)), axis=1))
        values -= np.mean(values)
        norm = float(np.linalg.norm(values))
        if norm == 0:
            # cannot really determine a difference for this audio portion
            continue
        prepared[video] = (values / norm, trim / sample_rate)

    reference_video = reference or next(
        (video for video in usable if video in prepared), None
    )
    if reference_video not in prepared or len(prepared) < 2:
        return offsets

    source, source_start = prepared[reference_video]
    # Below is largely AI-written to try and match videos/offset
    for video, (target, target_start) in prepared.items():
        if video is reference_video:
            continue
        length = source.size + target.size - 1
        fft_size = 1 << (length - 1).bit_length()
        raw = np.fft.irfft(
            np.fft.rfft(source, fft_size) * np.conj(np.fft.rfft(target, fft_size)),
            fft_size,
        )
        correlation = np.concatenate((raw[-(target.size - 1) :], raw[: source.size]))
        proposed = (
            np.arange(-(target.size - 1), source.size) * seconds_per_bin
            + source_start
            - target_start
        )
        keep = np.abs(proposed) <= max_lag_seconds
        if not np.any(keep):
            continue

        scores = correlation[keep]
        candidates = proposed[keep]
        best = int(np.argmax(scores))
        background = scores[
            np.abs(candidates - candidates[best]) > 5.0
        ]  # avoid comparing data within 5 seconds of this one
        spread = float(np.std(background)) if background.size else 0.0
        if spread == 0.0:
            continue
        # This is the most important identifying signal for our best offset candidate
        score = (float(scores[best]) - float(np.median(background))) / spread
        if score >= min_score:
            offsets[video] = float(candidates[best])
    return offsets
