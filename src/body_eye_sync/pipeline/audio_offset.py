"""Estimate glasses-video ``time_offset`` proposals from embedded audio."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from body_eye_sync.experiment.video import GlassesVideo


# Estimate offsets with only one signal: a low-rate log-energy audio envelope.
"""
That is effectively taking many samples ("low" for audio, high in reality) and comparing patterns across the clips
energy meaning how the samples change and envelope rate being
"""


def _read_mono_audio_samples(video_path: Path, sample_rate: int) -> np.ndarray | None:
    """
    Decode the videos first present audio screen into a format viable for signals sample matching, in this case,
    decode the audio into as mono float samples at 8k samples a second. Later, we will make envelopes that represent
    each 400 samples (so 0.05 seconds) and match these (with some logic to prevent repetitive/peaks from messing up
    our scoring/confidence)
    """
    with av.open(str(video_path)) as container:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            return None

        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32, copy=False)


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

    hop = max(1, round(sample_rate / envelope_rate))
    seconds_per_bin = hop / sample_rate
    prepared: dict[GlassesVideo, tuple[np.ndarray, float]] = {}
    for video in usable:
        video_path = video.video_path
        data = _read_mono_audio_samples(video_path, sample_rate)
        if data is None:
            # can happen e.g. a broken mp4 file or a video without audio.
            continue

        trim = round(data.size * (1.0 - middle_fraction) / 2.0)
        if trim:
            data = data[trim : data.size - trim]
        data = data[: data.size // hop * hop]
        if data.size == 0:
            continue

        audiofied_data = data.reshape(
            -1, hop
        )  # break up samples into a multi dimensional list of every 0.05 seconds
        waveformified_data = np.square(audiofied_data)  # applicable for audio data
        # The most important line of code: this helps to try and match the audio
        # overlap of videos by characterising them.
        values = np.log1p(np.mean(waveformified_data, axis=1))
        # With defaults, sample_rate=8000 and envelope_rate=20, so hop=400. That means each envelope point summarizes 400 raw samples, i.e. 0.05 seconds
        # It is not really loudness exactly but a smoothed representation. Logp1 means silence --> clap does not have an extreme range and outsized effect
        # (which could prevent effective pattern matching based on speech, which has less extreme energy-differnece-before-log-is-applied)
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
