"""Experiment contains the loaded inputs, their results and the pipeline to run"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import (
    CURRENT_VERSION,
    AudioInput,
    ExperimentConfig,
    FixedVideoInput,
    GlassesVideoInput,
    Pipeline,
    validate_input_id,
)
from body_eye_sync.experiment.video import FixedVideo, GlassesVideo, Video

logger = logging.getLogger(__name__)

# An experiment lives in a folder: this config file plus an outputs/ subfolder.
CONFIG_FILENAME = "experiment.yaml"
OUTPUTS_DIRNAME = "outputs"
RESULTS_FILENAME = "results.parquet"


class Experiment:
    """An experiment: its inputs, their results, and the pipeline to run.

    The inputs own their settings, each in the runtime class for its type;
    :class:`ExperimentConfig` is the on-disk form, converted to and from when
    the experiment is saved and loaded. Input ids are unique across the types,
    and inputs are added, removed and renamed through this class so they stay
    that way.
    """

    def __init__(self, config: ExperimentConfig, folder: str | Path | None = None):
        self.folder = Path(folder) if folder is not None else None
        self.pipeline: Pipeline = config.pipeline
        self.glasses_videos = [
            GlassesVideo(
                spec.id,
                self._resolve(spec.path),
                self._resolve(spec.gaze_path),
                spec.time_offset,
            )
            for spec in config.glasses_videos
        ]
        self.fixed_videos = [
            FixedVideo(spec.id, self._resolve(spec.path), spec.time_offset)
            for spec in config.fixed_videos
        ]
        glasses_by_id = {video.id: video for video in self.glasses_videos}
        self.audio = [
            Audio(
                spec.id,
                self._resolve(spec.path),
                spec.time_offset,
                glasses_by_id.get(spec.glasses_video),
            )
            for spec in config.audio
        ]
        for data in self.inputs:
            self._load_results(data)

    @property
    def inputs(self) -> list[Video | Audio]:
        """Every input of every type, for what applies to all of them."""
        return [*self.glasses_videos, *self.fixed_videos, *self.audio]

    def add_glasses_video(self, spec: GlassesVideoInput) -> GlassesVideo:
        """Add a glasses video input, returning its :class:`GlassesVideo`."""
        self._check_id(spec.id)
        video = GlassesVideo(
            spec.id,
            self._resolve(spec.path),
            self._resolve(spec.gaze_path),
            spec.time_offset,
        )
        self._discard_results(video.id)
        self.glasses_videos.append(video)
        return video

    def add_fixed_video(self, spec: FixedVideoInput) -> FixedVideo:
        """Add a fixed video input, returning its :class:`FixedVideo`."""
        self._check_id(spec.id)
        video = FixedVideo(spec.id, self._resolve(spec.path), spec.time_offset)
        self._discard_results(video.id)
        self.fixed_videos.append(video)
        return video

    def add_audio(self, spec: AudioInput) -> Audio:
        """Add an audio input, returning its :class:`Audio`.

        Raises :class:`ValueError` if it names a glasses video that is not in
        this experiment.
        """
        self._check_id(spec.id)
        glasses_video = None
        if spec.glasses_video is not None:
            glasses_video = next(
                (v for v in self.glasses_videos if v.id == spec.glasses_video), None
            )
            if glasses_video is None:
                raise ValueError(f"unknown glasses video id: {spec.glasses_video!r}")
        audio = Audio(
            spec.id, self._resolve(spec.path), spec.time_offset, glasses_video
        )
        self._discard_results(audio.id)
        self.audio.append(audio)
        return audio

    def remove_input(self, data: Video | Audio) -> None:
        """Remove an input from the experiment, leaving its output files alone.

        Raises :class:`ValueError` if it is a glasses video that audio inputs
        still refer to.
        """
        if isinstance(data, GlassesVideo):
            used_by = sorted(a.id for a in self.audio if a.glasses_video is data)
            if used_by:
                raise ValueError(
                    f"glasses video {data.id!r} is still used by audio inputs: {used_by}"
                )
        for inputs in (self.glasses_videos, self.fixed_videos, self.audio):
            if any(existing is data for existing in inputs):
                inputs.remove(data)
                return
        raise ValueError(f"input {data.id!r} is not in this experiment")

    def rename_input(self, data: Video | Audio, new_id: str) -> None:
        """Give an input a new id, moving its output directory if it exists."""
        if new_id == data.id:
            return
        self._check_id(new_id)
        if self.folder is not None:
            old_output_dir = self._input_output_dir(data.id)
            new_output_dir = self._input_output_dir(new_id)
            if new_output_dir.exists():
                raise ValueError(f"outputs already exist for input id: {new_id!r}")
            if old_output_dir.exists():
                old_output_dir.rename(new_output_dir)
        data.id = new_id

    def _check_id(self, input_id: str) -> None:
        """Check an id can name an output directory, and nothing else uses it."""
        validate_input_id(input_id)
        if any(data.id == input_id for data in self.inputs):
            raise ValueError(f"duplicate input id: {input_id!r}")

    def _require_folder(self) -> Path:
        if self.folder is None:
            raise ValueError("experiment has no folder; load or save it first")
        return self.folder

    def _resolve(self, path: Path) -> Path:
        """An input path as used at runtime: absolute where the folder allows."""
        if path.is_absolute() or self.folder is None:
            return path
        return (self.folder / path).resolve()

    def _store(self, path: Path) -> Path:
        """An input path as written to disk: relative where it is under the folder."""
        if self.folder is not None and path.is_relative_to(self.folder.resolve()):
            return path.relative_to(self.folder.resolve())
        return path

    @property
    def output_dir(self) -> Path:
        """Where per-input Parquet outputs live, inside the folder."""
        return self._require_folder() / OUTPUTS_DIRNAME

    def _input_output_dir(self, input_id: str) -> Path:
        """The application-owned output directory for one input."""
        return self.output_dir / input_id

    def output_path(self, data: Video | Audio) -> Path:
        """Main Parquet output path inside an input's output directory."""
        return self._input_output_dir(data.id) / RESULTS_FILENAME

    def _load_results(self, data: Video | Audio) -> None:
        """Fill an input from its output file, if the experiment has one, skip if invalid."""
        if self.folder is None or not self.output_path(data).exists():
            return
        try:
            data.load_parquet(self.output_path(data))
        except (OSError, ValueError) as exc:
            data.clear()
            logger.warning(
                "ignoring unreadable results for input %r in %s: %s",
                data.id,
                self.output_path(data),
                exc,
            )

    def _discard_results(self, input_id: str) -> None:
        """Delete the application-owned outputs left behind under an input id."""
        if self.folder is None:
            return
        path = self._input_output_dir(input_id)
        if path.exists():
            shutil.rmtree(path)

    def config(self) -> ExperimentConfig:
        """The experiment in its on-disk form."""
        return ExperimentConfig(
            glasses_videos=[
                GlassesVideoInput(
                    id=v.id,
                    path=self._store(v.video_path),
                    gaze_path=self._store(v.gaze_path),
                    time_offset=v.time_offset,
                )
                for v in self.glasses_videos
            ],
            fixed_videos=[
                FixedVideoInput(
                    id=v.id, path=self._store(v.video_path), time_offset=v.time_offset
                )
                for v in self.fixed_videos
            ],
            audio=[
                AudioInput(
                    id=a.id,
                    path=self._store(a.audio_path),
                    time_offset=a.time_offset,
                    glasses_video=(
                        a.glasses_video.id if a.glasses_video is not None else None
                    ),
                )
                for a in self.audio
            ],
            pipeline=self.pipeline,
        )

    @classmethod
    def load(cls, folder: str | Path) -> Experiment:
        """Load the experiment in ``folder``: its inputs and their results."""
        folder = Path(folder)
        with (folder / CONFIG_FILENAME).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        version = data.get("version", CURRENT_VERSION)
        if version > CURRENT_VERSION:
            raise ValueError(
                f"experiment version {version} is newer than supported "
                f"{CURRENT_VERSION}; please upgrade body-eye-sync"
            )
        return cls(ExperimentConfig.model_validate(data), folder)

    def save(self, folder: str | Path | None = None) -> None:
        """Write the experiment, and every input's results, into the folder.

        ``folder`` defaults to the current :attr:`folder` and becomes it when
        given. The config is always written; an input is written only once it
        has results.
        """
        if folder is not None:
            self.folder = Path(folder)
        folder = self._require_folder()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / CONFIG_FILENAME).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config().model_dump(mode="json"), f, sort_keys=False)
        for data in self.inputs:
            if data.data is not None:
                destination = self.output_path(data)
                destination.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(destination)
