"""Experiment contains the config, loaded inputs and tracking data / embeddings"""

from __future__ import annotations

from pathlib import Path

import yaml

from body_eye_sync.experiment.config import (
    CURRENT_VERSION,
    ExperimentConfig,
    VideoInput,
)
from body_eye_sync.experiment.video import Video

# An experiment lives in a folder: this config file plus an outputs/ subfolder.
CONFIG_FILENAME = "experiment.yaml"
OUTPUTS_DIRNAME = "outputs"


class Experiment:
    """An experiment: its :class:`ExperimentConfig` plus the loaded videos."""

    def __init__(self, config: ExperimentConfig, folder: str | Path | None = None):
        self.config = config
        self.folder = Path(folder) if folder is not None else None
        self._videos: dict[str, Video] = {}

    def _require_folder(self) -> Path:
        if self.folder is None:
            raise ValueError("experiment has no folder; load or save it first")
        return self.folder

    def resolved_input_path(self, spec: VideoInput) -> Path:
        """Absolute path for ``spec``; relative paths taken from the folder."""
        if spec.path.is_absolute():
            return spec.path
        return (self._require_folder() / spec.path).resolve()

    @property
    def output_dir(self) -> Path:
        """Where per-input Parquet outputs live, inside the folder."""
        return self._require_folder() / OUTPUTS_DIRNAME

    def output_path(self, spec: VideoInput) -> Path:
        """Parquet output path for ``spec`` under :attr:`output_dir`."""
        return self.output_dir / f"{spec.id}.parquet"

    def video(self, spec: VideoInput) -> Video:
        """The :class:`Video` for ``spec``, created on first access.

        Its cached outputs (data + embeddings) are loaded if they exist.
        """
        video = self._videos.get(spec.id)
        if video is None:
            video = Video()
            video.set_video(self.resolved_input_path(spec))
            if self.folder is not None and self.output_path(spec).exists():
                video.load_parquet(self.output_path(spec))
            self._videos[spec.id] = video
        return video

    @classmethod
    def load(cls, folder: str | Path) -> Experiment:
        """Load the experiment in ``folder`` (its config; videos load on access)."""
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
        """Write the config, and every loaded video's results, into the folder.

        ``folder`` defaults to the current :attr:`folder` and becomes it when
        given. The config is always written; a video is written only once it has
        results.
        """
        if folder is not None:
            self.folder = Path(folder)
        folder = self._require_folder()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / CONFIG_FILENAME).open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.model_dump(mode="json"), f, sort_keys=False)
        for spec in self.config.inputs:
            video = self._videos.get(spec.id)
            if video is not None and video.data is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                video.to_parquet(self.output_path(spec))
