from __future__ import annotations

from typing import Iterator

from body_eye_sync.experiment.config import BodyPoseStep
from body_eye_sync.experiment.video import Video
from body_eye_sync.gui.base_worker import BaseWorker


class BodyPoseWorker(BaseWorker):
    """Runs :func:`detect_body_poses` off the GUI thread, into a :class:`Video`.

    Body poses are detected inside the person boxes already tracked into the
    :class:`Video`. Each frame's poses are accumulated and emitted via
    ``new_frame`` so the GUI can draw them live, then folded onto the matching
    rows once the run finishes; a cancelled/failed pass keeps the tracked boxes.
    The detection arguments come from ``step``.
    """

    operation_name = "Body pose detection"

    def __init__(self, video: Video, step: BodyPoseStep) -> None:
        super().__init__(video)
        self._step = step

    def _items(self) -> Iterator:
        # lazy import to avoid making GUI startup slow due to module loading
        from body_eye_sync.pipeline.body_pose import detect_body_poses

        return detect_body_poses(
            self._video.video_path,
            self._video.all_boxes_by_frame(),
            **self._step.model_dump(),
        )

    def _accumulate(self, result) -> None:
        self._video.add_body_pose_frame(result)

    def _finalise(self) -> None:
        self._video.finish_body_pose_detection()

    def _discard(self) -> None:
        self._video.discard_body_pose_detection()
