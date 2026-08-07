"""The background workers that run a pipeline step off the GUI thread.

One worker per step, each a :class:`~body_eye_sync.gui.workers.base.BaseWorker`
subclass that reports its progress with signals and writes its results into the
:class:`~body_eye_sync.experiment.video.Video` or
:class:`~body_eye_sync.experiment.speech.Speech` it was given.
"""

from __future__ import annotations

from body_eye_sync.gui.workers.base import BaseWorker
from body_eye_sync.gui.workers.body_pose import BodyPoseWorker
from body_eye_sync.gui.workers.face_detection import FaceDetectionWorker
from body_eye_sync.gui.workers.object_tracking import ObjectTrackingWorker
from body_eye_sync.gui.workers.transcription import TranscriptionWorker

__all__ = [
    "BaseWorker",
    "BodyPoseWorker",
    "FaceDetectionWorker",
    "ObjectTrackingWorker",
    "TranscriptionWorker",
]
