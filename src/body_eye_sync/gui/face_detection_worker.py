from __future__ import annotations

from typing import Iterator

from body_eye_sync.gui.base_worker import BaseWorker


class FaceDetectionWorker(BaseWorker):
    """Runs :func:`detect_faces` off the GUI thread, into a :class:`Video`.

    Faces are detected inside the person boxes already tracked into the
    :class:`Video`. Each frame's faces are accumulated and emitted via
    ``new_frame`` so the GUI can draw them live, then folded onto the matching
    rows once the run finishes; a cancelled/failed pass keeps the tracked boxes.
    """

    operation_name = "Face detection"

    def _items(self) -> Iterator:
        # lazy import to avoid making GUI startup slow due to module loading
        from body_eye_sync.pipeline.face_detection import detect_faces

        return detect_faces(self._video.video_path, self._video.all_boxes_by_frame())

    def _accumulate(self, result) -> None:
        self._video.add_face_detection_frame(result)

    def _finalise(self) -> None:
        self._video.finish_face_detection()

    def _discard(self) -> None:
        self._video.discard_face_detection()
