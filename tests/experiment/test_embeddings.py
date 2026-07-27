"""Best-K embedding capture, reduction and companion-file round-trip."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from body_eye_sync.experiment.video import Video
from body_eye_sync.pipeline.object_tracking import BoundingBox
from body_eye_sync.pipeline.face_detection import FaceBox, FaceFrameResult

# BoxMOT tracks layout: x1, y1, x2, y2, id, conf, cls, det_ind.
_LANDMARKS = [(0.0, 0.0)] * 5


def _tracking_frame(frame_idx, rows, embeddings):
    return SimpleNamespace(
        frame_idx=frame_idx,
        tracks=np.array(rows, dtype=float),
        embeddings=np.array(embeddings, dtype=float),
    )


def _row(track_id, conf):
    return [0.0, 0.0, 10.0, 10.0, track_id, conf, 0, 0]


def test_body_embeddings_keep_best_k_per_track_by_confidence():
    video = Video()
    video.begin_object_tracking(embeddings_per_track=2)
    # Track 1 seen in three frames with confidences 0.5, 0.9, 0.7.
    video.add_object_tracking_frame(_tracking_frame(1, [_row(1, 0.5)], [[1, 0, 0, 0]]))
    video.add_object_tracking_frame(_tracking_frame(2, [_row(1, 0.9)], [[0, 1, 0, 0]]))
    video.add_object_tracking_frame(_tracking_frame(3, [_row(1, 0.7)], [[0, 0, 1, 0]]))
    video.finish_object_tracking()

    emb = video.body_embeddings
    assert emb is not None
    assert emb["track_id"].unique().tolist() == [1]
    # The two highest-confidence detections are kept, best first.
    assert list(emb["score"]) == pytest.approx([0.9, 0.7])
    assert list(emb["frame"]) == [1, 2]  # 0-based indices of frames 2 and 3
    assert emb["embedding"].iloc[0].dtype == np.float16


def test_body_embeddings_skip_nonfinite_predicted_tracks():
    video = Video()
    video.begin_object_tracking(embeddings_per_track=4)
    video.add_object_tracking_frame(
        _tracking_frame(1, [_row(1, 0.9)], [[np.nan, np.nan, np.nan]])
    )
    video.finish_object_tracking()
    assert video.body_embeddings is None


def test_embeddings_disabled_when_k_zero():
    video = Video()
    video.begin_object_tracking(embeddings_per_track=0)
    video.add_object_tracking_frame(_tracking_frame(1, [_row(1, 0.9)], [[1, 2, 3]]))
    video.finish_object_tracking()
    assert video.body_embeddings is None


def _tracked_video(rows):
    video = Video()
    video.set_data(
        pd.DataFrame(
            rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
        )
    )
    return video


def _face(track_id, score, embedding):
    box = BoundingBox(0.0, 0.0, 10.0, 10.0, track_id)
    return FaceBox(box, score, list(_LANDMARKS), np.array(embedding, dtype=float))


def test_face_embeddings_best_k_and_fp16_round_trip(tmp_path):
    video = _tracked_video([[0, 1, 0, 0, 10, 10, 0.9], [1, 1, 0, 0, 10, 10, 0.9]])
    video.begin_face_detection(embeddings_per_track=1)
    video.add_face_detection_frame(FaceFrameResult(0, [_face(1, 0.6, [1, 0, 0])]))
    video.add_face_detection_frame(FaceFrameResult(1, [_face(1, 0.8, [0, 1, 0])]))
    video.finish_face_detection()

    emb = video.face_embeddings
    assert emb is not None
    assert len(emb) == 1  # best-1 by face score
    assert emb["score"].iloc[0] == pytest.approx(0.8)
    # Face embeddings do not leak into the main dataframe.
    assert "embedding" not in video.data.columns

    path = tmp_path / "cam1" / "results.parquet"
    path.parent.mkdir()
    video.to_parquet(path)
    assert (path.parent / "face_embeddings.parquet").exists()
    assert not (path.parent / "body_embeddings.parquet").exists()

    loaded = Video.from_parquet(path)
    loaded_emb = loaded.face_embeddings
    assert len(loaded_emb) == 1
    assert loaded_emb["embedding"].iloc[0].dtype == np.float16
    np.testing.assert_array_equal(
        loaded_emb["embedding"].iloc[0], np.array([0, 1, 0], dtype=np.float16)
    )


def test_no_companion_files_written_without_embeddings(tmp_path):
    video = _tracked_video([[0, 1, 0, 0, 10, 10, 0.9]])
    path = tmp_path / "cam1" / "results.parquet"
    path.parent.mkdir()
    (path.parent / "face_embeddings.parquet").write_bytes(b"stale")
    (path.parent / "body_embeddings.parquet").write_bytes(b"stale")

    video.to_parquet(path)

    assert path.exists()
    assert not (path.parent / "face_embeddings.parquet").exists()
    assert not (path.parent / "body_embeddings.parquet").exists()
