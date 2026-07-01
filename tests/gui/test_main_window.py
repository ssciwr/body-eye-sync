import pytest

from body_eye_sync.gui import MainWindow


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_run_button_disabled_until_video_loaded(window):
    assert not window.run_button.isEnabled()


def test_window_has_icon(window):
    assert not window.windowIcon().isNull()


def test_load_video_enables_run_button(window, data_dir):
    video = data_dir / "three-people.mp4"
    window._load_video(video)

    assert window.video_viewer.frame_count == 5
    assert window.run_button.isEnabled()
    assert str(video) in window.file_label.text()


def test_run_object_tracking_populates_state_and_overlays(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    assert (
        window.video_viewer._overlay_items == []
    )  # nothing drawn before object tracking

    window._start_object_tracking()
    qtbot.waitUntil(lambda: window.video.data is not None, timeout=60000)

    assert window.video.data["track_id"].nunique() == 3
    # Object tracking drives the video, so it ends on the last frame with that frame's
    # boxes drawn (3 people -> 3 rects + 3 labels).
    assert window.video_viewer.current_frame == window.video_viewer.frame_count - 1
    assert len(window.video_viewer._overlay_items) == 6
    # Controls are restored and the worker thread is cleaned up once done.
    qtbot.waitUntil(lambda: window._thread is None, timeout=5000)
    assert window.run_button.isEnabled()
    assert window.video_viewer._play_button.isEnabled()
    assert not window.progress_bar.isVisible()
    assert not window.cancel_button.isVisible()


def test_face_button_disabled_until_object_tracking_done(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    # Face detection runs on tracked boxes, so it waits for object tracking.
    assert not window.face_button.isEnabled()

    window._start_object_tracking()
    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)
    assert window.face_button.isEnabled()


def test_run_face_detection_populates_state_and_overlays(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    window._start_object_tracking()
    qtbot.waitUntil(lambda: window.video.data is not None, timeout=60000)
    qtbot.waitUntil(lambda: window._thread is None, timeout=5000)

    window._start_face_detection()
    qtbot.waitUntil(
        lambda: window._thread is None and "face_score" in window.video.data.columns,
        timeout=120000,
    )

    data = window.video.data
    # The tracked rows are preserved and gain face columns.
    assert data["track_id"].nunique() == 3
    assert data["face_score"].notna().any()
    # Ends on the last frame drawing that frame's person boxes and faces.
    last = window.video_viewer.frame_count - 1
    assert window.video_viewer.current_frame == last
    assert window.video.faces_for_frame(last)
    assert window.run_button.isEnabled()
    assert window.face_button.isEnabled()


def test_transport_disabled_while_object_tracking(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    assert window.video_viewer._play_button.isEnabled()

    window._start_object_tracking()
    # The worker drives the frame during object tracking, so manual transport is locked.
    assert not window.video_viewer._play_button.isEnabled()
    assert not window.video_viewer._slider.isEnabled()

    qtbot.waitUntil(lambda: window._thread is None, timeout=60000)
    assert window.video_viewer._play_button.isEnabled()
