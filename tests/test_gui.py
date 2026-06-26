import pytest

from body_eye_sync.gui import MainWindow


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_run_button_disabled_until_video_loaded(window):
    assert not window.run_button.isEnabled()


def test_load_video_enables_run_button(window, data_dir):
    video = data_dir / "three-people.mp4"
    window._load_video(video)

    assert window.video_viewer.frame_count == 5
    assert window.run_button.isEnabled()
    assert str(video) in window.file_label.text()


def test_run_tracking_populates_state_and_overlays(qtbot, window, data_dir):
    window._load_video(data_dir / "three-people.mp4")
    assert window.video_viewer._overlay_items == []  # nothing drawn before tracking

    window._start_tracking()
    qtbot.waitUntil(lambda: window.state.tracklets is not None, timeout=60000)

    assert window.state.tracklets["track_id"].nunique() == 3
    # Boxes for the current frame are now drawn (3 people -> 3 rects + 3 labels).
    assert len(window.video_viewer._overlay_items) == 6
    # Controls are restored and the worker thread is cleaned up once done.
    qtbot.waitUntil(lambda: window._thread is None, timeout=5000)
    assert window.run_button.isEnabled()
    assert not window.progress_bar.isVisible()
    assert not window.cancel_button.isVisible()
