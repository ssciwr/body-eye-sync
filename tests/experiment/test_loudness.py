import pandas as pd
import pytest

from body_eye_sync.experiment.loudness import LOUDNESS_FILENAME, Loudness


def _levels() -> pd.DataFrame:
    return pd.DataFrame(
        {"time": [0.025, 0.075, 0.125], "level_db": [-60.0, -12.5, -30.0]}
    )


def test_a_new_loudness_has_nothing():
    assert Loudness().data is None


def test_measuring_a_recording_fills_the_levels(tmp_path, data_dir):
    loudness = Loudness()

    loudness.measure(data_dir / "three-people-glasses-1.opus")

    assert not loudness.data.empty
    assert loudness.data["time"].is_monotonic_increasing


def test_levels_are_saved_and_loaded(tmp_path):
    loudness = Loudness()
    loudness.set_data(_levels())

    loudness.save(tmp_path)
    loaded = Loudness()
    loaded.load(tmp_path)

    assert (tmp_path / LOUDNESS_FILENAME).exists()
    pd.testing.assert_frame_equal(loaded.data, _levels())


def test_loading_a_directory_without_levels_leaves_them_empty(tmp_path):
    loudness = Loudness()
    loudness.set_data(_levels())

    loudness.load(tmp_path)

    assert loudness.data is None


def test_saving_nothing_removes_levels_left_from_before(tmp_path):
    loudness = Loudness()
    loudness.set_data(_levels())
    loudness.save(tmp_path)

    loudness.clear()
    loudness.save(tmp_path)

    assert not (tmp_path / LOUDNESS_FILENAME).exists()


def test_a_table_without_the_measured_levels_is_rejected():
    with pytest.raises(ValueError, match="level_db"):
        Loudness().set_data(pd.DataFrame({"time": [0.025]}))
