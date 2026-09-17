import pandas as pd
import pytest

from body_eye_sync.experiment.identities import (
    IDENTITY_COLUMNS,
    IDENTITIES_FILENAME,
    Identities,
)


def _identities():
    return pd.DataFrame(
        {
            "video_id": ["cam1", "cam3", "cam1"],
            "track_id": [7, 12, 9],
            "participant_id": ["cam2", "cam2", None],
        }
    )


def test_new_identities_have_not_been_determined():
    identities = Identities()
    assert identities.data is None
    assert not identities.has_data()
    assert identities.participants == []
    assert identities.for_video("cam1").columns.tolist() == IDENTITY_COLUMNS


def test_resolved_and_unresolved_tracklets_can_be_looked_up_by_video():
    identities = Identities(_identities())
    assert identities.participants == ["cam2"]
    assert identities.for_video("cam1")["track_id"].tolist() == [7, 9]
    assert identities.for_video("unknown").empty


@pytest.mark.parametrize(
    "table", [_identities(), pd.DataFrame(columns=IDENTITY_COLUMNS)]
)
def test_save_load_round_trip_preserves_completed_results(tmp_path, table):
    Identities(table).save(tmp_path)
    loaded = Identities()
    loaded.load(tmp_path)
    assert loaded.has_data()
    pd.testing.assert_frame_equal(loaded.data, table)


def test_loading_a_missing_file_clears_existing_results(tmp_path):
    identities = Identities(_identities())
    identities.load(tmp_path)
    assert identities.data is None


def test_saving_cleared_results_removes_the_stored_table(tmp_path):
    identities = Identities(_identities())
    identities.save(tmp_path)
    identities.clear()
    identities.save(tmp_path)
    assert not (tmp_path / IDENTITIES_FILENAME).exists()


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        Identities(_identities().drop(columns="participant_id"))


def test_tracklets_are_unique_within_each_video():
    table = _identities()
    table.loc[1, ["video_id", "track_id"]] = ["cam1", 7]
    with pytest.raises(ValueError, match="duplicate video tracklets"):
        Identities(table)
    table.loc[1, "video_id"] = "cam3"
    assert Identities(table).has_data()


@pytest.mark.parametrize("categorical", [False, True])
def test_rename_updates_both_recording_and_participant_references(categorical):
    table = _identities()
    if categorical:
        participant_dtype = pd.CategoricalDtype(categories=["cam1", "cam2", "cam3"])
        table = table.astype(
            {"video_id": participant_dtype, "participant_id": participant_dtype}
        )
    identities = Identities(table)
    identities.rename_video("cam1", "renamed")
    identities.rename_video("cam2", "wearer")
    assert identities.for_video("cam1").empty
    assert identities.for_video("renamed")["track_id"].tolist() == [7, 9]
    assert identities.participants == ["wearer"]


def test_invalid_replacement_leaves_previous_results_intact():
    identities = Identities(_identities())
    with pytest.raises(ValueError):
        identities.set_data(pd.DataFrame())
    pd.testing.assert_frame_equal(identities.data, _identities())
