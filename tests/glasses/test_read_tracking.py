import gzip
import json

import numpy as np
import pytest

from body_eye_sync.glasses import (
    glasses2,
    glasses3,
    read_tracking,
    recording_folder,
    tsv,
)
from body_eye_sync.media import video_info


#: The columns of a gaze export, with and without the optional participant.
GAZE_HEADER = "rectimestamp\tgaze_x\tgaze_y\tpupil_left\tpupil_right\tgaze_video_time\n"
PARTICIPANT_GAZE_HEADER = f"participant\t{GAZE_HEADER}"


@pytest.fixture
def gaze_video(tmp_path, video_starting_late):
    """A scene video at the container's zero, 64x48 like the exports below."""
    return video_starting_late(tmp_path / "scene.mp4", delay=0.0)


def test_reads_a_glasses3_recording(glasses3_recording):
    folder = glasses3_recording()

    data = read_tracking(folder)

    assert len(data) == 4
    assert data.recording.device == glasses3.DEVICE
    assert data.recording.participant == "1401"
    assert data.recording.serial == "TG03G-010200450191"
    assert data.recording.resolution == (1920, 1080)
    assert data.recording.video_path == folder / "scenevideo.mp4"
    assert data.sample_rate == pytest.approx(50.0)


def test_glasses3_untracked_samples_are_gaps_not_positions(glasses3_recording):
    data = read_tracking(glasses3_recording())

    # The third sample is written as an empty object: nothing was tracked.
    assert list(data.valid) == [True, True, False, True]
    assert np.isnan(data.gaze[2]).all()
    assert np.isnan(data.gaze_3d[2]).all()
    # The second sample tracked one eye only, so it keeps its gaze point.
    assert data.valid[1]
    assert data.pupil[1, 0] == pytest.approx(3.1)
    assert np.isnan(data.pupil[1, 1])
    assert np.isnan(data.eye_direction[1, 1]).all()


def test_glasses3_times_reach_the_clock_the_sound_track_is_read_on(
    glasses3_recording, video_starting_late
):
    """Shift sensor timestamps, counted from the first frame, by the video start."""
    from body_eye_sync.glasses import read_streams

    folder = glasses3_recording()
    video_starting_late(folder / "scenevideo.mp4", delay=0.24)

    streams = read_streams(folder)

    assert video_info(folder / "scenevideo.mp4").start == pytest.approx(0.24)
    assert streams.tracking.time == pytest.approx([0.26, 0.28, 0.30, 0.32])
    assert streams.motion.acceleration.time == pytest.approx([0.25, 0.26])


def test_glasses3_falls_back_to_the_video_for_a_frame_size(glasses3_recording):
    folder = glasses3_recording(resolution=(64, 48))
    manifest = json.loads((folder / "recording.g3").read_text())
    del manifest["scenecamera"]["camera-calibration"]["resolution"]
    (folder / "recording.g3").write_text(json.dumps(manifest))

    assert read_tracking(folder).recording.resolution == (64, 48)


def test_glasses3_without_its_scene_video(glasses3_recording):
    """Gaze is timed from the first frame, so it cannot be read without the video."""
    folder = glasses3_recording()
    (folder / "scenevideo.mp4").unlink()

    with pytest.raises(ValueError, match="without its video"):
        read_tracking(folder)


def test_glasses3_naming_no_scene_video(glasses3_recording):
    folder = glasses3_recording()
    manifest = json.loads((folder / "recording.g3").read_text())
    del manifest["scenecamera"]["file"]
    (folder / "recording.g3").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="names no video"):
        read_tracking(folder)


def test_glasses2_without_its_scene_video(glasses2_recording):
    folder = glasses2_recording()
    (folder / "segments" / "1" / "fullstream.mp4").unlink()

    with pytest.raises(ValueError, match="without its video"):
        read_tracking(folder)


def test_glasses3_reports_a_missing_gaze_stream(glasses3_recording):
    folder = glasses3_recording()
    (folder / "gazedata.gz").unlink()

    with pytest.raises(FileNotFoundError, match="gazedata.gz"):
        read_tracking(folder)


def test_reads_a_glasses2_recording(glasses2_recording):
    folder = glasses2_recording()

    data = read_tracking(folder)

    assert len(data) == 3
    assert data.recording.device == glasses2.DEVICE
    assert data.recording.participant == "1403"
    assert data.recording.serial == "TG02G-010129906823"
    assert data.sample_rate == pytest.approx(50.0)


def test_glasses2_places_samples_on_the_video_clock(glasses2_recording):
    """The device clock is its own; ``vts`` says where the video starts on it."""
    data = read_tracking(glasses2_recording())

    # Eye tracking ran for 10 ms before the camera did, so it starts before it.
    assert data.time == pytest.approx([-0.01, 0.01, 0.03])


def test_glasses2_untracked_readings_are_gaps_not_zeros(glasses2_recording):
    data = read_tracking(glasses2_recording())

    # The device writes a zero for the middle sample and flags it with s != 0.
    assert list(data.valid) == [True, False, True]
    assert np.isnan(data.gaze[1]).all()
    assert np.isnan(data.gaze_3d[1]).all()
    assert np.isnan(data.pupil[1]).all()


def test_glasses2_joins_the_records_of_one_sample(glasses2_recording):
    data = read_tracking(glasses2_recording())

    assert data.gaze[0] == pytest.approx([0.5385, 0.2598])
    assert data.pupil[0] == pytest.approx([3.96, 4.12])
    assert data.eye_origin[0, 0] == pytest.approx([25.8, -24.5, -27.9])
    assert data.eye_direction[0, 0] == pytest.approx([-0.019, 0.288, 0.957])
    # Only the left eye was read for the first sample's centre and direction.
    assert np.isnan(data.eye_origin[0, 1]).all()


def test_glasses2_will_not_guess_at_a_multi_segment_recording(glasses2_recording):
    folder = glasses2_recording(segments=2)

    with pytest.raises(ValueError, match="2 segments"):
        read_tracking(folder)


def test_glasses2_reports_a_recording_it_cannot_place_against_its_video(
    glasses2_recording,
):
    folder = glasses2_recording(vts=False)

    with pytest.raises(ValueError, match="vts"):
        read_tracking(folder)


def test_reads_a_gaze_export(data_dir):
    data = read_tracking(
        data_dir / "three-people.tsv", video_path=data_dir / "three-people.mp4"
    )

    assert len(data) == 31
    assert data.recording.device == tsv.DEVICE
    assert data.recording.participant == "1403"
    # Placed against the video to within its 10 ms rounding, and spaced by the
    # device clock, which runs a shade fast of a nominal 50 Hz.
    assert data.time[0] == pytest.approx(0.0, abs=0.005)
    assert data.sample_rate == pytest.approx(50.03, abs=0.01)
    assert data.pupil[0] == pytest.approx([4.021, 4.091])
    assert data.gaze_pixels()[0] == pytest.approx([1385.0, 995.0])


def test_a_gaze_export_needs_the_video_its_pixels_belong_to(data_dir):
    with pytest.raises(ValueError, match="video pixels"):
        read_tracking(data_dir / "three-people.tsv")


def test_a_gaze_export_is_normalised_by_its_video_frame_size(tmp_path, gaze_video):
    export = tmp_path / "gaze.tsv"
    export.write_text(
        GAZE_HEADER + "5197.0\t16\t36\t4021\t4091\t0\n5217.0\t32\t24\t4021\t4091\t20\n"
    )

    data = read_tracking(export, video_path=gaze_video)

    assert data.recording.resolution == (64, 48)
    assert data.gaze[0] == pytest.approx([0.25, 0.75])
    assert data.gaze[1] == pytest.approx([0.5, 0.5])


def test_a_gaze_export_zero_is_an_eye_that_was_not_seen(tmp_path, gaze_video):
    export = tmp_path / "gaze.tsv"
    export.write_text(
        PARTICIPANT_GAZE_HEADER + "1403\t0.0\t0\t0\t0\t4091\t0\n"
        "1403\t20.0\t32\t24\t4021\t4091\t20\n"
    )

    data = read_tracking(export, video_path=gaze_video)

    assert np.isnan(data.gaze[0]).all()
    assert np.isnan(data.pupil[0, 0])
    assert data.pupil[0, 1] == pytest.approx(4.091)
    assert data.gaze[1] == pytest.approx([0.5, 0.5])


def test_a_gaze_export_without_the_columns_it_needs(tmp_path, gaze_video):
    export = tmp_path / "gaze.tsv"
    export.write_text("participant\tgaze_x\tgaze_y\n1403\t1\t2\n")

    with pytest.raises(ValueError, match="pupil_left"):
        read_tracking(export, video_path=gaze_video)


@pytest.mark.parametrize("generation", ["glasses2_recording", "glasses3_recording"])
def test_a_recording_filed_under_a_folder_of_its_own_is_still_found(
    request, generation, tmp_path
):
    """Accept a parent folder containing exactly one recording."""
    build = request.getfixturevalue(generation)
    outer = tmp_path / "1401"
    folder = build(outer / "recording-id")

    assert recording_folder(outer) == folder
    assert len(read_tracking(outer)) == len(read_tracking(folder))


def test_a_folder_of_several_recordings_names_none_of_them(
    glasses3_recording, tmp_path
):
    glasses3_recording(tmp_path / "one")
    glasses3_recording(tmp_path / "two")

    assert recording_folder(tmp_path) is None


def test_a_folder_that_is_not_a_recording(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing to read here")

    with pytest.raises(ValueError, match="not a glasses recording folder"):
        read_tracking(tmp_path)


def test_nothing_at_the_path_at_all(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_tracking(tmp_path / "gone.tsv")


def test_a_recording_names_its_video_of_each_generation(
    glasses2_recording, glasses3_recording
):
    """Resolve video paths even when the video files are missing."""
    from body_eye_sync.glasses import recording_info

    folder = glasses3_recording()
    assert recording_info(folder).video_path == folder / "scenevideo.mp4"

    folder = glasses2_recording()
    assert recording_info(folder).video_path == (
        folder / "segments" / "1" / "fullstream.mp4"
    )


def test_a_gaze_export_names_no_recording(data_dir):
    assert recording_folder(data_dir / "three-people.tsv") is None


def test_a_gaze_export_is_timed_by_the_clock_that_matches_the_video(
    tmp_path, gaze_video
):
    """Use device timestamps to recover spacing lost to 10 ms video-time rounding."""
    export = tmp_path / "gaze.tsv"
    spacing = 19.98
    rows = "".join(
        f"1403\t{1000 + i * spacing}\t32\t24\t4021\t4091"
        f"\t{round(i * spacing / 10) * 10}\n"
        for i in range(200)
    )
    export.write_text(PARTICIPANT_GAZE_HEADER + rows)

    data = read_tracking(export, video_path=gaze_video)

    # Evenly spaced, unlike the rounded column, and placed to well inside it.
    assert np.diff(data.time) == pytest.approx(spacing / 1e3)
    rounded = np.round(data.time * 100) / 100
    assert np.abs(data.time - rounded).max() < 0.005


def test_a_gaze_export_starts_where_its_video_time_says_it_does(tmp_path, gaze_video):
    export = tmp_path / "gaze.tsv"
    export.write_text(
        GAZE_HEADER + "5197.0\t32\t24\t4021\t4091\t500\n"
        "5217.0\t32\t24\t4021\t4091\t520\n"
    )

    data = read_tracking(export, video_path=gaze_video)

    assert data.time == pytest.approx([0.5, 0.52])


def test_a_gaze_export_without_the_device_clock_falls_back_to_video_time(
    tmp_path, gaze_video
):
    export = tmp_path / "gaze.tsv"
    export.write_text(
        "gaze_x\tgaze_y\tpupil_left\tpupil_right\tgaze_video_time\n"
        "32\t24\t4021\t4091\t0\n"
        "32\t24\t4021\t4091\t20\n"
    )

    data = read_tracking(export, video_path=gaze_video)

    assert data.time == pytest.approx([0.0, 0.02])


def test_a_gaze_export_marks_what_was_not_tracked_with_na(tmp_path, gaze_video):
    """What the device could not see is written as ``NA``, not as a value."""
    export = tmp_path / "gaze.tsv"
    export.write_text(
        PARTICIPANT_GAZE_HEADER + "1403\t5197.0\t32\t24\t4021\t4091\t0\n"
        "1403\t5217.0\tNA\tNA\tNA\tNA\t20\n"
        "1403\t5237.0\t32\t24\tNA\t4091\t40\n"
    )

    data = read_tracking(export, video_path=gaze_video)

    assert list(data.valid) == [True, False, True]
    assert np.isnan(data.pupil[1]).all()
    # An eye can go untracked while the gaze point itself is still known.
    assert np.isnan(data.pupil[2, 0])
    assert data.pupil[2, 1] == pytest.approx(4.091)


def test_glasses2_times_reach_the_clock_the_sound_track_is_read_on(
    glasses2_recording, video_starting_late
):
    """Add the video stream start to the times derived from ``vts``."""
    folder = glasses2_recording()
    video_starting_late(folder / "segments" / "1" / "fullstream.mp4", delay=0.24)

    data = read_tracking(folder)

    assert data.time == pytest.approx([-0.01 + 0.24, 0.01 + 0.24, 0.03 + 0.24])


def test_a_gaze_export_reaches_the_clock_its_sound_track_is_read_on(
    tmp_path, video_starting_late
):
    video = video_starting_late(tmp_path / "glasses.mp4", delay=0.24)
    export = tmp_path / "gaze.tsv"
    export.write_text(
        GAZE_HEADER + "5197.0\t32\t24\t4021\t4091\t0\n5217.0\t32\t24\t4021\t4091\t20\n"
    )

    data = read_tracking(export, video_path=video)

    assert data.time == pytest.approx([0.24, 0.26])


def test_recording_info_reads_what_a_recording_is_without_its_samples(
    glasses3_recording, glasses2_recording
):
    """Read metadata without loading sensor streams."""
    from body_eye_sync.glasses import recording_info

    folder = glasses3_recording()
    recording = recording_info(folder)
    assert recording.participant == "1401"
    assert recording.device == glasses3.DEVICE
    assert recording.serial == "TG03G-010200450191"

    folder = glasses2_recording()
    recording = recording_info(folder)
    assert recording.participant == "1403"
    assert recording.device == glasses2.DEVICE


def test_recording_info_of_something_that_is_not_a_recording(tmp_path):
    from body_eye_sync.glasses import recording_info

    assert recording_info(tmp_path) is None


def test_find_recordings_finds_a_card_full_of_them(glasses3_recording, tmp_path):
    card = tmp_path / "card"
    first = glasses3_recording(card / "20220728T161118Z")
    second = glasses3_recording(card / "20220728T170000Z")

    from body_eye_sync.glasses import find_recordings

    assert find_recordings(card) == [first, second]
    # A recording names itself, and anything else names none.
    assert find_recordings(first) == [first]
    assert find_recordings(tmp_path / "nowhere") == []


def test_find_recordings_looks_below_the_folder_it_is_given(
    glasses3_recording, glasses2_recording, tmp_path
):
    """Cards file recordings deep: Glasses 2 keeps them three folders down."""
    from body_eye_sync.glasses import find_recordings

    session = tmp_path / "session"
    shallow = glasses3_recording(session / "1401" / "20220728T161118Z")
    deep = glasses2_recording(session / "projects" / "sr2ixeu" / "recordings" / "id")

    assert find_recordings(session) == [shallow, deep]


def test_find_recordings_does_not_search_inside_a_recording(
    glasses3_recording, tmp_path
):
    """Whatever a recording holds, it is one recording, not several."""
    from body_eye_sync.glasses import find_recordings

    recording = glasses3_recording(tmp_path / "20220728T161118Z")
    (recording / "meta" / "20220101T000000Z").mkdir(parents=True, exist_ok=True)

    assert find_recordings(recording) == [recording]


def test_find_recordings_gives_up_before_walking_a_whole_drive(
    glasses3_recording, tmp_path
):
    from body_eye_sync.glasses import find_recordings

    buried = tmp_path.joinpath(*[f"level{n}" for n in range(8)])
    glasses3_recording(buried / "20220728T161118Z")

    assert find_recordings(tmp_path) == []
    assert len(find_recordings(tmp_path, depth=20)) == 1


def test_find_recordings_ignores_what_the_tools_leave_behind(
    glasses3_recording, tmp_path
):
    from body_eye_sync.glasses import find_recordings

    glasses3_recording(tmp_path / ".Trash" / "20220728T161118Z")

    assert find_recordings(tmp_path) == []


def test_reads_glasses3_head_movement(glasses3_recording):
    from body_eye_sync.glasses import read_streams

    streams = read_streams(glasses3_recording())

    motion = streams.motion
    # The magnetometer reading between them is passed over.
    assert len(motion.acceleration) == 2
    assert len(motion.angular_velocity) == 2
    assert motion.acceleration.time == pytest.approx([0.01, 0.02])
    # Both sensors read at the same moment, so they share their times.
    assert motion.angular_velocity.time == pytest.approx(motion.acceleration.time)
    assert motion.recording.device == glasses3.DEVICE


def test_glasses3_before_firmware_129_has_its_axes_turned(glasses3_recording):
    """Older firmware leaves the IMU in the sensor's own axes, not the head's."""
    from body_eye_sync.glasses import read_motion

    motion = read_motion(glasses3_recording(firmware="1.14.3+nudelsoppa"))

    # A head unit at rest reads ~+9.8 upward; the raw reading was -9.8 on Y.
    # The 12 degree mounting angle puts part of it on Z, so the turn is a
    # rotation as well as a flip, and only the magnitude is unchanged.
    reading = motion.acceleration.values[0]
    assert reading == pytest.approx(
        [0.0, 9.8 * np.cos(np.radians(12)), -9.8 * np.sin(np.radians(12))]
    )
    assert np.linalg.norm(reading) == pytest.approx(9.8)


def test_glasses3_from_firmware_129_is_left_alone(glasses3_recording):
    """The device aligns its own readings from 1.29, so turning them would spoil them."""
    from body_eye_sync.glasses import read_motion

    motion = read_motion(glasses3_recording(firmware="1.29.0"))

    assert motion.acceleration.values[0] == pytest.approx([0.0, -9.8, 0.0])


def test_glasses3_of_unstated_firmware_is_taken_to_be_recent(glasses3_recording):
    from body_eye_sync.glasses import read_motion

    motion = read_motion(glasses3_recording(firmware=None))

    assert motion.acceleration.values[0] == pytest.approx([0.0, -9.8, 0.0])


def test_reads_glasses2_head_movement_from_the_one_stream(glasses2_recording):
    """Glasses 2 interleaves every sensor, so one pass gives eye and head both."""
    from body_eye_sync.glasses import read_streams

    streams = read_streams(glasses2_recording())

    assert len(streams.tracking) == 3
    motion = streams.motion
    assert len(motion.acceleration) == 2
    assert len(motion.angular_velocity) == 1
    # Its Y axis is reversed, as Glasses 3's is before 1.29.
    assert motion.acceleration.values[0, 1] == pytest.approx(9.821)
    # And its readings are on the same clock as the eye tracking.
    assert motion.acceleration.time[0] == pytest.approx(streams.tracking.time[0])


@pytest.mark.parametrize("status", [1, 2])
def test_glasses2_invalid_motion_keeps_timestamps_but_not_values(
    glasses2_recording, status
):
    from body_eye_sync.glasses import read_motion

    folder = glasses2_recording()
    with gzip.open(folder / "segments" / "1" / "livedata.json.gz", "at") as stream:
        for sensor in ("ac", "gy"):
            stream.write(
                json.dumps({"ts": 337_788_295, "s": status, sensor: [0, 0, 0]}) + "\n"
            )

    motion = read_motion(folder)

    assert motion.acceleration.time == pytest.approx([-0.01, 0.01, 0.03])
    assert motion.angular_velocity.time == pytest.approx([0.0, 0.03])
    for series in (motion.acceleration, motion.angular_velocity):
        assert np.isfinite(series.values[:-1]).all()
        assert np.isnan(series.values[-1]).all()


def test_a_gaze_export_has_no_head_movement(data_dir):
    from body_eye_sync.glasses import read_streams

    streams = read_streams(
        data_dir / "three-people.tsv", video_path=data_dir / "three-people.mp4"
    )

    assert len(streams.tracking) == 31
    assert len(streams.motion) == 0
    assert streams.motion.recording.device == tsv.DEVICE
