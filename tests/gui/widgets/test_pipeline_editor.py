from body_eye_sync.experiment.config import (
    BodyPoseStep,
    DiarizationStep,
    FaceDetectionStep,
    ObjectTrackingStep,
    SpeechPipeline,
    TranscriptionStep,
    VideoPipeline,
)
from body_eye_sync.gui.widgets.pipeline_editor import (
    SPEECH_STEPS,
    VIDEO_STEPS,
    PipelineEditor,
)


def _editor(qtbot, steps=VIDEO_STEPS):
    editor = PipelineEditor(steps)
    qtbot.addWidget(editor)
    return editor


def _pipeline(**overrides):
    return VideoPipeline(**overrides)


def test_reset_yields_only_mandatory_step(qtbot):
    editor = _editor(qtbot)
    editor.reset()

    # Object tracking is required; the optional passes start disabled.
    assert [type(s) for s in editor.enabled_steps()] == [ObjectTrackingStep]


def test_set_from_enables_present_optional_steps(qtbot):
    editor = _editor(qtbot)
    editor.set_from(_pipeline(face_detection=FaceDetectionStep()))

    assert [type(s) for s in editor.enabled_steps()] == [
        ObjectTrackingStep,
        FaceDetectionStep,
    ]


def test_set_from_populates_step_arguments(qtbot):
    editor = _editor(qtbot)
    editor.set_from(
        _pipeline(
            object_tracking=ObjectTrackingStep(detector="yolov8m"),
            face_detection=FaceDetectionStep(det_thresh=0.7),
        )
    )

    steps = {type(s): s for s in editor.enabled_steps()}
    assert steps[ObjectTrackingStep].detector == "yolov8m"
    assert steps[FaceDetectionStep].det_thresh == 0.7


def test_enabled_steps_are_in_canonical_order(qtbot):
    editor = _editor(qtbot)
    editor.set_from(
        _pipeline(face_detection=FaceDetectionStep(), body_pose=BodyPoseStep())
    )

    assert [type(s) for s in editor.enabled_steps()] == [
        ObjectTrackingStep,
        FaceDetectionStep,
        BodyPoseStep,
    ]


def test_toggling_optional_step_changes_pipeline(qtbot):
    editor = _editor(qtbot)
    editor.reset()
    assert [type(s) for s in editor.enabled_steps()] == [ObjectTrackingStep]

    # Find the body-pose section and enable it.
    for section in editor._sections:
        if section.step_type is BodyPoseStep:
            section.setChecked(True)

    assert [type(s) for s in editor.enabled_steps()] == [
        ObjectTrackingStep,
        BodyPoseStep,
    ]


def test_apply_to_writes_enabled_steps_and_none_for_disabled(qtbot):
    editor = _editor(qtbot)
    editor.reset()  # tracking only, optional passes off

    pipeline = _pipeline(face_detection=FaceDetectionStep(), body_pose=BodyPoseStep())
    editor.apply_to(pipeline)

    assert isinstance(pipeline.object_tracking, ObjectTrackingStep)
    assert pipeline.face_detection is None
    assert pipeline.body_pose is None


def test_changed_fires_on_toggle(qtbot):
    editor = _editor(qtbot)
    editor.reset()

    face_section = next(s for s in editor._sections if s.step_type is FaceDetectionStep)
    with qtbot.waitSignal(editor.changed):
        face_section.setChecked(True)


def test_set_from_does_not_emit_changed(qtbot):
    editor = _editor(qtbot)
    fired = []
    editor.changed.connect(lambda: fired.append(True))

    editor.set_from(_pipeline(face_detection=FaceDetectionStep()))
    assert fired == []


def test_step_run_button_emits_run_requested_with_its_step_type(qtbot):
    editor = _editor(qtbot)
    tracking = next(s for s in editor._sections if s.step_type is ObjectTrackingStep)
    tracking.set_run_enabled(True)

    with qtbot.waitSignal(editor.run_requested) as blocker:
        tracking.run_button.click()

    assert blocker.args == [ObjectTrackingStep]


def test_run_all_button_emits_run_all_requested(qtbot):
    editor = _editor(qtbot)
    editor.set_run_all_enabled(True)

    with qtbot.waitSignal(editor.run_all_requested):
        editor.run_all_button.click()


def test_run_buttons_start_disabled(qtbot):
    editor = _editor(qtbot)

    assert not any(s.run_button.isEnabled() for s in editor._sections)
    assert not editor.run_all_button.isEnabled()


def test_speech_steps_edit_a_speech_pipeline(qtbot):
    editor = _editor(qtbot, SPEECH_STEPS)
    editor.set_from(SpeechPipeline(transcription=TranscriptionStep(beam_size=3)))

    steps = {type(s): s for s in editor.enabled_steps()}
    assert list(steps) == [DiarizationStep, TranscriptionStep]
    assert steps[TranscriptionStep].beam_size == 3


def test_speech_transcription_is_the_optional_step(qtbot):
    editor = _editor(qtbot, SPEECH_STEPS)
    editor.reset()
    assert [type(s) for s in editor.enabled_steps()] == [DiarizationStep]

    pipeline = SpeechPipeline(transcription=TranscriptionStep())
    editor.apply_to(pipeline)

    assert isinstance(pipeline.diarization, DiarizationStep)
    assert pipeline.transcription is None
