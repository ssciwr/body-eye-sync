import pytest
from pydantic import ValidationError
from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from body_eye_sync.experiment.config import (
    FaceDetectionStep,
    ObjectTrackingStep,
)
from body_eye_sync.gui.pydantic_form import PydanticForm


def test_defaults_round_trip(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)
    assert form.to_model() == FaceDetectionStep()


def test_widgets_match_field_types(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)

    # choices -> editable combobox; bounded int/float -> spin boxes.
    assert isinstance(form._widgets["model_name"], QComboBox)
    assert form._widgets["model_name"].isEditable()
    assert isinstance(form._widgets["det_size"], QSpinBox)
    assert isinstance(form._widgets["det_thresh"], QDoubleSpinBox)


def test_choices_populate_combobox(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)

    combo = form._widgets["model_name"]
    items = [combo.itemText(i) for i in range(combo.count())]
    assert "buffalo_l" in items and "antelopev2" in items


def test_bounds_applied_to_spinboxes(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)

    thresh = form._widgets["det_thresh"]
    assert thresh.minimum() == 0.0 and thresh.maximum() == 1.0


def test_editing_values_reflected_in_model(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)

    form._widgets["det_thresh"].setValue(0.7)
    form._widgets["model_name"].setCurrentText("antelopev2")

    model = form.to_model()
    assert model.det_thresh == 0.7
    assert model.model_name == "antelopev2"


def test_custom_combobox_value_is_allowed(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)

    # Editable combobox: a value outside the suggestions is accepted.
    form._widgets["model_name"].setCurrentText("/path/to/custom.onnx")
    assert form.to_model().model_name == "/path/to/custom.onnx"


def test_list_field_round_trips(qtbot):
    form = PydanticForm(ObjectTrackingStep(object_classes=[0, 32]))
    qtbot.addWidget(form)

    line = form._widgets["object_classes"]
    assert isinstance(line, QLineEdit)
    assert line.text() == "0, 32"

    line.setText("0, 15, 16")
    assert form.to_model().object_classes == [0, 15, 16]


def test_invalid_list_entry_raises(qtbot):
    form = PydanticForm(ObjectTrackingStep())
    qtbot.addWidget(form)

    form._widgets["object_classes"].setText("person, dog")
    with pytest.raises((ValidationError, ValueError)):
        form.to_model()


def test_changed_signal_fires_on_edit(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)

    with qtbot.waitSignal(form.changed):
        form._widgets["det_thresh"].setValue(0.9)


def test_from_model_repopulates_widgets(qtbot):
    form = PydanticForm(FaceDetectionStep())
    qtbot.addWidget(form)

    form.from_model(FaceDetectionStep(det_size=1280, det_thresh=0.3))
    assert form._widgets["det_size"].value() == 1280
    assert form._widgets["det_thresh"].value() == 0.3
