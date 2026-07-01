"""Editor for an experiment's pipeline: which steps run, and their arguments.

The pipeline structure is hard-coded here -- the known steps and their order --
while each step's arguments are edited by an auto-generated :class:`PydanticForm`.
Object tracking is the mandatory base pass; face and body-pose detection are
optional passes that run over its tracked boxes, so they are toggled on/off.

This widget concerns itself only with the *pipeline*; managing the experiment's
inputs (videos, gaze data, ...) is a separate interface.
"""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from body_eye_sync.experiment.config import (
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
    StepSpec,
)
from body_eye_sync.gui.pydantic_form import PydanticForm

#: The pipeline steps the GUI offers, in run order: the ``Experiment`` field
#: each maps to, its model type, the title shown, and whether it is optional.
#: Object tracking is required; the later passes consume its boxes and are opt-in.
_STEPS: list[tuple[str, type, str, bool]] = [
    ("object_tracking", ObjectTrackingStep, "Object tracking", False),
    ("face_detection", FaceDetectionStep, "Face detection", True),
    ("body_pose", BodyPoseStep, "Body pose", True),
]


class _StepSection(QWidget):
    """One step's enable toggle (optional steps), its form and a run button.

    The "Run" button is deliberately a *sibling* of the checkable group box,
    not laid out inside it: a checkable ``QGroupBox`` disables its own layout's
    contents while unchecked and re-applies that whenever its effective
    enabled state changes (e.g. the whole editor being disabled and
    re-enabled around a run), which would clobber the run button's own
    enabled state. A step's arguments can still be run interactively even
    when the step itself isn't toggled into the saved pipeline.
    """

    changed = Signal()
    run_requested = Signal()

    def __init__(
        self, attr_name: str, step_type: type, title: str, optional: bool, parent=None
    ) -> None:
        super().__init__(parent)
        self.attr_name = attr_name
        self.step_type = step_type
        self._optional = optional

        self.group = QGroupBox(title)
        self.form = PydanticForm(step_type())
        group_layout = QVBoxLayout(self.group)
        group_layout.addWidget(self.form)

        self.run_button = QPushButton("Run")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_requested)
        header = QHBoxLayout()
        header.addStretch(1)
        header.addWidget(self.run_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.group)

        if optional:
            # A checkable group box puts a checkbox in the title and greys out
            # its contents when unchecked -- exactly the enable/disable we want.
            self.group.setCheckable(True)
            self.group.setChecked(False)
            self.group.toggled.connect(lambda _on: self.changed.emit())
        self.form.changed.connect(self.changed)

    def is_enabled(self) -> bool:
        return not self._optional or self.group.isChecked()

    def setChecked(self, checked: bool) -> None:
        self.group.setChecked(checked)

    def set_from(self, step: StepSpec | None) -> None:
        """Enable and fill from ``step`` if present, otherwise disable."""
        if self._optional:
            self.group.setChecked(step is not None)
        if step is not None:
            self.form.from_model(step)

    def reset(self) -> None:
        """Restore default arguments, unchecked if the step is optional."""
        self.form.from_model(self.step_type())
        if self._optional:
            self.group.setChecked(False)

    def to_step(self) -> StepSpec:
        return self.form.to_model()

    def set_run_enabled(self, enabled: bool) -> None:
        self.run_button.setEnabled(enabled)


class PipelineEditor(QWidget):
    """Edit an experiment's pipeline steps and their arguments.

    Populate from an experiment with :meth:`set_from` (or :meth:`reset` to
    defaults), and write the edited values back into one with :meth:`apply_to`.
    ``changed`` fires on any toggle or field edit. Each step has its own "Run"
    button (``run_requested``, with the step's type); ``run_all_requested`` fires
    from the button that runs every enabled step in order. Running is out of
    scope for this widget -- it only reports the requests, and its buttons'
    enabled state is driven from outside via
    :meth:`set_run_enabled`/:meth:`set_run_all_enabled`.
    """

    changed = Signal()
    run_requested = Signal(object)
    run_all_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sections: list[_StepSection] = []
        layout = QVBoxLayout(self)
        for attr_name, step_type, title, optional in _STEPS:
            section = _StepSection(attr_name, step_type, title, optional)
            section.changed.connect(self.changed)
            section.run_requested.connect(
                lambda step_type=step_type: self.run_requested.emit(step_type)
            )
            self._sections.append(section)
            layout.addWidget(section)

        self.run_all_button = QPushButton("Run all")
        self.run_all_button.setEnabled(False)
        self.run_all_button.clicked.connect(self.run_all_requested)
        layout.addWidget(self.run_all_button)
        layout.addStretch(1)

    def set_from(self, experiment: Experiment) -> None:
        """Populate the editor from ``experiment``'s pipeline (no ``changed``)."""
        for section in self._sections:
            section.blockSignals(True)
            section.set_from(getattr(experiment, section.attr_name))
            section.blockSignals(False)

    def reset(self) -> None:
        """Reset every step to its defaults, optional steps switched off."""
        for section in self._sections:
            section.blockSignals(True)
            section.reset()
            section.blockSignals(False)

    def apply_to(self, experiment: Experiment) -> None:
        """Write the edited steps back onto ``experiment``'s pipeline fields.

        Disabled optional steps become ``None``. All steps are validated before
        anything is assigned, so an invalid field leaves ``experiment`` intact.
        Raises :class:`pydantic.ValidationError` / :class:`ValueError` if any
        step's arguments are invalid.
        """
        values = {
            s.attr_name: (s.to_step() if s.is_enabled() else None)
            for s in self._sections
        }
        for name, value in values.items():
            setattr(experiment, name, value)

    def enabled_steps(self) -> list[StepSpec]:
        """The enabled steps, in order, built and validated from the widgets.

        Raises :class:`pydantic.ValidationError` / :class:`ValueError` if any
        enabled step's arguments are invalid.
        """
        return [s.to_step() for s in self._sections if s.is_enabled()]

    def _section(self, step_type: type) -> _StepSection:
        """The section editing ``step_type``, or ``KeyError`` if unknown."""
        for section in self._sections:
            if section.step_type is step_type:
                return section
        raise KeyError(step_type)

    def config_for(self, step_type: type) -> StepSpec:
        """The validated config for one step, whether or not it is enabled.

        Lets an interactive run of a single pass use the arguments the user has
        set, independent of whether the step is toggled into the saved pipeline.
        Raises :class:`pydantic.ValidationError` / :class:`ValueError` if invalid.
        """
        return self._section(step_type).to_step()

    def set_run_enabled(self, step_type: type, enabled: bool) -> None:
        """Enable/disable one step's "Run" button (e.g. while its inputs aren't ready)."""
        self._section(step_type).set_run_enabled(enabled)

    def set_run_all_enabled(self, enabled: bool) -> None:
        self.run_all_button.setEnabled(enabled)
