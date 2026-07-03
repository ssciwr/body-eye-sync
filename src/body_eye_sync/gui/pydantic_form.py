"""Auto-generate an editing form for a flat pydantic model.

:class:`PydanticForm` builds one input widget per model field from the field's
type, constraints and metadata, so the pydantic schema stays the single source
of truth for both serialisation and the GUI. It supports the scalar field types
the pipeline step models use (``str``, ``int``, ``float``, ``bool``, a
``list`` of scalars) plus ``choices`` metadata; it is not a general recursive
form and does not descend into nested models.

Widget mapping:

* ``choices`` in ``json_schema_extra`` -> editable :class:`QComboBox`
* ``bool``                             -> :class:`QCheckBox`
* ``int`` (with ``ge``/``le`` bounds)  -> :class:`QSpinBox`
* ``float`` (with ``ge``/``le`` bounds)-> :class:`QDoubleSpinBox`
* ``list[...]``                        -> :class:`QLineEdit` (comma separated)
* anything else / ``str``              -> :class:`QLineEdit`

``Literal`` fields (the discriminator tags) are fixed and not shown.
"""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

import annotated_types
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

_INT_LIMIT = 1_000_000
_FLOAT_LIMIT = 1e9


def _bounds(field: FieldInfo) -> tuple[float | None, float | None]:
    """Lower/upper numeric bounds declared via ``Field(ge=..., le=...)``."""
    low = high = None
    for constraint in field.metadata:
        if isinstance(constraint, annotated_types.Ge):
            low = constraint.ge
        elif isinstance(constraint, annotated_types.Gt):
            low = constraint.gt
        elif isinstance(constraint, annotated_types.Le):
            high = constraint.le
        elif isinstance(constraint, annotated_types.Lt):
            high = constraint.lt
    return low, high


def _choices(field: FieldInfo) -> list | None:
    extra = field.json_schema_extra
    if isinstance(extra, dict):
        return extra.get("choices")
    return None


class PydanticForm(QWidget):
    """An editing form for one flat pydantic model instance.

    Populate from a model with :meth:`from_model`, read the edited values back
    (validated) with :meth:`to_model`. ``changed`` fires on any edit.
    """

    changed = Signal()

    def __init__(self, model: BaseModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model_type = type(model)
        self._widgets: dict[str, QWidget] = {}
        self._field_info: dict[str, FieldInfo] = {}

        layout = QFormLayout(self)
        for name, field in self._model_type.model_fields.items():
            if get_origin(field.annotation) is Literal:
                continue  # discriminator tag: fixed, not user-editable
            widget = self._make_widget(field)
            self._widgets[name] = widget
            self._field_info[name] = field
            label = name.replace("_", " ").capitalize()
            if field.description:
                widget.setToolTip(field.description)
            layout.addRow(label, widget)

        self.from_model(model)

    def _make_widget(self, field: FieldInfo) -> QWidget:
        choices = _choices(field)
        if choices:
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems([str(c) for c in choices])
            combo.currentTextChanged.connect(self.changed)
            return combo

        annotation = field.annotation
        if annotation is bool:
            check = QCheckBox()
            check.toggled.connect(self.changed)
            return check
        if annotation is int:
            spin = QSpinBox()
            low, high = _bounds(field)
            spin.setMinimum(int(low) if low is not None else -_INT_LIMIT)
            spin.setMaximum(int(high) if high is not None else _INT_LIMIT)
            spin.valueChanged.connect(self.changed)
            return spin
        if annotation is float:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            low, high = _bounds(field)
            spin.setMinimum(float(low) if low is not None else -_FLOAT_LIMIT)
            spin.setMaximum(float(high) if high is not None else _FLOAT_LIMIT)
            spin.valueChanged.connect(self.changed)
            return spin

        line = QLineEdit()
        line.textChanged.connect(self.changed)
        return line

    def from_model(self, model: BaseModel) -> None:
        """Populate the widgets from ``model``'s current values."""
        for name, widget in self._widgets.items():
            value = getattr(model, name)
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QLineEdit):
                if isinstance(value, (list, tuple)):
                    widget.setText(", ".join(str(v) for v in value))
                else:
                    widget.setText(str(value))

    def to_model(self) -> BaseModel:
        """Build a validated model from the current widget values.

        Raises :class:`pydantic.ValidationError` (or :class:`ValueError` from
        list parsing) if the edited values are invalid.
        """
        return self._model_type(**self._values())

    def _values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            field = self._field_info[name]
            if isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                if get_origin(field.annotation) is list:
                    values[name] = _parse_list(widget.text(), field)
                else:
                    values[name] = widget.text()
        return values


def _parse_list(text: str, field: FieldInfo) -> list:
    """Parse a comma-separated line edit into a list of the field's item type."""
    args = get_args(field.annotation)
    convert = args[0] if args else str
    return [convert(part.strip()) for part in text.split(",") if part.strip()]
