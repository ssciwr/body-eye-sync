"""Input files tab: the input video and audio files for the experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import (
    AudioInput,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import GlassesVideo, Video
from body_eye_sync.gui.tabs.base import BaseTab

VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
AUDIO_FILTER = "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg *.opus);;All files (*)"
GAZE_FILTER = "Gaze files (*.tsv *.csv *.txt);;All files (*)"

#: The columns every section has; a kind's extra columns follow them.
_ID, _FILE = range(2)
_COLUMNS = ["Id", "File"]

#: Shown in the glasses video column of an audio input aimed at nobody.
_NO_GLASSES = "—"


@dataclass(frozen=True)
class _ExtraColumn:
    """A column only some input types have, holding a widget per row."""

    title: str
    #: Builds the cell's widget for one of this section's inputs.
    widget: Callable[["_InputSection", Video | Audio], QWidget]


#: The gaze file a glasses video was recorded with, which it cannot go without.
GAZE_FILE = _ExtraColumn(
    title="Gaze file",
    widget=lambda section, video: section.gaze_button(video),
)

#: The glasses video an audio input was recorded alongside, if any.
GLASSES_VIDEO = _ExtraColumn(
    title="Glasses video",
    widget=lambda section, audio: section.glasses_combo(audio),
)


@dataclass(frozen=True)
class _InputKind:
    """One type of input: what it is called, where they live, how to add one."""

    title: str
    #: Title of the file dialog that adds one.
    dialog_title: str
    #: The file types that dialog offers.
    file_filter: str
    #: The experiment's inputs of this type, in order.
    inputs: Callable[[Experiment], list[Video | Audio]]
    #: Adds one to the experiment, given an unused id, the file it holds and
    #: whatever :attr:`gather` collected.
    add: Callable[..., None]
    #: Any further files this type needs, asked for as one is added. ``None``
    #: back means the user did not supply them and the input is not added.
    gather: Callable[["_InputSection", Path], dict[str, Path] | None] = (
        lambda section, path: {}
    )
    #: The columns this type has beyond :data:`_COLUMNS`.
    extra_columns: tuple[_ExtraColumn, ...] = ()


def gaze_file_for(section: _InputSection, video: Path) -> dict[str, Path] | None:
    """The gaze file to record a glasses video with: found beside it, or chosen.

    Devices export the gaze samples next to the video, so that is looked for
    first and the user is only asked when it is not obvious.
    """
    beside = video.with_suffix(".tsv")
    if beside.exists():
        return {"gaze_path": beside}
    chosen, _ = QFileDialog.getOpenFileName(
        section, f"Gaze file for {video.name}", str(video.parent), GAZE_FILTER
    )
    if not chosen:
        section.status_message.emit(f"{video.name} not added: it needs a gaze file")
        return None
    return {"gaze_path": Path(chosen)}


GLASSES_VIDEOS = _InputKind(
    title="Glasses videos",
    dialog_title="Add glasses video",
    file_filter=VIDEO_FILTER,
    inputs=lambda experiment: experiment.glasses_videos,
    add=lambda experiment, input_id, path, gaze_path: experiment.add_glasses_video(
        GlassesVideoInput(id=input_id, path=path, gaze_path=gaze_path)
    ),
    gather=gaze_file_for,
    extra_columns=(GAZE_FILE,),
)

FIXED_VIDEOS = _InputKind(
    title="Fixed videos",
    dialog_title="Add fixed video",
    file_filter=VIDEO_FILTER,
    inputs=lambda experiment: experiment.fixed_videos,
    add=lambda experiment, input_id, path: experiment.add_fixed_video(
        FixedVideoInput(id=input_id, path=path)
    ),
)

AUDIO = _InputKind(
    title="Audio",
    dialog_title="Add audio",
    file_filter=AUDIO_FILTER,
    inputs=lambda experiment: experiment.audio,
    add=lambda experiment, input_id, path: experiment.add_audio(
        AudioInput(id=input_id, path=path)
    ),
    extra_columns=(GLASSES_VIDEO,),
)

#: The sections of the tab, in order: the main input type first.
INPUT_KINDS = (GLASSES_VIDEOS, FIXED_VIDEOS, AUDIO)


def _input_path(data: Video | Audio) -> Path:
    """The file an input was recorded to, whichever kind of input it is."""
    return data.video_path if isinstance(data, Video) else data.audio_path


def _file_label(path: Path | None) -> str:
    """How a file reads in a cell: its name, flagged if it is not there."""
    if path is None:
        return _NO_GLASSES
    return path.name if path.exists() else f"{path.name} (not found)"


def _remove_inputs(
    parent: QWidget, experiment: Experiment, inputs: list[Video | Audio]
) -> list[Video | Audio]:
    """Remove what can be removed, reporting what cannot, and say what went.

    A glasses video that audio inputs still refer to is kept: the audio has to
    be repointed or removed first.
    """
    removed = []
    for data in inputs:
        try:
            experiment.remove_input(data)
        except ValueError as exc:
            QMessageBox.critical(parent, "Could not remove input", str(exc))
            continue
        removed.append(data)
    return removed


class _InputSection(QGroupBox):
    """One input type's inputs: a table of them, and buttons to add and remove.

    The section edits the experiment in place and reports that with ``changed``;
    re-reading it afterwards is the tab's job, since a change here can show up in
    another section (an added glasses video becomes a choice for every audio
    input).
    """

    #: An input of this type was added, removed or edited.
    changed = Signal()
    #: A row here was selected, so the other sections should drop theirs.
    selected = Signal()
    status_message = Signal(str)

    def __init__(self, kind: _InputKind, experiment: Experiment) -> None:
        super().__init__(kind.title)
        self.kind = kind
        self.experiment = experiment
        #: The input each table row shows, in row order.
        self._inputs: list[Video | Audio] = []
        #: Set while the table is being rebuilt, so its own edits are ignored.
        self._updating = False

        headers = [*_COLUMNS, *(column.title for column in kind.extra_columns)]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            _FILE, QHeaderView.ResizeMode.Stretch
        )
        # The section grows with its rows and the page scrolls, rather than each
        # table scrolling within a height the user has to set.
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.empty_label = QLabel(f"No {kind.title.lower()} yet")
        self.empty_label.setEnabled(False)

        self.add_button = QPushButton("Add…")
        self.add_button.clicked.connect(self._choose_files)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_selected)

        # The empty-state text shares the button row, so a section with nothing
        # in it is a single line rather than a mostly empty box.
        header = QHBoxLayout()
        header.addWidget(self.empty_label)
        header.addStretch(1)
        header.addWidget(self.add_button)
        header.addWidget(self.remove_button)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.table)

        self.refresh()

    def refresh(self) -> None:
        """Rebuild the table from the experiment's inputs of this type."""
        selected = {data.id for data in self.selected_inputs()}
        self._updating = True
        try:
            self._inputs = list(self.kind.inputs(self.experiment))
            self.table.clearContents()
            self.table.setRowCount(len(self._inputs))
            for row, data in enumerate(self._inputs):
                self._fill_row(row, data)
        finally:
            self._updating = False
        # An empty section says so in a line of text rather than showing a table
        # with nothing but headers in it.
        self.table.setVisible(bool(self._inputs))
        self.empty_label.setVisible(not self._inputs)
        for row, data in enumerate(self._inputs):
            if data.id in selected:
                self.table.selectRow(row)
        self._fit_to_rows()
        self._update_button_state()

    def selected_inputs(self) -> list[Video | Audio]:
        """The inputs of this section's selected rows, in table order."""
        rows = sorted(
            index.row() for index in self.table.selectionModel().selectedRows()
        )
        return [self._inputs[row] for row in rows if row < len(self._inputs)]

    def clear_selection(self) -> None:
        self.table.clearSelection()

    def add_files(self, paths: list[Path]) -> None:
        """Add each path as an input of this type, ids taken from the filenames.

        A type that needs more than the one file says so through its
        :attr:`_InputKind.gather`; a path it cannot complete is left out.
        """
        added = False
        for path in paths:
            path = Path(path)
            extra = self.kind.gather(self, path)
            if extra is None:
                continue
            self.kind.add(self.experiment, self._unused_id(path), path, **extra)
            added = True
        if added:
            self.changed.emit()

    def _fit_to_rows(self) -> None:
        """Make the table exactly as tall as its header and rows."""
        height = self.table.horizontalHeader().height() + 2 * self.table.frameWidth()
        for row in range(self.table.rowCount()):
            height += self.table.rowHeight(row)
        self.table.setFixedHeight(height)

    def _fill_row(self, row: int, data: Video | Audio) -> None:
        self.table.setItem(row, _ID, QTableWidgetItem(data.id))

        path = _input_path(data)
        path_item = QTableWidgetItem(str(path))
        path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if path is not None and not path.exists():
            path_item.setText(f"{path} (not found)")
            path_item.setToolTip("File not found")
        self.table.setItem(row, _FILE, path_item)

        for offset, column in enumerate(self.kind.extra_columns):
            self.table.setCellWidget(
                row, len(_COLUMNS) + offset, column.widget(self, data)
            )

    def gaze_button(self, video: GlassesVideo) -> QPushButton:
        """A chooser for the gaze file a glasses video was recorded with."""
        button = QPushButton(_file_label(video.gaze_path))
        button.setToolTip(str(video.gaze_path))
        button.setFlat(True)
        button.clicked.connect(
            lambda _checked=False, video=video: self._choose_gaze(video)
        )
        return button

    def _choose_gaze(self, video: GlassesVideo) -> None:
        start = video.gaze_path.parent if video.gaze_path is not None else ""
        chosen, _ = QFileDialog.getOpenFileName(
            self, f"Gaze file for {video.id}", str(start), GAZE_FILTER
        )
        if not chosen or Path(chosen) == video.gaze_path:
            return
        video.set_gaze(chosen)
        self.changed.emit()

    def glasses_combo(self, audio: Audio) -> QComboBox:
        """A chooser for the glasses video an audio input was recorded alongside."""
        combo = QComboBox()
        combo.addItem(_NO_GLASSES, None)
        for video in self.experiment.glasses_videos:
            combo.addItem(video.id, video)
        combo.setCurrentIndex(max(0, combo.findData(audio.glasses_video)))
        combo.currentIndexChanged.connect(
            lambda _index, audio=audio, combo=combo: self._set_glasses_video(
                audio, combo.currentData()
            )
        )
        return combo

    def _set_glasses_video(self, audio: Audio, video: GlassesVideo | None) -> None:
        if audio.glasses_video is video:
            return
        audio.glasses_video = video
        self.changed.emit()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Apply an edited id, or put the old value back."""
        if self._updating or item.row() >= len(self._inputs):
            return
        if item.column() == _ID:
            self._rename(self._inputs[item.row()], item.text().strip())

    def _rename(self, data: Video | Audio, new_id: str) -> None:
        if new_id == data.id:
            return
        try:
            self.experiment.rename_input(data, new_id)
        except (ValueError, OSError) as exc:
            self.status_message.emit(f"Could not rename input: {exc}")
            self.refresh()
            return
        self.changed.emit()

    def _unused_id(self, path: Path) -> str:
        """An id based on ``path``'s filename that no input is using yet.

        Ids are unique across every type, not just this section's.
        """
        stem = Path(path).stem or "input"
        used = {data.id for data in self.experiment.inputs}
        if stem not in used:
            return stem
        suffix = 2
        while f"{stem}-{suffix}" in used:
            suffix += 1
        return f"{stem}-{suffix}"

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, self.kind.dialog_title, "", self.kind.file_filter
        )
        self.add_files([Path(path) for path in paths])

    def _remove_selected(self) -> None:
        if _remove_inputs(self, self.experiment, self.selected_inputs()):
            self.changed.emit()

    def _on_selection_changed(self) -> None:
        self._update_button_state()
        # An empty selection is what clearing the other sections looks like, so
        # only a real one takes the selection away from them.
        if self.selected_inputs():
            self.selected.emit()

    def _update_button_state(self) -> None:
        self.remove_button.setEnabled(bool(self.selected_inputs()))


class InputFilesTab(BaseTab):
    """Name the experiment, and add, remove and edit its input files."""

    title = "Input files"

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)

        #: One section per input type, in :data:`INPUT_KINDS` order.
        self.sections = [_InputSection(kind, experiment) for kind in INPUT_KINDS]
        self.glasses_section, self.fixed_section, self.audio_section = self.sections

        sections_layout = QVBoxLayout()
        for section in self.sections:
            section.changed.connect(self._on_section_changed)
            section.status_message.connect(self.status_message)
            section.selected.connect(lambda source=section: self._select_only(source))
            sections_layout.addWidget(section)
        sections_layout.addStretch(1)

        sections = QWidget()
        sections.setLayout(sections_layout)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(sections)

        layout = QVBoxLayout(self)
        layout.addWidget(area)

    def refresh(self) -> None:
        """Rebuild every section from the experiment."""
        for section in self.sections:
            section.refresh()

    def set_experiment(self, experiment: Experiment) -> None:
        for section in self.sections:
            section.experiment = experiment
        super().set_experiment(experiment)

    def selected_inputs(self) -> list[Video | Audio]:
        """The selected inputs; at most one section has a selection at a time."""
        return [data for section in self.sections for data in section.selected_inputs()]

    def add_glasses_videos(self, paths: list[Path]) -> None:
        """Add each path as a glasses video input, ids taken from the filenames."""
        self.glasses_section.add_files(paths)

    def add_fixed_videos(self, paths: list[Path]) -> None:
        """Add each path as a fixed video input, ids taken from the filenames."""
        self.fixed_section.add_files(paths)

    def add_audio(self, paths: list[Path]) -> None:
        """Add each path as an audio input, ids taken from the filenames."""
        self.audio_section.add_files(paths)

    def remove_inputs(self, inputs: list[Video | Audio]) -> None:
        """Remove ``inputs`` from the experiment, reporting any that cannot go."""
        if _remove_inputs(self, self.experiment, inputs):
            self._on_section_changed()

    def _on_section_changed(self) -> None:
        """A section edited the experiment: re-read it, and pass the news on.

        Every section is rebuilt, not just the one that changed: adding, renaming
        or removing a glasses video changes what the audio section offers.
        """
        self.refresh()
        self.experiment_changed.emit()

    def _select_only(self, source: _InputSection) -> None:
        """Keep the selection in one section, so Remove is never ambiguous."""
        for section in self.sections:
            if section is not source:
                section.clear_selection()
