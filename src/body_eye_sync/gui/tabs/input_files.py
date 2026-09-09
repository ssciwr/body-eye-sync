"""Input files tab: the input video and audio files for the experiment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListView,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidgetItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from body_eye_sync.experiment.audio import Audio
from body_eye_sync.experiment.config import (
    RESERVED_ID_CHARACTERS,
    RESERVED_IDS,
    AudioInput,
    FixedVideoInput,
    GlassesVideoInput,
)
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.experiment.video import GlassesVideo, Video
from body_eye_sync.glasses import (
    Recording,
    find_recordings,
    recording_for_video,
    recording_info,
)
from body_eye_sync.gui.tabs.base import BaseTab
from body_eye_sync.gui.widgets.auto_height_table import AutoHeightTable

VIDEO_FILTER = (
    "Video (formats with audio like MP4 will include the related audio) "
    "(*.mp4 *.avi *.mov *.mkv);;All files (*)"
)
AUDIO_FILTER = "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg *.opus);;All files (*)"
GAZE_FILTER = "Gaze files (*.tsv *.csv *.txt);;All files (*)"

#: Gaze source menu labels.
GAZE_FOLDER_ACTION = "Recording folder…"
GAZE_FILE_ACTION = "Gaze file…"

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
    widget: Callable[[_InputSection, Video | Audio], QWidget]
    #: Share spare width with the file column.
    stretch: bool = False


#: The gaze file a glasses video was recorded with, which it cannot go without.
GAZE_FILE = _ExtraColumn(
    title="Gaze file",
    widget=lambda section, video: section.gaze_button(video),
    stretch=True,
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
    add: Callable[..., Video | Audio]
    #: Any further files this type needs, asked for as one is added. ``None``
    #: back means the user did not supply them and the input is not added.
    gather: Callable[[_InputSection, Path], dict[str, Path] | None] = (
        lambda section, path: {}
    )
    #: Input name derived from the file and gathered metadata.
    name: Callable[[Path, dict[str, Path]], str] = lambda path, extra: path.stem
    #: File chooser button and column labels.
    add_button: str = "Add…"
    file_column: str = "File"
    #: The columns this type has beyond :data:`_COLUMNS`.
    extra_columns: tuple[_ExtraColumn, ...] = ()
    #: Optional recording-folder import: the button label, and what a found
    #: folder holds, as the file to add and whatever else goes with it. Raises
    #: ``ValueError`` for a folder without a usable file.
    folder_button: str | None = None
    from_folder: Callable[[Path], tuple[Path, dict[str, Path]]] | None = None


def choose_folders(parent: QWidget, title: str) -> list[Path]:
    """Select multiple folders using a non-native Qt dialog."""
    dialog = QFileDialog(parent, title)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    for view in [
        *dialog.findChildren(QListView),
        *dialog.findChildren(QTreeView),
    ]:
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    if not dialog.exec():
        return []
    return [Path(chosen) for chosen in dialog.selectedFiles()]


def recording_name(recording: Recording, folder: Path) -> str:
    """Use the participant name, falling back to the folder name."""
    return recording.participant or folder.name


def glasses_input_name(video: Path, gaze_path: Path) -> str:
    """Use recording metadata for the input name, or the video stem for exports."""
    recording = recording_info(gaze_path)
    return video.stem if recording is None else recording_name(recording, gaze_path)


def glasses_recording(folder: Path) -> tuple[Path, dict[str, Path]]:
    """The video a recording folder holds, with the folder as its gaze source."""
    recording = recording_info(folder)
    video = recording.video_path if recording is not None else None
    if video is None or not video.is_file():
        raise ValueError("it has no video to go with it")
    return video, {"gaze_path": folder}


def gaze_file_for(section: _InputSection, video: Path) -> dict[str, Path] | None:
    """Use the recording folder or adjacent TSV; otherwise ask for a gaze file."""
    if (folder := recording_for_video(video)) is not None:
        return {"gaze_path": folder}
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
    add_button="Add video…",
    file_column="Video file",
    file_filter=VIDEO_FILTER,
    inputs=lambda experiment: experiment.glasses_videos,
    add=lambda experiment, input_id, path, gaze_path: experiment.add_glasses_video(
        GlassesVideoInput(id=input_id, path=path, gaze_path=gaze_path)
    ),
    gather=gaze_file_for,
    name=lambda path, extra: glasses_input_name(path, extra["gaze_path"]),
    extra_columns=(GAZE_FILE,),
    folder_button="Add recording folder…",
    from_folder=glasses_recording,
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


def _file_label(path: Path | None) -> str:
    """Label a source by name, marking recording folders and missing paths."""
    if path is None:
        return _NO_GLASSES
    if not path.exists():
        return f"{path.name} (not found)"
    return f"{path.name}/" if path.is_dir() else path.name


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

        headers = [
            *_COLUMNS[:_FILE],
            kind.file_column,
            *(column.title for column in kind.extra_columns),
        ]
        self.table = AutoHeightTable(headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        # Share spare width between file columns.
        header.setSectionResizeMode(_FILE, QHeaderView.ResizeMode.Stretch)
        for offset, column in enumerate(kind.extra_columns):
            if column.stretch:
                header.setSectionResizeMode(
                    len(_COLUMNS) + offset, QHeaderView.ResizeMode.Stretch
                )
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.empty_label = QLabel(f"No {kind.title.lower()} yet")
        self.empty_label.setEnabled(False)

        self.add_folder_button: QPushButton | None = None
        if kind.folder_button is not None:
            self.add_folder_button = QPushButton(kind.folder_button)
            self.add_folder_button.clicked.connect(self._choose_folder)
        self.add_button = QPushButton(kind.add_button)
        self.add_button.clicked.connect(self._choose_files)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_selected)

        # The buttons sit below the current rows, where a new input will appear. Preserve sold "line" appearance vs empty box.
        actions = QHBoxLayout()
        actions.addStretch(1)
        if self.add_folder_button is not None:
            actions.addWidget(self.add_folder_button)
        actions.addWidget(self.add_button)
        actions.addWidget(self.remove_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.table)
        layout.addLayout(actions)

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
        self.table.fit_to_rows()
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
        if self._add_each(paths, self._from_file):
            self.changed.emit()

    def _from_file(self, path: Path) -> tuple[Path, dict[str, Path]] | None:
        """The file itself, and whatever its kind gathers to go with it."""
        # Refuse a duplicate before asking for any accompanying files.
        self.experiment.check_path(path)
        extra = self.kind.gather(self, path)
        return None if extra is None else (path, extra)

    def _add_each(
        self,
        paths: list[Path],
        describe: Callable[[Path], tuple[Path, dict[str, Path]] | None],
    ) -> int:
        """Add an input per path, reporting each rejection; count those added.

        ``describe`` turns a path into the file to add and what goes with it,
        or ``None`` when the user declined to complete it.
        """
        added = 0
        for path in paths:
            path = Path(path)
            try:
                described = describe(path)
                if described is None:
                    continue
                video, extra = described
                name = self.kind.name(video, extra)
                self.kind.add(self.experiment, self._unused_id(name), video, **extra)
            except (ValueError, OSError) as exc:
                self.status_message.emit(f"{path} not added: {exc}")
                continue
            added += 1
        return added

    def _fill_row(self, row: int, data: Video | Audio) -> None:
        self.table.setItem(row, _ID, QTableWidgetItem(data.id))

        path = data.path
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
        """Build a menu for selecting a recording folder or gaze file."""
        button = QPushButton(_file_label(video.gaze_path))
        button.setToolTip(str(video.gaze_path))
        button.setFlat(True)
        menu = QMenu(button)
        menu.addAction(
            GAZE_FOLDER_ACTION, lambda video=video: self._choose_gaze_folder(video)
        )
        menu.addAction(
            GAZE_FILE_ACTION, lambda video=video: self._choose_gaze_file(video)
        )
        button.setMenu(menu)
        return button

    def _gaze_start_folder(self, video: GlassesVideo) -> str:
        """Starting directory for gaze source dialogs."""
        if video.gaze_path is None:
            return ""
        return str(
            video.gaze_path if video.gaze_path.is_dir() else video.gaze_path.parent
        )

    def _choose_gaze_file(self, video: GlassesVideo) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            f"Gaze file for {video.id}",
            self._gaze_start_folder(video),
            GAZE_FILTER,
        )
        if not chosen or Path(chosen) == video.gaze_path:
            return
        video.set_gaze(chosen)
        self.changed.emit()

    def _choose_gaze_folder(self, video: GlassesVideo) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, f"Recording folder for {video.id}", self._gaze_start_folder(video)
        )
        if not chosen:
            return
        recording = recording_info(chosen)
        if recording is None:
            self.status_message.emit(
                f"{Path(chosen).name} is not a glasses recording folder"
            )
            return
        folder = recording.source
        if folder == video.gaze_path:
            return
        self._warn_if_video_differs(video, recording)
        video.set_gaze(folder)
        self.changed.emit()

    def _warn_if_video_differs(self, video: GlassesVideo, recording: Recording) -> None:
        """Warn when the gaze recording names a different video."""
        if recording.video_path is None or video.video_path is None:
            return
        if recording.video_path == video.video_path:
            return
        self.status_message.emit(
            f"{video.id}: {recording.source.name} was recorded with "
            f"{recording.video_path.name}, not {video.video_path.name} — check "
            "these are the same recording, uncut"
        )

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

    def _unused_id(self, name: str) -> str:
        """Sanitise ``name`` and make it unique across all input types."""
        stem = name.translate(
            str.maketrans({char: "_" for char in RESERVED_ID_CHARACTERS})
        ).strip()
        if not stem or stem in RESERVED_IDS:
            stem = "input"
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

    def _choose_folder(self) -> None:
        """Find and import the glasses recordings in the selected folders."""
        assert self.kind.from_folder is not None
        chosen = choose_folders(self, "Add glasses recording folder")
        if not chosen:
            return
        # Deduplicate recordings found through overlapping selections.
        found = dict.fromkeys(
            recording for folder in chosen for recording in find_recordings(folder)
        )
        if not found:
            names = ", ".join(folder.name for folder in chosen)
            self.status_message.emit(
                f"No glasses recording in {names}: open a recording folder, or "
                "one with recordings somewhere inside it"
            )
            return
        added = self._add_each(list(found), self.kind.from_folder)
        if added:
            self.status_message.emit(
                f"Added {added} glasses recording{'s' if added != 1 else ''}"
            )
            self.changed.emit()

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
