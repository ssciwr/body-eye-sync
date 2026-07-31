"""The main window: the File menu, and the tabs an experiment is worked through."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

from qtpy.QtGui import QAction, QIcon, QKeySequence
from qtpy.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from pydantic import ValidationError

from body_eye_sync.experiment.config import ExperimentConfig
from body_eye_sync.experiment.experiment import Experiment
from body_eye_sync.gui.tabs import TAB_TYPES, BaseTab

#: Base window title; the open experiment's folder is appended to it.
_BASE_TITLE = "body-eye-sync"

#: Stands in for the folder name of an experiment that has yet to be saved.
_UNSAVED_TITLE = "unsaved experiment"

#: Offered as the name of the folder a new experiment is saved into.
_DEFAULT_FOLDER_NAME = "my-experiment"


def _new_experiment() -> Experiment:
    return Experiment(ExperimentConfig())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        with as_file(files(__package__) / "resources" / "icon.ico") as icon_path:
            self.setWindowIcon(QIcon(str(icon_path)))

        self.experiment = _new_experiment()
        self._update_title()
        self._busy = False
        self._dirty = False

        self._build_menu_bar()

        self.tabs = QTabWidget()
        self.tab_widgets: list[BaseTab] = []
        for tab_type in TAB_TYPES:
            tab = tab_type(self.experiment)
            tab.status_message.connect(self.statusBar().showMessage)
            tab.experiment_changed.connect(
                lambda source=tab: self._on_experiment_changed(source)
            )
            tab.busy_changed.connect(
                lambda busy, source=tab: self._set_busy(busy, source)
            )
            self.tabs.addTab(tab, tab_type.title)
            self.tab_widgets.append(tab)
        self.setCentralWidget(self.tabs)

    def tab(self, tab_type: type[BaseTab]) -> BaseTab:
        """The window's instance of ``tab_type``."""
        return next(tab for tab in self.tab_widgets if isinstance(tab, tab_type))

    def load_experiment(self, folder: str | Path) -> None:
        self._load_experiment(Path(folder))

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.new_action = QAction("&New", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_action.triggered.connect(self._new_experiment)
        file_menu.addAction(self.new_action)

        self.open_action = QAction("&Open…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self._choose_experiment)
        file_menu.addAction(self.open_action)

        self.save_action = QAction("&Save", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._save_experiment)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

    def _new_experiment(self) -> None:
        """Start again with an empty experiment, discarding the current one."""
        if self._busy or not self._confirm_discarding_changes():
            return
        self._set_experiment(_new_experiment())

    def _choose_experiment(self) -> None:
        if self._busy or not self._confirm_discarding_changes():
            return
        folder = QFileDialog.getExistingDirectory(self, "Open experiment folder")
        if folder:
            self._load_experiment(Path(folder))

    def _load_experiment(self, folder: Path) -> None:
        try:
            experiment = Experiment.load(folder)
        except (OSError, ValueError, ValidationError) as exc:
            QMessageBox.critical(self, "Could not open experiment", str(exc))
            return
        self._set_experiment(experiment)
        self.statusBar().showMessage(f"Opened experiment {folder}")

    def _set_experiment(self, experiment: Experiment) -> None:
        """Hand ``experiment`` to every tab, in place of the current one."""
        self.experiment = experiment
        self._dirty = False
        for tab in self.tab_widgets:
            tab.set_experiment(experiment)
        self._update_title()

    def _on_experiment_changed(self, source: BaseTab) -> None:
        """One tab changed the experiment: the others re-read it, and it is dirty."""
        self._dirty = True
        for tab in self.tab_widgets:
            if tab is not source:
                tab.refresh()

    def _save_experiment(self) -> bool:
        """Write the experiment (config and any computed results) to its folder.

        Returns True if the save was successful.
        """
        if self._busy:
            self.statusBar().showMessage("Cannot save while a step is running")
            return False
        folder = None
        if self.experiment.folder is None:
            folder = self._ask_save_folder()
            if folder is None:
                return False
        try:
            self.experiment.save(folder)
        except (OSError, ValueError, ValidationError) as exc:
            QMessageBox.critical(self, "Could not save experiment", str(exc))
            return False
        finally:
            # A failed save can still have created the folder, so the title is
            # brought up to date either way.
            self._update_title()
        self._dirty = False
        self.statusBar().showMessage(f"Saved experiment to {self.experiment.folder}")
        return True

    def _confirm_discarding_changes(self) -> bool:
        """Offer to save unsaved changes, and say whether to carry on.

        Returns True if the user discarded the changes, otherwise saves them and returns False
        """
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "This experiment has changes that have not been saved.",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save_experiment()
        return answer == QMessageBox.StandardButton.Discard

    def _ask_save_folder(self) -> Path | None:
        """Ask where to save an experiment that has no folder yet."""
        location = QFileDialog.getExistingDirectory(self, "Save experiment in…")
        if not location:
            return None
        name, chosen = QInputDialog.getText(
            self,
            "Save experiment",
            f"Name of the folder to create in {location}\n(leave empty to use it):",
            text=_DEFAULT_FOLDER_NAME,
        )
        if not chosen:
            return None
        name = name.strip()
        return Path(location) / name if name else Path(location)

    def _update_title(self) -> None:
        folder = self.experiment.folder
        name = folder.name if folder is not None else _UNSAVED_TITLE
        self.setWindowTitle(f"{_BASE_TITLE} :: [{name}]")

    def _set_busy(self, busy: bool, source: BaseTab | None = None) -> None:
        """Lock other tabs and actions while the source tab is busy."""
        self._busy = busy
        self.new_action.setEnabled(not busy)
        self.open_action.setEnabled(not busy)
        self.save_action.setEnabled(not busy)
        for index, tab in enumerate(self.tab_widgets):
            self.tabs.setTabEnabled(index, not busy or tab is source)

    def closeEvent(self, event) -> None:
        if not self._confirm_discarding_changes():
            event.ignore()
            return
        for tab in self.tab_widgets:
            tab.shutdown()
        super().closeEvent(event)
