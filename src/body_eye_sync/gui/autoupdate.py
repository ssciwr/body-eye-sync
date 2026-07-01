"""Check GitHub for a newer version of body-eye-sync and offer to install it.

The version currently comes from the ``pyproject.toml`` on the ``main`` branch on
GitHub, and updates are installed straight from a source archive of that branch.

If any deps are missing we instead point the user at the full installer.

TODO: Once body-eye-sync is published on PyPI, check the version there
(``https://pypi.org/pypi/body-eye-sync/json``) and install with
``pip install --upgrade body-eye-sync`` instead of using GitHub.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError, version
from urllib.request import urlopen

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from qtpy.QtCore import QObject, Signal, Slot
from qtpy.QtWidgets import QMessageBox, QProgressDialog, QWidget

PACKAGE = "body-eye-sync"

# TODO: replace with the PyPI JSON API once the package is published there.
_PYPROJECT_URL = (
    "https://raw.githubusercontent.com/ssciwr/body-eye-sync/main/pyproject.toml"
)
# TODO: replace with ``pip install --upgrade body-eye-sync`` once on PyPI.
_ARCHIVE_URL = "https://github.com/ssciwr/body-eye-sync/archive/refs/heads/main.zip"
# Where to send users whose update needs new dependencies we can't install here.
_RELEASES_URL = "https://github.com/ssciwr/body-eye-sync/releases"


@dataclass
class UpdateInfo:
    """A newer version that is available to install."""

    version: str
    missing_dependencies: list[str]


def _installed_version() -> str | None:
    try:
        return version(PACKAGE)
    except PackageNotFoundError:
        return None


def _is_editable_install() -> bool:
    """True when running from an editable dev install (``uv run``, ``pip install -e``)."""
    try:
        direct_url = Distribution.from_name(PACKAGE).read_text("direct_url.json")
    except PackageNotFoundError:
        return False
    if not direct_url:
        return False
    try:
        info = json.loads(direct_url)
    except json.JSONDecodeError:
        return False
    return bool(info.get("dir_info", {}).get("editable"))


def _fetch_pyproject(timeout: float = 10.0) -> dict:
    with urlopen(_PYPROJECT_URL, timeout=timeout) as response:  # noqa: S310
        return tomllib.loads(response.read().decode("utf-8"))


def _missing_dependencies(pyproject: dict) -> list[str]:
    """Requirements the new version needs that this environment does not satisfy."""
    missing: list[str] = []
    for spec in pyproject.get("project", {}).get("dependencies", []):
        try:
            req = Requirement(spec)
        except InvalidRequirement:
            continue
        # Skip requirements that don't apply to this platform / Python version.
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            missing.append(spec)
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            missing.append(spec)
    return missing


def check_for_update() -> UpdateInfo | None:
    """Return info about a newer version if found, otherwise None."""
    installed = _installed_version()
    if installed is None:
        return None
    if _is_editable_install():
        # A developer running from a source checkout: don't overwrite their tree.
        return None
    try:
        pyproject = _fetch_pyproject()
    except Exception:
        # offline, GitHub unreachable, malformed pyproject, etc
        return None
    latest = pyproject.get("project", {}).get("version")
    if latest is None:
        return None
    try:
        if Version(latest) <= Version(installed):
            return None
    except InvalidVersion:
        return None
    return UpdateInfo(latest, _missing_dependencies(pyproject))


def _run_pip_install() -> subprocess.CompletedProcess:
    """Install the latest version from GitHub into the current environment using --no-deps."""
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            _ARCHIVE_URL,
        ],
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )


class AutoUpdater(QObject):
    _update_found = Signal(object)  # UpdateInfo
    _install_finished = Signal(bool, str)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._parent = parent
        self._progress: QProgressDialog | None = None
        self._update_found.connect(self._on_update_found)
        self._install_finished.connect(self._on_install_finished)

    def start(self) -> None:
        threading.Thread(target=self._check, daemon=True).start()

    def _check(self) -> None:
        info = check_for_update()
        if info is not None:
            self._update_found.emit(info)

    @Slot(object)
    def _on_update_found(self, info: UpdateInfo) -> None:
        if info.missing_dependencies:
            QMessageBox.information(
                self._parent,
                "Update available",
                f"A newer version of {PACKAGE} ({info.version}) is available, but "
                "it needs updated components that the in-app updater cannot "
                "install:\n\n  "
                + "\n  ".join(info.missing_dependencies)
                + "\n\nPlease download and run the latest installer instead:\n"
                + _RELEASES_URL,
            )
            return

        reply = QMessageBox.question(
            self._parent,
            "Update available",
            f"A newer version of {PACKAGE} ({info.version}) is available "
            f"(you have {_installed_version()}).\n\n"
            "Download and install it now?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._progress = QProgressDialog(
            "Downloading and installing update…", "", 0, 0, self._parent
        )
        self._progress.setWindowTitle("Updating")
        self._progress.setCancelButton(None)  # the install cannot be interrupted
        self._progress.setMinimumDuration(0)
        self._progress.show()

        threading.Thread(target=self._install, daemon=True).start()

    def _install(self) -> None:
        try:
            result = _run_pip_install()
        except Exception as exc:
            self._install_finished.emit(False, str(exc))
            return
        self._install_finished.emit(
            result.returncode == 0, result.stderr or result.stdout
        )

    @Slot(bool, str)
    def _on_install_finished(self, ok: bool, output: str) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None

        if not ok:
            QMessageBox.critical(
                self._parent,
                "Update failed",
                f"The update could not be installed:\n\n{output}",
            )
            return

        QMessageBox.information(
            self._parent,
            "Update installed",
            "The update has been installed. Please restart body-eye-sync to "
            "use the new version.",
        )
        # close the app so the user restarts into the freshly installed version.
        self._parent.window().close()
