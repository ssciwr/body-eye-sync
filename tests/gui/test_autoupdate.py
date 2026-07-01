import subprocess

import pytest
from qtpy.QtWidgets import QMessageBox, QWidget

from body_eye_sync.gui import autoupdate
from body_eye_sync.gui.autoupdate import AutoUpdater, UpdateInfo


@pytest.fixture
def parent(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def not_editable(monkeypatch):
    monkeypatch.setattr(autoupdate, "_is_editable_install", lambda: False)


def _pyproject(version, dependencies=()):
    return {"project": {"version": version, "dependencies": list(dependencies)}}


# --- check_for_update -------------------------------------------------------


def test_returns_update_when_remote_is_newer(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(autoupdate, "_fetch_pyproject", lambda: _pyproject("0.1.0"))
    info = autoupdate.check_for_update()
    assert info == UpdateInfo("0.1.0", [])


def test_returns_none_when_up_to_date(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.1.0")
    monkeypatch.setattr(autoupdate, "_fetch_pyproject", lambda: _pyproject("0.1.0"))
    assert autoupdate.check_for_update() is None


def test_returns_none_when_remote_is_older(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.2.0")
    monkeypatch.setattr(autoupdate, "_fetch_pyproject", lambda: _pyproject("0.1.0"))
    assert autoupdate.check_for_update() is None


def test_returns_none_when_offline(not_editable, monkeypatch):
    def boom():
        raise OSError("offline")

    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(autoupdate, "_fetch_pyproject", boom)
    assert autoupdate.check_for_update() is None


def test_returns_none_when_not_installed(monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: None)
    assert autoupdate.check_for_update() is None


def test_editable_install_never_updates(monkeypatch):
    # A dev checkout (uv run / pip install -e): skip without even fetching.
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(autoupdate, "_is_editable_install", lambda: True)

    def fail():
        raise AssertionError("should not fetch for an editable install")

    monkeypatch.setattr(autoupdate, "_fetch_pyproject", fail)
    assert autoupdate.check_for_update() is None


# --- dependency pre-flight check -------------------------------------------


def test_flags_a_brand_new_dependency(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate,
        "_fetch_pyproject",
        lambda: _pyproject("0.1.0", ["definitely-not-installed-xyz"]),
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == ["definitely-not-installed-xyz"]


def test_flags_a_pin_beyond_what_is_installed(not_editable, monkeypatch):
    # numpy is installed (it's a real dependency); an impossible pin is unmet.
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate, "_fetch_pyproject", lambda: _pyproject("0.1.0", ["numpy==999.0.0"])
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == ["numpy==999.0.0"]


def test_satisfied_dependencies_are_not_flagged(not_editable, monkeypatch):
    # numpy is installed and any version satisfies a bare requirement.
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate, "_fetch_pyproject", lambda: _pyproject("0.1.0", ["numpy"])
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == []


def test_dependency_not_for_this_platform_is_skipped(not_editable, monkeypatch):
    # A marker that never applies must not be reported as missing.
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate,
        "_fetch_pyproject",
        lambda: _pyproject("0.1.0", ["not-installed-pkg; python_version < '3.0'"]),
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == []


# --- AutoUpdater flow -------------------------------------------------------


def _fake_completed(returncode, output=""):
    return subprocess.CompletedProcess([], returncode, stdout=output, stderr=output)


def test_missing_deps_sends_user_to_installer_without_installing(
    qtbot, parent, monkeypatch
):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a))
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a))
    installed = []
    monkeypatch.setattr(
        autoupdate,
        "_run_pip_install",
        lambda: installed.append(True) or _fake_completed(0),
    )

    AutoUpdater(parent)._on_update_found(UpdateInfo("9.9.9", ["new-dep"]))

    assert len(infos) == 1  # pointed at the installer
    assert asked == []  # never offered the in-app update
    assert installed == []  # nothing installed


def test_declining_update_does_not_install(qtbot, parent, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    called = []
    monkeypatch.setattr(
        autoupdate,
        "_run_pip_install",
        lambda: called.append(True) or _fake_completed(0),
    )

    AutoUpdater(parent)._on_update_found(UpdateInfo("9.9.9", []))

    assert called == []


def test_successful_update_installs_and_closes(qtbot, parent, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(autoupdate, "_run_pip_install", lambda: _fake_completed(0))

    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a))
    closed = []
    monkeypatch.setattr(parent, "close", lambda: closed.append(True))

    updater = AutoUpdater(parent)
    with qtbot.waitSignal(updater._install_finished, timeout=5000):
        updater._on_update_found(UpdateInfo("9.9.9", []))

    # The queued slot runs on the GUI thread; let the event loop deliver it.
    qtbot.waitUntil(lambda: closed == [True], timeout=5000)
    assert len(infos) == 1  # "please restart" message shown


def test_failed_update_reports_error_and_stays_open(qtbot, parent, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(
        autoupdate, "_run_pip_install", lambda: _fake_completed(1, "pip exploded")
    )

    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    closed = []
    monkeypatch.setattr(parent, "close", lambda: closed.append(True))

    updater = AutoUpdater(parent)
    with qtbot.waitSignal(updater._install_finished, timeout=5000):
        updater._on_update_found(UpdateInfo("9.9.9", []))

    qtbot.waitUntil(lambda: len(errors) == 1, timeout=5000)
    assert closed == []  # app is not closed on a failed update
