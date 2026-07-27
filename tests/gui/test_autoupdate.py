import subprocess
from io import BytesIO

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


def _metadata(version, dependencies=()):
    return {"info": {"version": version, "requires_dist": list(dependencies)}}


def test_fetches_latest_release_metadata_from_pypi(monkeypatch):
    requested = []

    def fake_urlopen(url, timeout):
        requested.append((url, timeout))
        return BytesIO(b'{"info": {"version": "0.1.0", "requires_dist": []}}')

    monkeypatch.setattr(autoupdate, "urlopen", fake_urlopen)

    assert autoupdate._fetch_package_metadata() == _metadata("0.1.0")
    assert requested == [("https://pypi.org/pypi/body-eye-sync/json", 10.0)]


# --- check_for_update -------------------------------------------------------


def test_returns_update_when_remote_is_newer(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate, "_fetch_package_metadata", lambda: _metadata("0.1.0")
    )
    info = autoupdate.check_for_update()
    assert info == UpdateInfo("0.1.0", [])


def test_returns_none_when_up_to_date(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.1.0")
    monkeypatch.setattr(
        autoupdate, "_fetch_package_metadata", lambda: _metadata("0.1.0")
    )
    assert autoupdate.check_for_update() is None


def test_returns_none_when_remote_is_older(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.2.0")
    monkeypatch.setattr(
        autoupdate, "_fetch_package_metadata", lambda: _metadata("0.1.0")
    )
    assert autoupdate.check_for_update() is None


def test_returns_none_when_offline(not_editable, monkeypatch):
    def boom():
        raise OSError("offline")

    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(autoupdate, "_fetch_package_metadata", boom)
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

    monkeypatch.setattr(autoupdate, "_fetch_package_metadata", fail)
    assert autoupdate.check_for_update() is None


# --- dependency pre-flight check -------------------------------------------


def test_flags_a_brand_new_dependency(not_editable, monkeypatch):
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate,
        "_fetch_package_metadata",
        lambda: _metadata("0.1.0", ["definitely-not-installed-xyz"]),
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == ["definitely-not-installed-xyz"]


def test_flags_a_pin_beyond_what_is_installed(not_editable, monkeypatch):
    # numpy is installed (it's a real dependency); an impossible pin is unmet.
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate,
        "_fetch_package_metadata",
        lambda: _metadata("0.1.0", ["numpy==999.0.0"]),
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == ["numpy==999.0.0"]


def test_satisfied_dependencies_are_not_flagged(not_editable, monkeypatch):
    # numpy is installed and any version satisfies a bare requirement.
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate,
        "_fetch_package_metadata",
        lambda: _metadata("0.1.0", ["numpy"]),
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == []


def test_dependency_not_for_this_platform_is_skipped(not_editable, monkeypatch):
    # A marker that never applies must not be reported as missing.
    monkeypatch.setattr(autoupdate, "_installed_version", lambda: "0.0.1")
    monkeypatch.setattr(
        autoupdate,
        "_fetch_package_metadata",
        lambda: _metadata("0.1.0", ["not-installed-pkg; python_version < '3.0'"]),
    )
    info = autoupdate.check_for_update()
    assert info.missing_dependencies == []


# --- AutoUpdater flow -------------------------------------------------------


def _fake_completed(returncode, output=""):
    return subprocess.CompletedProcess([], returncode, stdout=output, stderr=output)


def test_pip_install_uses_pypi_package(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _fake_completed(0)

    monkeypatch.setattr(autoupdate.subprocess, "run", fake_run)

    autoupdate._run_pip_install()

    command, kwargs = calls[0]
    assert command == [
        autoupdate.sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "body-eye-sync",
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False


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
