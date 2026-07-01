from importlib.metadata import version
import sys

from click.testing import CliRunner

from body_eye_sync.__main__ import _ensure_standard_streams, main


def test_main_version():
    # --version is eager and short-circuits before any QApplication is created,
    # so this runs headless without a Qt fixture.
    result = CliRunner().invoke(main, ("--version",))
    assert result.exit_code == 0
    assert version("body-eye-sync") in result.output


def test_pythonw_launch_has_writable_standard_streams(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "__stdout__", None)
    monkeypatch.setattr(sys, "__stderr__", None)

    _ensure_standard_streams()

    assert sys.stdout is not None
    assert sys.stderr is not None
    sys.stdout.write("")
    sys.stderr.write("")
    assert sys.__stdout__ is sys.stdout
    assert sys.__stderr__ is sys.stderr
