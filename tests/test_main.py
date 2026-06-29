from importlib.metadata import version

from click.testing import CliRunner

from body_eye_sync.__main__ import main


def test_main_version():
    # --version is eager and short-circuits before any QApplication is created,
    # so this runs headless without a Qt fixture.
    result = CliRunner().invoke(main, ("--version",))
    assert result.exit_code == 0
    assert version("body-eye-sync") in result.output
