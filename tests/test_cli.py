from body_eye_sync.cli import main

from click.testing import CliRunner


def test_body_eye_sync_cli():
    runner = CliRunner()
    result = runner.invoke(main, ())
    assert result.exit_code == 0


def test_body_eye_sync_cli_version():
    from importlib.metadata import version

    runner = CliRunner()
    result = runner.invoke(main, ("--version",))
    assert result.exit_code == 0
    assert version("body-eye-sync") in result.output
