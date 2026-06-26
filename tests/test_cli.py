from body_eye_sync.cli import main

from click.testing import CliRunner


def test_body_eye_sync_cli():
    runner = CliRunner()
    result = runner.invoke(main, ())
    assert result.exit_code == 0
