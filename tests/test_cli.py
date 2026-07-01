from body_eye_sync import cli
from body_eye_sync.cli import main
from body_eye_sync.experiment.config import (
    BodyPoseStep,
    Experiment,
    FaceDetectionStep,
    ObjectTrackingStep,
)

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


def test_init_scaffolds_loadable_experiment_folder(tmp_path):
    folder = tmp_path / "my-experiment"
    runner = CliRunner()
    result = runner.invoke(main, ("init", str(folder)))

    assert result.exit_code == 0
    exp = Experiment.from_dir(folder)
    assert exp.name == "my-experiment"
    assert exp.face_detection is not None
    assert exp.body_pose is not None
    assert [type(s) for s in exp.steps] == [
        ObjectTrackingStep,
        FaceDetectionStep,
        BodyPoseStep,
    ]


def test_init_refuses_existing_without_force(tmp_path):
    folder = tmp_path / "exp"
    runner = CliRunner()
    assert runner.invoke(main, ("init", str(folder))).exit_code == 0
    (folder / "experiment.yaml").write_text("touched")

    result = runner.invoke(main, ("init", str(folder)))
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert (folder / "experiment.yaml").read_text() == "touched"

    forced = runner.invoke(main, ("init", str(folder), "--force"))
    assert forced.exit_code == 0
    assert Experiment.from_dir(folder).name == "exp"


def test_run_parses_options_and_invokes_run_experiment(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    CliRunner().invoke(main, ("init", str(folder)))

    captured = {}

    def fake_run_experiment(experiment, **kwargs):
        captured["experiment"] = experiment
        captured["kwargs"] = kwargs
        return {"cam1": folder / "outputs" / "cam1.parquet"}

    monkeypatch.setattr(cli, "run_experiment", fake_run_experiment)

    result = CliRunner().invoke(
        main,
        (
            "run",
            str(folder),
            "--device",
            "cpu",
            "--providers",
            "CPUExecutionProvider, CUDAExecutionProvider",
            "--force",
        ),
    )

    assert result.exit_code == 0, result.output
    assert isinstance(captured["experiment"], Experiment)
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["providers"] == [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
    ]
    assert captured["kwargs"]["force"] is True
    assert "cam1:" in result.output


def test_run_accepts_a_yaml_file_directly(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    CliRunner().invoke(main, ("init", str(folder)))
    monkeypatch.setattr(cli, "run_experiment", lambda experiment, **kwargs: {})

    result = CliRunner().invoke(main, ("run", str(folder / "experiment.yaml")))
    assert result.exit_code == 0, result.output


def test_run_defaults_providers_to_none(tmp_path, monkeypatch):
    folder = tmp_path / "exp"
    CliRunner().invoke(main, ("init", str(folder)))

    captured = {}
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda experiment, **kwargs: captured.update(kwargs) or {},
    )

    result = CliRunner().invoke(main, ("run", str(folder)))
    assert result.exit_code == 0, result.output
    assert captured["providers"] is None
    assert captured["device"] is None
