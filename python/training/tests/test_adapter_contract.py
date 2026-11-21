import json
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from odylab_training import event_record, task_overrides, write_config_package, write_zip_entry


from odylab_training.worker import WorkerRun
from odylab_training.worker import training_stage

def test_standalone_worker_package_carries_config_and_preserves_executable_mode() -> None:
    data = BytesIO()
    with ZipFile(data, "w") as archive:
        write_zip_entry(archive, "run", "#!/bin/sh\n", executable=True)
        write_config_package(archive)
    with ZipFile(data) as archive:
        assert archive.getinfo("run").external_attr >> 16 == 0o100755
        assert "odyssey_config/__init__.py" in archive.namelist()
        assert archive.getinfo("odyssey_config/__init__.py").external_attr >> 16 == 0o100644


def test_worker_config_and_event_keep_game_specific_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "task.json"
    config.write_text(json.dumps({"training": {"seed": 23}, "monitoring": {"metrics": ["loss"]}}))
    monkeypatch.setenv("ODY_TASK_CONFIG", str(config))

    overrides = task_overrides()
    assert overrides == {"training.seed": 23, "monitoring.metrics": ["loss"]}
    event = event_record("training.progress", "fixture", {"seed": overrides["training.seed"]})
    assert json.loads(json.dumps(event))["attributes"] == {"seed": 23}


def test_managed_worker_uses_scene_defaults_and_task_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    scene, output = tmp_path / "scene", tmp_path / "output"
    task = tmp_path / "task.json"
    task.write_text('{"training":{"seed":23,"iterations":4}}')
    monkeypatch.setenv("ODY_SCENE_DIR", str(scene))
    monkeypatch.setenv("ODY_OUTPUT_DIR", str(output))
    monkeypatch.setenv("ODY_TASK_CONFIG", str(task))

    worker = WorkerRun.from_environment()
    def load_options(_path: Path | None, **options: object) -> dict[str, object]:
        return options

    options = worker.load_config(load_options, overrides={"training.seed": 42})
    assert worker.output_dir == output
    assert options["default_path"] == scene / "training.yaml"
    assert options["overrides"] == {"training.seed": 42, "training.iterations": 4}
    assert not output.exists()

    monkeypatch.delenv("ODY_SCENE_DIR")
    with pytest.raises(ValueError, match="ODY_SCENE_DIR and ODY_OUTPUT_DIR are required"):
        WorkerRun.from_environment()


@pytest.mark.parametrize("failure", [False, True])
def test_training_stage_emits_process_evidence_and_keeps_error(
    failure: bool, capsys: pytest.CaptureFixture[str],
) -> None:

    error = RuntimeError("private diagnostic")
    expectation = pytest.raises(RuntimeError) if failure else nullcontext()
    with expectation as raised, training_stage("execution"):
        if failure:
            raise error
    if failure:
        assert raised is not None
        assert raised.value is error
    output = capsys.readouterr().out
    assert "private diagnostic" not in output
    assert [json.loads(line)["attributes"] for line in output.splitlines()] == [
        {"stage": "execution", "status": "started"},
        {"stage": "execution", "status": "failed" if failure else "completed"},
    ]


def test_training_stage_decorator_preserves_result_stdout(capsys: pytest.CaptureFixture[str]) -> None:

    @training_stage("execution", stderr=True)
    def train() -> int:
        print('{"result": 42}')
        return 42

    assert train() == 42
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"result": 42}
    assert [json.loads(line)["attributes"]["status"] for line in captured.err.splitlines()] == ["started", "completed"]
