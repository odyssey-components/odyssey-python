"""Common types and their shared command-line loader."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NotRequired, TypedDict

import pytest
from odyssey_config import Database, Loader, Logging, ObjectStorage, Web, config_snapshot, run_app
from pydantic import BaseModel

from pydantic import ValidationError

CASES = Path(__file__).resolve().parents[3] / "docs/configuration/contract/cases/common-types.json"
TYPES: dict[str, type[BaseModel]] = {"database": Database, "web": Web, "object_storage": ObjectStorage, "logging": Logging}



class CommonTypeCase(TypedDict):
    section: str
    value: dict[str, object]
    valid: bool


class DiscoveryCase(TypedDict):
    files: dict[str, str]
    args: list[str]
    environment: NotRequired[dict[str, str]]
    error: NotRequired[bool]
    expected: NotRequired[str]
    labels: NotRequired[list[str]]
    enabled: NotRequired[bool]


@pytest.mark.parametrize("case", json.loads(CASES.read_text()))
def test_common_type_contract(case: CommonTypeCase, tmp_path: Path) -> None:
    section = case["section"]
    loader = Loader(TYPES[section], "COMMON_TEST")
    path = tmp_path / "application.yaml"
    path.write_text(json.dumps(case["value"]))
    if case["valid"]:
        config = loader.load(path)
        assert "private-value" not in repr(config_snapshot(config))
    else:
        with pytest.raises(ValueError) as error:
            loader.load(path)
        assert "private-value" not in str(error.value)


def test_shared_cli_loads_and_overrides_nested_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    class Settings(BaseModel):
        database: Database
        web: Web

    path = tmp_path / "app.yaml"
    path.write_text("database: {host: localhost, port: 5432}\nweb: {listen_addr: ':8080'}\n")
    (tmp_path / "app-test.yaml").write_text("database: {port: 5433}\nweb: {read_timeout: 2s}\n")
    monkeypatch.setenv("COMMON_TEST_DATABASE_PORT", "5434")
    monkeypatch.setenv("COMMON_TEST_DATABASE_PASSWORD", "private-value")

    def start(settings: Settings, args: argparse.Namespace) -> int:
        assert settings.database.port == 5435
        assert settings.web.read_timeout == "2s"
        assert config_snapshot(settings)["database"]["password"] == "<redacted>"
        assert vars(args) == {"output": "result"}
        return 0

    def arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output", required=True)

    def invoke(
        argv: Sequence[str],
        add_arguments: Callable[[argparse.ArgumentParser], object] = arguments,
    ) -> int:
        def bind(parser: argparse.ArgumentParser) -> None:
            add_arguments(parser)

        return run_app(
            argv=argv, name="app", loader=Loader(Settings, "COMMON_TEST", path),
            add_arguments=bind, start=start, version=lambda: "1",
        )

    assert invoke(["--profile", "test", "--database.port", "5435", "--output", "result"]) == 0
    with pytest.raises(SystemExit) as exit_info:
        invoke(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--database.port" in help_text
    assert "--database.password" not in help_text

    with pytest.raises(ValueError, match="configuration destination"):
        invoke([], lambda parser: parser.add_argument("--other", dest="profile"))
    with pytest.raises(argparse.ArgumentError):
        invoke([], lambda parser: parser.add_argument("--profile"))


def test_shared_settings_revalidates_aliased_fields_and_hides_inputs():

    original = Database(host="localhost", schema="training")
    copy = original.validate_settings()
    assert copy is not original
    assert copy.database_schema == "training"
    original.host = "https://private-value"
    with pytest.raises(ValidationError) as error:
        original.validate_settings()
    assert "private-value" not in str(error.value)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Database.model_validate({"host": "localhost", "unknown": 1})


@pytest.mark.parametrize("case", json.loads(CASES.with_name("discovery.json").read_text()))
def test_shared_discovery_contract(
    case: DiscoveryCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    class Server(BaseModel):
        listen_addr: str = ""
        enabled: bool = False
        labels: list[str] = []

    class Settings(BaseModel):
        server: Server = Server()

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP_CONFIG_FILE", raising=False)
    monkeypatch.delenv("APP_PROFILE", raising=False)
    for key, value in case.get("environment", {}).items():
        monkeypatch.setenv(key, value)
    for name, content in case["files"].items():
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    started: list[Server] = []

    def start(config: Settings, _args: argparse.Namespace) -> None:
        started.append(config.server)

    def invoke() -> int:
        return run_app(
            argv=case["args"], name="app", loader=Loader(Settings, "APP", directory_name="app"),
            start=start, version=lambda: "1",
        )

    if case.get("error"):
        with pytest.raises(SystemExit) as error:
            invoke()
        assert error.value.code == 2
        assert not started
        assert "private-value" not in capsys.readouterr().err
    else:
        assert invoke() == 0
        assert len(started) == 1
        assert "expected" in case
        assert started[0].listen_addr == case["expected"]
        if "labels" in case:
            assert started[0].labels == case["labels"]
        if "enabled" in case:
            assert started[0].enabled == case["enabled"]
