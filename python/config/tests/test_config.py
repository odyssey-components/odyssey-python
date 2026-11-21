import argparse
import json
import tempfile
import unittest
from pathlib import Path

from odyssey_config import bind_config_arguments, config_snapshot, load_config
from pydantic import BaseModel, ConfigDict, Field, SecretStr


from pydantic import model_validator
from pydantic_core import PydanticCustomError
import pytest
from odyssey_config import Database

class Training(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iterations: int = Field(default=20, gt=0)
    metrics: list[str] = Field(default_factory=lambda: ["reward"])


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training: Training = Field(default_factory=Training)
    token: SecretStr | None = None


class TypedConfigTests(unittest.TestCase):
    def test_sources_are_bound_and_validated_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("training:\n  iterations: 30\n")
            settings = load_config(Settings, path=path, env_prefix="ODYLAB_MAZE", environment={"ODYLAB_MAZE_TRAINING_ITERATIONS": "40"}, overrides={"training.iterations": 50})
        self.assertIsInstance(settings, Settings)
        self.assertEqual(settings.training.iterations, 50)

    def test_environment_lists_and_platform_bindings(self):
        settings = load_config(Settings, env_prefix="ODYLAB_MAZE", environment={"ODYLAB_MAZE_TRAINING_METRICS": '["loss", "reward"]', "ODY_OUTPUT_TOKEN": "private"}, env_bindings={"ODY_OUTPUT_TOKEN": "token"})
        self.assertEqual(settings.training.metrics, ["loss", "reward"])
        assert settings.token is not None
        self.assertEqual(settings.token.get_secret_value(), "private")
        with self.assertRaises(ValueError):
            load_config(Settings, env_prefix="ODYLAB_MAZE", environment={"ODYLAB_MAZE_TRAINING_ITERATIONS": "0"})

    def test_secret_from_file_or_environment_but_not_command_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text('token: private\n')
            settings = load_config(Settings, path=path, env_prefix="ODYLAB_MAZE", environment={})
            assert settings.token is not None
            self.assertEqual(settings.token.get_secret_value(), "private")
        with self.assertRaises(ValueError):
            load_config(Settings, env_prefix="ODYLAB_MAZE", environment={}, overrides={"token": "private"})
        settings = load_config(Settings, env_prefix="ODYLAB_MAZE", environment={"ODYLAB_MAZE_TOKEN": "private"})
        self.assertEqual(config_snapshot(settings)["token"], "<redacted>")

    def test_yaml_and_yml_with_unknown_nested_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, content in (("yaml", "training:\n  iterations: 31\n"), ("yml", "training:\n  iterations: 31\n")):
                path = Path(directory) / f"config.{suffix}"
                path.write_text(content)
                self.assertEqual(load_config(Settings, path=path, environment={}).training.iterations, 31)
                path.unlink()
            path = Path(directory) / "invalid.yaml"
            path.write_text("unknown: {}\n")
            with self.assertRaises(ValueError):
                load_config(Settings, path=path, environment={})

    def test_unsupported_extensions_are_rejected(self):
        for suffix in ("json",):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "extension"):
                load_config(Settings, path=Path("config." + suffix), environment={})

    def test_rejects_colliding_environment_paths(self):
        class Nested(BaseModel):
            value: int = 1

        class Collision(BaseModel):
            nested: Nested = Field(default_factory=Nested)
            nested_value: int = 2

        with self.assertRaisesRegex(ValueError, "collision"):
            load_config(Collision, env_prefix="ODYLAB_MAZE", environment={})

    def test_cli_only_overrides_explicit_values_and_excludes_secrets(self):
        parser = argparse.ArgumentParser()
        bind_config_arguments(parser, Settings)
        self.assertNotIn("--token", parser.format_help())
        self.assertEqual(vars(parser.parse_args([])), {})
        overrides = vars(parser.parse_args(["--training.iterations", "51"]))
        settings = load_config(Settings, env_prefix="ODYLAB_MAZE", environment={"ODYLAB_MAZE_TRAINING_ITERATIONS": "40"}, overrides=overrides)
        self.assertEqual(settings.training.iterations, 51)



def test_validation_errors_are_useful_without_input_values(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text('training:\n  iterations: private-value\n')
    try:
        load_config(Settings, path=path, environment={})
    except ValueError as error:
        assert 'training.iterations' in str(error)
        assert 'invalid integer' in str(error)
        assert 'private-value' not in str(error)
    else:
        raise AssertionError('invalid value was accepted')


def test_custom_validation_message_and_sensitive_input_are_not_exposed():

    class Unsafe(BaseModel):
        token: str

        @model_validator(mode='after')
        def validate_token(self):
            raise PydanticCustomError('unknown_custom_error', 'secret: {token}', {'token': self.token})

    try:
        load_config(Unsafe, environment={'APP_TOKEN': 'private-token'}, env_prefix='APP')
    except ValueError as error:
        assert 'private-token' not in str(error)
        assert 'invalid value' in str(error)
        assert error.__suppress_context__
    else:
        raise AssertionError('invalid token was accepted')


def test_profile_and_toml_precedence(tmp_path: Path) -> None:
    base = tmp_path / "app.toml"
    base.write_text('[training]\niterations = 30\nmetrics = ["base"]\n')
    (tmp_path / "app-prod.toml").write_text('[training]\niterations = 35\nmetrics = ["profile"]\n')
    settings = load_config(Settings, path=base, env_prefix="APP", environment={"APP_PROFILE": "prod", "APP_TRAINING_ITERATIONS": "40"}, overrides={"training.iterations": "50"})
    assert settings.training.iterations == 50
    assert settings.training.metrics == ["profile"]


def test_profile_failures_and_duplicate_yaml(tmp_path: Path) -> None:
    base = tmp_path / "app.yaml"
    base.write_text('training:\n  iterations: 20\n  iterations: 30\n')
    with pytest.raises(ValueError):
        load_config(Settings, path=base, environment={})
    base.write_text('training: {}\n')
    with pytest.raises(ValueError):
        load_config(Settings, path=base, profile="../prod", environment={})
    with pytest.raises(ValueError):
        load_config(Settings, path=base, profile="prod", environment={})


def test_shared_contract_source_cases(tmp_path: Path) -> None:

    class Server(BaseModel):
        listen_addr: str
        labels: list[str]
        enabled: bool = False

    class ContractSettings(BaseModel):
        server: Server

    directory = Path(__file__).resolve().parents[3] / 'docs/configuration/contract/cases'
    case = json.loads((directory / 'sources.json').read_text())
    # Required typed fields are supplied by the fixtures.
    for format in ('yaml', 'toml'):
        for name in (f'base.{format}', f'base-prod.{format}'):
            (tmp_path / name).write_bytes((directory / name).read_bytes())
        settings = load_config(
            ContractSettings, path=tmp_path / f'base.{format}',
            env_prefix='DEMO', environment=case['environment'],
            profile='prod', overrides=case['arguments'],
        )
        assert settings.model_dump() == case['expected']
        (tmp_path / f'base.{format}').unlink()
        (tmp_path / f'base-prod.{format}').unlink()


def test_shared_contract_error_cases(tmp_path: Path) -> None:


    class Server(BaseModel):
        listen_addr: str = ''
        labels: list[str] = Field(default_factory=list)
        enabled: bool = False

    class ContractSettings(BaseModel):
        server: Server = Field(default_factory=Server)

    directory = Path(__file__).resolve().parents[3] / 'docs/configuration/contract/cases'
    for case in json.loads((directory / 'errors.json').read_text()):
        if case['id'] in {'format_mismatch', 'profile_unknown'}:
            continue
        path = tmp_path / case['file']
        path.write_bytes((directory / case['file']).read_bytes())
        with pytest.raises(ValueError) as error:
            load_config(ContractSettings, path=path, environment={}, profile=case.get('profile'))
        assert 'private-value' not in str(error.value), case['id']
        path.unlink()


def test_shared_database_type_uses_canonical_keys_and_redacts():

    class Application(BaseModel):
        database: Database

    settings = load_config(Application, env_prefix='APP', environment={
        'APP_DATABASE_HOST': 'localhost',
        'APP_DATABASE_SCHEMA': 'private', 'APP_DATABASE_PASSWORD': 'secret',
    })
    assert settings.database.database_schema == 'private'
    assert config_snapshot(settings)['database']['password'] == '<redacted>'


def test_discovers_toml_and_arbitrary_profile(tmp_path: Path) -> None:
    base = tmp_path / 'config.toml'
    base.write_text('[training]\niterations = 20\n')
    (tmp_path / 'config-staging.yaml').write_text('training:\n  iterations: 30\n')
    assert load_config(Settings, path=base, profile='staging', environment={}).training.iterations == 30


def test_conflicting_formats_rejected(tmp_path: Path) -> None:
    base = tmp_path / 'config.yaml'
    base.write_text('training: {}\n')
    (tmp_path / 'config.toml').write_text('[training]\n')
    with pytest.raises(ValueError, match='multiple'):
        load_config(Settings, path=base, environment={})
