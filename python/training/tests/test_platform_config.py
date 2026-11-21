"""Frozen platform inputs use the existing config loader without host overrides."""

import hashlib
from pathlib import Path

import pytest
from odylab_training import platform
from odylab_training.platform import load_frozen_config, platform_schema
from odyssey_config import load_config
from pydantic import BaseModel


class Training(BaseModel):
    steps: int


class Settings(BaseModel):
    training: Training


def test_frozen_config_ignores_host_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "training.yaml"
    data = b'{"training":{"steps":42}}'
    path.write_bytes(data)
    monkeypatch.setenv("ODYLAB_TRAINING_STEPS", "999")
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert load_frozen_config(Settings, path, digest).training.steps == 42
    path.write_bytes(b'{"training":{"steps":99}}')
    with pytest.raises(ValueError, match="digest"):
        load_frozen_config(Settings, path, digest)


def test_published_schema_closes_model_fields() -> None:
    schema = platform_schema(Settings)
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Training"]["additionalProperties"] is False
    assert "additionalProperties" not in Settings.model_json_schema()


def test_loader_reads_the_verified_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "training.yaml"
    data = b'{"training":{"steps":42}}'
    path.write_bytes(data)
    original_loader = load_config

    def swap_source(model: type[Settings], **kwargs: object) -> Settings:
        path.write_bytes(b'{"training":{"steps":99}}')
        return original_loader(model, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(platform, "load_config", swap_source)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    assert load_frozen_config(Settings, path, digest).training.steps == 42
