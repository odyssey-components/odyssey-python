"""Reusable declaration for direct loading and command-line startup."""

from argparse import ArgumentParser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ._settings import bind_config_arguments, load_config


@dataclass(frozen=True)
class Loader[ConfigT: BaseModel]:
    model: type[ConfigT]
    env_prefix: str
    default_path: Path | None = None
    safe_error_messages: Mapping[str, str] | None = None
    sensitive_paths: frozenset[str] = frozenset()

    directory_name: str | None = None
    default_profile: str | None = None
    base_required: bool = True

    def bind_arguments(self, parser: ArgumentParser) -> None:
        if self.directory_name is not None:
            parser.add_argument("--config-dir")
        bind_config_arguments(parser, self.model, sensitive_paths=self.sensitive_paths)

    def load(
        self,
        path: str | Path | None = None,
        *,
        profile: str | None = None,
        config_dir: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        default_path: Path | None = None,
    ) -> ConfigT:
        return load_config(
            self.model,
            path=Path(path).expanduser() if path is not None else None,
            profile=profile,
            config_dir=config_dir,
            directory_name=self.directory_name,
            default_profile=self.default_profile,
            base_required=self.base_required,
            overrides=overrides,
            default_path=self.default_path if default_path is None else default_path,
            env_prefix=self.env_prefix,
            sensitive_paths=self.sensitive_paths,
            safe_error_messages=self.safe_error_messages,
        )
