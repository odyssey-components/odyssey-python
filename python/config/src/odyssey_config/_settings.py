"""Bind configuration sources to a validated Pydantic settings model."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, SecretBytes, SecretStr, TypeAdapter, ValidationError
from pydantic_core import PydanticUndefined

from ._sources import config_paths
from .documents import read_document
from .errors import ConfigurationError

SettingsT = TypeVar("SettingsT", bound=BaseModel)


def _model_type(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _is_secret(annotation: Any) -> bool:
    return annotation in (SecretStr, SecretBytes) or any(_is_secret(arg) for arg in get_args(annotation))


def _fields(model: type[BaseModel], prefix: str = "") -> dict[str, tuple[Any, bool]]:
    result: dict[str, tuple[Any, bool]] = {}
    for name, field in model.model_fields.items():
        name = field.alias if isinstance(field.alias, str) else name
        key = f"{prefix}.{name}" if prefix else name
        nested = _model_type(field.annotation)
        if nested:
            result.update(_fields(nested, key))
        else:
            result[key] = (field.annotation, _is_secret(field.annotation))
    return result


def _defaults(model: type[BaseModel]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        name = field.alias if isinstance(field.alias, str) else name
        nested = _model_type(field.annotation)
        default = field.get_default(call_default_factory=True)
        if default is not PydanticUndefined:
            result[name] = default.model_dump(by_alias=True) if isinstance(default, BaseModel) else copy.deepcopy(default)
        elif nested:
            result[name] = _defaults(nested)
    return result


def bind_config_arguments(parser: argparse.ArgumentParser, model: type[BaseModel], *, sensitive_paths: set[str] | frozenset[str] = frozenset(), allowed_sections: set[str] | frozenset[str] | None = None) -> None:
    for key, (_, secret) in _fields(model).items():
        if secret or key in sensitive_paths or (allowed_sections is not None and key.split('.')[0] not in allowed_sections):
            continue
        flag = "--" + ".".join(part.replace("_", "-") for part in key.split("."))
        parser.add_argument(flag, dest=key, default=argparse.SUPPRESS)


def _flatten[KeyT](values: Mapping[KeyT, Any], fields: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        if key in fields:
            if key in result:
                raise ValueError(f"duplicate configuration key: {key}")
            result[key] = value
        elif isinstance(value, Mapping) and any(field.startswith(key + ".") for field in fields):
            for child, item in _flatten(cast("Mapping[object, Any]", value), fields, key).items():
                if child in result:
                    raise ValueError(f"duplicate configuration key: {child}")
                result[child] = item
        else:
            raise ConfigurationError("unknown_key", "unknown configuration key")
    return result


def _set(values: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = values
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            raise ValueError(f"configuration value {part} must be an object")  # noqa: TRY004
        cursor = cast("dict[str, Any]", nested)
    cursor[parts[-1]] = value



def _safe_validation_error(error: ValidationError, fields: Mapping[str, Any], fallback: str = "configuration", safe_errors: Mapping[str, str] | None = None) -> str:
    details: list[str] = []
    categories = {
        "missing": "required", "literal_error": "unsupported choice",
        "greater_than_equal": "below minimum", "less_than_equal": "above maximum",
        "greater_than": "below minimum", "less_than": "above maximum",
        "int_parsing": "invalid integer", "int_type": "invalid integer",
        "float_parsing": "invalid number", "float_type": "invalid number",
        "string_type": "invalid string", "bool_parsing": "invalid boolean",
        "list_type": "invalid list", "finite_number": "invalid number",
    }
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        parts: list[str] = []
        for part in item["loc"]:
            candidate = ".".join([*parts, str(part)])
            if candidate not in fields and not any(key.startswith(candidate + ".") for key in fields):
                break
            parts.append(str(part))
        location = ".".join(parts) or fallback
        category = (safe_errors or {}).get(item["type"], categories.get(item["type"], "invalid value"))
        details.append(f"{location}: {category}")
    return "; ".join(details)


def _validate_file_value(key: str, annotation: Any, value: Any) -> None:
    # File values must have their declared type. Paths and durations are
    # represented as strings in both portable file formats.

    candidates = (annotation, *get_args(annotation))
    string_encoded = any(
        item in (Path, timedelta) or (isinstance(item, type) and issubclass(item, Enum))
        for item in candidates
    )
    try:
        TypeAdapter(annotation).validate_python(value, strict=not string_encoded)
    except ValidationError as error:
        raise ConfigurationError("invalid_type", _safe_validation_error(error, {key: annotation}, key)) from None
    except TypeError:
        raise ValueError(f"invalid configuration value: {key}") from None


def _decode_override(key: str, annotation: Any, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    alternatives = tuple(item for item in get_args(annotation) if item is not type(None))
    if type(None) in get_args(annotation) and len(alternatives) == 1:
        annotation = alternatives[0]
    if annotation is bool:
        if value in {"1", "t", "T", "true", "TRUE", "True"}:
            return True
        if value in {"0", "f", "F", "false", "FALSE", "False"}:
            return False
        raise ConfigurationError("invalid_type", f"{key}: invalid boolean")
    origin = get_origin(annotation)
    if origin in (list, tuple, set, dict):
        expected = dict if origin is dict else list
        try:
            decoded: object = json.loads(value)
        except ValueError:
            raise ConfigurationError("invalid_type", f"{key}: invalid JSON configuration value") from None
        if not isinstance(decoded, expected):
            raise ConfigurationError("invalid_type", f"{key}: invalid JSON configuration value")
        return cast("dict[object, Any] | list[Any]", decoded)
    return value


def load_config(model: type[SettingsT], *, path: str | Path | None = None,  # noqa: UP047
                      environment: Mapping[str, str] | None = None,
                      env_bindings: Mapping[str, str] | None = None,
                      sensitive_paths: set[str] | frozenset[str] = frozenset(),
                      env_prefix: str = "", overrides: Mapping[str, Any] | None = None,
                      allowed_sections: set[str] | frozenset[str] | None = None,
                      profile: str | None = None, base_required: bool = True,
                      default_path: str | Path | None = None,
                      directory_name: str | None = None, config_dir: str | None = None,
                      default_profile: str | None = None,
                      safe_error_messages: Mapping[str, str] | None = None) -> SettingsT:
    if _model_type(model) is None:
        raise TypeError("configuration schema must be a Pydantic model")
    all_fields = _fields(model)
    fields = {key: value for key, value in all_fields.items() if allowed_sections is None or key.split('.')[0] in allowed_sections}
    secrets = {key for key, (_, secret) in fields.items() if secret} | set(sensitive_paths)
    names: dict[str, str] = {}
    for key in fields:
        env_name = f"{env_prefix}_{key.replace('.', '_')}".upper() if env_prefix else ""
        if env_name:
            if env_name in names:
                raise ConfigurationError("environment_collision", "configuration environment name collision")
            names[env_name] = key
    for env_name, key in (env_bindings or {}).items():
        if key not in fields or (env_name in names and names[env_name] != key):
            raise ValueError(f"invalid environment binding: {env_name}")
        names[env_name] = key
    env = os.environ if environment is None else environment
    result = _defaults(model)
    for document in config_paths(
        path=path, default_path=default_path, profile=profile, default_profile=default_profile,
        env=env, prefix=env_prefix.upper(), base_required=base_required,
        directory_name=directory_name, config_dir=config_dir,
    ):
        for key, value in _flatten(read_document(document), fields).items():
            _validate_file_value(key, fields[key][0], value)
            _set(result, key, value)
    text_overrides: dict[str, Any] = {}
    for name, value in env.items():
        if env_prefix and name in {f"{env_prefix.upper()}_PROFILE", f"{env_prefix.upper()}_CONFIG_FILE"}:
            continue
        key = names.get(name)
        if key is None:
            if env_prefix and name.startswith(env_prefix.upper() + "_"):
                raise ConfigurationError("unknown_key", "unknown configuration environment key")
            continue
        if key in text_overrides:
            raise ConfigurationError("environment_collision", f"{key}: supplied by multiple environment variables")
        text_overrides[key] = value
    for key, value in (overrides or {}).items():
        if key not in fields:
            raise ConfigurationError("unknown_key", "unknown configuration key")
        if key in secrets:
            raise ValueError(f"sensitive configuration value {key} must come from the environment")
        text_overrides[key] = value
    for key, value in text_overrides.items():
        _set(result, key, _decode_override(key, fields[key][0], value))
    try:
        return model.model_validate(result, extra="forbid")
    except ValidationError as error:
        raise ConfigurationError("invalid_value", _safe_validation_error(error, fields, safe_errors=safe_error_messages)) from None


def config_snapshot(model: BaseModel, *, sensitive_paths: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return a JSON-compatible configuration snapshot with secrets redacted."""
    result = model.model_dump(mode="json", by_alias=True)
    for key, (_, secret) in _fields(type(model)).items():
        if secret or key in sensitive_paths:
            _set(result, key, "<redacted>")
    return result
