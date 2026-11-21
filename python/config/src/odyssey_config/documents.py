"""Read a declared configuration format without exposing parser input."""

import copy
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import yaml

from .errors import ConfigurationError


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject repeated mapping keys, including keys nested in lists."""


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[object, Any]:
    loader.flatten_mapping(node)
    # PyYAML's stub omits the node and result types for construct_object.
    construct_object = cast("Callable[[yaml.Node, bool], object]", loader.construct_object)  # pyright: ignore[reportUnknownMemberType]
    result: dict[object, Any] = {}
    for key_node, value_node in node.value:
        key = construct_object(key_node, deep)
        if key in result:
            raise ValueError("duplicate configuration key")
        result[key] = construct_object(value_node, deep)
    return result


# Match YAML 1.2 boolean resolution instead of PyYAML's YAML 1.1 yes/no rules.
UniqueKeyLoader.yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)
for initial, resolvers in UniqueKeyLoader.yaml_implicit_resolvers.items():
    UniqueKeyLoader.yaml_implicit_resolvers[initial] = [
        (tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
# The module-level registration also mutates the global dumper.
UniqueKeyLoader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.IGNORECASE), list("tTfF"))  # pyright: ignore[reportUnknownMemberType]

UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def find_document(base: Path) -> Path | None:
    paths = [Path(f"{base}{extension}") for extension in (".yaml", ".yml", ".toml")]
    found = [path for path in paths if path.exists()]
    if len(found) > 1:
        raise ConfigurationError("duplicate_candidate", "multiple configuration formats")
    return found[0] if found else None


def parse_yaml_document(raw: bytes | str, *, allow_empty: bool = False) -> dict[str, Any]:
    """Parse an in-memory YAML configuration mapping using the file loader's rules."""
    try:
        result: object = yaml.load(raw, Loader=UniqueKeyLoader)
    except (TypeError, ValueError, yaml.YAMLError):
        raise ConfigurationError("invalid_document", "cannot parse YAML configuration") from None
    if result is None and allow_empty:
        return {}
    if not isinstance(result, dict):
        raise ConfigurationError("invalid_document", "configuration document must contain a mapping")
    return cast("dict[str, Any]", result)


def read_document(path: Path) -> dict[str, Any]:
    extension = path.suffix.lower()
    if extension not in {".yaml", ".yml", ".toml"}:
        raise ValueError("unsupported configuration file extension")
    find_document(path.with_suffix(""))
    format = "toml" if extension == ".toml" else "yaml"
    try:
        raw = path.read_bytes()
        result = tomllib.loads(raw.decode("utf-8")) if format == "toml" else None
    except (OSError, ValueError):
        raise ConfigurationError("invalid_document", f"cannot read or parse {format} configuration file") from None
    if format == "yaml":
        return parse_yaml_document(raw, allow_empty=True)
    if not isinstance(result, dict):
        raise ValueError("configuration document must contain a mapping")  # noqa: TRY004
    return result
