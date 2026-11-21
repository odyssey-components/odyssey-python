"""Resolve configuration files in ascending merge order."""

import re
from collections.abc import Iterator, Mapping
from pathlib import Path

from .documents import find_document
from .errors import ConfigurationError


def config_paths(
    *,
    path: str | Path | None,
    default_path: str | Path | None,
    profile: str | None,
    default_profile: str | None,
    env: Mapping[str, str],
    prefix: str,
    base_required: bool,
    directory_name: str | None,
    config_dir: str | None,
) -> Iterator[Path]:
    profile = profile if profile is not None else env.get(f"{prefix}_PROFILE", default_profile)
    if profile is not None and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", profile):
        raise ConfigurationError("invalid_profile", "invalid configuration profile")
    env_path = env.get(f"{prefix}_CONFIG_FILE") or None
    if directory_name and path is None and (config_dir is not None or env_path is None):
        directories = (config_dir if config_dir is not None else env.get("CONFIG_DIR", "").strip() or ".").split(",")
        if any(not directory.strip() for directory in directories):
            raise ConfigurationError("invalid_directory")
        for suffix in ("", f"-{profile}") if profile is not None else ("",):
            found = False
            for directory in reversed(directories):
                root = Path(directory.strip())
                for parent in (root, root / "config"):
                    if candidate := find_document(parent / (directory_name + suffix)):
                        found = True
                        yield candidate
            if not found and (suffix or base_required or config_dir is not None):
                raise ConfigurationError("profile_missing" if suffix else "base_missing")
        return
    selected = path if path is not None else env_path or default_path
    if selected is None:
        if profile is not None:
            raise ConfigurationError("profile_missing", "configuration profile requires a base path")
        return
    base = Path(selected)
    if base.suffix.lower() not in {".yaml", ".yml", ".toml"}:
        raise ConfigurationError("format_mismatch", "unsupported configuration file extension")
    discovered = find_document(base.with_suffix(""))
    if path is None and env_path is None and discovered is not None:
        base = discovered
    if base.exists() or base_required or (directory_name and (path is not None or env_path is not None)):
        yield base
    if profile is not None:
        overlay = find_document(base.with_name(f"{base.stem}-{profile}"))
        if overlay is None:
            raise ConfigurationError("profile_missing", "profile_missing (configuration profile file not found)")
        yield overlay
