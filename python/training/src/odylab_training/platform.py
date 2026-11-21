"""Public config helpers for file-driven, hermetic platform runs.

Legacy standalone packages need not bundle this optional module. Ordinary CLI
configuration retains its existing source precedence in odyssey_config.
"""

from __future__ import annotations

import copy
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, cast

from odyssey_config import load_config
from pydantic import BaseModel

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_CONFIG_BYTES = 256 * 1024


def load_frozen_config[Config: BaseModel](settings: type[Config], path: Path, digest: str) -> Config:
    """Verify the mounted snapshot before loading it, without environment/CLI overrides."""
    if not _DIGEST.fullmatch(digest) or path.is_symlink() or not path.is_file():
        raise ValueError("invalid frozen configuration digest or path")
    with path.open("rb") as source:
        data = source.read(_MAX_CONFIG_BYTES + 1)
    if len(data) > _MAX_CONFIG_BYTES or "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("frozen configuration digest mismatch")
    # Load precisely the bytes that were hashed, even if the original path is
    # replaced between validation and the existing file loader's read.
    with tempfile.TemporaryDirectory(prefix="ody-training-config-") as directory:
        snapshot = Path(directory) / ("frozen" + path.suffix)
        snapshot.write_bytes(data)
        snapshot.chmod(0o400)
        return load_config(settings, path=snapshot, environment={}, overrides={})


def platform_schema(settings: type[BaseModel]) -> dict[str, Any]:
    """Publish the model's schema with the loader's existing unknown-key rejection.

    Explicit dictionary schemas retain their declared additionalProperties;
    ordinary model objects are closed. Secret settings belong in runtime refs.
    """
    schema = copy.deepcopy(settings.model_json_schema())

    def close(value: Any) -> None:
        if isinstance(value, dict):
            node = cast(dict[str, Any], value)
            if node.get("writeOnly") is True:
                raise ValueError("training schema must use runtime references for secrets")
            if node.get("type") == "object" and "properties" in node:
                node.setdefault("additionalProperties", False)
            for child in node.values():
                close(child)
        elif isinstance(value, list):
            for child in cast(list[Any], value):
                close(child)

    close(schema)
    return schema
