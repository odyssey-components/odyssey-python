"""Read typed task overrides written by the OdyLab Worker."""

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast


def task_overrides() -> dict[str, object]:
    path = os.environ.get("ODY_TASK_CONFIG")
    if not path:
        return {}
    document: object = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("task configuration must be an object")  # noqa: TRY004 - invalid JSON content.

    def leaves(section: Mapping[str, object], prefix: str = "") -> Iterator[tuple[str, object]]:
        for name, value in section.items():
            key = f"{prefix}{name}"
            if isinstance(value, dict):
                yield from leaves(cast(Mapping[str, object], value), f"{key}.")
            else:
                yield key, value

    return dict(leaves(cast(Mapping[str, object], document)))
