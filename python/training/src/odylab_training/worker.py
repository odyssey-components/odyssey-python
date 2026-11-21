"""Shared entry contract for training commands launched by the OdyLab Worker."""

import json
import os
import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from .events import event_record
from .task_config import task_overrides


class ConfigLoader[Config](Protocol):
    def __call__(
        self,
        path: Path | None,
        /,
        *,
        profile: str | None,
        default_path: Path,
        overrides: Mapping[str, object],
    ) -> Config: ...


@dataclass(frozen=True)
class WorkerRun:
    scene_dir: Path
    output_dir: Path

    @classmethod
    def from_environment(cls) -> Self:
        scene = os.environ.get("ODY_SCENE_DIR")
        output = os.environ.get("ODY_OUTPUT_DIR")
        if not scene or not output:
            raise ValueError("ODY_SCENE_DIR and ODY_OUTPUT_DIR are required")
        return cls(Path(scene), Path(output))

    def load_config[Config](
        self,
        load: ConfigLoader[Config],
        path: Path | None = None,
        *,
        profile: str | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> Config:
        return load(
            path,
            profile=profile,
            default_path=self.scene_dir / "training.yaml",
            overrides={**task_overrides(), **(overrides or {})},
        )


@contextmanager
def training_stage(
    stage: Literal["preparation", "execution", "publication"], *, stderr: bool = False,
) -> Generator[None]:
    """Emit command-level progress while leaving Attempt state to the Runner."""
    stream = sys.stderr if stderr else sys.stdout

    def emit(status: str, level: str = "INFO") -> None:
        record = event_record("training.stage", "训练阶段", {"stage": stage, "status": status}, level=level)
        print(json.dumps(record), file=stream, flush=True)

    emit("started")
    try:
        yield
    except Exception:
        emit("failed", "ERROR")
        raise
    else:
        emit("completed")
