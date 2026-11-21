"""Stable shared training APIs used by OdyLab game trainers."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .archive import write_config_package, write_zip_entry
from .events import event_record
from .monitoring import (
    COMMON_METRICS,
    LocalRunReporter,
    Metric,
    MetricSpec,
    MonitorEvent,
    local_reporter,
    memory_snapshot,
    phase_costs_seconds,
    sync_pending_runs,
    validate_event,
)
from .task_config import task_overrides

if TYPE_CHECKING:
    from . import continuous_ppo as continuous_ppo

__all__ = ["COMMON_METRICS", "LocalRunReporter", "Metric", "MetricSpec", "MonitorEvent", "continuous_ppo", "event_record", "local_reporter", "memory_snapshot", "phase_costs_seconds", "sync_pending_runs", "task_overrides", "validate_event", "write_config_package", "write_zip_entry"]


def __getattr__(name: str) -> ModuleType:
    if name == "continuous_ppo":
        return import_module(".continuous_ppo", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
