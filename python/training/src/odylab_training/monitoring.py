"""Game-neutral monitor events and a failure-isolated local OdyLab reporter."""

from __future__ import annotations

import atexit
import ipaddress
import json
import math
import os
import re
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict, cast
from urllib.parse import urlparse
from uuid import uuid4

_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_GAMES = {"go1", "miniduck", "maze", "gomoku", "xiangqi"}
_MAX_METRICS = 50
_MAX_EVENT_BYTES = 64 * 1024
_MAX_BATCH_BYTES = 900 * 1024


class MetricRecord(TypedDict):
    key: str
    label: str
    value: int | float
    display: str
    unit: NotRequired[str]
    max: NotRequired[int | float]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    value: int | float
    unit: str = ""
    display: str = "value"
    maximum: int | float | None = None

    def record(self) -> MetricRecord:
        item: MetricRecord = {"key": self.key, "label": self.label, "value": self.value, "display": self.display}
        if self.unit:
            item["unit"] = self.unit
        if self.maximum is not None:
            item["max"] = self.maximum
        return item


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    display: str = "line"


COMMON_METRICS = {
    spec.key: spec for spec in (
        MetricSpec("common.elapsed_seconds", "运行耗时", "秒"),
        MetricSpec("common.cpu_seconds", "CPU 时间", "秒"),
        MetricSpec("common.peak_memory_mib", "峰值内存", "MiB"),
        MetricSpec("common.gpu_utilization_percent", "GPU 利用率", "%"),
        MetricSpec("common.throughput_per_second", "处理速率", "/秒"),
    )
}


class MonitorAttributes(TypedDict):
    run_id: str
    sequence: int
    game: str
    phase: str
    step: int
    metrics: list[MetricRecord]
    quantity: NotRequired[dict[str, object]]
    phase_cost_seconds: NotRequired[dict[str, float]]
    device: NotRequired[str]
    peak_memory: NotRequired[dict[str, object]]


class MonitorRecord(TypedDict):
    event_id: str
    timestamp: str
    level: str
    event: str
    message: str
    attributes: MonitorAttributes


class MonitorEvent:
    """Build events compatible with the existing monitor.metrics log parser."""

    @staticmethod
    def create(
        *, run_id: str, sequence: int, game: str, phase: str, step: int,
        metrics: list[Metric], quantity: dict[str, object] | None = None,
        costs: dict[str, float] | None = None, device: str = "",
        peak_memory: dict[str, object] | None = None, message: str = "训练指标",
    ) -> MonitorRecord:
        attributes: MonitorAttributes = {
            "run_id": run_id, "sequence": sequence, "game": game, "phase": phase,
            "step": step, "metrics": [metric.record() for metric in metrics],
        }
        event: MonitorRecord = {
            "event_id": f"{run_id}:{sequence}",
            "timestamp": datetime.now(UTC).isoformat(), "level": "INFO", "event": "monitor.metrics", "message": message,
            "attributes": attributes,
        }
        if quantity is not None:
            attributes["quantity"] = quantity
        if costs is not None:
            attributes["phase_cost_seconds"] = costs
        if device:
            attributes["device"] = device
        if peak_memory is not None:
            attributes["peak_memory"] = peak_memory
        validate_event(event)
        return event


def _fields(value: object) -> Mapping[object, object] | None:
    if isinstance(value, dict):
        return cast(Mapping[object, object], value)
    return None


def validate_event(event: Mapping[str, object]) -> None:
    attributes = _fields(event.get("attributes"))
    if event.get("event") != "monitor.metrics" or attributes is None:
        raise ValueError("event must be a monitor.metrics record")
    run_id = attributes.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("attributes.run_id is required")
    sequence = attributes.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise ValueError("attributes.sequence must be a positive integer")
    if attributes.get("game") not in _GAMES:
        raise ValueError("attributes.game must name a supported OdyLab game")
    step = attributes.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("attributes.step must be a non-negative integer")
    metrics_value = attributes.get("metrics")
    metrics = cast(Sequence[object], metrics_value) if isinstance(metrics_value, list) else None
    if metrics is None or len(metrics) > _MAX_METRICS:
        raise ValueError(f"attributes.metrics must contain at most {_MAX_METRICS} items")
    for item in metrics:
        metric = _fields(item)
        if metric is None or not isinstance(key := metric.get("key"), str) or not _KEY.fullmatch(key):
            raise ValueError("metric key must match the stable identifier format")
        label = metric.get("label")
        if not isinstance(label, str) or not label or len(label) > 80:
            raise ValueError("metric label must contain 1 to 80 characters")
        if key.startswith("common.") and key not in COMMON_METRICS:
            raise ValueError("unknown cross-game common metric")
        if len(str(metric.get("unit", ""))) > 32:
            raise ValueError("metric unit exceeds 32 characters")
        if metric.get("display", "value") not in {"value", "line", "progress"}:
            raise ValueError("metric display must be value, line, or progress")
        value = metric.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("metric value must be a finite number")
        if metric.get("display") == "progress":
            maximum = metric.get("max")
            if not isinstance(maximum, (int, float)) or isinstance(maximum, bool) or not math.isfinite(maximum) or maximum <= 0:
                raise ValueError("progress metric requires a positive finite max")
    quantity_value = attributes.get("quantity")
    if quantity_value is not None:
        quantity = _fields(quantity_value)
        if quantity is None or not isinstance(value := quantity.get("value"), (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("quantity value must be a non-negative finite number")
        unit = quantity.get("unit")
        if not isinstance(unit, str) or not unit or len(unit) > 32:
            raise ValueError("quantity unit is required and limited to 32 characters")
    costs = _fields(attributes.get("phase_cost_seconds", {}))
    if costs is None or len(costs) > 50 or any(not isinstance(key, str) or not _KEY.fullmatch(key) or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for key, value in costs.items()):
        raise ValueError("phase costs must be finite non-negative seconds keyed by stable identifiers")
    memory_value = attributes.get("peak_memory")
    if memory_value is not None:
        memory = _fields(memory_value)
        if (
            memory is None
            or not isinstance(value := memory.get("value"), (int, float))
            or isinstance(value, bool) or not math.isfinite(value) or value < 0
            or not isinstance(source := memory.get("source"), str) or not source
            or memory.get("scope") not in {"process", "device", "container", "node"}
        ):
            raise ValueError("peak memory requires a finite value, source, and supported scope")
    encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded) > _MAX_EVENT_BYTES:
        raise ValueError(f"event exceeds {_MAX_EVENT_BYTES} bytes")


def phase_costs_seconds(started: float, phases: dict[str, float]) -> dict[str, float]:
    """Convert monotonic phase start times to elapsed costs."""
    now = time.monotonic()
    return {name: round(max(0.0, now - start), 6) for name, start in phases.items() if start >= started}


def memory_snapshot(value: float, unit: str, source: str, scope: str) -> dict[str, object]:
    if value < 0 or not math.isfinite(value):
        raise ValueError("memory value must be finite and non-negative")
    if not source or scope not in {"process", "device", "container", "node"}:
        raise ValueError("memory source and scope are required")
    return {"value": value, "unit": unit, "source": source, "scope": scope}


Transport = Callable[[str, dict[str, object], float], dict[str, object]]


def local_reporter(game: str, algorithm: str, *, device: str = "", name: str = "") -> LocalRunReporter | None:
    """Create the reporter only when a host-local training invocation opts in."""
    base_url = os.environ.get("ODYLAB_REPORT_URL", "").strip()
    if not base_url:
        return None
    root = Path(os.environ.get("ODYLAB_REPORT_SPOOL", Path.home() / ".odyssey/odylab/runs"))
    return LocalRunReporter(base_url, root, game=game, algorithm=algorithm, device=device, name=name)


def _http_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class LocalRunReporter:
    """Persistently buffer events and best-effort upload them to loopback OdyLab."""

    def __init__(self, base_url: str, spool_root: str | Path, *, game: str, algorithm: str, run_id: str | None = None,
                 device: str = "", name: str = "", transport: Transport = _http_post, timeout: float = 2.0) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not self._is_loopback(parsed.hostname):
            raise ValueError("local training reporting only supports a loopback OdyLab API URL")
        if game not in _GAMES:
            raise ValueError(f"unsupported game: {game}")
        self.base_url = base_url.rstrip("/")
        self.run_id = run_id or str(uuid4())
        self.game, self.algorithm, self.device, self.name = game, algorithm, device, name
        self.directory = Path(spool_root) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.directory.chmod(0o700)
        except OSError:
            pass
        self.pending = self.directory / "pending.jsonl"
        self.finished = self.directory / "finished.json"
        self.manifest = self.directory / "run.json"
        try:
            metadata = json.loads(self.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {"run_id":self.run_id,"game":game,"algorithm":algorithm,"device":device,"name":name,"started_at":datetime.now(UTC).isoformat()}
            self.manifest.write_text(json.dumps(metadata,ensure_ascii=False),encoding="utf-8")
        self.metadata = metadata
        self.sequence = self._last_sequence() + 1
        self.transport, self.timeout = transport, timeout
        self._registered = False
        self._finished = False
        atexit.register(self._mark_interrupted)

    @staticmethod
    def _is_loopback(host: str) -> bool:
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _last_sequence(self) -> int:
        try:
            lines = self.pending.read_text(encoding="utf-8").splitlines()
            return max((int(json.loads(line)["attributes"]["sequence"]) for line in lines), default=0)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return 0

    def emit(self, phase: str, step: int, metrics: list[Metric], *, quantity: dict[str, object] | None = None,
             costs: dict[str, float] | None = None, peak_memory: dict[str, object] | None = None) -> None:
        try:
            event = MonitorEvent.create(run_id=self.run_id, sequence=self.sequence, game=self.game, phase=phase,
                                        step=step, metrics=metrics, quantity=quantity, costs=costs, device=self.device,
                                        peak_memory=peak_memory)
            with self.pending.open("a", encoding="utf-8") as spool:
                spool.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                spool.flush()
                os.fsync(spool.fileno())
            self.sequence += 1
            self.flush()
        except Exception:  # noqa: BLE001 - telemetry must never fail training.
            # Monitoring must never alter a training job's success/failure.
            return

    def finish(self, status: str = "completed") -> None:
        if status not in {"completed", "failed", "interrupted"}:
            return
        try:
            self.finished.write_text(json.dumps({"status": status}), encoding="utf-8")
            self._finished = True
            self.flush()
        except Exception:  # noqa: BLE001 - retain terminal state for a later retry.
            return

    def _mark_interrupted(self) -> None:
        if not self._finished:
            self.finish("interrupted")

    def flush(self) -> bool:
        try:
            if not self._registered:
                self.transport(self.base_url + "/api/v1/training-runs/local", {
                    "run_id": self.run_id, "game": self.game, "algorithm": self.algorithm,
                    "device": self.device, "name": self.name, "started_at": self.metadata["started_at"],
                }, self.timeout)
                self._registered = True
            if self.pending.exists():
                lines = self.pending.read_text(encoding="utf-8").splitlines()
                if lines:
                    batch: list[object] = []
                    batch_bytes = 0
                    for line in lines:
                        encoded = line.encode("utf-8") + b"\n"
                        if len(batch) >= 100 or (batch and batch_bytes + len(encoded) > _MAX_BATCH_BYTES):
                            break
                        batch.append(json.loads(line))
                        batch_bytes += len(encoded)
                    self.transport(self.base_url + f"/api/v1/training-runs/local/{self.run_id}/events", {"events": batch}, self.timeout)
                    remainder = lines[len(batch):]
                    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.directory, delete=False) as spool:
                        spool.write("".join(line + "\n" for line in remainder))
                        replacement = Path(spool.name)
                    replacement.replace(self.pending)
                    if remainder:
                        return False
            if self.finished.exists():
                self.transport(self.base_url + f"/api/v1/training-runs/local/{self.run_id}/finish",
                               json.loads(self.finished.read_text(encoding="utf-8")), self.timeout)
                self.finished.unlink(missing_ok=True)
            return True
        except Exception:  # noqa: BLE001 - uploads are best effort and must keep the spool.
            return False


def sync_pending_runs(base_url: str, spool_root: str | Path, *, transport: Transport = _http_post) -> dict[str, int]:
    """Retry every locally buffered run after the OdyLab API becomes available."""
    root = Path(spool_root)
    result = {"synced": 0, "pending": 0, "invalid": 0}
    if not root.exists():
        return result
    for manifest in root.glob("*/run.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            reporter = LocalRunReporter(base_url, root, game=data["game"], algorithm=data["algorithm"],
                                         run_id=data["run_id"], device=data.get("device", ""), name=data.get("name", ""), transport=transport)
            if reporter.flush():
                result["synced"] += 1
            else:
                result["pending"] += 1
        except Exception:  # noqa: BLE001 - malformed local queue entries are isolated.
            result["invalid"] += 1
    return result
