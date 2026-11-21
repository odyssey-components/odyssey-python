import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Never, cast
from urllib.error import URLError

import pytest
from odylab_training.monitoring import LocalRunReporter, Metric, MonitorEvent, sync_pending_runs, validate_event


def unavailable(*_args: object) -> Never:
    raise URLError("offline")


def test_event_validates_common_and_game_specific_metrics() -> None:
    event = MonitorEvent.create(
        run_id="run-1",
        sequence=1,
        game="maze",
        phase="ppo",
        step=128,
        metrics=[Metric("steps", "训练步数", 128, "step", "progress", 1000)],
        quantity={"value": 128, "unit": "step"},
        costs={"ppo": 3.5},
        peak_memory={"value": 512, "unit": "MiB", "source": "pytorch_peak_allocated", "scope": "process"},
    )

    validate_event(event)
    assert event["attributes"]["metrics"][0]["key"] == "steps"


@pytest.mark.parametrize("bad", [
    {"sequence": 0},
    {"sequence": 1, "attributes": {"metrics": [{"key": "bad key", "value": 1}]}},
    {"sequence": 1, "attributes": {"metrics": [{"key": "loss", "value": float("nan")}]}},
])
def test_invalid_monitor_events_are_rejected(bad: dict[str, object]) -> None:
    event = {"event": "monitor.metrics", "attributes": {"run_id": "x", **bad}}
    with pytest.raises(ValueError):
        validate_event(event)


def test_reporter_buffers_and_never_propagates_transport_failure(tmp_path: Path) -> None:
    reporter = LocalRunReporter(
        "http://127.0.0.1:8787", tmp_path, game="maze", algorithm="ppo", run_id="run-1", transport=unavailable
    )
    reporter.emit("ppo", 4, [Metric("loss", "Loss", 0.2, display="line")])

    pending = tmp_path / "run-1" / "pending.jsonl"
    assert pending.exists()
    assert json.loads(pending.read_text().splitlines()[0])["attributes"]["sequence"] == 1


def test_reporter_flush_retries_pending_events(tmp_path: Path) -> None:
    received: list[object] = []

    def transport(url: str, payload: dict[str, object], _timeout: float) -> dict[str, object]:
        events = payload.get("events", [])
        assert isinstance(events, list)
        batch = cast(Sequence[object], events)
        received.extend(batch)
        return {"accepted": len(batch), "registered": url.endswith("/local")}

    reporter = LocalRunReporter("http://localhost:8787", tmp_path, game="gomoku", algorithm="selfplay", run_id="g-1", transport=transport)
    reporter.emit("train", 10, [Metric("loss", "Loss", 0.4)])
    assert reporter.flush()
    assert len(received) == 1
    assert not (tmp_path / "g-1" / "pending.jsonl").read_text().strip()


def test_reporter_rejects_non_loopback_endpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LocalRunReporter("https://example.com", tmp_path, game="maze", algorithm="ppo", run_id="x")


def test_offline_spool_can_sync_later_and_finish(tmp_path: Path) -> None:
    reporter = LocalRunReporter("http://127.0.0.1:8787", tmp_path, game="xiangqi", algorithm="policy", run_id="retry-1", transport=unavailable)
    reporter.emit("train", 2, [Metric("loss", "Loss", 0.3)])
    reporter.finish()
    posted: list[tuple[str, dict[str, object]]] = []

    def online(url: str, payload: dict[str, object], _timeout: float) -> dict[str, object]:
        posted.append((url, payload))
        return {"ok": True}

    result = sync_pending_runs("http://127.0.0.1:8787", tmp_path, transport=online)
    assert result == {"synced": 1, "pending": 0, "invalid": 0}
    assert any(url.endswith("/events") for url, _ in posted)
    assert any(url.endswith("/finish") and payload["status"] == "completed" for url, payload in posted)
    assert (tmp_path / "retry-1" / "pending.jsonl").read_text() == ""


def test_unfinished_reporter_queues_interrupted_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exit_callbacks: list[Callable[[], None]] = []
    monkeypatch.setattr("atexit.register", exit_callbacks.append)
    LocalRunReporter(
        "http://127.0.0.1:8787", tmp_path, game="maze", algorithm="ppo", run_id="stopped-1",
        transport=unavailable,
    )
    assert len(exit_callbacks) == 1
    exit_callbacks[0]()
    assert json.loads((tmp_path / "stopped-1" / "finished.json").read_text()) == {"status": "interrupted"}
