"""Shared OdyLab structured event envelope."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def event_record(event: str, message: str, attributes: dict[str, Any], level: str = "INFO") -> dict[str, Any]:
    """Build the common timestamped event record consumed by OdyLab workers."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "event": event,
        "message": message,
        "attributes": attributes,
    }
