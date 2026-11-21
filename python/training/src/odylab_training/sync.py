"""Retry locally buffered OdyLab training monitor events."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .monitoring import sync_pending_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("ODYLAB_REPORT_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--spool", type=Path, default=Path(os.environ.get("ODYLAB_REPORT_SPOOL", Path.home() / ".odyssey/odylab/runs")))
    args = parser.parse_args()
    result = sync_pending_runs(args.api_url, args.spool)
    print(json.dumps(result, ensure_ascii=False))
    if result["pending"] or result["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
