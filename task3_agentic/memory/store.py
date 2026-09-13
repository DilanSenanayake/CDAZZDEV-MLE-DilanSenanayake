"""Persistent and short-term memory helpers for Task 3C."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def cache_key(ticker: str, on: date | None = None) -> str:
    on = on or datetime.now(timezone.utc).date()
    return f"{ticker.upper()}_{on.isoformat()}"


def cache_path(ticker: str, on: date | None = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{cache_key(ticker, on)}.json"


def save_brief(ticker: str, brief: dict[str, Any], on: date | None = None) -> Path:
    path = cache_path(ticker, on)
    payload = {
        "ticker": ticker.upper(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "brief": brief,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_brief(ticker: str, on: date | None = None) -> dict[str, Any] | None:
    path = cache_path(ticker, on)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class ShortTermMemory:
    """In-session store so follow-ups can reuse prior tool results."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def snapshot(self) -> dict[str, Any]:
        return dict(self._store)
