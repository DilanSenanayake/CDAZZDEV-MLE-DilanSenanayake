"""
Observability: append every tool call to task3_agentic/logs/agent_trace.jsonl.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

LOG_PATH = Path(__file__).resolve().parent / "logs" / "agent_trace.jsonl"


def _truncate(obj: Any, limit: int = 200) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    text = text.replace("\n", " ")
    return text[:limit]


def log_tool_call(
    tool_name: str,
    inputs: dict[str, Any],
    output: Any,
    duration_ms: float,
    path: Path = LOG_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "inputs": inputs,
        "output": _truncate(output, 200),
        "duration_ms": round(duration_ms, 3),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def traced(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        error = None
        result: Any = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            result = {"error": error}
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            # Bind positional args to names best-effort
            inputs = dict(kwargs)
            for i, val in enumerate(args):
                inputs[f"arg{i}"] = val
            log_tool_call(fn.__name__, inputs, result if error is None else {"error": error}, duration_ms)

    return wrapper
