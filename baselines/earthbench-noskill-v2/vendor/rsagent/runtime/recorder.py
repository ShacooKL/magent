"""Small JSONL event recorder for agent runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from rsagent.runtime.clock import TIMEZONE_NAME, now_beijing


MAX_STRING_LENGTH = 8_000
TRAJECTORY_SCHEMA_VERSION = 2
SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "x-api-key",
}


class Recorder:
    def __init__(self, log_dir: str | Path) -> None:
        directory = Path(log_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            directory.chmod(0o700)
        now = now_beijing()
        self.session_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        self.path = directory / f"{self.session_id}.jsonl"
        self._lock = Lock()

    def log(self, event_type: str, **data: Any) -> None:
        record = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "time": now_beijing().isoformat(),
            "timezone": TIMEZONE_NAME,
            "session_id": self.session_id,
            "type": event_type,
            **_sanitize(data),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)
            if os.name == "posix":
                self.path.chmod(0o600)


class NullRecorder:
    session_id = "disabled"
    path: Path | None = None

    def log(self, event_type: str, **data: Any) -> None:
        return None


def append_event(path: str | Path, event_type: str, **data: Any) -> None:
    """Append a post-run event to an existing trajectory."""
    target = Path(path)
    record = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "time": now_beijing().isoformat(),
        "timezone": TIMEZONE_NAME,
        "session_id": target.stem,
        "type": event_type,
        **_sanitize(data),
    }
    with target.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    if os.name == "posix":
        target.chmod(0o600)


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key and key.lower() in SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _sanitize(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + "\n...[truncated]"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)
