"""Persistent, resumable session snapshots for the coding agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from rsagent.runtime.clock import TIMEZONE_NAME, now_beijing


SESSION_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,127}$")


@dataclass
class SessionHandle:
    store: "SessionStore"
    session_id: str
    created_at: str
    metadata: dict[str, Any]
    resumed: bool = False

    @property
    def path(self) -> Path:
        return self.store.directory / f"{self.session_id}.json"

    def save(self, context: dict[str, Any]) -> None:
        payload = {
            "version": SESSION_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": now_beijing().isoformat(),
            "timezone": TIMEZONE_NAME,
            **self.metadata,
            "context": context,
        }
        self.store._write(self.path, payload)


class SessionStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            self.directory.chmod(0o700)

    def new(self, **metadata: Any) -> SessionHandle:
        now = now_beijing()
        session_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        return SessionHandle(self, session_id, now.isoformat(), metadata)

    def resume(self, identifier: str, workspace: Path) -> tuple[SessionHandle, dict[str, Any]]:
        if identifier == "latest":
            payload = self._latest_payload(workspace.resolve())
        else:
            payload = self._load_id(identifier)
        stored_workspace = Path(str(payload.get("workspace", ""))).resolve()
        if stored_workspace != workspace.resolve():
            raise ValueError(
                f"Session workspace mismatch: saved={stored_workspace}, current={workspace.resolve()}"
            )
        metadata = {
            key: payload[key]
            for key in ("workspace", "provider", "model", "profile", "sandbox", "read_only_paths")
            if key in payload
        }
        handle = SessionHandle(
            self,
            str(payload["session_id"]),
            str(payload["created_at"]),
            metadata,
            resumed=True,
        )
        context = payload.get("context")
        if not isinstance(context, dict):
            raise ValueError(f"Session has no valid context snapshot: {handle.session_id}")
        return handle, context

    def list(self, workspace: Path | None = None, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self._candidate_paths():
            try:
                payload = self._load(path)
            except ValueError:
                continue
            if workspace is not None and Path(str(payload.get("workspace", ""))).resolve() != workspace.resolve():
                continue
            context = payload.get("context", {})
            rows.append({
                "session_id": payload.get("session_id", path.stem),
                "updated_at": payload.get("updated_at", ""),
                "workspace": payload.get("workspace", ""),
                "model": payload.get("model", ""),
                "turns": len(context.get("turns", [])) if isinstance(context, dict) else 0,
                "summaries": len(context.get("summary", [])) if isinstance(context, dict) else 0,
            })
            if len(rows) >= limit:
                break
        return rows

    def _latest_payload(self, workspace: Path) -> dict[str, Any]:
        for path in self._candidate_paths():
            try:
                payload = self._load(path)
            except ValueError:
                continue
            if Path(str(payload.get("workspace", ""))).resolve() == workspace:
                return payload
        raise ValueError(f"No saved session found for workspace: {workspace}")

    def _candidate_paths(self) -> list[Path]:
        return sorted(self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)

    def _load_id(self, session_id: str) -> dict[str, Any]:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(f"Invalid session id: {session_id}")
        return self._load(self.directory / f"{session_id}.json")

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Session not found: {path.stem}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read session {path}: {error}") from error
        if not isinstance(payload, dict) or payload.get("version") != SESSION_VERSION:
            raise ValueError(f"Unsupported or invalid session file: {path}")
        return payload

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            if os.name == "posix":
                path.chmod(0o600)
        finally:
            if temporary.exists():
                temporary.unlink()
