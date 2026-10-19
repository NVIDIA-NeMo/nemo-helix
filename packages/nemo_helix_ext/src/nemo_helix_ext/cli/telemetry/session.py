# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local telemetry session state with time-bounded identifier rotation."""

from __future__ import annotations

import json
import logging
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SESSION_STATE_FILENAME = "telemetry-state.json"
_SESSION_ROTATION_DAYS = 30
_SESSION_ROTATION_INTERVAL = timedelta(days=_SESSION_ROTATION_DAYS)
_SESSION_ID_KEY = "session_id"
_CREATED_AT_KEY = "created_at"
_STATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
_STATE_DIR_MODE = stat.S_IRWXU


@dataclass(frozen=True)
class _SessionState:
    session_id: str
    created_at: datetime


_cached_state: _SessionState | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _default_state_root() -> Path:
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "nhx"
    return Path.home() / ".local" / "state" / "nhx"


def _session_state_path() -> Path:
    return _default_state_root() / _SESSION_STATE_FILENAME


def _format_created_at(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_created_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_session_state(path: Path) -> _SessionState | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    session_id = data.get(_SESSION_ID_KEY)
    created_at = _parse_created_at(data.get(_CREATED_AT_KEY))
    if not isinstance(session_id, str) or not session_id.strip() or created_at is None:
        return None
    return _SessionState(session_id=session_id, created_at=created_at)


def _requires_rotation(state: _SessionState, now: datetime) -> bool:
    if state.created_at > now:
        return True
    return now - state.created_at >= _SESSION_ROTATION_INTERVAL


def _new_session_state(now: datetime) -> _SessionState:
    return _SessionState(session_id=uuid.uuid4().hex, created_at=now)


def _write_session_state(path: Path, state: _SessionState) -> None:
    created_parent = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if created_parent:
        try:
            os.chmod(path.parent, _STATE_DIR_MODE)
        except OSError:
            pass

    payload = {
        _SESSION_ID_KEY: state.session_id,
        _CREATED_AT_KEY: _format_created_at(state.created_at),
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _STATE_FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            json.dump(payload, f, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
        os.chmod(path, _STATE_FILE_MODE)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _load_or_rotate_session_state(path: Path, now: datetime) -> _SessionState:
    state = _read_session_state(path)
    if state is not None and not _requires_rotation(state, now):
        return state

    state = _new_session_state(now)
    _write_session_state(path, state)
    return state


def get_session_id() -> str:
    """Return the current random telemetry session identifier, rotating every 30 days.

    A missing, corrupt, dateless, future-dated, or expired state file is replaced
    with a new random identifier and fresh creation timestamp. If local state cannot
    be read or written, fall back to a process-local random identifier so telemetry
    remains best-effort and never blocks command execution.
    """
    global _cached_state

    now = _now_utc()
    if _cached_state is not None and not _requires_rotation(_cached_state, now):
        return _cached_state.session_id

    try:
        _cached_state = _load_or_rotate_session_state(_session_state_path(), now)
        return _cached_state.session_id
    except Exception:
        logger.debug("Failed to resolve telemetry session state; using an ephemeral identifier", exc_info=True)
        _cached_state = _new_session_state(now)
        return _cached_state.session_id
