# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Read Ray session logs after a training failure and name the usual startup causes.

Ray writes the runtime_env_agent and dashboard_agent logs under ``/tmp/ray/session_latest/logs``,
not to the driver. When one of those agents dies, the raylet shuts down with it and prints a hint
that lists every possible cause. The cause itself is in the agent log. These helpers print that log
and return one line the training error rules can classify.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from nhx.customization_common.training.errors.parser import find_exceptions

# Printed, in this order, when a job fails. Worker logs are omitted: a session holds one file per task.
RAY_FAILURE_LOG_NAMES = (
    "runtime_env_agent.log",
    "runtime_env_agent.err",
    "runtime_env_agent.out",
    "dashboard_agent.log",
    "dashboard_agent.err",
    "dashboard_agent.out",
    "raylet.err",
    "raylet.out",
    "gcs_server.err",
)

# Lines the training error rules match. The fate-share notice is a symptom: a real cause printed
# earlier, such as a GPU running out of memory, still wins. The others are causes.
RAY_AGENT_PORT_CONFLICT = "Ray agent failed to start because its port is already in use"
RAY_AGENT_GRPCIO_MISMATCH = "Ray agent crashed during startup because grpcio does not match Ray's requirement"
RAY_AGENT_KILLED = "Ray agent was killed by the operating system during startup"
RAY_AGENT_STARTUP_ERROR = "Ray agent failed during startup:"
RAY_AGENT_FATE_SHARE = "failed and raylet fate-shares with it"

_TAIL_LINES = 200
_TAIL_BYTES = 256 * 1024

# Ray prints this whenever an agent dies, naming grpcio, a port conflict, and an OOM in one paragraph.
# It is not evidence of any one of them.
_CANNED_HINT_RE = re.compile(
    r"The raylet fate shares with the agent\. This can happen because.*?out of memory\)\.",
    re.DOTALL,
)
_PORT_RE = re.compile(r"Address already in use|EADDRINUSE|Errno 98|Failed to bind", re.IGNORECASE)
_SEGV_RE = re.compile(r"Segmentation fault|SIGSEGV|core dumped|\bsignal 11\b|exit code 139", re.IGNORECASE)
_GRPCIO_VERSION_RE = re.compile(
    r"grpcio.{0,80}(requirement|incompatible|not supported|version mismatch)"
    r"|(requirement|incompatible|not supported).{0,40}grpcio",
    re.IGNORECASE,
)
_OOM_RE = re.compile(
    r"OOM killer|\bSIGKILL\b|\bsignal 9\b|exit code 137|(?:^|\n)Killed(?:\n|$)",
    re.IGNORECASE,
)
_FATE_SHARE_AGENTS = ("runtime_env_agent", "dashboard_agent")

_AGENT_LOG_NAMES = frozenset(
    name for name in RAY_FAILURE_LOG_NAMES if name.startswith(("runtime_env_agent", "dashboard_agent"))
)


def ray_session_log_dir() -> Path | None:
    """Return the current session's log directory, or None when Ray has not created one.

    ``RAY_TMPDIR`` is the session root Ray was given. When it is set, a missing session there is
    not replaced with ``/tmp/ray``, which can belong to a different process.
    """
    root = Path(os.environ["RAY_TMPDIR"]) if os.environ.get("RAY_TMPDIR") else Path("/tmp/ray")
    logs = root / "session_latest" / "logs"
    if logs.is_dir():
        return logs.resolve()
    return None


def read_ray_failure_logs(log_dir: Path) -> dict[str, list[str]]:
    """Return the tail of each startup log that exists, including empty files."""
    found: dict[str, list[str]] = {}
    for name in RAY_FAILURE_LOG_NAMES:
        path = log_dir / name
        if path.is_file():
            found[name] = _tail(path)
    return found


def render_ray_failure_logs(log_dir: Path, logs: dict[str, list[str]]) -> str:
    """Format the logs for the job output."""
    if not logs:
        return f"[ray_bootstrap] Ray session log directory {log_dir} has no agent or raylet logs."
    parts = [f"[ray_bootstrap] Ray session logs from {log_dir} follow."]
    for name in RAY_FAILURE_LOG_NAMES:
        if name not in logs:
            continue
        parts.append(f"===== {log_dir / name} =====")
        parts.extend(logs[name] or ["(empty)"])
    return "\n".join(parts)


def diagnose_ray_startup(logs: dict[str, list[str]], *, startup_failure: bool) -> str | None:
    """Return one ``RuntimeError`` line for the rules, or None when these logs do not explain the failure.

    During training, a diagnosis is reported only when the raylet actually fate-shared with an agent.
    A stale "address already in use" from an earlier attempt in the same session must not replace the
    training error. A failure before the driver starts is a startup failure, so the agent log is the cause
    even when the raylet never got as far as the fate-share line.
    """
    agent_text = _text(logs, _AGENT_LOG_NAMES)
    raylet_text = _text(logs, set(RAY_FAILURE_LOG_NAMES) - _AGENT_LOG_NAMES)
    evidence = agent_text + "\n" + _CANNED_HINT_RE.sub("", raylet_text)
    fate_agent = _fate_share_agent(raylet_text)
    if not startup_failure and fate_agent is None:
        return None

    if _PORT_RE.search(evidence):
        return f"RuntimeError: {RAY_AGENT_PORT_CONFLICT}"
    if _SEGV_RE.search(evidence) or _GRPCIO_VERSION_RE.search(agent_text):
        return f"RuntimeError: {RAY_AGENT_GRPCIO_MISMATCH}"
    if _OOM_RE.search(evidence):
        return f"RuntimeError: {RAY_AGENT_KILLED}"

    excerpt = _last_exception(logs)
    if startup_failure and (excerpt or fate_agent):
        detail = excerpt or f"{fate_agent} {RAY_AGENT_FATE_SHARE}"
        return f"RuntimeError: {RAY_AGENT_STARTUP_ERROR} {detail}"
    if fate_agent in _FATE_SHARE_AGENTS:
        return f"RuntimeError: {fate_agent} {RAY_AGENT_FATE_SHARE}"
    if fate_agent is not None:
        return "RuntimeError: Ray agent failed and the raylet shut down with it"
    return None


def _text(logs: dict[str, list[str]], names: frozenset[str] | set[str]) -> str:
    return "\n".join("\n".join(lines) for name, lines in logs.items() if name in names)


def _fate_share_agent(raylet_text: str) -> str | None:
    for agent in _FATE_SHARE_AGENTS:
        if f"{agent} {RAY_AGENT_FATE_SHARE}" in raylet_text:
            return agent
    if "The raylet fate shares with the agent" in raylet_text:
        return "agent"
    return None


def _last_exception(logs: dict[str, list[str]]) -> str | None:
    lines = [line for name in RAY_FAILURE_LOG_NAMES if name in _AGENT_LOG_NAMES for line in logs.get(name, ())]
    found = find_exceptions(lines)
    if not found:
        return None
    last = found[-1]
    text = " ".join(f"{last.exception_type}: {last.message}".split())
    if len(text) > 240:
        return text[:237] + "..."
    return text


def _tail(path: Path) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - _TAIL_BYTES))
        data = handle.read().decode("utf-8", errors="replace")
    return data.splitlines()[-_TAIL_LINES:]
