# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve evaluation log lines from dispatch work dirs and runner containers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scaled_evals.api import artifacts
from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.dispatch.registry import get_backend_capabilities
from scaled_evals.dispatch.runtime_backend import RuntimeBackendCapabilities


def tail_log_lines(lines: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    return lines[-n:]


def _finalize_lines(lines: list[str], tail_lines: int | None) -> list[str]:
    redacted = [redact_secret_text(line) for line in lines]
    return tail_log_lines(redacted, tail_lines) if tail_lines is not None else redacted


def backend_handle_map(value: Any) -> Mapping[str, Any]:
    import json

    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"external_id": value}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def dispatch_work_dir(evaluation_id: str, runtime: str | None) -> Path | None:
    capabilities = _capabilities(runtime)
    if capabilities is None:
        return None
    return capabilities.dispatch_work_dir(evaluation_id)


def _capabilities(runtime: str | None) -> RuntimeBackendCapabilities | None:
    if not runtime:
        return None
    try:
        return get_backend_capabilities(str(runtime))
    except ValueError:
        return None


def dispatch_log_path(evaluation_id: str, runtime: str | None) -> Path | None:
    """Primary dispatch metadata log (gym.log / harbor.log)."""
    capabilities = _capabilities(runtime)
    if capabilities is None:
        return None
    return capabilities.dispatch_log_path(evaluation_id)


def resolve_log_path(row: Mapping[str, Any]) -> Path | None:
    handle = backend_handle_map(row.get("backend_handle"))
    raw_handle = handle.get("raw") if isinstance(handle.get("raw"), Mapping) else handle
    log_path = raw_handle.get("log") if isinstance(raw_handle, Mapping) else None
    if log_path:
        return Path(str(log_path)).expanduser()
    execution_id = str(handle.get("external_id") or row["id"])
    return dispatch_log_path(execution_id, row.get("runtime"))


def _log_file_candidates(row: Mapping[str, Any]) -> list[Path]:
    handle = backend_handle_map(row.get("backend_handle"))
    evaluation_id = str(handle.get("external_id") or row["id"])
    runtime = row.get("runtime")
    paths: list[Path] = []

    primary = resolve_log_path(row)
    if primary is not None:
        paths.append(primary)

    capabilities = _capabilities(str(runtime) if runtime else None)
    if capabilities is not None:
        for path in capabilities.log_file_candidates(evaluation_id):
            if path not in paths:
                paths.append(path)

    return paths


def docker_runner_log_lines(evaluation_id: str, runtime: str | None) -> list[str]:
    """Stdout/stderr from the one-shot gym-runner or harbor-runner container."""
    capabilities = _capabilities(runtime)
    if capabilities is None:
        return []
    container_name = capabilities.runner_container_name(evaluation_id)
    if container_name is None:
        return []
    try:
        from docker.errors import NotFound

        import docker

        text = docker.from_env().containers.get(container_name).logs(tail=20_000).decode("utf-8", errors="replace")
    except NotFound:
        return []
    except Exception:  # noqa: BLE001 — docker socket unavailable in unit tests
        return []
    return text.splitlines()


def read_log_files(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    seen: set[Path] = set()
    chunks: list[tuple[str, list[str]]] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        file_lines = resolved.read_text(errors="replace").splitlines()
        if file_lines:
            chunks.append((resolved.name, file_lines))
    if len(chunks) > 1:
        for name, file_lines in chunks:
            lines.append(f"--- {name} ---")
            lines.extend(file_lines)
    elif chunks:
        lines.extend(chunks[0][1])
    return lines


def remote_live_log_lines(row: Mapping[str, Any]) -> list[str]:
    """Read the runner snapshot when the API has a separate filesystem."""
    try:
        text = artifacts.read_text_object_if_exists(
            artifacts.evaluation_live_log_key(
                str(row["id"]),
                int(row.get("current_execution") or 1),
            )
        )
    except Exception:  # noqa: BLE001 — live logs remain best-effort observability
        return []
    return text.splitlines() if text else []


def collect_log_lines(row: Mapping[str, Any], *, tail_lines: int | None = 100) -> list[str]:
    """Merge dispatch metadata, run logs on disk, and live runner container output."""
    result = row.get("result")
    if isinstance(result, Mapping):
        raw_logs = result.get("logs") or result.get("log")
        if isinstance(raw_logs, str):
            lines = raw_logs.splitlines()
        elif isinstance(raw_logs, list):
            lines = [str(line) for line in raw_logs]
        else:
            lines = []
        if lines:
            return _finalize_lines(lines, tail_lines)

    lines = read_log_files(_log_file_candidates(row))
    if not lines:
        lines = remote_live_log_lines(row)
    handle = backend_handle_map(row.get("backend_handle"))
    execution_id = str(handle.get("external_id") or row["id"])
    docker_lines = docker_runner_log_lines(execution_id, row.get("runtime"))
    if docker_lines:
        if lines:
            lines.append("--- runner ---")
        lines.extend(docker_lines)

    if not lines and row.get("status_detail"):
        lines = [str(row["status_detail"])]

    return _finalize_lines(lines, tail_lines)
