# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded diagnostics retained with the workspace-scoped deployment entity."""

import asyncio
from pathlib import Path

from nemo_agents_plugin.runner.backend import LocalLog, RunnerBackend

STARTUP_LOG_LINES = 500
STARTUP_LOG_BYTES = 64 * 1024
STARTUP_LOG_TIMEOUT = 10


def _bounded_tail(text: str) -> str:
    return (
        "\n".join(text.splitlines()[-STARTUP_LOG_LINES:])
        .encode("utf-8")[-STARTUP_LOG_BYTES:]
        .decode("utf-8", errors="ignore")
    )


def _read_local_tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - STARTUP_LOG_BYTES))
        return _bounded_tail(stream.read(STARTUP_LOG_BYTES).decode("utf-8", errors="replace"))


async def capture_startup_logs(backend: RunnerBackend, workspace: str, name: str) -> str:
    """Read before teardown; callers handle unavailable channels as best effort."""
    async with asyncio.timeout(STARTUP_LOG_TIMEOUT):
        get_logs = getattr(backend, "get_logs", None)
        if get_logs is not None:
            result = await get_logs(workspace=workspace, name=name, tail=STARTUP_LOG_LINES)
            return _bounded_tail("\n".join(result.lines))
        location = backend.get_log_location(workspace, name)
        if isinstance(location, LocalLog):
            return await asyncio.to_thread(_read_local_tail, location.path)
        return ""
