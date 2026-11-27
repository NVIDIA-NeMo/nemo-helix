# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why an OpenSandbox sandbox stopped, from the server's supported API.

Reads ``GET /sandboxes/{id}`` and, for a sandbox that stopped, the stable runtime events route.
Neither names an OOM kill on server 1.1 with controller 0.2: Kubernetes records it only in the
container's status, and the controller leaves a sandbox whose pod died in its ``Pending`` phase,
which the server reports as is. Those are diagnosed ``unknown``, with what the server said.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Protocol

import httpx
from nhx_sandbox.core.errors import SandboxBackendError, SandboxNotFound
from nhx_sandbox.core.termination import TerminationCause, TerminationDiagnosis

BACKEND = "opensandbox"
# States of a sandbox that is still there to use. Anything else, once it was ready, means it stopped:
# controller 0.2 leaves a sandbox whose pod died in Pending.
_ALIVE_STATES = frozenset({"Running", "Pausing", "Paused", "Resuming"})
# Server's own reason for a sandbox its lifetime ended.
_EXPIRED_REASON = "SANDBOX_EXPIRED"
# A runtime event line: "[<time>] <Type> <Reason> <message>".
_EVENT_LINE = re.compile(r"^\[[^\]]+\]\s+(?P<type>\S+)\s+(?P<reason>\S+)\s*(?P<message>.*)$")
# At most this many warning events are kept, the latest ones.
MAX_EVENTS = 10
# The NeMo extension's message for a stopped container: "container <name>: <reason> (exit <code>)".
_CONTAINER_MESSAGE = re.compile(r"^container (?P<name>\S+): ")
# The agent's own container in the sandbox pod; the rest are the task's compose services.
_SANDBOX_CONTAINER = "sandbox"


class ServerClient(Protocol):
    """Reads from the OpenSandbox server's API, as the environment's own server client does."""

    async def get(self, path: str, *, timeout: float, deadline: float | None = None) -> httpx.Response:
        """``GET`` a path under the API root; the response is returned whatever its status."""
        ...


class OpenSandboxTerminationDiagnostics:
    """``TerminationDiagnostics`` for OpenSandbox, over the server's API.

    Args:
        server: A client for the server's API, with the caller's API key.
    """

    backend = BACKEND

    def __init__(self, server: ServerClient) -> None:
        """Diagnose through ``server``."""
        self._server = server

    async def diagnose_termination(self, sandbox_id: str, *, deadline: float) -> TerminationDiagnosis | None:
        """The diagnosis for a sandbox that was ready once; ``None`` while it is alive.

        The status decides the diagnosis and is read by ``deadline``; the events only add detail,
        so they get whatever time is left and are left out when there is none.
        """
        response = await self._server.get(
            f"/sandboxes/{sandbox_id}", timeout=max(1.0, deadline - time.monotonic()), deadline=deadline
        )

        if response.status_code == 404:
            raise SandboxNotFound(f"OpenSandbox has no sandbox {sandbox_id}")

        if response.status_code != 200:
            raise SandboxBackendError(f"reading OpenSandbox sandbox {sandbox_id} failed ({response.status_code})")

        status = response.json().get("status") or {}
        state = str(status.get("state") or "")
        if state in _ALIVE_STATES:
            return None

        events = await self._warning_events(sandbox_id, deadline)

        return diagnose_from_status(
            state=state, reason=status.get("reason"), message=status.get("message"), events=events
        )

    async def _warning_events(self, sandbox_id: str, deadline: float) -> list[str]:
        """The sandbox pod's warning events by ``deadline``, or none when the server can't give them in time."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return []
        try:
            async with asyncio.timeout(remaining):
                response = await self._server.get(
                    f"/sandboxes/{sandbox_id}/diagnostics/events?scope=runtime", timeout=remaining, deadline=deadline
                )
            if response.status_code != 200:
                return []
            content = response.json().get("content") or ""
        except (TimeoutError, httpx.HTTPError, ValueError, AttributeError):
            return []
        return warning_events(str(content))


def warning_events(content: str) -> list[str]:
    """The ``Warning`` lines of the events route's text, as ``<Reason>: <message>``, latest ``MAX_EVENTS``."""
    warnings = []
    for line in content.splitlines():
        match = _EVENT_LINE.match(line.strip())
        if match and match["type"] == "Warning":
            warnings.append(f"{match['reason']}: {match['message'].strip()}")
    return warnings[-MAX_EVENTS:]


def diagnose_from_status(
    *, state: str, reason: str | None, message: str | None, events: list[str]
) -> TerminationDiagnosis:
    """The diagnosis for a sandbox the server still knows but that is no longer alive."""
    evidence: dict[str, Any] = {"state": state, "reason": reason, "message": message, "events": events}
    reported = f"server reported {state}" + (f" ({reason})" if reason else "") + (f": {message}" if message else "")

    if reason == _EXPIRED_REASON:
        return TerminationDiagnosis(TerminationCause.LIFETIME_REACHED, reported, evidence)

    if state == "Stopping":
        return TerminationDiagnosis(TerminationCause.DELETED, reported, evidence)

    evicted = next((event for event in events if event.startswith("Evicted:")), None)

    if evicted is not None:
        detail = f"Kubernetes evicted the sandbox pod: {evicted.removeprefix('Evicted:').strip()}"
        return TerminationDiagnosis(TerminationCause.EVICTED, detail, evidence)

    container = _CONTAINER_MESSAGE.match(message or "")
    if container is not None and container["name"] != _SANDBOX_CONTAINER:
        # A compose service stopped, not the agent's container: its memory limit is the task's, not the sandbox's.
        service = container["name"]
        killed = "ran out of memory" if "OOMKilled" in (message or "") else "stopped"
        detail = f"the sandbox's {service} service container {killed}; {reported}"
        return TerminationDiagnosis(TerminationCause.UNKNOWN, detail, evidence)

    if any("OOMKilled" in text for text in [message or "", *events]):
        return TerminationDiagnosis(
            TerminationCause.OUT_OF_MEMORY, f"the sandbox ran out of memory; {reported}", evidence
        )

    if state == "Pending":
        # A sandbox that was ready once and reads Pending again died; "being created" would mislead.
        reported = (
            f"server reported {state}" + (f" ({reason})" if reason else "") + ", as it does for a sandbox whose "
            "container stopped; the NeMo OpenSandbox extension (nemo-opensandbox-ext) reports the container's reason"
        )
    detail = f"the sandbox stopped; {reported}" + (f"; events: {'; '.join(events)}" if events else "")

    return TerminationDiagnosis(TerminationCause.UNKNOWN, detail, evidence)
