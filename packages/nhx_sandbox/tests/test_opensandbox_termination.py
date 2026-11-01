# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnosing why an OpenSandbox sandbox stopped, from what server 1.1 and 1.2 report.

The event text and states are what server 1.1.1 with controller 0.2.0 reported for a sandbox
OOM-killed and one evicted for exceeding its ephemeral-storage limit.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from nhx_sandbox.backends.opensandbox.termination import (
    MAX_EVENTS,
    OpenSandboxTerminationDiagnostics,
    diagnose_state,
    warning_events,
)
from nhx_sandbox.core.errors import SandboxBackendError, SandboxNotFound
from nhx_sandbox.core.termination import TerminationCause

# The runtime events route's content for an evicted sandbox, as server 1.1.1 returned it.
EVICTED_EVENTS = """\
[2026-10-30 20:03:53.641627+00:00] Normal   Scheduled            Successfully assigned jgulabrai-dev/45c30f94-0 to 10.54.20.134
[2026-10-30 20:03:55+00:00] Normal   Started              Started container sandbox
[2026-10-30 20:05:07+00:00] Warning  Evicted              Pod ephemeral local storage usage exceeds the total limit of containers 200Mi.
[2026-10-30 20:05:07+00:00] Normal   Killing              Stopping container sandbox
"""
EVICTED = "Evicted: Pod ephemeral local storage usage exceeds the total limit of containers 200Mi."


class Server:
    """The server's API as the diagnostics read it: canned answers by path, and every path asked for."""

    def __init__(self, answers: dict[str, httpx.Response]) -> None:
        self._answers = answers
        self.paths: list[str] = []

    async def get(self, path: str, *, timeout: float, deadline: float | None = None) -> httpx.Response:
        self.paths.append(path)
        return self._answers.get(path, httpx.Response(404))


def _status(state: str, reason: str | None = None, message: str | None = None) -> httpx.Response:
    return httpx.Response(200, json={"id": "sbx-1", "status": {"state": state, "reason": reason, "message": message}})


def _events(content: str) -> httpx.Response:
    return httpx.Response(200, json={"kind": "events", "scope": "runtime", "delivery": "inline", "content": content})


async def _diagnose(answers: dict[str, httpx.Response]) -> tuple[Any, Server]:
    server = Server(answers)
    return await OpenSandboxTerminationDiagnostics(server, timeout_sec=5).diagnose_termination("sbx-1"), server


@pytest.mark.parametrize("state", ["Running", "Paused", "Pausing", "Resuming"])
async def test_a_running_or_paused_sandbox_is_alive_and_its_events_are_not_read(state: str) -> None:
    diagnosis, server = await _diagnose({"/sandboxes/sbx-1": _status(state)})

    assert diagnosis is None
    assert server.paths == ["/sandboxes/sbx-1"]


async def test_an_eviction_reported_as_pending_is_named_from_its_event() -> None:
    """Server 1.1 with controller 0.2 reports an evicted sandbox as Pending/CREATING; only its event says why."""
    diagnosis, _ = await _diagnose(
        {
            "/sandboxes/sbx-1": _status("Pending", "CREATING", "Sandbox is being created"),
            "/sandboxes/sbx-1/diagnostics/events?scope=runtime": _events(EVICTED_EVENTS),
        }
    )

    assert diagnosis.cause is TerminationCause.EVICTED
    assert diagnosis.detail == f"Kubernetes evicted the sandbox pod: {EVICTED.removeprefix('Evicted: ')}"
    assert diagnosis.evidence == {
        "state": "Pending",
        "reason": "CREATING",
        "message": "Sandbox is being created",
        "events": [EVICTED],
    }


async def test_an_oom_kill_the_server_does_not_name_is_unknown_with_what_it_said() -> None:
    """An OOM kill leaves no event and the state is Pending, so the cause can't be told; the detail says so."""
    diagnosis, _ = await _diagnose(
        {
            "/sandboxes/sbx-1": _status("Pending", "CREATING", "Sandbox is being created"),
            "/sandboxes/sbx-1/diagnostics/events?scope=runtime": _events(""),
        }
    )

    assert diagnosis.cause is TerminationCause.UNKNOWN
    assert diagnosis.detail == "the sandbox stopped; server reported Pending (CREATING): Sandbox is being created"


async def test_unreadable_events_still_diagnose_from_the_state() -> None:
    diagnosis, _ = await _diagnose({"/sandboxes/sbx-1": _status("Failed", "FAILED", "pod failed")})

    assert diagnosis.cause is TerminationCause.UNKNOWN
    assert diagnosis.evidence["events"] == []


async def test_a_sandbox_the_server_does_not_know_is_not_found() -> None:
    with pytest.raises(SandboxNotFound):
        await _diagnose({"/sandboxes/sbx-1": httpx.Response(404, json={"code": "KUBERNETES::SANDBOX_NOT_FOUND"})})


async def test_any_other_status_read_failure_says_nothing() -> None:
    with pytest.raises(SandboxBackendError) as raised:
        await _diagnose({"/sandboxes/sbx-1": httpx.Response(503)})

    assert not isinstance(raised.value, SandboxNotFound)


def test_an_oom_kill_in_the_servers_message_is_named() -> None:
    diagnosis = diagnose_state(state="Failed", reason="FAILED", message="container sandbox: OOMKilled", events=[])

    assert diagnosis.cause is TerminationCause.OUT_OF_MEMORY


def test_server_1_2s_expiry_reason_is_the_lifetime() -> None:
    diagnosis = diagnose_state(state="Terminated", reason="SANDBOX_EXPIRED", message="Sandbox expired", events=[])

    assert diagnosis.cause is TerminationCause.LIFETIME_REACHED


def test_a_sandbox_being_deleted_was_deleted() -> None:
    diagnosis = diagnose_state(state="Stopping", reason="DELETING", message="Sandbox is being deleted", events=[])

    assert diagnosis.cause is TerminationCause.DELETED


def test_unknown_lists_the_warning_events() -> None:
    diagnosis = diagnose_state(
        state="Failed", reason="TASK_FAILED", message="1 sandbox task(s) failed", events=["BackOff: x"]
    )

    assert diagnosis.detail.endswith("; events: BackOff: x")


def test_warning_events_keeps_only_warnings_as_reason_and_message() -> None:
    assert warning_events(EVICTED_EVENTS) == [EVICTED]


def test_warning_events_keeps_the_latest_ones() -> None:
    content = "\n".join(f"[2026-10-30 20:00:{n:02}+00:00] Warning  BackOff  attempt {n}" for n in range(MAX_EVENTS + 5))

    events = warning_events(content)

    assert len(events) == MAX_EVENTS
    assert events[-1] == f"BackOff: attempt {MAX_EVENTS + 4}"
