# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read why a sandbox's container stopped from OpenSandbox's diagnostics report.

OpenSandbox reports container state only as a plain-text report from
``GET /v1/sandboxes/{id}/diagnostics/inspect``; its status endpoint can't see a container that has
exited on Kubernetes. Each runtime prints the report differently. Only the Kubernetes runtime's
format is read today; on the Docker runtime the report isn't recognized, so nothing is recorded.

This module imports only the standard library, so both ``NemoOpenSandboxEnvironment`` (Harbor's
venv) and the service can use it.
"""

from __future__ import annotations

import re
from typing import Any

# Name OpenSandbox gives the container that runs the task image in a sandbox's pod.
SANDBOX_CONTAINER_NAME = "sandbox"
# Pod phases in which every container has stopped, so a container whose state can't be read has died.
_ENDED_POD_PHASES = frozenset({"Failed", "Succeeded"})

# Patterns for OpenSandbox v0.2.1's Kubernetes report; see sandbox_exit_from_inspect for its layout.
_POD_PHASE = re.compile(r"^Phase:[ \t]*(?P<phase>\S+)", re.MULTILINE)
# The sandbox container's fields: the four-space-indented lines after "  sandbox:" in the
# "Containers:" section. The lazy skip only crosses indented lines, so it never reaches the
# separate "Init Containers:" section.
_SANDBOX_FIELDS = re.compile(
    rf"^Containers:\n(?:[ \t].*\n)*?  {SANDBOX_CONTAINER_NAME}:\n(?P<fields>(?:    .*(?:\n|$))*)", re.MULTILINE
)
# "Last State:" (an earlier restart) doesn't match: only the current state counts.
_STATE = re.compile(r"^[ \t]+State:[ \t]*(?P<state>.+)$", re.MULTILINE)
_MESSAGE = re.compile(r"^[ \t]+Message:[ \t]*(?P<message>.+)$", re.MULTILINE)
# The server prints missing values as Python's "None".
_TERMINATED_STATE = re.compile(r"Terminated \(exit=(?P<exit_code>[^,]*), reason=(?P<reason>[^)]*)\)")


def sandbox_exit_from_inspect(inspect_text: str) -> dict[str, Any] | None:
    """Return how the sandbox container stopped, from OpenSandbox's inspect report.

    Returns ``None`` while the container is running or waiting to start. If the report has
    changed so its state can't be read, a pod that has ended still counts as a stopped container,
    with reason ``"unknown"``. Raises ``ValueError`` when the report can't tell either way, so a
    format change never blames the sandbox for a failure it didn't cause.

    The Kubernetes report is indented text, not JSON (from ``k8s_diagnostics.py`` in OpenSandbox
    v0.2.1)::

        Phase:          Failed
        Containers:
          sandbox:
            State:          Terminated (exit=137, reason=OOMKilled)
            Message:        ...
        Init Containers:
          execd-installer:
            State:          Terminated (exit=0, reason=Completed)

    Only the ``sandbox`` entry under ``Containers:`` counts. OpenSandbox's own init containers are
    listed separately and always end terminated.
    """
    phase_match = _POD_PHASE.search(inspect_text)
    pod_phase = phase_match["phase"] if phase_match else None
    fields_match = _SANDBOX_FIELDS.search(inspect_text)
    fields = fields_match["fields"] if fields_match else ""
    state_match = _STATE.search(fields)
    state = state_match["state"].strip() if state_match else None

    terminated = _TERMINATED_STATE.fullmatch(state) if state else None
    if terminated and state_match:
        exit_code = terminated["exit_code"].strip()
        reason = terminated["reason"].strip()
        result: dict[str, Any] = {
            "pod_phase": pod_phase,
            "container": SANDBOX_CONTAINER_NAME,
            "state": state,
            "exit_code": int(exit_code) if exit_code.lstrip("-").isdigit() else None,
            "reason": None if reason == "None" else reason,
        }
        # A message printed under the state explains it, for example why a container was killed.
        message = _MESSAGE.search(fields, state_match.end())
        if message:
            result["message"] = message["message"].strip()
        return result

    if state and state.startswith(("Running", "Waiting")):
        return None
    if pod_phase in _ENDED_POD_PHASES:
        return {
            "pod_phase": pod_phase,
            "container": SANDBOX_CONTAINER_NAME,
            "state": state,
            "exit_code": None,
            "reason": "unknown",
        }
    raise ValueError(f"unrecognized OpenSandbox inspect report (pod phase {pod_phase!r}, sandbox state {state!r})")
