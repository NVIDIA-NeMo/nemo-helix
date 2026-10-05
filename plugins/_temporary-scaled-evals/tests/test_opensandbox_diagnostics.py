# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading a sandbox container's exit from OpenSandbox's inspect report."""

from __future__ import annotations

import pytest

pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")

from scaled_evals.opensandbox_diagnostics import sandbox_exit_from_inspect

# OpenSandbox v0.2.1's full report for a sandbox whose container exceeded its 256 MiB limit,
# captured on the dev cluster; only names and addresses are replaced. If an upgrade changes the
# format, refresh this from the new server.
OOM_INSPECT = """\
Pod Name:       sbx-1-0
Namespace:      sandboxes
Node:           10.0.0.1
Phase:          Failed
Pod IP:         10.244.0.2
Host IP:        10.0.0.1
Start Time:     2026-10-05 16:07:07+00:00

Containers:
  sandbox:
    Ready:          False
    Restart Count:  0
    Image:          registry.example/hello:1
    State:          Terminated (exit=137, reason=OOMKilled)

Init Containers:
  execd-installer:
    Ready:          True
    State:          Terminated (exit=0, reason=Completed)

Conditions:
  PodReadyToStartContainers: False (reason=N/A)
  Initialized: True (reason=N/A)
  Ready: False (reason=PodFailed)
  ContainersReady: False (reason=PodFailed)
  PodScheduled: True (reason=N/A)

Labels:
  batch-sandbox.sandbox.opensandbox.io/name=sbx-1
  batch-sandbox.sandbox.opensandbox.io/pod-index=0
  opensandbox.io/id=sbx-1

Resources:
  sandbox:
    Requests: {'cpu': '500m', 'memory': '256Mi'}
    Limits:   {'cpu': '500m', 'memory': '256Mi'}
"""

# The same sandbox's report while it was still running, trimmed to the parts that matter.
RUNNING_INSPECT = """\
Pod Name:       sbx-1-0
Phase:          Running

Containers:
  sandbox:
    Ready:          True
    State:          Running (since 2026-10-05 16:07:07+00:00)

Init Containers:
  execd-installer:
    Ready:          True
    State:          Terminated (exit=0, reason=Completed)
"""


def test_reads_the_sandbox_container_termination() -> None:
    assert sandbox_exit_from_inspect(OOM_INSPECT) == {
        "pod_phase": "Failed",
        "container": "sandbox",
        "state": "Terminated (exit=137, reason=OOMKilled)",
        "exit_code": 137,
        "reason": "OOMKilled",
    }


def test_ignores_a_running_sandbox_and_its_finished_init_container() -> None:
    assert sandbox_exit_from_inspect(RUNNING_INSPECT) is None


def test_keeps_the_termination_message() -> None:
    text = OOM_INSPECT.replace("reason=OOMKilled)\n", "reason=Error)\n    Message:        node lost\n")

    result = sandbox_exit_from_inspect(text)

    assert result is not None
    assert (result["reason"], result["message"]) == ("Error", "node lost")


def test_an_ended_pod_with_an_unreadable_state_is_an_unknown_exit() -> None:
    text = OOM_INSPECT.replace("State:          Terminated (exit=137, reason=OOMKilled)", "Status:  exited")

    assert sandbox_exit_from_inspect(text) == {
        "pod_phase": "Failed",
        "container": "sandbox",
        "state": None,
        "exit_code": None,
        "reason": "unknown",
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not a pod report",
        RUNNING_INSPECT.replace("State:          Running", "Status:  up"),
    ],
)
def test_a_report_that_cannot_tell_raises(text: str) -> None:
    with pytest.raises(ValueError, match="unrecognized OpenSandbox inspect report"):
        sandbox_exit_from_inspect(text)
