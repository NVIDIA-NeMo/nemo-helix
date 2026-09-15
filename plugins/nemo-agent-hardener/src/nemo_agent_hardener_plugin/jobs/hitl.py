# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge the ``agent-hardener serve`` synth HITL to the platform ``status_details`` channel.

The war-game job drives the synth service (interview rounds, then benign-suite review) and relays each
checkpoint to the operator via the job's ``status_details`` — Studio renders it and PATCHes a response.
:func:`drive_synth_hitl` is the transport-agnostic loop (``publish``/``await_response`` injected so it is
unit-testable); :class:`StatusDetailsChannel` implements those over the typed Jobs client for the real job.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_HITL_TIMEOUT,
    CATEGORY_NETWORK,
    CATEGORY_SYNTH_SERVICE,
    AgentHardenerRunError,
)
from nemo_agent_hardener_plugin.jobs.synth_client import SynthClient
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.jobs.client import JobsClient
from nemo_platform_plugin.jobs.types import JobStatusDetailsUpdate

logger = logging.getLogger(__name__)

# Consecutive publish failures tolerated before the interview is abandoned as a control-plane outage.
_PUBLISH_MAX_ATTEMPTS = 3

# publish(kind, payload) -> None ; await_response(kind) -> list of answer/suite rows
Publish = Callable[[str, dict[str, Any]], None]
AwaitResponse = Callable[[str], list[dict[str, Any]]]


def drive_synth_hitl(
    client: SynthClient, config: str, publish: Publish, await_response: AwaitResponse, *, validator: str | None = None
) -> str:
    """Run the synth service to completion, relaying each interview round + the review via the channel.

    Returns the path to the written ``requests.csv``. Loops interview rounds (``publish`` questions →
    ``await_response`` → ``POST /answers``) until the service reports ``review``, then relays the suite for
    editing and writes it back.
    """
    step = client.start(config, validator=validator)
    while step.get("status") == "interview":
        publish("interview", {"questions": step.get("questions", [])})
        answers = await_response("interview")
        step = client.answers(step["thread_id"], answers)
    if step.get("status") != "review":
        raise AgentHardenerRunError(
            CATEGORY_SYNTH_SERVICE, f"synth service returned unexpected status {step.get('status')!r}"
        )
    publish("review", {"suite": step.get("suite", [])})
    edited = await_response("review")
    done = client.write_suite(step["thread_id"], edited)
    return str(done.get("benign_csv", ""))


class StatusDetailsChannel:
    """Implements ``publish``/``await_response`` over the job's ``status_details`` (Studio is the peer).

    Each ``publish`` stamps an incrementing ``round`` so a multi-round interview never reads a stale answer;
    Studio echoes the round in its ``{kind}_response``. Polls (the job stays ``active`` — no platform pause).
    """

    def __init__(
        self, sdk: Any, *, name: str, workspace: str, poll_interval: float = 2.0, timeout: float = 1800.0
    ) -> None:
        self._sdk = sdk
        self._name = name
        self._workspace = workspace
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._round = 0
        # Interview answers accumulated across rounds, kept so the run can persist the Q&A for display.
        self.interview: list[dict[str, Any]] = []

    def publish(self, kind: str, payload: dict[str, Any]) -> None:
        self._round += 1
        body = {kind: {**payload, "round": self._round}}
        # Publishing the prompt is a write the operator's UI depends on; retry a transient control-plane
        # blip, but a persistent failure must abort the run loudly (a silently-dropped prompt would hang
        # the interview until the poll deadline with no explanation).
        for attempt in range(1, _PUBLISH_MAX_ATTEMPTS + 1):
            try:
                client_from_platform(self._sdk, JobsClient).update_job_status_details(
                    name=self._name, workspace=self._workspace, body=JobStatusDetailsUpdate(root=body)
                )
                return
            except Exception:
                if attempt == _PUBLISH_MAX_ATTEMPTS:
                    raise AgentHardenerRunError(
                        CATEGORY_NETWORK,
                        f"could not publish the {kind} prompt to the job after {_PUBLISH_MAX_ATTEMPTS} attempts",
                    )
                logger.warning("status_details publish failed for job %s; retrying", self._name, exc_info=True)
                time.sleep(self._poll_interval)

    def await_response(self, kind: str) -> list[dict[str, Any]]:
        key = f"{kind}_response"
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                job = (
                    client_from_platform(self._sdk, JobsClient)
                    .get_job(name=self._name, workspace=self._workspace)
                    .data()
                )
            except Exception:  # a transient poll failure must not abort a minutes-long human wait
                logger.warning("status_details poll failed for job %s; retrying", self._name, exc_info=True)
                time.sleep(self._poll_interval)
                continue
            resp = (getattr(job, "status_details", None) or {}).get(key)
            if isinstance(resp, dict) and resp.get("round") == self._round:
                rows = list(resp.get("answers") or resp.get("suite") or [])
                if kind == "interview":
                    self.interview.extend(row for row in rows if isinstance(row, dict))
                return rows
            time.sleep(self._poll_interval)
        raise AgentHardenerRunError(
            CATEGORY_HITL_TIMEOUT,
            f"no operator response to the {kind} prompt (round {self._round}) within {self._timeout:.0f}s",
        )
