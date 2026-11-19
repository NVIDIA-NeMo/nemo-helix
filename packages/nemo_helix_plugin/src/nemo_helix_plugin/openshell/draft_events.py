# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn OpenShell draft policy chunks into notification events.

A sandbox's pending network requests live on the gateway as draft policy chunks.
The gateway has no usable push signal for them, so the Jobs and Deployments
backends hand the chunks they read on their own poll cycle to a
``DraftPolicyTracker``, which reports only what changed since the last poll.

Prototype scope: tracker state is in memory, so a controller restart reports
every chunk that is still pending again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from nemo_helix_plugin.openshell.network_policy import agent_from_labels, agent_network_link

logger = logging.getLogger(__name__)

EVENT_TYPE_PREFIX = "openshell.policy.draft"
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
PUBLISH_TIMEOUT_SECONDS = 5.0

_SEVERITY = {"pending": "warning", "approved": "success", "rejected": "error"}
_LINK_LABEL = {"job": "View job", "deployment": "View deployments"}


class _Endpoint(Protocol):
    host: str
    port: int


class _Rule(Protocol):
    @property
    def endpoints(self) -> Iterable[_Endpoint]: ...


class DraftChunk(Protocol):
    """The ``openshell.v1.PolicyChunk`` fields an event carries."""

    id: str
    status: str
    rule_name: str
    rationale: str
    security_notes: str
    binary: str
    hit_count: int
    stage: str
    validation_result: str
    rejection_reason: str

    @property
    def proposed_rule(self) -> _Rule: ...


@dataclass(frozen=True)
class DraftEventSource:
    """The Helix workload a sandbox belongs to."""

    kind: Literal["job", "deployment"]
    workspace: str
    name: str
    sandbox: str
    step: str | None = None
    link: str | None = None
    notify_emails: tuple[str, ...] = ()
    agent: str | None = None
    """The agent the workload runs, in the workload's workspace. Its page is where requests are reviewed."""

    @classmethod
    def for_sandbox(
        cls,
        labels: Mapping[str, str],
        *,
        kind: Literal["job", "deployment"],
        workspace: str,
        name: str,
        sandbox: str,
        link: str,
        step: str | None = None,
        notify_emails: tuple[str, ...] = (),
    ) -> DraftEventSource:
        """The source for a sandbox, pointing at its agent when its labels name one."""
        owner = agent_from_labels(labels)
        agent = owner[1] if owner is not None and owner[0] == workspace else None
        return cls(
            kind=kind,
            workspace=workspace,
            name=name,
            sandbox=sandbox,
            step=step,
            link=agent_network_link(workspace, agent) if agent else link,
            notify_emails=notify_emails,
            agent=agent,
        )


def _endpoints(chunk: DraftChunk) -> list[str]:
    return [f"{endpoint.host}:{endpoint.port}" for endpoint in chunk.proposed_rule.endpoints]


def build_event(chunk: DraftChunk, source: DraftEventSource) -> dict[str, Any]:
    """The notif publish body for a chunk in its current status."""
    endpoints = _endpoints(chunk)
    target = ", ".join(endpoints) or chunk.rule_name
    binary = chunk.binary.rsplit("/", 1)[-1] or "A process"
    workload = f"{source.kind} {source.name}"
    if source.agent:
        workload = f"agent {source.agent} ({workload})"
    if chunk.status == "pending":
        title = f"Network access requested: {target}"
        message = f"{binary} in {workload} was blocked from {target}. {chunk.rationale}".strip()
    elif chunk.status == "approved":
        title = f"Network access approved: {target}"
        message = f"{binary} in {workload} can now reach {target}."
    else:
        title = f"Network access {chunk.status}: {target}"
        message = f"{binary} in {workload} stays blocked from {target}. {chunk.rejection_reason}".strip()
    if source.agent:
        link_label = "Review request" if chunk.status == "pending" else "View agent"
    else:
        link_label = _LINK_LABEL[source.kind]
    # The notif service emails these people, e.g. the creator of the job that is blocked.
    email_to = {"email_to": list(source.notify_emails)} if source.notify_emails else {}
    return {
        "type": f"{EVENT_TYPE_PREFIX}.{chunk.status}",
        "title": title[:200],
        "message": message[:2000],
        "severity": _SEVERITY.get(chunk.status, "info"),
        "link": source.link,
        "data": {
            # Studio labels the notification's link with this.
            "link_label": link_label,
            "agent": source.agent,
            "source": {"kind": source.kind, "workspace": source.workspace, "name": source.name, "step": source.step},
            "sandbox": source.sandbox,
            "chunk_id": chunk.id,
            "status": chunk.status,
            "rule_name": chunk.rule_name,
            "binary": chunk.binary,
            "endpoints": endpoints,
            "rationale": chunk.rationale,
            "security_notes": chunk.security_notes,
            "validation_result": chunk.validation_result,
            "hit_count": chunk.hit_count,
            "stage": chunk.stage,
            **email_to,
        },
    }


class DraftPolicyTracker:
    """Remembers each sandbox's chunk statuses and reports the changes."""

    def __init__(
        self,
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._last_poll: dict[str, float] = {}
        self._statuses: dict[str, dict[str, str]] = {}

    def due(self, sandbox: str) -> bool:
        """Whether this sandbox's drafts should be read now. Marks the poll as taken."""
        now = self._clock()
        last = self._last_poll.get(sandbox)
        if last is not None and now - last < self._poll_interval_seconds:
            return False
        self._last_poll[sandbox] = now
        return True

    def observe(self, source: DraftEventSource, chunks: Iterable[DraftChunk]) -> list[dict[str, Any]]:
        """Events for chunks that are new or changed status since the last call."""
        seen = self._statuses.setdefault(source.sandbox, {})
        events: list[dict[str, Any]] = []
        for chunk in chunks:
            previous = seen.get(chunk.id)
            if previous == chunk.status:
                continue
            seen[chunk.id] = chunk.status
            # A chunk first seen already decided was never shown as pending, so its
            # decision is not news.
            if previous is None and chunk.status != "pending":
                continue
            events.append(build_event(chunk, source))
        return events

    def forget(self, sandbox: str) -> None:
        self._last_poll.pop(sandbox, None)
        self._statuses.pop(sandbox, None)


def publish_events(
    base_url: str,
    workspace: str,
    events: Iterable[Mapping[str, Any]],
    *,
    headers: Mapping[str, str] | None = None,
    client: httpx.Client | None = None,
) -> int:
    """Publish events to the notif service. Returns how many were accepted.

    Best effort: a missing notif service or a failed publish is logged and dropped,
    because a notification must never fail the reconcile that produced it.
    """
    url = f"{base_url.rstrip('/')}/apis/notif/v2/workspaces/{workspace}/events"
    published = 0
    with client or httpx.Client(timeout=PUBLISH_TIMEOUT_SECONDS) as http:
        for event in events:
            try:
                http.post(url, json=dict(event), headers=dict(headers or {})).raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Failed to publish %s to %s: %s", event.get("type"), url, exc)
                continue
            published += 1
    return published
