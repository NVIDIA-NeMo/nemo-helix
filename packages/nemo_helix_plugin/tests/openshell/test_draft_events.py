# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
from nemo_helix_plugin.openshell.draft_events import DraftEventSource, DraftPolicyTracker, publish_events


@dataclass
class Endpoint:
    host: str
    port: int


@dataclass
class Rule:
    endpoints: list[Endpoint] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    status: str = "pending"
    rule_name: str = "allow_example_com_443"
    rationale: str = "Allow python3.13 to connect to example.com:443."
    security_notes: str = ""
    binary: str = "/usr/local/bin/python3.13"
    hit_count: int = 1
    stage: str = "initial"
    validation_result: str = ""
    rejection_reason: str = ""
    proposed_rule: Rule = field(default_factory=lambda: Rule([Endpoint("example.com", 443)]))


SOURCE = DraftEventSource(
    kind="job", workspace="team", name="train", sandbox="nhx-abc", step="run", link="/workspaces/team/jobs/train"
)


def test_pending_chunk_is_reported_once() -> None:
    tracker = DraftPolicyTracker()

    [event] = tracker.observe(SOURCE, [Chunk("c1")])

    assert event["type"] == "openshell.policy.draft.pending"
    assert event["severity"] == "warning"
    assert event["title"] == "Network access requested: example.com:443"
    assert event["link"] == "/workspaces/team/jobs/train"
    assert event["data"]["link_label"] == "View job"
    assert event["data"]["source"] == {"kind": "job", "workspace": "team", "name": "train", "step": "run"}
    assert event["data"]["chunk_id"] == "c1"
    assert event["data"]["endpoints"] == ["example.com:443"]
    assert tracker.observe(SOURCE, [Chunk("c1")]) == []


def test_decision_on_a_reported_chunk_is_reported() -> None:
    tracker = DraftPolicyTracker()
    tracker.observe(SOURCE, [Chunk("c1"), Chunk("c2")])

    events = tracker.observe(
        SOURCE, [Chunk("c1", status="approved"), Chunk("c2", status="rejected", rejection_reason="Too broad.")]
    )

    assert [event["type"] for event in events] == [
        "openshell.policy.draft.approved",
        "openshell.policy.draft.rejected",
    ]
    assert events[1]["message"].endswith("Too broad.")


def test_chunk_first_seen_decided_is_not_reported() -> None:
    tracker = DraftPolicyTracker()

    assert tracker.observe(SOURCE, [Chunk("c1", status="approved")]) == []


def test_sandboxes_are_tracked_separately_and_forgotten() -> None:
    tracker = DraftPolicyTracker()
    other = DraftEventSource(kind="deployment", workspace="team", name="agent", sandbox="nhx-def")
    tracker.observe(SOURCE, [Chunk("c1")])

    assert len(tracker.observe(other, [Chunk("c1")])) == 1
    tracker.forget(SOURCE.sandbox)
    assert len(tracker.observe(SOURCE, [Chunk("c1")])) == 1


def test_due_throttles_per_sandbox() -> None:
    now = 100.0
    tracker = DraftPolicyTracker(poll_interval_seconds=5.0, clock=lambda: now)

    assert tracker.due("a")
    assert not tracker.due("a")
    assert tracker.due("b")
    now = 105.0
    assert tracker.due("a")


def test_publish_posts_each_event_and_survives_failures() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404 if json.loads(request.content)["type"] == "missing" else 201)

    published = publish_events(
        "http://platform:8080/",
        "team",
        [{"type": "missing"}, {"type": "ok"}],
        headers={"X-Test": "1"},
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert published == 1
    assert [str(request.url) for request in requests] == [
        "http://platform:8080/apis/notif/v2/workspaces/team/events"
    ] * 2
    assert requests[1].headers["X-Test"] == "1"
