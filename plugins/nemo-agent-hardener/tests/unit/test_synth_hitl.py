# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the synth HTTP client and the status_details HITL bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
from nemo_agent_hardener_plugin.jobs import hitl
from nemo_agent_hardener_plugin.jobs.synth_client import SynthClient


def test_synth_client_maps_endpoints() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        bodies = {
            "/healthz": {"status": "ok"},
            "/synth": {"thread_id": "t1", "status": "interview", "questions": [{"gap": "g"}]},
            "/synth/t1/answers": {"thread_id": "t1", "status": "review", "suite": [{"tool": "clock"}]},
            "/synth/t1/suite": {"thread_id": "t1", "status": "done", "benign_csv": "/x/requests.csv"},
        }
        return httpx.Response(200, json=bodies[path]) if path in bodies else httpx.Response(404)

    with SynthClient("http://svc", transport=httpx.MockTransport(handler)) as client:
        assert client.healthz() is True
        assert client.start("m.yaml")["status"] == "interview"
        assert client.answers("t1", [{"gap": "g", "answer": "a"}])["status"] == "review"
        assert client.write_suite("t1", [{"tool": "clock", "payload": "p"}])["benign_csv"].endswith("requests.csv")


class _FakeSynthService:
    """Duck-typed SynthClient: one interview round, then review, then done."""

    def __init__(self) -> None:
        self.answered: list[list[dict[str, Any]]] = []
        self.written: list[dict[str, Any]] = []

    def start(self, _config: str, *, validator: str | None = None) -> dict[str, Any]:
        return {"thread_id": "t", "status": "interview", "questions": [{"gap": "g1"}]}

    def answers(self, _thread_id: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
        self.answered.append(answers)
        return {"thread_id": "t", "status": "review", "suite": [{"tool": "clock", "payload": "time?"}]}

    def write_suite(self, _thread_id: str, suite: list[dict[str, Any]]) -> dict[str, Any]:
        self.written = suite
        return {"thread_id": "t", "status": "done", "benign_csv": "/x/requests.csv"}


def test_drive_synth_hitl_relays_interview_then_review() -> None:
    published: list[tuple[str, dict[str, Any]]] = []
    responses = {"interview": [{"gap": "g1", "answer": "a"}], "review": [{"tool": "clock", "payload": "edited"}]}
    service = _FakeSynthService()

    path = hitl.drive_synth_hitl(
        cast(SynthClient, service),
        "m.yaml",
        lambda kind, payload: published.append((kind, payload)),
        lambda kind: responses[kind],
    )

    assert [kind for kind, _ in published] == ["interview", "review"]
    assert service.answered == [[{"gap": "g1", "answer": "a"}]]
    assert service.written == [{"tool": "clock", "payload": "edited"}]
    assert path.endswith("requests.csv")


def test_status_details_channel_publishes_and_matches_round() -> None:
    published: dict[str, Any] = {}

    job = {
        "id": "job1",
        "attempt_id": "att-1",
        "name": "job1",
        "workspace": "default",
        "source": "test",
        "spec": {},
        "platform_spec": {
            "steps": [{"name": "step-one", "executor": {"provider": "cpu", "container": {"image": "x"}}}]
        },
        "fileset": "fs-1",
        "status": "active",
        "status_details": {"interview_response": {"round": 1, "answers": [{"gap": "g", "answer": "a"}]}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PATCH" and request.url.path.endswith("/status-details"):
            if request.content:
                published.update(json.loads(request.content))
            return httpx.Response(200, json={})
        if request.method == "GET" and request.url.path.endswith("/jobs/job1"):
            return httpx.Response(200, json=job)
        return httpx.Response(404)

    # The channel wraps the SDK via ``client_from_platform`` into a typed JobsClient;
    # model that with a real JobsClient over a mocked transport.
    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://platform:8080")
    platform = SimpleNamespace(
        base_url="http://platform:8080",
        workspace="default",
        _custom_headers={"Authorization": "Bearer x"},
        _client=http_client,
        timeout=None,
        max_retries=2,
        _prepare_url=lambda url: url,
    )

    channel = hitl.StatusDetailsChannel(platform, name="job1", workspace="default", poll_interval=0.0)
    channel.publish("interview", {"questions": [{"gap": "g"}]})
    assert published["interview"]["round"] == 1
    assert channel.await_response("interview") == [{"gap": "g", "answer": "a"}]
    # Interview answers are accumulated so the run can persist the Q&A on the manifest for display.
    assert channel.interview == [{"gap": "g", "answer": "a"}]
