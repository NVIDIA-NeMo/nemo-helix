# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from nhx.studio import assistant_topic_gate


@pytest.fixture(autouse=True)
def reset_warned_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(assistant_topic_gate, "_warned_failures", set())


@pytest.fixture
def gate_requests(monkeypatch: pytest.MonkeyPatch) -> Callable[[httpx.Response | Exception], list[httpx.Request]]:
    def install(outcome: httpx.Response | Exception) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def client(**kwargs: Any) -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(assistant_topic_gate, "AsyncClient", client)
        return requests

    return install


async def _check(message: str = "How do I bake a cheesecake?", workspace: str = "team-a") -> bool:
    return await assistant_topic_gate.is_off_topic(
        message,
        base_url="http://platform.test",
        workspace=workspace,
        headers={"authorization": "Bearer token"},
    )


@pytest.mark.parametrize(
    ("gate_model", "default_model", "expected"),
    [
        (None, None, None),
        (None, "default/nemotron", "default/nemotron"),
        ("default/judge", "default/nemotron", "default/judge"),
        ("off", "default/nemotron", None),
        ("  ", "default/nemotron", None),
    ],
)
def test_topic_gate_model(
    monkeypatch: pytest.MonkeyPatch, gate_model: str | None, default_model: str | None, expected: str | None
):
    for name, value in (
        (assistant_topic_gate.TOPIC_GATE_MODEL_ENV, gate_model),
        ("NEMO_DEFAULT_MODEL", default_model),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert assistant_topic_gate.topic_gate_model() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "expected"), [("blocked", True), ("success", False)])
async def test_is_off_topic_follows_check_status(
    monkeypatch: pytest.MonkeyPatch, gate_requests, status: str, expected: bool
):
    monkeypatch.setenv(assistant_topic_gate.TOPIC_GATE_MODEL_ENV, "default/judge")
    requests = gate_requests(httpx.Response(200, json={"status": status, "rails_status": {}}))

    assert await _check() is expected

    [request] = requests
    assert str(request.url) == "http://platform.test/apis/guardrails/v2/workspaces/default/checks"
    assert request.headers["authorization"] == "Bearer token"
    body = json.loads(request.content)
    assert body["model"] == "default/judge"
    assert body["messages"] == [{"role": "user", "content": "How do I bake a cheesecake?"}]
    assert body["guardrails"]["config"]["rails"] == {"input": {"flows": ["self check input"]}}
    prompt = body["guardrails"]["config"]["prompts"][0]
    assert prompt["task"] == "self_check_input"
    assert "{{ user_input }}" in prompt["content"]


@pytest.mark.asyncio
async def test_is_off_topic_uses_caller_workspace_for_unqualified_model(monkeypatch: pytest.MonkeyPatch, gate_requests):
    monkeypatch.setenv(assistant_topic_gate.TOPIC_GATE_MODEL_ENV, "judge")
    requests = gate_requests(httpx.Response(200, json={"status": "success", "rails_status": {}}))

    await _check(workspace="team-a")

    assert requests[0].url.path == "/apis/guardrails/v2/workspaces/team-a/checks"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        httpx.Response(500, json={"detail": "boom"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["blocked"]),
        httpx.ConnectError("unreachable"),
    ],
)
async def test_is_off_topic_fails_open(
    monkeypatch: pytest.MonkeyPatch, gate_requests, outcome: httpx.Response | Exception
):
    monkeypatch.setenv(assistant_topic_gate.TOPIC_GATE_MODEL_ENV, "default/judge")
    gate_requests(outcome)

    assert await _check() is False


@pytest.mark.asyncio
async def test_is_off_topic_fails_open_at_the_wall_clock_deadline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(assistant_topic_gate.TOPIC_GATE_MODEL_ENV, "default/judge")
    monkeypatch.setattr(assistant_topic_gate, "TOPIC_GATE_TIMEOUT_SECONDS", 0.05)

    class StalledClient:
        """A response that keeps the request open past the deadline, as an incomplete stream would."""

        async def __aenter__(self) -> "StalledClient":
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
            await asyncio.sleep(3600)
            raise AssertionError("the deadline should have cancelled the post")

    monkeypatch.setattr(assistant_topic_gate, "AsyncClient", lambda **kwargs: StalledClient())

    assert await _check() is False


@pytest.mark.asyncio
async def test_is_off_topic_skips_check_when_disabled(monkeypatch: pytest.MonkeyPatch, gate_requests):
    monkeypatch.setenv(assistant_topic_gate.TOPIC_GATE_MODEL_ENV, "off")
    requests = gate_requests(httpx.Response(200, json={"status": "blocked"}))

    assert await _check() is False
    assert requests == []


@pytest.mark.asyncio
async def test_is_off_topic_warns_once_per_failure_kind(
    monkeypatch: pytest.MonkeyPatch, gate_requests, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv(assistant_topic_gate.TOPIC_GATE_MODEL_ENV, "default/judge")
    gate_requests(httpx.Response(401, json={"detail": "unauthorized"}))

    with caplog.at_level(logging.WARNING, logger=assistant_topic_gate.logger.name):
        await _check()
        await _check()

    [record] = caplog.records
    assert "HTTP 401" in record.getMessage()
    assert "not being filtered" in record.getMessage()
    assert record.exc_info is None


def test_terminal_topic_gate_policy_matches_studio_policy() -> None:
    # The terminal assistant's VirtualModel gate (agents/nemo-studio-assistant) must enforce the same policy as Studio.
    repo_root = Path(__file__).resolve().parents[4]
    terminal_gate = (repo_root / "agents/nemo-studio-assistant/src/nemo_studio_assistant/topic_gate.py").read_text()

    assert assistant_topic_gate.TOPIC_POLICY in terminal_gate
