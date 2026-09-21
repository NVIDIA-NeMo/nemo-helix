# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_evaluator.jobs.agent_sessions import (
    SESSION_ID_HEADER,
    AgentSessionBroker,
    deployment_ref_from_url,
)
from nemo_evaluator_sdk.agent_inference import AgentInferenceContext
from nemo_evaluator_sdk.values import GenericAgent
from nemo_platform_plugin.agents.types import AgentSession


def _agent() -> GenericAgent:
    return GenericAgent(
        url="http://agent.invalid",
        name="calc",
        body={"messages": []},
        response_path="$.choices[0].message.content",
    )


def _session(session_id: str) -> AgentSession:
    return AgentSession(id=session_id, name=f"session-{session_id}", deployment_id="deployment-1")


def _broker(sessions: list[str], closed: list[str] | None = None) -> AgentSessionBroker:
    remaining = list(sessions)
    return AgentSessionBroker(
        open_session=lambda: _session(remaining.pop(0)),
        close_session=lambda name: (closed if closed is not None else []).append(name),
    )


@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "http://localhost:8080/apis/agents/v2/workspaces/default/deployments/calc-1/-/v1/chat/completions",
            ("default", "calc-1"),
        ),
        (
            "http://localhost:8080/apis/agents/v2/workspaces/my%20ws/deployments/calc-1/-/v1/chat/completions",
            ("my ws", "calc-1"),
        ),
        ("http://localhost:8080/apis/agents/v2/workspaces/default/agents/calc/-/v1/chat/completions", None),
        ("http://example.com/v1/chat/completions", None),
    ],
)
def test_deployment_ref_from_url(url: str, expected: tuple[str, str] | None) -> None:
    assert deployment_ref_from_url(url) == expected


def test_open_assigns_one_session_per_task() -> None:
    broker = _broker(["a", "b"])

    broker.open(["task-1", "task-2"])

    assert broker.sessions == {"task-1": "a", "task-2": "b"}


def test_a_refused_session_leaves_its_task_unsessioned() -> None:
    def open_session() -> AgentSession:
        raise RuntimeError("gateway said no")

    broker = AgentSessionBroker(open_session=open_session, close_session=lambda name: None)

    broker.open(["task-1"])

    assert broker.sessions == {}
    assert broker.session_for("task-1") is None


def test_close_closes_every_open_session() -> None:
    closed: list[str] = []
    broker = _broker(["a", "b"], closed)
    broker.open(["task-1", "task-2"])

    broker.close()

    assert closed == ["session-a", "session-b"]


def test_close_failure_does_not_propagate() -> None:
    def close_session(name: str) -> None:
        raise RuntimeError("already gone")

    broker = AgentSessionBroker(open_session=lambda: _session("a"), close_session=close_session)
    broker.open(["task-1"])

    broker.close()


async def _headers_seen(
    broker: AgentSessionBroker, monkeypatch: pytest.MonkeyPatch, *, task_id: str
) -> dict[str, str] | None:
    """Drive the wrapper the factory builds, with a probe standing in for SDK inference."""
    import nemo_evaluator.jobs.agent_sessions as module

    captured: dict[str, dict[str, str] | None] = {"headers": None}

    async def probe(agent: GenericAgent, request: dict, **kwargs: object) -> dict:
        headers = kwargs.get("default_headers")
        captured["headers"] = (
            {str(key): str(value) for key, value in headers.items()} if isinstance(headers, dict) else None
        )
        return {}

    monkeypatch.setattr(module, "make_agent_inference_fn", lambda context: probe)
    fn = broker.inference_fn_factory()(AgentInferenceContext(metadata={"task_id": task_id}))
    await fn(_agent(), {}, max_retries=1)
    return captured["headers"]


async def test_inference_fn_stamps_the_task_session_header(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(["a"])
    broker.open(["task-1"])

    assert await _headers_seen(broker, monkeypatch, task_id="task-1") == {SESSION_ID_HEADER: "a"}


async def test_unsessioned_task_gets_the_plain_inference_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(["a"])
    broker.open(["task-1"])

    assert await _headers_seen(broker, monkeypatch, task_id="task-2") is None
