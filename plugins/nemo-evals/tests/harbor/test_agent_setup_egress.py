# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent wrapper that runs a Harbor agent's setup() inside the sandbox's setup egress window."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from harbor.agents.factory import AgentFactory
from harbor.agents.nop import NopAgent
from harbor.models.trial.config import AgentConfig
from nemo_evals.harbor.agent_setup_egress import INNER_AGENT_KWARG, SETUP_EGRESS_AGENT, AgentSetupEgress


class _WindowEnvironment:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    @asynccontextmanager
    async def agent_setup_egress(self) -> AsyncGenerator[None, None]:
        self.events.append("open")
        try:
            yield
        finally:
            self.events.append("close")


def _agent(tmp_path, inner: str = "nop") -> Any:
    config = AgentConfig(import_path=SETUP_EGRESS_AGENT, kwargs={INNER_AGENT_KWARG: inner})
    return AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)


def test_harbor_builds_the_real_agent_from_the_wrapper_import_path(tmp_path):
    agent = _agent(tmp_path)
    assert isinstance(agent, NopAgent)
    assert type(agent).__name__ == "NopAgent" and agent.name() == NopAgent.name()


async def test_setup_runs_inside_the_window(tmp_path, monkeypatch):
    events: list[str] = []

    async def setup(self, environment) -> None:
        events.append("setup")

    monkeypatch.setattr(NopAgent, "setup", setup)
    await _agent(tmp_path).setup(_WindowEnvironment(events))
    assert events == ["open", "setup", "close"]


async def test_a_failed_setup_still_closes_the_window(tmp_path, monkeypatch):
    events: list[str] = []

    async def setup(self, environment) -> None:
        raise RuntimeError("install failed")

    monkeypatch.setattr(NopAgent, "setup", setup)
    with pytest.raises(RuntimeError, match="install failed"):
        await _agent(tmp_path).setup(_WindowEnvironment(events))
    assert events == ["open", "close"]


async def test_an_environment_without_a_window_runs_setup_unchanged(tmp_path, monkeypatch):
    events: list[str] = []

    async def setup(self, environment) -> None:
        events.append("setup")

    monkeypatch.setattr(NopAgent, "setup", setup)
    await _agent(tmp_path).setup(object())
    assert events == ["setup"]


def test_preflight_is_the_real_agents_without_the_wrapper_kwarg(monkeypatch):
    seen: list[dict[str, Any]] = []

    def preflight(cls, kwargs=None, env=None) -> None:
        seen.append(dict(kwargs or {}))

    monkeypatch.setattr(NopAgent, "preflight", classmethod(preflight))
    AgentFactory.run_preflight(
        AgentConfig(import_path=SETUP_EGRESS_AGENT, kwargs={INNER_AGENT_KWARG: "nop", "version": "1"})
    )
    assert seen == [{"version": "1"}]


def test_the_inner_agent_may_be_an_import_path(tmp_path):
    agent = _agent(tmp_path, inner="harbor.agents.nop:NopAgent")
    assert isinstance(agent, NopAgent)


def test_the_wrapper_requires_the_inner_agent(tmp_path):
    with pytest.raises(KeyError, match=INNER_AGENT_KWARG):
        AgentSetupEgress(logs_dir=tmp_path)


def test_an_acp_shorthand_keeps_its_registry_spec(tmp_path):
    from harbor.agents.installed.acp import AcpAgent

    assert isinstance(_agent(tmp_path, inner="acp:claude-code"), AcpAgent)
