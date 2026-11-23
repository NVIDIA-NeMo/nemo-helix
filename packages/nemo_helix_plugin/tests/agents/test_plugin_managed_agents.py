# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugin-author contract for plugin-managed agent types."""

from __future__ import annotations

import json
from typing import ClassVar

import httpx
import pytest
from nemo_helix_plugin.agents.plugin_managed_agents import (
    CONFIGURED_DEFAULT_MODEL,
    HELIX_DEFAULT_SCHEMA_KEY,
    PluginManagedAgent,
    PreparedRun,
    helix_default,
)
from nemo_helix_plugin.agents.types import Agent, PluginManagedAgentSpec, RegisterAgentRequest
from nemo_helix_plugin.client.client import AsyncNemoClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError

BASE = "http://test:8000"


class _Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    model: str = Field(json_schema_extra=helix_default(CONFIGURED_DEFAULT_MODEL))


class _PreparedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    notes: str


class _OneModelType(PluginManagedAgent[_Context]):
    type_id: ClassVar[str] = "example.one"
    title: ClassVar[str] = "One model"
    adapter_id: ClassVar[str] = "nvidia.example.adapter"
    post_run_actions: ClassVar[frozenset[str]] = frozenset({"agents.noop"})
    default_name: ClassVar[str] = "example-one"


class _TwoModelType(PluginManagedAgent[_Context, _PreparedContext]):
    type_id: ClassVar[str] = "example.two"
    title: ClassVar[str] = "Two models"
    adapter_id: ClassVar[str] = "nvidia.example.adapter"
    post_run_actions: ClassVar[frozenset[str]] = frozenset({"agents.noop"})
    default_name: ClassVar[str] = "example-two"


def test_one_generic_argument_uses_the_context_model_for_both_contracts() -> None:
    assert _OneModelType.context_model is _Context
    assert _OneModelType.prepared_context_model is _Context


def test_two_generic_arguments_set_separate_models() -> None:
    assert _TwoModelType.context_model is _Context
    assert _TwoModelType.prepared_context_model is _PreparedContext


def test_a_non_model_generic_argument_is_rejected() -> None:
    with pytest.raises(TypeError, match="pydantic BaseModel"):

        class _Broken(PluginManagedAgent[dict]):  # ty: ignore[invalid-type-arguments]
            pass


def test_default_config_is_one_harness_of_the_adapter() -> None:
    config = _OneModelType.config(name="my-agent")

    assert config["name"] == "my-agent"
    assert config["config_format"] == "nemo-agents-spec-v1"
    assert config["harnesses"][config["default_harness"]] == {"kind": "nvidia.example.adapter"}
    assert config["environment"] == {"provider": "local"}


def test_context_schema_is_the_user_facing_model_with_its_hints() -> None:
    schema = _TwoModelType.context_schema()

    assert set(schema["properties"]) == {"target", "model"}
    assert schema["properties"]["model"][HELIX_DEFAULT_SCHEMA_KEY] == CONFIGURED_DEFAULT_MODEL
    assert schema["additionalProperties"] is False


def test_helix_default_rejects_an_unknown_hint() -> None:
    with pytest.raises(ValueError, match="Unknown x-helix-default hint"):
        helix_default("configured_nothing")


def test_parse_context_drops_the_platform_key_and_parses_the_prepared_model() -> None:
    parsed = _TwoModelType.parse_context(
        {"target": "bot", "notes": "be thorough", "nemo_helix": {"job_id": "job-1", "job_workspace": "default"}}
    )

    assert parsed == _PreparedContext(target="bot", notes="be thorough")


def test_parse_context_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        _TwoModelType.parse_context({"target": "bot", "notes": "x", "surprise": 1})


@pytest.mark.asyncio
async def test_default_prepare_passes_a_single_model_context_through() -> None:
    context = _Context(target="bot", model="default/m")

    prepared = await _OneModelType.prepare(context, workspace="default", client=AsyncNemoClient(base_url=BASE))

    assert prepared == PreparedRun(context=context)


@pytest.mark.asyncio
async def test_default_prepare_refuses_to_guess_a_separate_prepared_model() -> None:
    with pytest.raises(NotImplementedError, match="must implement prepare"):
        await _TwoModelType.prepare(
            _Context(target="bot", model="default/m"), workspace="default", client=AsyncNemoClient(base_url=BASE)
        )


def test_is_instance_matches_only_agents_of_the_type() -> None:
    assert _OneModelType.is_instance(Agent(name="a", plugin_managed_agent=PluginManagedAgentSpec(type="example.one")))
    assert not _OneModelType.is_instance(Agent(name="a", plugin_managed_agent=PluginManagedAgentSpec(type="other.one")))
    assert not _OneModelType.is_instance(Agent(name="a"))


@pytest.mark.asyncio
@pytest.mark.parametrize("found", [True, False])
async def test_find_agent_filters_by_type(found: bool) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        data = [{"name": "example-one", "plugin_managed_agent": {"type": "example.one"}}] if found else []
        return httpx.Response(
            200,
            json={
                "data": data,
                "pagination": {
                    "page": 1,
                    "page_size": 1,
                    "current_page_size": len(data),
                    "total_pages": 1,
                    "total_results": len(data),
                },
            },
        )

    client = AsyncNemoClient(base_url=BASE, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    agent = await _OneModelType.find_agent(workspace="research", client=client)

    assert (agent.name if agent else None) == ("example-one" if found else None)
    [request] = seen
    assert request.url.path == "/apis/agents/v2/workspaces/research/agents"
    assert json.loads(request.url.params["filter"]) == {"plugin_managed_agent.type": "example.one"}


class TestRegisterAgentRequest:
    def test_accepts_a_type_with_name_and_metadata(self) -> None:
        body = RegisterAgentRequest.model_validate(
            {"name": "analyst", "description": "d", "labels": {"team": "x"}, "plugin_managed_agent": {"type": "a.b"}}
        )

        assert body.plugin_managed_agent == PluginManagedAgentSpec(type="a.b")

    @pytest.mark.parametrize(
        "field, value",
        [
            ("config", {"config_format": "nemo-agents-spec-v1"}),
            ("config_format", "nemo-agents-spec-v1"),
            ("context_schema", {"type": "object"}),
            ("default_input", "go"),
            ("post_run_actions", ["agents.noop"]),
            ("reference", {"container_image": "agent:1"}),
        ],
    )
    def test_rejects_what_the_type_supplies(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError, match=f"{field} cannot be set with plugin_managed_agent"):
            RegisterAgentRequest.model_validate(
                {"name": "analyst", "plugin_managed_agent": {"type": "a.b"}, field: value}
            )

    @pytest.mark.parametrize("type_id", ["compass", "Compass.Analyst", ".analyst", "compass.", "compass analyst.x"])
    def test_rejects_a_malformed_type_id(self, type_id: str) -> None:
        with pytest.raises(ValidationError):
            RegisterAgentRequest.model_validate({"name": "analyst", "plugin_managed_agent": {"type": type_id}})
