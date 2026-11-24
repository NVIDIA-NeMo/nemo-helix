# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-managed agents, through the real services, entity store, and jobs routes.

Unit tests cover loading, derivation and job preparation against mocks; these
check what they cannot: that the stored row round-trips, that the type filters
work against the real store, and that a run's subject can be listed by.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from nemo_agents_plugin import plugin_managed_agents
from nemo_agents_plugin.fabric import adapters
from nemo_agents_plugin.sdk import AgentsResource
from nemo_agents_plugin.service import AgentsService
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.plugin_managed_agents import (
    PluginManagedAgent,
    PluginManagedAgentInputError,
    PreparedRun,
)
from nemo_helix_plugin.agents.types import PluginManagedAgentSpec, RegisterAgentRequest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client, subprocess_job_executor_patch
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.integration


class _Context(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str


class _PreparedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    workspace: str


class _Analyst(PluginManagedAgent[_Context, _PreparedContext]):
    type_id: ClassVar[str] = "example.analyst"
    title: ClassVar[str] = "Example analyst"
    # A plugin adapter (installed by ``ctx``): a type's runs carry a context, which stock adapters drop.
    adapter_id: ClassVar[str] = "acme.example.analyst"
    post_run_actions: ClassVar[frozenset[str]] = frozenset({"agents.noop"})
    default_name: ClassVar[str] = "example-analyst"
    subject_label: ClassVar[str | None] = "Analyzed agent"
    default_input: ClassVar[str | None] = "Analyze the target agent."

    @classmethod
    def config(cls, *, name: str) -> dict[str, Any]:
        # The adapter takes its model from the config.
        return {**super().config(name=name), "models": {"default": {"provider": "nvidia", "model": "test/model"}}}

    @classmethod
    async def prepare(
        cls, context: _Context, *, workspace: str, client: AsyncNemoClient
    ) -> PreparedRun[_PreparedContext]:
        if context.target == "missing":
            raise PluginManagedAgentInputError("No traces for agent 'missing'.")
        return PreparedRun(
            context=_PreparedContext(target=context.target, workspace=workspace),
            subject=context.target,
        )


class _TestAgentsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(AgentsService())


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[ClientContext]:
    manifest = tmp_path / "nemo-fabric/adapters/acme/example-analyst.fabric-adapter.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"adapter_id": _Analyst.adapter_id, "config": {"accepts": ["models"]}}))
    monkeypatch.setattr(adapters, "ADAPTER_MANIFEST_ROOT", tmp_path / "nemo-fabric")
    monkeypatch.delenv(adapters.ADAPTER_PYTHON_ENV, raising=False)
    registered = {
        _Analyst.type_id: [SimpleNamespace(name=_Analyst.type_id, value="tests:_Analyst", load=lambda: _Analyst)]
    }
    monkeypatch.setattr(plugin_managed_agents, "_registered_entry_points", lambda: registered)
    # Pin the CPU job profile, which Jobs otherwise derives from whether Docker is running.
    with (
        subprocess_job_executor_patch(),
        create_test_client(
            _TestAgentsService, FilesService, JobsService, client_type=ClientContext, workspace="default"
        ) as ctx,
    ):
        yield ctx


def _create(ctx: ClientContext, name: str = "example-analyst") -> dict[str, Any]:
    body = RegisterAgentRequest(name=name, plugin_managed_agent=PluginManagedAgentSpec(type="example.analyst"))
    return (
        AgentsClient.from_client(ctx.client)
        .register_agent(workspace="default", body=body)
        .data()
        .model_dump(mode="json")
    )


def _execute(ctx: ClientContext, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    return AgentsResource(ctx.client).jobs.execute.create(name=name, spec=spec)


def test_an_agent_of_a_type_is_stored_by_type_and_read_with_its_config(ctx: ClientContext) -> None:
    created = _create(ctx)
    stored = AgentsResource(ctx.client).get("example-analyst")

    for agent in (created, stored):
        assert agent["plugin_managed_agent"] == {"type": "example.analyst"}
        assert agent["config"]["harnesses"]["default"]["kind"] == "acme.example.analyst"
        assert agent["context_schema"] == _Analyst.context_schema()
        assert agent["default_input"] == "Analyze the target agent."
        assert agent["post_run_actions"] == ["agents.noop"]

    client = AgentsClient.from_client(ctx.client)
    by_type = client.list_agents(
        workspace="default", query_params={"filter": json.dumps({"plugin_managed_agent.type": "example.analyst"})}
    )
    assert [agent.name for agent in by_type.page().items] == ["example-analyst"]
    [type_info] = client.list_plugin_managed_agent_types(workspace="default").data()
    assert type_info.type == "example.analyst"
    assert type_info.agent == "example-analyst"


def test_a_workspace_holds_one_agent_per_type(ctx: ClientContext) -> None:
    _create(ctx)

    with pytest.raises(NemoHTTPError) as excinfo:
        _create(ctx, name="second-analyst")

    assert excinfo.value.status_code == 409
    assert "already has an agent of type 'example.analyst'" in str(excinfo.value)


def test_a_run_records_its_prepared_context_and_can_be_listed_by_subject(ctx: ClientContext) -> None:
    _create(ctx)

    job = _execute(ctx, "analyze-bot", {"agent": "example-analyst", "context": {"target": "bot"}})
    _execute(ctx, "analyze-other", {"agent": "example-analyst", "context": {"target": "other"}})

    spec = job["spec"]
    assert spec["request"]["context"] == {"target": "bot"}
    assert spec["context"] == {"target": "bot", "workspace": "default"}
    assert spec["subject"] == "bot"
    assert spec["plugin_managed_agent_type"] == "example.analyst"
    assert spec["input"] == "Analyze the target agent."
    # The stored spec records the profile the run used.
    assert spec["job_profile"] == "default"
    runs = AgentsClient.from_client(ctx.client).list_agent_jobs(
        workspace="default",
        collection="execute",
        query_params={
            "filter": json.dumps({"spec.plugin_managed_agent_type": "example.analyst", "spec.subject": "bot"})
        },
    )
    assert [run.name for run in runs.page().items] == ["analyze-bot"]


def test_a_context_prepare_rejects_fails_the_create_request(ctx: ClientContext) -> None:
    _create(ctx)

    with pytest.raises(NemoHTTPError) as excinfo:
        _execute(ctx, "analyze-missing", {"agent": "example-analyst", "context": {"target": "missing"}})

    assert 400 <= excinfo.value.status_code < 500
    assert "No traces for agent 'missing'." in str(excinfo.value)
