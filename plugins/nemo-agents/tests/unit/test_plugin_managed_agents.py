# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-managed agent types, host side: loading, derivation, the agent routes and execute jobs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin import plugin_managed_agents
from nemo_agents_plugin.api.v2 import agents as agents_router_module
from nemo_agents_plugin.api.v2 import plugin_managed_agent_types as types_router_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client, get_files_client
from nemo_agents_plugin.api.v2.deployments import _resolve_deployment_config
from nemo_agents_plugin.entities import NEMO_AGENTS_SPEC_CONFIG_FORMAT, Agent, PluginManagedAgentSpec
from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult
from nemo_agents_plugin.jobs.execute import (
    ExecuteAgentJob,
    ExecuteAgentJobConfig,
    ExecuteAgentStepConfig,
    ResolvedAgentConfig,
)
from nemo_agents_plugin.plugin_managed_agents import (
    PluginManagedAgentTypeError,
    derive_agent,
    list_plugin_managed_agent_types,
    load_plugin_managed_agent_type,
)
from nemo_agents_plugin.schema import AgentFilter
from nemo_helix_plugin.agents.plugin_managed_agents import (
    CONFIGURED_DEFAULT_MODEL,
    PluginManagedAgent,
    PluginManagedAgentInputError,
    PreparedRun,
    helix_default,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.entity_client import NemoPaginationInfo
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator
from nemo_helix_plugin.job_context import JobContext
from pydantic import BaseModel, ConfigDict, Field

InstallTypes = Callable[..., None]

# A plugin adapter the test types run on. A type's runs always carry a context,
# which the stock adapters drop, so a type cannot run on one.
TEST_ADAPTER = "acme.example.analyst"


@pytest.fixture(autouse=True)
def _test_adapter_installed(installed_adapter_manifests: Path) -> None:
    """Install TEST_ADAPTER, which takes its model from the config like most adapters."""
    manifest = installed_adapter_manifests / "adapters/acme/example-analyst.fabric-adapter.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"adapter_id": TEST_ADAPTER, "config": {"accepts": ["models"]}}), encoding="utf-8")


def _with_a_model(config: dict[str, Any]) -> dict[str, Any]:
    """TEST_ADAPTER takes its model from the config, so a test type running it must name one."""
    return {**config, "models": {"default": {"provider": "nvidia", "model": "test/model"}}}


class AnalystContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    model: str = Field(json_schema_extra=helix_default(CONFIGURED_DEFAULT_MODEL))
    since: str | None = None


class AnalystPreparedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    model: str
    notes: str | None = None


class FakeAnalyst(PluginManagedAgent[AnalystContext, AnalystPreparedContext]):
    type_id: ClassVar[str] = "example.analyst"
    title: ClassVar[str] = "Example analyst"
    description: ClassVar[str] = "Analyzes an agent."
    adapter_id: ClassVar[str] = TEST_ADAPTER
    post_run_actions: ClassVar[frozenset[str]] = frozenset({"agents.noop"})
    default_name: ClassVar[str] = "example-analyst"
    subject_label: ClassVar[str | None] = "Analyzed agent"
    default_input: ClassVar[str | None] = "Analyze the target agent."

    calls: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def config(cls, *, name: str) -> dict[str, Any]:
        return _with_a_model(super().config(name=name))

    @classmethod
    async def prepare(
        cls, context: AnalystContext, *, workspace: str, client: AsyncNemoClient
    ) -> PreparedRun[AnalystPreparedContext]:
        cls.calls.append({"context": context, "workspace": workspace, "client": client})
        if context.target == "missing":
            raise PluginManagedAgentInputError("No agent named 'missing' has traces here.")
        if context.target == "forbidden":
            raise PermissionDeniedError(httpx.Response(403, json={"detail": "nope"}))
        if context.target == "broken":
            return PreparedRun(context=context)  # ty: ignore[invalid-return-type]
        model = context.model if "/" in context.model else f"{workspace}/{context.model}"
        return PreparedRun(
            context=AnalystPreparedContext(target=context.target, model=model, notes="prepared"),
            subject=context.target,
            job_profile="analyst-profile",
        )


class _SimpleContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str


class FakeReporter(PluginManagedAgent[_SimpleContext]):
    type_id: ClassVar[str] = "example.reporter"
    title: ClassVar[str] = "Example reporter"
    adapter_id: ClassVar[str] = TEST_ADAPTER
    post_run_actions: ClassVar[frozenset[str]] = frozenset({"agents.noop"})
    default_name: ClassVar[str] = "example-reporter"

    @classmethod
    def config(cls, *, name: str) -> dict[str, Any]:
        return _with_a_model(super().config(name=name))


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    FakeAnalyst.calls.clear()


def _stored(name: str = "example-analyst", type_id: str = "example.analyst", workspace: str = "default") -> Agent:
    """A plugin-managed agent as the entity store returns it: no config of its own."""
    agent = Agent(
        name=name,
        workspace=workspace,
        config={},
        config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        plugin_managed_agent=PluginManagedAgentSpec(type=type_id),
    )
    agent._id = f"agent-{name}-id"
    return agent


def _list_response(items: list[Agent], *, page: int = 1, total_pages: int = 1) -> MagicMock:
    response = MagicMock()
    response.data = items
    response.pagination = NemoPaginationInfo(
        page=page, page_size=20, current_page_size=len(items), total_pages=total_pages, total_results=len(items)
    )
    return response


def _raise_from_config(cls: Any, *, name: str) -> dict[str, Any]:
    raise RuntimeError("plugin bug")


# ---------------------------------------------------------------------------
# Loading and load-time checks
# ---------------------------------------------------------------------------


class TestLoading:
    def test_loads_an_installed_type(self, install_plugin_managed_agent_types: InstallTypes) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        assert load_plugin_managed_agent_type("example.analyst") is FakeAnalyst

    def test_an_unknown_type_names_what_is_installed(self, install_plugin_managed_agent_types: InstallTypes) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        with pytest.raises(PluginManagedAgentTypeError, match=r"Unknown .*'nope.x'.*example\.analyst"):
            load_plugin_managed_agent_type("nope.x")

    def test_the_same_implementation_registered_twice_loads(
        self, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        # The aggregate wheel re-declares every bundled plugin's entry points.
        install_plugin_managed_agent_types(FakeAnalyst, FakeAnalyst)

        assert load_plugin_managed_agent_type("example.analyst") is FakeAnalyst

    def test_two_implementations_of_one_type_conflict(self, install_plugin_managed_agent_types: InstallTypes) -> None:
        install_plugin_managed_agent_types(FakeAnalyst, ("example.analyst", FakeReporter))

        with pytest.raises(PluginManagedAgentTypeError, match="Conflicting"):
            load_plugin_managed_agent_type("example.analyst")

    def test_listing_skips_a_broken_type(self, install_plugin_managed_agent_types: InstallTypes) -> None:
        install_plugin_managed_agent_types(FakeAnalyst, ("example.mismatch", FakeReporter))

        assert list_plugin_managed_agent_types() == [FakeAnalyst]


def _variant(base: type[PluginManagedAgent], **attributes: Any) -> type[PluginManagedAgent]:
    return type(f"{base.__name__}Variant", (base,), attributes)


class _OpenContext(BaseModel):
    topic: str


class _OptionalContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str | None = None


class _ReservedKeyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nemo_helix: str


class _Nested(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class _NestedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nested: _Nested


def _with_context(model: type[BaseModel]) -> type[PluginManagedAgent]:
    class _Typed(PluginManagedAgent[model]):  # ty: ignore[invalid-type-form]
        type_id: ClassVar[str] = "example.reporter"
        title: ClassVar[str] = "Typed"
        adapter_id: ClassVar[str] = TEST_ADAPTER
        post_run_actions: ClassVar[frozenset[str]] = frozenset({"agents.noop"})
        default_name: ClassVar[str] = "example-reporter"

        @classmethod
        def config(cls, *, name: str) -> dict[str, Any]:
            return _with_a_model(super().config(name=name))

    return _Typed


@pytest.mark.parametrize(
    "type_cls, problem",
    [
        (object, "PluginManagedAgent subclass"),
        (_variant(FakeReporter, type_id="example.other"), "must match its entry point name"),
        (_variant(FakeReporter, title=""), "must set title"),
        (_variant(FakeReporter, default_name="Not A Name"), "not a valid agent name"),
        (_variant(FakeReporter, post_run_actions={"agents.noop"}), "frozenset"),
        (_with_context(_OpenContext), 'extra="forbid"'),
        (_with_context(_ReservedKeyContext), "reserves"),
        (_with_context(_NestedContext), r"\$ref"),
        (_variant(FakeReporter, adapter_id="nvidia.nowhere"), "not an installed Fabric adapter"),
        (
            _variant(
                FakeReporter,
                config=classmethod(
                    lambda cls, *, name: {**PluginManagedAgent.config.__func__(cls, name=name), "default_harness": None}
                ),
            ),
            "config\\(\\) is not a valid",
        ),
        (
            _variant(
                FakeReporter,
                config=classmethod(
                    lambda cls, *, name: {
                        **PluginManagedAgent.config.__func__(cls, name=name),
                        "harnesses": {"default": {"kind": "codex"}},
                    }
                ),
            ),
            "default harness runs 'nvidia.fabric.codex'",
        ),
        (
            # The default config names no model, which the test adapter needs.
            _variant(FakeReporter, config=classmethod(PluginManagedAgent.config.__func__)),
            "cannot run on Fabric: .*no models.default",
        ),
        (
            _variant(FakeReporter, adapter_id="nvidia.fabric.claude"),
            "context declares topic, but its adapter 'nvidia.fabric.claude' does not read the request context",
        ),
        (
            # An optional field is just as unreachable as a required one.
            _variant(_with_context(_OptionalContext), adapter_id="nvidia.fabric.claude"),
            "context declares topic, but its adapter 'nvidia.fabric.claude' does not read",
        ),
        (_variant(FakeReporter, post_run_actions=frozenset({"example.uninstalled"})), "post-run action"),
        # A bug in the plugin's own code makes the type unusable rather than failing every read.
        (_variant(FakeReporter, config=classmethod(_raise_from_config)), "failed its load checks: plugin bug"),
    ],
)
def test_a_broken_type_fails_to_load_naming_the_problem(
    install_plugin_managed_agent_types: InstallTypes, type_cls: Any, problem: str
) -> None:
    install_plugin_managed_agent_types(("example.reporter", type_cls))

    with pytest.raises(PluginManagedAgentTypeError, match=problem):
        load_plugin_managed_agent_type("example.reporter")


class _EmptyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")


def test_a_type_without_context_fields_may_run_on_a_stock_adapter(
    install_plugin_managed_agent_types: InstallTypes,
) -> None:
    """Such a type only manages its agent's config; its runs take their prompt from input."""
    type_cls = _variant(_with_context(_EmptyContext), adapter_id="nvidia.fabric.claude")
    install_plugin_managed_agent_types(("example.reporter", type_cls))

    assert load_plugin_managed_agent_type("example.reporter") is type_cls


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


class TestDerivation:
    def test_fills_what_the_type_supplies(self, install_plugin_managed_agent_types: InstallTypes) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        derived = derive_agent(_stored(name="my-analyst"))

        assert derived.config == FakeAnalyst.config(name="my-analyst")
        assert derived.config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT
        assert derived.context_schema == FakeAnalyst.context_schema()
        assert derived.default_input == "Analyze the target agent."
        assert derived.post_run_actions == {"agents.noop"}
        assert derived.id == "agent-my-analyst-id"

    def test_an_agent_whose_type_is_not_installed_reads_as_stored(
        self, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types()
        stored = _stored()

        assert derive_agent(stored) is stored

    def test_an_ordinary_agent_passes_through(self) -> None:
        agent = Agent(name="calc", workspace="default")

        assert derive_agent(agent) is agent

    def test_a_deployment_runs_the_derived_config(self, install_plugin_managed_agent_types: InstallTypes) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        config = _resolve_deployment_config(_stored(), workspace="default")

        assert config["harnesses"]["default"]["kind"] == TEST_ADAPTER


# ---------------------------------------------------------------------------
# Agent routes
# ---------------------------------------------------------------------------


@pytest.fixture
def entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    for router in (agents_router_module.router, types_router_module.router):
        app.include_router(router, prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    app.dependency_overrides[get_files_client] = lambda: AsyncMock(spec=AsyncFilesClient)
    return TestClient(app, raise_server_exceptions=False)


_AGENTS = "/apis/agents/v2/workspaces/default/agents"
_TYPES = "/apis/agents/v2/workspaces/default/plugin-managed-agent-types"


class TestRegister:
    def test_stores_only_the_name_and_type_and_returns_the_derived_agent(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)
        entity_client.list.return_value = _list_response([])
        entity_client.create.side_effect = lambda agent: agent

        response = client.post(
            _AGENTS,
            json={"name": "my-analyst", "labels": {"team": "x"}, "plugin_managed_agent": {"type": "example.analyst"}},
        )

        assert response.status_code == 201, response.text
        stored: Agent = entity_client.create.call_args.args[0]
        assert stored.config == {}
        assert stored.config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT
        assert stored.context_schema is None
        assert stored.plugin_managed_agent == PluginManagedAgentSpec(type="example.analyst")
        assert stored.labels == {"team": "x"}
        body = response.json()
        assert body["config"] == FakeAnalyst.config(name="my-analyst")
        assert body["context_schema"] == FakeAnalyst.context_schema()
        assert body["plugin_managed_agent"] == {"type": "example.analyst"}
        lookup = entity_client.list.call_args.kwargs["filter_operation"]
        assert lookup == ComparisonOperation(
            operator=FilterOperator.EQ, field="data.plugin_managed_agent.type", value="example.analyst"
        )

    def test_rejects_a_second_agent_of_the_type(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)
        entity_client.list.return_value = _list_response([_stored(name="first")])

        response = client.post(_AGENTS, json={"name": "second", "plugin_managed_agent": {"type": "example.analyst"}})

        assert response.status_code == 409
        assert "already has an agent of type 'example.analyst': 'first'" in response.json()["detail"]
        entity_client.create.assert_not_called()

    def test_rejects_an_uninstalled_type(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        response = client.post(_AGENTS, json={"name": "x", "plugin_managed_agent": {"type": "example.nothing"}})

        assert response.status_code == 400
        assert "Unknown plugin-managed agent type 'example.nothing'" in response.json()["detail"]
        entity_client.create.assert_not_called()

    def test_rejects_a_config_alongside_a_type(self, client: TestClient) -> None:
        response = client.post(
            _AGENTS,
            json={"name": "x", "plugin_managed_agent": {"type": "example.analyst"}, "default_input": "go"},
        )

        assert response.status_code == 422


class TestRead:
    def test_get_derives_a_plugin_managed_agent(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)
        entity_client.get.return_value = _stored()

        body = client.get(f"{_AGENTS}/example-analyst").json()

        assert body["config"] == FakeAnalyst.config(name="example-analyst")
        assert body["post_run_actions"] == ["agents.noop"]

    def test_list_derives_plugin_managed_agents_and_leaves_others(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)
        ordinary = Agent(name="calc", workspace="default", config={"workflow": {}})
        entity_client.list.return_value = _list_response(
            [_stored(), ordinary, _stored(name="orphan", type_id="gone.x")]
        )

        data = client.get(_AGENTS).json()["data"]

        assert data[0]["config"] == FakeAnalyst.config(name="example-analyst")
        assert data[1]["config"] == {"workflow": {}}
        # An agent whose type is no longer installed still lists, as stored.
        assert data[2]["config"] == {}
        assert data[2]["plugin_managed_agent"] == {"type": "gone.x"}

    def test_the_type_filter_targets_the_stored_field(self) -> None:
        translated = AgentFilter.translate_operation(
            ComparisonOperation(operator=FilterOperator.EQ, field="plugin_managed_agent.type", value="example.analyst")
        )

        assert translated == ComparisonOperation(
            operator=FilterOperator.EQ, field="data.plugin_managed_agent.type", value="example.analyst"
        )

    def test_list_route_accepts_the_type_filter(self, client: TestClient, entity_client: AsyncMock) -> None:
        entity_client.list.return_value = _list_response([])

        response = client.get(_AGENTS, params={"filter[plugin_managed_agent.type]": "example.analyst"})

        assert response.status_code == 200, response.text
        operation = entity_client.list.call_args.kwargs["filter_operation"]
        assert operation.field == "data.plugin_managed_agent.type"
        assert operation.value == "example.analyst"


class TestTypesRoute:
    def test_lists_types_with_this_workspaces_agent_of_each(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst, FakeReporter)
        entity_client.list.return_value = _list_response([_stored(name="my-analyst")])

        response = client.get(_TYPES)

        assert response.status_code == 200, response.text
        analyst, reporter = response.json()
        assert analyst == {
            "type": "example.analyst",
            "title": "Example analyst",
            "description": "Analyzes an agent.",
            "default_name": "example-analyst",
            "adapter_id": TEST_ADAPTER,
            "context_schema": FakeAnalyst.context_schema(),
            "post_run_actions": ["agents.noop"],
            "subject_label": "Analyzed agent",
            "agent": "my-analyst",
        }
        assert reporter["type"] == "example.reporter"
        assert reporter["agent"] is None
        lookup = entity_client.list.call_args.kwargs["filter_operation"]
        assert lookup.field == "data.plugin_managed_agent.type"
        assert sorted(lookup.value) == ["example.analyst", "example.reporter"]

    def test_finds_agents_beyond_the_first_page(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst, FakeReporter)
        entity_client.list.side_effect = [
            _list_response([_stored(name="my-analyst")], page=1, total_pages=2),
            _list_response([_stored(name="my-reporter", type_id="example.reporter")], page=2, total_pages=2),
        ]

        analyst, reporter = client.get(_TYPES).json()

        assert (analyst["agent"], reporter["agent"]) == ("my-analyst", "my-reporter")
        assert [call.kwargs["page"] for call in entity_client.list.call_args_list] == [1, 2]

    def test_lists_nothing_when_no_types_are_installed(
        self, client: TestClient, entity_client: AsyncMock, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types()

        assert client.get(_TYPES).json() == []
        entity_client.list.assert_not_called()


# ---------------------------------------------------------------------------
# Execute jobs
# ---------------------------------------------------------------------------


async def _to_spec(agent: Agent, *, sdk: Any = None, **request: Any) -> ExecuteAgentStepConfig:
    entity_client = AsyncMock()
    entity_client.get.return_value = agent
    spec = await ExecuteAgentJob.to_spec(
        ExecuteAgentJobConfig(agent=agent.name, **request),
        workspace="default",
        entity_client=entity_client,
        async_sdk=sdk if sdk is not None else MagicMock(),
        is_local=False,
    )
    return ExecuteAgentStepConfig.model_validate(spec)


class TestExecute:
    @pytest.mark.asyncio
    async def test_prepares_the_run_as_the_caller_and_snapshots_it(
        self, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)
        sdk = MagicMock()
        request_context = {"target": "bot", "model": "fast-one"}

        step = await _to_spec(_stored(), sdk=sdk, context=request_context)

        [call] = FakeAnalyst.calls
        assert call["context"] == AnalystContext(target="bot", model="fast-one")
        assert call["workspace"] == "default"
        assert call["client"] is sdk
        assert step.request.context == request_context
        assert step.context == {"target": "bot", "model": "default/fast-one", "notes": "prepared"}
        assert step.subject == "bot"
        assert step.job_profile == "analyst-profile"
        assert step.plugin_managed_agent_type == "example.analyst"
        assert step.agent_ref == "default/example-analyst"
        assert step.input == "Analyze the target agent."
        assert step.agent.config["harnesses"]["default"]["kind"] == TEST_ADAPTER
        assert step.post_run_action.kind == "agents.noop"

    @pytest.mark.asyncio
    async def test_the_callers_input_overrides_the_types_default_input(
        self, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        step = await _to_spec(_stored(), context={"target": "bot", "model": "m"}, input="Only look at errors.")

        assert step.input == "Only look at errors."

    @pytest.mark.asyncio
    async def test_input_is_required_when_the_type_declares_no_default_input(
        self, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeReporter)

        with pytest.raises(
            ValueError, match="'input' is required: agent 'default/example-reporter' declares no default_input"
        ):
            await _to_spec(_stored(name="example-reporter", type_id="example.reporter"), context={"topic": "x"})

    @pytest.mark.asyncio
    async def test_a_single_model_type_runs_the_callers_context(
        self, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types(FakeReporter)

        step = await _to_spec(
            _stored(name="example-reporter", type_id="example.reporter"), context={"topic": "x"}, input="Go."
        )

        assert step.context == {"topic": "x"}
        assert step.subject is None
        assert step.job_profile is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "type_cls, requested, expected",
        [
            (FakeAnalyst, None, "analyst-profile"),
            (FakeAnalyst, "sandbox", "sandbox"),
            (FakeReporter, None, "default"),
        ],
    )
    async def test_runs_on_the_requested_profile_then_the_types(
        self,
        install_plugin_managed_agent_types: InstallTypes,
        type_cls: type[PluginManagedAgent],
        requested: str | None,
        expected: str,
    ) -> None:
        install_plugin_managed_agent_types(type_cls)
        step = await _to_spec(
            _stored(name=type_cls.default_name, type_id=type_cls.type_id),
            context={"target": "bot", "model": "m"} if type_cls is FakeAnalyst else {"topic": "x"},
            input="Go.",
        )

        compiled = await ExecuteAgentJob.compile(
            workspace="default",
            spec=step,
            entity_client=MagicMock(),
            job_name=None,
            async_sdk=MagicMock(),
            profile=requested,
        )

        assert compiled.steps[0].executor.profile == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "context, error",
        [
            ({"target": "bot"}, "'model' is a required property"),
            ({"target": "bot", "model": "m", "extra": 1}, "'extra' was unexpected"),
        ],
    )
    async def test_rejects_a_context_that_does_not_match_the_type(
        self, install_plugin_managed_agent_types: InstallTypes, context: dict[str, Any], error: str
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        with pytest.raises(ValueError, match=error):
            await _to_spec(_stored(), context=context)
        assert FakeAnalyst.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "target, error_type, message",
        [
            ("missing", ValueError, "No agent named 'missing'"),
            ("forbidden", PermissionError, "403"),
            ("broken", ValueError, "is broken: prepare\\(\\) must return"),
        ],
    )
    async def test_maps_prepare_failures_for_the_caller(
        self,
        install_plugin_managed_agent_types: InstallTypes,
        target: str,
        error_type: type[Exception],
        message: str,
    ) -> None:
        install_plugin_managed_agent_types(FakeAnalyst)

        with pytest.raises(error_type, match=message):
            await _to_spec(_stored(), context={"target": target, "model": "m"})

    @pytest.mark.asyncio
    async def test_refuses_an_agent_whose_type_is_not_installed(
        self, install_plugin_managed_agent_types: InstallTypes
    ) -> None:
        install_plugin_managed_agent_types()

        with pytest.raises(ValueError, match="of plugin-managed agent type 'example.analyst', which cannot run"):
            await _to_spec(_stored(), context={"target": "bot", "model": "m"})

    def test_run_hands_the_prepared_context_to_the_agent_and_the_action(
        self, ctx: JobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        step = ExecuteAgentStepConfig(
            request=ExecuteAgentJobConfig(agent="example-analyst", context={"target": "bot", "model": "m"}),
            agent=ResolvedAgentConfig(
                name="example-analyst",
                workspace="default",
                config=FakeAnalyst.config(name="example-analyst"),
                config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
            ),
            input="Analyze bot.",
            context={"target": "bot", "model": "default/m", "notes": "prepared"},
            subject="bot",
            plugin_managed_agent_type="example.analyst",
        )
        seen: dict[str, Any] = {}

        async def _invoke(request: Any) -> FabricRuntimeResult:
            seen["caller_context"] = request.caller_context
            return FabricRuntimeResult(status="succeeded")

        class _Action:
            def after_invoke(self, context: Any) -> None:
                seen["run_context"] = context.run_context

        monkeypatch.setattr("nemo_agents_plugin.jobs.execute.invoke_agent_config_request_once", _invoke)
        monkeypatch.setattr("nemo_agents_plugin.jobs.execute.resolve_post_run_action", lambda kind: _Action)

        ExecuteAgentJob().run(step.model_dump(mode="json"), ctx=ctx)

        assert seen["caller_context"] == {
            "target": "bot",
            "model": "default/m",
            "notes": "prepared",
            "nemo_helix": {"job_id": ctx.job_id, "job_workspace": ctx.workspace, "agent": "default/example-analyst"},
        }
        assert seen["run_context"] == {"target": "bot", "model": "default/m", "notes": "prepared"}

    def test_a_step_stored_before_it_recorded_its_context_runs_the_requests(self) -> None:
        stored = {
            "request": {"agent": "calc", "input": "hi", "context": {"since": "2026-10-01"}},
            "agent": {"name": "calc", "workspace": "default", "config": {}, "config_format": "nemo-agents-spec-v1"},
            "input": "hi",
        }

        step = ExecuteAgentStepConfig.model_validate(stored)

        assert step.context == {"since": "2026-10-01"}
        assert step.subject is None
        assert step.plugin_managed_agent_type is None


def test_the_types_module_reads_its_own_entry_point_group(monkeypatch: pytest.MonkeyPatch) -> None:
    groups: list[str] = []
    monkeypatch.setattr(plugin_managed_agents, "entry_points", lambda group: groups.append(group) or [])

    assert plugin_managed_agents._registered_entry_points() == {}
    assert groups == ["nemo.agents.plugin-managed-agent-types"]
