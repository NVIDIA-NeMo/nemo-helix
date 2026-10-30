# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent run declarations, through the real services, entity store, and jobs routes.

Unit tests cover how a job resolves against an agent's declarations; these
check what they cannot: that the declarations survive the entity store, and
that a job the declarations reject fails the create request with a 4xx that
names the problem.
"""

from collections.abc import Iterator
from importlib.metadata import EntryPoint, entry_points
from typing import Any

import pytest
from nemo_agents_plugin.jobs import post_run_actions
from nemo_agents_plugin.sdk import AgentsResource
from nemo_agents_plugin.service import AgentsService
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.post_run_actions import POST_RUN_ACTION_ENTRY_POINT_GROUP
from nemo_helix_plugin.agents.types import RegisterAgentRequest
from nemo_helix_plugin.client.errors import NemoHTTPError
from nhx.core.files.service import FilesService
from nhx.core.jobs.service import JobsService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

pytestmark = pytest.mark.integration

_NOOP_TARGET = "nemo_agents_plugin.jobs.post_run_actions:NoopPostRunAction"

# Includes ``$``-prefixed keywords a document store might treat specially.
_CONTEXT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$comment": "Run parameters for the analyst.",
    "type": "object",
    "properties": {"since": {"type": "string", "format": "date-time"}},
    "required": ["since"],
    "additionalProperties": False,
}


class _TestAgentsService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(AgentsService())


def _agent_config(name: str) -> dict[str, Any]:
    return {
        "config_format": "nemo-agents-spec-v1",
        "name": name,
        "default_harness": "hermes",
        "harnesses": {"hermes": {"kind": "hermes"}},
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
    }


@pytest.fixture
def ctx() -> Iterator[ClientContext]:
    with create_test_client(
        _TestAgentsService, FilesService, JobsService, client_type=ClientContext, workspace="default"
    ) as ctx:
        yield ctx


def _register(ctx: ClientContext, name: str, **declarations: Any) -> dict[str, Any]:
    body = RegisterAgentRequest(name=name, config=_agent_config(name), **declarations)
    return (
        AgentsClient.from_client(ctx.client)
        .register_agent(workspace="default", body=body)
        .data()
        .model_dump(mode="json")
    )


def _execute(ctx: ClientContext, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    return AgentsResource(ctx.client).jobs.execute.create(name=name, spec=spec)


def _execute_error(ctx: ClientContext, name: str, spec: dict[str, Any]) -> NemoHTTPError:
    with pytest.raises(NemoHTTPError) as excinfo:
        _execute(ctx, name, spec)
    return excinfo.value


def test_declarations_survive_the_entity_store(ctx: ClientContext) -> None:
    created = _register(
        ctx,
        "analyst",
        context_schema=_CONTEXT_SCHEMA,
        default_input="Analyze recent runs.",
        post_run_actions={"insights.analysis", "agents.noop"},
    )
    stored = AgentsResource(ctx.client).get("analyst")

    for agent in (created, stored):
        assert agent["context_schema"] == _CONTEXT_SCHEMA
        assert agent["default_input"] == "Analyze recent runs."
        assert agent["post_run_actions"] == ["agents.noop", "insights.analysis"]


def test_an_agent_without_declarations_allows_only_noop(ctx: ClientContext) -> None:
    _register(ctx, "calc")

    assert AgentsResource(ctx.client).get("calc")["post_run_actions"] == ["agents.noop"]
    error = _execute_error(
        ctx, "run-calc", {"agent": "calc", "input": "hello", "post_run_action": {"kind": "insights.analysis"}}
    )
    assert 400 <= error.status_code < 500
    assert "'insights.analysis' is not one this agent declares" in str(error)


def test_register_rejects_a_context_schema_with_a_reference(ctx: ClientContext) -> None:
    schema = {"properties": {"since": {"$ref": "http://169.254.169.254/latest/meta-data/schema.json"}}}

    with pytest.raises(NemoHTTPError) as excinfo:
        _register(ctx, "analyst", context_schema=schema)

    assert excinfo.value.status_code == 400
    assert "context_schema does not support $ref" in str(excinfo.value)


def test_register_rejects_an_uninstalled_post_run_action(ctx: ClientContext) -> None:
    with pytest.raises(NemoHTTPError) as excinfo:
        _register(ctx, "calc", post_run_actions={"acme.missing"})

    assert excinfo.value.status_code == 400
    assert "Unknown post-run action 'acme.missing'" in str(excinfo.value)


def test_a_job_that_omits_input_stores_and_reads_back_the_default(ctx: ClientContext) -> None:
    _register(ctx, "analyst", context_schema=_CONTEXT_SCHEMA, default_input="Analyze recent runs.")

    created = _execute(ctx, "run-analyst", {"agent": "analyst", "context": {"since": "2026-10-01T00:00:00Z"}})
    fetched = AgentsResource(ctx.client).jobs.execute.get("run-analyst")

    for job in (created, fetched):
        assert job["spec"]["input"] == "Analyze recent runs."
        assert job["spec"]["request"].get("input") is None
        assert job["spec"]["post_run_action"]["kind"] == "agents.noop"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        pytest.param(
            {"context": {"since": "2026-10-01T00:00:00Z", "sinse": "x"}},
            "'sinse' was unexpected",
            id="unknown-context-key",
        ),
        pytest.param({"context": {"since": "yesterday"}}, "is not a 'date-time'", id="malformed-date-time"),
        pytest.param({"context": {}}, "'since' is a required property", id="missing-required-key"),
    ],
)
def test_a_context_that_does_not_match_the_schema_fails_job_create(
    ctx: ClientContext, spec: dict[str, Any], message: str
) -> None:
    _register(ctx, "analyst", context_schema=_CONTEXT_SCHEMA, default_input="Analyze recent runs.")

    error = _execute_error(ctx, "run-analyst", {"agent": "analyst", **spec})

    assert 400 <= error.status_code < 500
    assert message in str(error)


def test_omitting_input_fails_job_create_without_a_default(ctx: ClientContext) -> None:
    _register(ctx, "calc")

    error = _execute_error(ctx, "run-calc", {"agent": "calc"})

    assert 400 <= error.status_code < 500
    assert "agent 'default/calc' declares no default_input" in str(error)


@pytest.fixture
def _two_report_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install two extra kinds beside the real ones, so a declaration can offer several without noop."""
    installed = list(entry_points(group=POST_RUN_ACTION_ENTRY_POINT_GROUP))
    extra = [
        EntryPoint(name=kind, value=_NOOP_TARGET, group=POST_RUN_ACTION_ENTRY_POINT_GROUP)
        for kind in ("acme.report", "acme.digest")
    ]
    monkeypatch.setattr(post_run_actions, "entry_points", lambda group: [*installed, *extra])


@pytest.mark.usefixtures("_two_report_actions")
@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        pytest.param({"agents.noop", "acme.report"}, "agents.noop", id="noop-declared"),
        pytest.param({"acme.report"}, "acme.report", id="single-member"),
        pytest.param({"acme.report", "acme.digest"}, None, id="several-must-choose"),
    ],
)
def test_an_omitted_post_run_action_resolves_against_the_stored_declaration(
    ctx: ClientContext, declared: set[str], expected: str | None
) -> None:
    _register(ctx, "analyst", post_run_actions=declared)
    spec = {"agent": "analyst", "input": "hello"}

    if expected is None:
        error = _execute_error(ctx, "run-analyst", spec)
        assert 400 <= error.status_code < 500
        assert "set post_run_action.kind to one of: acme.digest, acme.report" in str(error)
        return
    assert _execute(ctx, "run-analyst", spec)["spec"]["post_run_action"]["kind"] == expected
