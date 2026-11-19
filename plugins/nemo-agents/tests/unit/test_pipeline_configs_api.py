# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pipeline config routes, workspace validation, the CLI group and the example file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import pipeline_configs as pipeline_configs_router_module
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.cli import AgentsCLI
from nemo_agents_plugin.entities import Agent, AgentPipelineConfig
from nemo_agents_plugin.pipeline_validation import check_pipeline_config, static_issues
from nemo_helix_plugin.agents.types import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT
from nemo_helix_plugin.client.errors import AuthenticationError, PermissionDeniedError
from nemo_helix_plugin.entity_client import (
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
)
from typer.testing import CliRunner

NOW = datetime.now(timezone.utc)
BASE = "/apis/agents/v2/workspaces/default/pipeline-configs"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "pipeline-configs" / "pr-pipeline.yaml"

PIPELINE: dict[str, Any] = {
    "version": 1,
    "inputs": {"input": {}},
    "name": "email-triage",
    "description": "Triage then reply.",
    "steps": [
        {"name": "triage", "agent": "triage-agent", "response_format": "json"},
        {
            "name": "reply",
            "agent": "default/reply-agent",
            "input": "Verdict: {{ previous.response.verdict }} / {{ input }}",
            "context": {"indicators": "{{ steps.triage.response.indicators }}", "tone": "formal"},
        },
    ],
}


def _stamp(entity: AgentPipelineConfig) -> AgentPipelineConfig:
    entity._id = f"agent_pipeline_config-{entity.name}-id"
    entity._created_at = NOW
    return entity


def _stored(data: dict[str, Any] | None = None) -> AgentPipelineConfig:
    return _stamp(AgentPipelineConfig.model_validate({**(data or PIPELINE), "workspace": "default"}))


def _agent(
    name: str,
    *,
    workspace: str = "default",
    config_format: str = NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    schema: dict | None = None,
) -> Agent:
    return Agent(
        name=name, workspace=workspace, config={"name": name}, config_format=config_format, context_schema=schema
    )


def _agents_by_name(
    *agents: Agent, denied_workspaces: tuple[str, ...] = (), unauthenticated: bool = False
) -> AsyncMock:
    by_ref = {(agent.workspace, agent.name): agent for agent in agents}

    async def get(entity_type: type, *, name: str, workspace: str) -> Any:
        if unauthenticated:
            raise AuthenticationError(httpx.Response(401, request=httpx.Request("GET", "http://entities")))
        if workspace in denied_workspaces:
            raise PermissionDeniedError(httpx.Response(403, request=httpx.Request("GET", "http://entities")))
        if entity_type is Agent and (workspace, name) in by_ref:
            return by_ref[(workspace, name)]
        raise NemoEntityNotFoundError(name)

    entity_client = AsyncMock()
    entity_client.get = AsyncMock(side_effect=get)
    return entity_client


def _test_client(entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(pipeline_configs_router_module.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# CRUD routes
# ---------------------------------------------------------------------------


class TestPipelineRoutes:
    def test_create(self) -> None:
        entity_client = AsyncMock()
        entity_client.create = AsyncMock(side_effect=_stamp)

        resp = _test_client(entity_client).post(BASE, json=PIPELINE)

        assert resp.status_code == 201, resp.text
        created: AgentPipelineConfig = entity_client.create.call_args[0][0]
        assert created.name == "email-triage"
        assert created.workspace == "default"
        assert [step.name for step in created.steps] == ["triage", "reply"]
        assert resp.json()["steps"][1]["context"]["indicators"] == "{{ steps.triage.response.indicators }}"

    def test_create_does_not_resolve_agents(self) -> None:
        entity_client = AsyncMock()
        entity_client.create = AsyncMock(side_effect=_stamp)
        _test_client(entity_client).post(BASE, json=PIPELINE)
        entity_client.get.assert_not_awaited()

    @pytest.mark.parametrize(
        "change",
        [
            {"steps": [{"name": "a", "agent": "x", "model": "gpt"}]},
            {"steps": [{"name": "a", "agent": "x", "input": "{{ steps.b.response }}"}, {"name": "b", "agent": "y"}]},
            {"version": 9},
        ],
    )
    def test_create_rejects_static_errors(self, change: dict[str, Any]) -> None:
        entity_client = AsyncMock()
        resp = _test_client(entity_client).post(BASE, json={**PIPELINE, **change})
        assert resp.status_code == 422
        entity_client.create.assert_not_awaited()

    def test_create_store_rejection_hides_the_error_text(self) -> None:
        entity_client = AsyncMock()
        entity_client.create = AsyncMock(side_effect=NemoEntityValidationError("internal detail"))
        resp = _test_client(entity_client).post(BASE, json=PIPELINE)
        assert resp.status_code == 422
        assert "internal detail" not in resp.text

    def test_create_conflict(self) -> None:
        entity_client = AsyncMock()
        entity_client.create = AsyncMock(side_effect=NemoEntityConflictError("exists"))
        assert _test_client(entity_client).post(BASE, json=PIPELINE).status_code == 409

    def test_get_and_not_found(self) -> None:
        entity_client = AsyncMock()
        entity_client.get = AsyncMock(return_value=_stored())
        resp = _test_client(entity_client).get(f"{BASE}/email-triage")
        assert resp.status_code == 200
        assert resp.json()["version"] == 1

        entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("gone"))
        assert _test_client(entity_client).get(f"{BASE}/email-triage").status_code == 404

    def test_list(self) -> None:
        entity_client = AsyncMock()
        entity_client.list = AsyncMock(return_value=type("R", (), {"data": [_stored()], "pagination": None})())
        resp = _test_client(entity_client).get(BASE)
        assert resp.status_code == 200
        assert [item["name"] for item in resp.json()["data"]] == ["email-triage"]

    def test_put_replaces_the_config(self) -> None:
        entity_client = AsyncMock()
        entity_client.get = AsyncMock(return_value=_stored())
        entity_client.update = AsyncMock(side_effect=lambda entity: entity)
        body = {"version": 1, "inputs": {"input": {}}, "steps": [{"name": "only", "agent": "x"}]}

        resp = _test_client(entity_client).put(f"{BASE}/email-triage", json=body)

        assert resp.status_code == 200, resp.text
        updated: AgentPipelineConfig = entity_client.update.call_args[0][0]
        assert [step.name for step in updated.steps] == ["only"]
        assert updated.description == ""  # replaced, not merged
        assert updated.name == "email-triage"

    def test_put_rejects_static_errors_without_updating(self) -> None:
        entity_client = AsyncMock()
        entity_client.get = AsyncMock(return_value=_stored())
        body = {
            "version": 1,
            "inputs": {"input": {}},
            "steps": [{"name": "a", "agent": "x", "input": "{{ steps.a.response }}"}],
        }
        resp = _test_client(entity_client).put(f"{BASE}/email-triage", json=body)
        assert resp.status_code == 422
        entity_client.update.assert_not_awaited()

    def test_put_store_rejection_hides_the_error_text(self) -> None:
        entity_client = AsyncMock()
        entity_client.get = AsyncMock(return_value=_stored())
        entity_client.update = AsyncMock(side_effect=NemoEntityValidationError("internal detail"))
        body = {"version": 1, "inputs": {"input": {}}, "steps": [{"name": "a", "agent": "x"}]}
        resp = _test_client(entity_client).put(f"{BASE}/email-triage", json=body)
        assert resp.status_code == 422
        assert "internal detail" not in resp.text

    @pytest.mark.parametrize(
        "stale_step",
        [
            {"name": "a", "agent": "x", "input": "{{ steps.zz.response }}"},  # a static check fails
            {"name": "a", "agent": "x", "retries": 9, "new_key": 1},  # a field rule fails, an unknown key
        ],
    )
    def test_stored_config_that_fails_current_rules_still_loads(self, stale_step: dict[str, Any]) -> None:
        # e.g. saved before a rule was tightened, or by a newer release: reads must keep working so it can be
        # fixed or deleted.
        stale = {**PIPELINE, "steps": [stale_step]}
        entity_client = AsyncMock()
        entity_client.get = AsyncMock(return_value=_stored(stale))
        entity_client.list = AsyncMock(return_value=SimpleNamespace(data=[_stored(stale)], pagination=None))
        client = _test_client(entity_client)
        assert client.get(f"{BASE}/email-triage").status_code == 200
        assert client.get(BASE).status_code == 200
        assert client.delete(f"{BASE}/email-triage").status_code == 204

    def test_put_rejects_a_rename(self) -> None:
        entity_client = AsyncMock()
        body = {"version": 1, "inputs": {"input": {}}, "name": "other", "steps": [{"name": "a", "agent": "x"}]}
        resp = _test_client(entity_client).put(f"{BASE}/email-triage", json=body)
        assert resp.status_code == 422
        assert "renaming is not supported" in resp.json()["detail"]
        entity_client.update.assert_not_awaited()

    def test_put_not_found_and_conflict(self) -> None:
        body = {"version": 1, "inputs": {"input": {}}, "steps": [{"name": "a", "agent": "x"}]}
        entity_client = AsyncMock()
        entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("gone"))
        assert _test_client(entity_client).put(f"{BASE}/email-triage", json=body).status_code == 404

        entity_client.get = AsyncMock(return_value=_stored())
        entity_client.update = AsyncMock(side_effect=NemoEntityConflictError("stale"))
        assert _test_client(entity_client).put(f"{BASE}/email-triage", json=body).status_code == 409

    def test_delete(self) -> None:
        entity_client = AsyncMock()
        entity_client.get = AsyncMock(return_value=_stored())
        assert _test_client(entity_client).delete(f"{BASE}/email-triage").status_code == 204
        entity_client.delete_by_id.assert_awaited_once()


# ---------------------------------------------------------------------------
# Validate route and workspace checks
# ---------------------------------------------------------------------------


class TestValidatePipeline:
    def test_valid_pipeline(self) -> None:
        entity_client = _agents_by_name(_agent("triage-agent"), _agent("reply-agent"))
        resp = _test_client(entity_client).post(f"{BASE}/validate", json=PIPELINE)
        assert resp.status_code == 200
        assert resp.json() == {"valid": True, "issues": []}

    def test_static_errors_are_reported_not_raised(self) -> None:
        entity_client = AsyncMock()
        body = {**PIPELINE, "steps": [{"name": "a", "agent": "x", "image": "i"}]}
        resp = _test_client(entity_client).post(f"{BASE}/validate", json=body)
        assert resp.status_code == 200
        result = resp.json()
        assert result["valid"] is False
        assert result["issues"][0]["location"] == "steps[0]"
        assert "run settings" in result["issues"][0]["message"]
        entity_client.get.assert_not_awaited()

    def test_unregistered_agent(self) -> None:
        entity_client = _agents_by_name(_agent("triage-agent"))
        result = _test_client(entity_client).post(f"{BASE}/validate", json=PIPELINE).json()
        assert result["valid"] is False
        assert result["issues"] == [
            {
                "location": "steps[reply].agent",
                "message": "Agent 'default/reply-agent' is not registered. Register it from its agent.yaml first.",
            }
        ]

    def test_agent_format_must_be_runnable(self) -> None:
        entity_client = _agents_by_name(
            _agent("triage-agent", config_format=NAT_WORKFLOW_CONFIG_FORMAT), _agent("reply-agent")
        )
        result = _test_client(entity_client).post(f"{BASE}/validate", json=PIPELINE).json()
        assert result["valid"] is False
        assert "supports only 'nemo-agents-spec-v1'" in result["issues"][0]["message"]

    async def test_context_schema_skips_templated_values_and_checks_literals(self) -> None:
        schema = {
            "type": "object",
            "properties": {"indicators": {"type": "array"}, "tone": {"enum": ["formal", "casual"]}},
            "required": ["indicators", "tone"],
        }
        entity_client = _agents_by_name(_agent("triage-agent"), _agent("reply-agent", schema=schema))
        valid = await check_pipeline_config(PIPELINE, workspace="default", entity_client=entity_client)
        assert valid.valid, valid.issues  # the templated 'indicators' string is not checked as an array

        bad = json.loads(json.dumps(PIPELINE))
        bad["steps"][1]["context"]["tone"] = "angry"
        result = await check_pipeline_config(bad, workspace="default", entity_client=entity_client)
        assert not result.valid
        assert [issue.location for issue in result.issues] == ["steps[reply].context.tone"]

    async def test_agent_in_another_workspace(self) -> None:
        body = {
            "version": 1,
            "inputs": {"input": {}},
            "name": "cross",
            "steps": [{"name": "a", "agent": "team-b/shared-agent"}],
        }
        found = await check_pipeline_config(
            body, workspace="default", entity_client=_agents_by_name(_agent("shared-agent", workspace="team-b"))
        )
        assert found.valid, found.issues

        missing = await check_pipeline_config(body, workspace="default", entity_client=_agents_by_name())
        assert [i.message for i in missing.issues] == [
            "Agent 'team-b/shared-agent' is not registered. Register it from its agent.yaml first."
        ]

    def test_unreadable_agent_is_an_issue_not_an_error_response(self) -> None:
        body = {
            "version": 1,
            "inputs": {"input": {}},
            "name": "cross",
            "steps": [{"name": "a", "agent": "team-b/shared-agent"}],
        }
        entity_client = _agents_by_name(denied_workspaces=("team-b",))
        resp = _test_client(entity_client).post(f"{BASE}/validate", json=body)
        assert resp.status_code == 200
        assert resp.json()["issues"] == [
            {
                "location": "steps[a].agent",
                "message": "Agent 'team-b/shared-agent' could not be read; check that you have access to it.",
            }
        ]

    def test_other_client_errors_are_not_reported_as_access_problems(self) -> None:
        resp = _test_client(_agents_by_name(unauthenticated=True)).post(f"{BASE}/validate", json=PIPELINE)
        assert resp.status_code != 200

    def test_a_stored_config_as_returned_by_get_can_be_validated(self) -> None:
        entity_client = _agents_by_name(_agent("triage-agent"), _agent("reply-agent"))
        body = _stored().model_dump(mode="json")
        assert {"id", "workspace", "db_version"} <= set(body)
        result = _test_client(entity_client).post(f"{BASE}/validate", json=body).json()
        assert result == {"valid": True, "issues": []}

    async def test_raw_only_context_is_checked_as_the_agent_receives_it(self) -> None:
        schema = {"type": "object", "properties": {"tone": {"enum": ["formal"]}}}
        body = {
            "version": 1,
            "inputs": {"input": {}},
            "name": "raw",
            "steps": [{"name": "a", "agent": "x", "context": {"tone": "{% raw %}formal{% endraw %}"}}],
        }
        result = await check_pipeline_config(
            body, workspace="default", entity_client=_agents_by_name(_agent("x", schema=schema))
        )
        assert result.valid, result.issues

    def test_an_unused_input_is_allowed(self) -> None:
        entity_client = _agents_by_name(_agent("x"))
        body = {
            "version": 1,
            "inputs": {"input": {}},
            "name": "fixed",
            "steps": [{"name": "a", "agent": "x", "input": "fixed text"}],
        }
        assert _test_client(entity_client).post(f"{BASE}/validate", json=body).json() == {"valid": True, "issues": []}


# ---------------------------------------------------------------------------
# Example file and CLI
# ---------------------------------------------------------------------------


def _example() -> dict[str, Any]:
    return yaml.safe_load(EXAMPLE.read_text())


def test_example_pipeline_passes_static_checks() -> None:
    assert static_issues(_example()) == []


async def test_example_pipeline_passes_workspace_checks() -> None:
    entity_client = _agents_by_name(_agent("pr-planner"), _agent("pr-writer"), _agent("pr-reviewer"))
    result = await check_pipeline_config(_example(), workspace="default", entity_client=entity_client)
    assert result.valid, result.issues


def test_cli_validate_local(tmp_path: Path) -> None:
    app = AgentsCLI().get_cli()
    ok = CliRunner().invoke(app, ["pipeline-configs", "validate", "--local", "--spec-file", str(EXAMPLE)])
    assert ok.exit_code == 0, ok.output
    assert "'pr-pipeline' is valid" in ok.output

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "inputs": {"input": {}},
                "name": "bad",
                "steps": [{"name": "a", "agent": "x", "input": "{{ x | y }}"}],
            }
        )
    )
    result = CliRunner().invoke(app, ["pipeline-configs", "validate", "--local", "-f", str(bad)])
    assert result.exit_code == 1
    assert "steps[a].input: Filters are not supported" in result.output


def test_cli_explain_prints_the_schema() -> None:
    result = CliRunner().invoke(AgentsCLI().get_cli(), ["pipeline-configs", "explain"])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.output)
    assert {"version", "name", "steps"} <= set(schema["required"])
    assert "in" in schema["$defs"]["ConditionIn"]["properties"]


class _State(SimpleNamespace):
    def get_base_url(self, default: str | None = None) -> str:
        return "http://test"

    def get_workspace(self) -> str:
        return "default"


def _invoke_with_client(args: list[str], client: MagicMock) -> Any:
    with (
        patch("nemo_agents_plugin.cli._agents_client", return_value=client),
        patch("nemo_agents_plugin.cli._json_from_response", side_effect=lambda response: response),
    ):
        return CliRunner().invoke(AgentsCLI().get_cli(), args, obj=_State())


def test_cli_create_sends_the_file(tmp_path: Path) -> None:
    spec = tmp_path / "pipeline.yaml"
    spec.write_text(yaml.safe_dump(PIPELINE))
    client = MagicMock()
    client.create_pipeline_config.return_value = {"name": "email-triage"}

    result = _invoke_with_client(["pipeline-configs", "create", "-f", str(spec), "--workspace", "default"], client)

    assert result.exit_code == 0, result.output
    body = client.create_pipeline_config.call_args.kwargs["body"]
    assert body.name == "email-triage" and [step.name for step in body.steps] == ["triage", "reply"]


def test_cli_create_stops_on_static_errors(tmp_path: Path) -> None:
    spec = tmp_path / "pipeline.yaml"
    spec.write_text(yaml.safe_dump({**PIPELINE, "version": 9}))
    client = MagicMock()
    result = _invoke_with_client(["pipeline-configs", "create", "-f", str(spec), "--workspace", "default"], client)
    assert result.exit_code == 1
    assert "Unsupported version 9" in result.output
    client.create_pipeline_config.assert_not_called()


def test_cli_update_replaces_and_guards_renames(tmp_path: Path) -> None:
    spec = tmp_path / "pipeline.yaml"
    spec.write_text(yaml.safe_dump(PIPELINE))
    client = MagicMock()
    client.update_pipeline_config.return_value = {"name": "email-triage"}

    ok = _invoke_with_client(
        ["pipeline-configs", "update", "email-triage", "-f", str(spec), "--workspace", "default"], client
    )
    assert ok.exit_code == 0, ok.output
    assert client.update_pipeline_config.call_args.kwargs["name"] == "email-triage"

    renamed = _invoke_with_client(
        ["pipeline-configs", "update", "other", "-f", str(spec), "--workspace", "default"], client
    )
    assert renamed.exit_code == 2
    assert "renaming is not supported" in renamed.output
    assert client.update_pipeline_config.call_count == 1
