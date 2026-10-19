# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for resumable sample-agent provisioning."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from nemo_agents_plugin.api.v2 import sample_agent as sample
from nemo_helix_plugin.agents.types import Agent, AgentDeployment, CreateSampleAgentRequest, SampleAgentStreamEvent
from nemo_helix_plugin.files.types import FilesetPurpose
from nemo_helix_plugin.models.refs import ResolvedModelReference
from nemo_helix_plugin.workspaces.constants import SAMPLE_WORKSPACE_DESCRIPTION
from nemo_helix_plugin.workspaces.types import Workspace


class _Page:
    def __init__(self, items: list) -> None:
        self._items = items

    async def items(self):
        for item in self._items:
            yield item


def _wire(value):
    response = MagicMock()
    response.data.return_value = value
    return response


def _model(name: str = "my-model") -> ResolvedModelReference:
    return ResolvedModelReference(
        url=f"http://test/apis/inference-gateway/v2/workspaces/default/model/{name}/-/v1",
        name=name,
        host_url=None,
    )


def _config(model: ResolvedModelReference) -> dict:
    return {"models": {"default": {"model": model.name, "base_url": model.url}}}


def _workspace(name: str, owner: str = "alice", description: str = SAMPLE_WORKSPACE_DESCRIPTION) -> Workspace:
    now = datetime.now(timezone.utc)
    return Workspace(
        id=f"workspace-{name}",
        name=name,
        description=description,
        created_at=now,
        created_by=owner,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_existing_sample_uses_marker_and_owner() -> None:
    workspaces = MagicMock()
    workspaces.list_workspaces = AsyncMock(
        return_value=_Page(
            [
                _workspace("sample-other", owner="bob"),
                _workspace("sample-fake", description="Not created by Helix"),
                _workspace("sample-owned"),
            ]
        )
    )
    assert (await sample._existing_sample(workspaces, "alice", auth_enabled=True)).name == "sample-owned"


@pytest.mark.asyncio
async def test_existing_sample_is_shared_without_auth() -> None:
    workspaces = MagicMock()
    workspaces.list_workspaces = AsyncMock(return_value=_Page([_workspace("sample-shared", owner="other")]))

    assert (await sample._existing_sample(workspaces, "alice", auth_enabled=False)).name == "sample-shared"


@pytest.mark.asyncio
async def test_model_change_allowed_before_deployment() -> None:
    old_model = _model("old-model")
    new_model = _model("new-model")
    agents = MagicMock()
    agents.list_deployments = AsyncMock(return_value=_Page([]))
    agents.get_agent = AsyncMock(return_value=_wire(Agent(name=sample._AGENT_NAME, config=_config(old_model))))
    agents.delete_agent = AsyncMock()
    agents.create_agent = AsyncMock()
    agents.create_deployment = AsyncMock(return_value=_wire(AgentDeployment(name="dep-new", agent=sample._AGENT_NAME)))

    deployment, agent_created, deployment_submitted = await sample._agent_and_deployment(
        agents, "sample-owned", new_model, _config(new_model), "nemo-agents-spec-v1"
    )

    assert agent_created and deployment_submitted and deployment.name == "dep-new"
    agents.delete_agent.assert_awaited_once()
    agents.create_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_change_rejected_after_deployment_submission() -> None:
    old_model = _model("old-model")
    agents = MagicMock()
    agents.list_deployments = AsyncMock(
        return_value=_Page(
            [AgentDeployment(name="dep-pending", agent=sample._AGENT_NAME, config=_config(old_model), status="pending")]
        )
    )
    agents.delete_agent = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await sample._agent_and_deployment(agents, "sample-owned", _model("new-model"), {}, "nemo-agents-spec-v1")

    assert exc_info.value.status_code == 409
    agents.delete_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_sample_files_streams_each_upload() -> None:
    files = MagicMock()
    files.create_fileset = AsyncMock(return_value=_wire(SimpleNamespace(purpose=FilesetPurpose.DATASET)))
    files.list_files = AsyncMock(return_value=_wire(SimpleNamespace(data=[])))
    files.upload_file = AsyncMock()

    events = [event async for event in sample._sample_files(files, "sample-owned", b"dataset", b"dataset: old\n")]

    assert [(event.component, event.status) for event in events] == [
        ("dataset", "uploaded"),
        ("evaluation_config", "uploaded"),
    ]
    assert files.upload_file.await_count == 2
    assert (
        files.upload_file.await_args_list[1].kwargs["content"]
        == b"dataset: sample-owned/esec-eval-data#dataset.jsonl\n"
    )


@pytest.mark.asyncio
async def test_incomplete_workspace_resumes_without_creating_another(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _workspace("sample-owned")
    deployment = AgentDeployment(name="dep-pending", agent=sample._AGENT_NAME, status="pending")
    model = _model()
    monkeypatch.setattr(sample, "_existing_sample", AsyncMock(return_value=workspace))
    monkeypatch.setattr(
        sample.AsyncModelsClient,
        "from_client",
        lambda client: SimpleNamespace(resolve_model_reference=AsyncMock(return_value=model)),
    )
    monkeypatch.setattr(
        sample, "_sample_config", lambda model: (_config(model), "nemo-agents-spec-v1", b"dataset", b"eval")
    )
    agent_step = AsyncMock(return_value=(deployment, False, False))

    async def file_events(*_args):
        yield SampleAgentStreamEvent(kind="progress", component="dataset", status="uploaded", workspace=workspace.name)
        yield SampleAgentStreamEvent(
            kind="progress", component="evaluation_config", status="existing", workspace=workspace.name
        )

    file_step = MagicMock(side_effect=file_events)
    monkeypatch.setattr(sample, "_agent_and_deployment", agent_step)
    monkeypatch.setattr(sample, "_sample_files", file_step)
    authorize = AsyncMock()
    monkeypatch.setattr(sample, "platform_auth_enabled", lambda: True)

    response = await sample.create_sample_agent(
        body=CreateSampleAgentRequest(model="default/my-model"),
        client=MagicMock(),
        principal_id="alice",
        authorize=authorize,
    )
    frames = [SampleAgentStreamEvent.model_validate_json(chunk) async for chunk in response.body_iterator]

    assert [(frame.kind, frame.component) for frame in frames] == [
        ("progress", "workspace"),
        ("progress", "agent"),
        ("progress", "deployment"),
        ("progress", "dataset"),
        ("progress", "evaluation_config"),
        ("done", None),
    ]
    assert frames[-1].result is not None
    assert frames[-1].result.status == "resumed"
    assert frames[-1].result.workspace == "sample-owned"
    assert frames[-1].result.deployment_status == "pending"
    assert response.status_code == 200
    authorize.assert_not_awaited()
    file_step.assert_called_once()


@pytest.mark.asyncio
async def test_failure_preserves_workspace_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _workspace("sample-owned")
    model = _model()
    monkeypatch.setattr(sample, "_existing_sample", AsyncMock(return_value=workspace))
    monkeypatch.setattr(
        sample.AsyncModelsClient,
        "from_client",
        lambda client: SimpleNamespace(resolve_model_reference=AsyncMock(return_value=model)),
    )
    monkeypatch.setattr(
        sample, "_sample_config", lambda model: (_config(model), "nemo-agents-spec-v1", b"dataset", b"eval")
    )
    monkeypatch.setattr(
        sample,
        "_agent_and_deployment",
        AsyncMock(return_value=(AgentDeployment(name="dep", agent=sample._AGENT_NAME), False, False)),
    )

    async def file_events(*_args):
        yield SampleAgentStreamEvent(kind="progress", component="dataset", status="uploaded", workspace=workspace.name)
        raise RuntimeError("files unavailable")

    monkeypatch.setattr(sample, "_sample_files", file_events)
    monkeypatch.setattr(sample, "platform_auth_enabled", lambda: True)

    response = await sample.create_sample_agent(
        body=CreateSampleAgentRequest(model="default/my-model"),
        client=MagicMock(),
        principal_id="alice",
        authorize=AsyncMock(),
    )
    frames = [SampleAgentStreamEvent.model_validate_json(chunk) async for chunk in response.body_iterator]

    assert response.status_code == 200
    assert frames[-2].component == "dataset"
    assert frames[-1].kind == "error"
    assert frames[-1].workspace == "sample-owned"
    assert frames[-1].failed_step == "sample files"
    assert frames[-1].retryable is True


@pytest.mark.asyncio
async def test_model_conflict_ends_stream_without_uploading_files(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _workspace("sample-owned")
    model = _model()
    monkeypatch.setattr(sample, "_existing_sample", AsyncMock(return_value=workspace))
    monkeypatch.setattr(
        sample.AsyncModelsClient,
        "from_client",
        lambda client: SimpleNamespace(resolve_model_reference=AsyncMock(return_value=model)),
    )
    monkeypatch.setattr(
        sample, "_sample_config", lambda model: (_config(model), "nemo-agents-spec-v1", b"dataset", b"eval")
    )
    monkeypatch.setattr(sample, "_agent_and_deployment", AsyncMock(side_effect=HTTPException(409, "model locked")))
    file_step = MagicMock()
    monkeypatch.setattr(sample, "_sample_files", file_step)
    monkeypatch.setattr(sample, "platform_auth_enabled", lambda: True)

    response = await sample.create_sample_agent(
        body=CreateSampleAgentRequest(model="default/my-model"),
        client=MagicMock(),
        principal_id="alice",
        authorize=AsyncMock(),
    )
    frames = [SampleAgentStreamEvent.model_validate_json(chunk) async for chunk in response.body_iterator]

    assert frames[-1].kind == "error"
    assert frames[-1].status_code == 409
    assert frames[-1].message == "model locked"
    file_step.assert_not_called()


def test_packaged_sample_assets_load() -> None:
    config, config_format, dataset, eval_config = sample._sample_config(_model())
    assert config_format == "nemo-agents-spec-v1"
    assert config["models"]["default"]["model"] == "my-model"
    assert dataset and eval_config


def test_sample_agent_openapi_responses() -> None:
    app = FastAPI()
    app.include_router(sample.router, prefix="/apis/agents/v2")

    responses = app.openapi()["paths"]["/apis/agents/v2/sample-agent"]["post"]["responses"]

    assert {"200", "201"} <= responses.keys()
    for status in ("200", "201"):
        assert responses[status]["content"]["application/x-ndjson"]["schema"]["$ref"].endswith(
            "/SampleAgentStreamEvent"
        )
