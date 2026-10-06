# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision the packaged sample agent and its resources."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit
from uuid import uuid4

import yaml
from email_security_triage.resources import sample_file
from fastapi import APIRouter, Depends, HTTPException, Response
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.utils import expand_env_vars
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.types import (
    AgentDeployment,
    CreateAgentRequest,
    CreateDeploymentRequest,
    CreateSampleAgentRequest,
    SampleAgentConflictResponse,
    SampleAgentResponse,
    SampleAgentRetryResponse,
)
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.dependencies import (
    RequestAuthorizer,
    get_effective_principal_id,
    get_nemo_client,
    get_request_authorizer,
)
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose, UpdateFilesetRequest
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.refs import ResolvedModelReference, parse_workspace_name_ref
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_helix_plugin.workspaces.constants import SAMPLE_WORKSPACE_DESCRIPTION, SAMPLE_WORKSPACE_PREFIX
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest, Workspace

logger = logging.getLogger(__name__)
router = APIRouter()

_AGENT_NAME = "email-security-triage"
_AGENT_DESCRIPTION = "Email security triage sample agent created by the NeMo Helix sample flow."
_FILESET = "esec-eval-data"
_DATASET = "dataset.jsonl"
_EVAL_SOURCE = "eval-config.dataset-driven.yml"
_EVAL_CONFIG = "eval-config.yaml"


def _studio_url(workspace: str) -> str:
    return f"/studio/workspaces/{workspace}/dashboard"


async def _existing_sample(
    workspaces: AsyncWorkspacesClient, principal_id: str, auth_enabled: bool
) -> Workspace | None:
    result = await workspaces.list_workspaces()
    samples = [
        workspace
        async for workspace in result.items()
        if workspace.name.startswith(SAMPLE_WORKSPACE_PREFIX)
        and workspace.description == SAMPLE_WORKSPACE_DESCRIPTION
        and (not auth_enabled or workspace.created_by == principal_id)
    ]
    return min(samples, key=lambda workspace: workspace.created_at) if samples else None


def _model_matches(config: dict, model: ResolvedModelReference) -> bool:
    model_config = config.get("models", {}).get("default", {})
    return (
        model_config.get("model") == model.name
        and urlsplit(model_config.get("base_url", "")).path == urlsplit(model.url).path
    )


def _sample_config(model: ResolvedModelReference) -> tuple[dict, str, bytes, bytes]:
    """Read packaged assets and bind the selected model before any workspace write."""
    config = yaml.safe_load(sample_file("agent.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Packaged sample agent config must be a mapping")
    config = expand_env_vars(config, vars_dict={"NEMO_DEFAULT_MODEL": model.name})
    config_format = config["config_format"]
    model_config = config["models"]["default"]
    model_config["model"] = model.name
    model_config["base_url"] = model.url
    dataset = sample_file(_DATASET).read_bytes()
    eval_config = sample_file(_EVAL_SOURCE).read_text(encoding="utf-8").encode()
    return config, config_format, dataset, eval_config


async def _agent_and_deployment(
    agents: AsyncAgentsClient,
    workspace: str,
    model: ResolvedModelReference,
    config: dict,
    config_format: str,
) -> tuple[AgentDeployment, bool]:
    deployments = [
        deployment
        async for deployment in (await agents.list_deployments(workspace=workspace)).items()
        if deployment.agent == _AGENT_NAME
    ]
    if any(not _model_matches(deployment.config, model) for deployment in deployments):
        raise HTTPException(status_code=409, detail="The sample agent has already been deployed with another model")

    changed = False
    try:
        agent = (await agents.get_agent(workspace=workspace, name=_AGENT_NAME)).data()
    except NotFoundError:
        agent = None
    if agent is not None and not _model_matches(agent.config, model):
        if deployments:
            raise HTTPException(status_code=409, detail="The sample agent has already been deployed with another model")
        await agents.delete_agent(workspace=workspace, name=_AGENT_NAME)
        agent = None
        changed = True
    if agent is None:
        await agents.create_agent(
            workspace=workspace,
            body=CreateAgentRequest(
                name=_AGENT_NAME,
                description=_AGENT_DESCRIPTION,
                config=config,
                config_format=config_format,
            ),
        )
        changed = True

    active = next(
        (deployment for deployment in deployments if deployment.status in {"pending", "starting", "running"}), None
    )
    if active is not None:
        return active, changed
    deployment = (
        await agents.create_deployment(workspace=workspace, body=CreateDeploymentRequest(agent=_AGENT_NAME))
    ).data()
    return deployment, True


async def _sample_files(files: AsyncFilesClient, workspace: str, dataset: bytes, eval_source: bytes) -> bool:
    fileset = (
        await files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=_FILESET,
                description="Evaluation dataset for the sample email security agent.",
                purpose=FilesetPurpose.DATASET,
            ),
            exist_ok=True,
        )
    ).data()
    changed = False
    if fileset.purpose != FilesetPurpose.DATASET:
        await files.update_fileset(
            workspace=workspace,
            name=_FILESET,
            body=UpdateFilesetRequest(purpose=FilesetPurpose.DATASET),
        )
        changed = True

    present = {item.path for item in (await files.list_files(workspace=workspace, name=_FILESET)).data().data}
    if _DATASET not in present:
        await files.upload_file(workspace=workspace, name=_FILESET, path=_DATASET, content=dataset)
        changed = True
    if _EVAL_CONFIG not in present:
        eval_config = yaml.safe_load(eval_source)
        if not isinstance(eval_config, dict):
            raise ValueError("Packaged sample evaluation config must be a mapping")
        eval_config["dataset"] = f"{workspace}/{_FILESET}#{_DATASET}"
        await files.upload_file(
            workspace=workspace,
            name=_FILESET,
            path=_EVAL_CONFIG,
            content=yaml.safe_dump(eval_config, sort_keys=False).encode(),
        )
        changed = True
    return changed


@router.post(
    "/sample-agent",
    response_model=SampleAgentResponse,
    status_code=201,
    response_description="Sample agent and workspace created; deployment may still be pending.",
    responses={
        200: {"model": SampleAgentResponse, "description": "Existing sample agent resumed or already provisioned."},
        409: {
            "model": SampleAgentConflictResponse,
            "description": "The existing sample agent deployment uses another model.",
        },
        502: {
            "model": SampleAgentRetryResponse,
            "description": "Provisioning failed; the workspace is retained so the request can be retried.",
        },
    },
    tags=["Sample Agent"],
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
async def create_sample_agent(
    body: CreateSampleAgentRequest,
    response: Response,
    client: AsyncNemoClient = Depends(get_nemo_client),
    principal_id: str = Depends(get_effective_principal_id),
    authorize: RequestAuthorizer = Depends(get_request_authorizer),
) -> SampleAgentResponse:
    """Create or resume the caller's sample agent and resources; deployment remains asynchronous."""
    try:
        parse_workspace_name_ref(body.model, label="Model reference")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    workspaces = AsyncWorkspacesClient.from_client(client)
    auth_enabled = platform_auth_enabled()
    workspace = await _existing_sample(workspaces, principal_id, auth_enabled)
    created = workspace is None

    try:
        model = await AsyncModelsClient.from_client(client).resolve_model_reference(body.model)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found") from exc
    try:
        config, config_format, dataset, eval_source = _sample_config(model)
    except Exception as exc:
        logger.exception("Packaged sample assets are unavailable")
        raise HTTPException(status_code=500, detail="Packaged sample assets are unavailable") from exc

    if created:
        await authorize("POST", "/apis/entities/v2/workspaces")
        name = f"{SAMPLE_WORKSPACE_PREFIX}{uuid4().hex}"
        async with get_async_nemo_client(
            as_service="agents",
            internal=True,
            on_behalf_of=principal_id if auth_enabled else None,
        ) as service_client:
            workspace = (
                await AsyncWorkspacesClient.from_client(service_client).create_workspace(
                    body=CreateWorkspaceRequest(name=name, description=SAMPLE_WORKSPACE_DESCRIPTION)
                )
            ).data()

    step = "agent deployment"
    try:
        deployment, agent_changed = await _agent_and_deployment(
            AsyncAgentsClient.from_client(client), workspace.name, model, config, config_format
        )
        step = "sample files"
        files_changed = await _sample_files(AsyncFilesClient.from_client(client), workspace.name, dataset, eval_source)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Sample agent provisioning failed at %s", step)
        raise HTTPException(
            status_code=502,
            detail={
                "workspace": workspace.name,
                "studio_url": _studio_url(workspace.name),
                "failed_step": step,
                "retryable": True,
            },
        ) from exc

    status = "created" if created else "resumed" if agent_changed or files_changed else "already_exists"
    if not created:
        response.status_code = 200
    return SampleAgentResponse(
        status=status,
        workspace=workspace.name,
        studio_url=_studio_url(workspace.name),
        agent=_AGENT_NAME,
        deployment=deployment.name,
        deployment_status=deployment.status,
    )
