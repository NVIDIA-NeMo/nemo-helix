# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent CRUD routes — POST/GET/LIST/DELETE for Agent entities.

All routes are mounted at ``/apis/agents/v2/workspaces/{workspace}/agents``
by the platform (``/apis/agents`` prefix from the platform + ``/v2/workspaces/{workspace}``
prefix from the RouterSpec).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_agents_plugin.agent_config_formats import AgentConfigFormatError, validate_agent_config
from nemo_agents_plugin.api.v2._perms import AgentPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.deployment_routing import get_deployment_endpoint, is_deployment_routable
from nemo_agents_plugin.entities import Agent, AgentDeployment
from nemo_agents_plugin.mcp_status import (
    McpServerStatus,
    McpStatusResponse,
    config_from_dict,
    declared_mcp_servers,
)
from nemo_agents_plugin.schema import (
    AgentDeploymentStatus,
    AgentFilter,
    AgentPage,
    AgentStatus,
    CreateAgentRequest,
)
from nemo_helix_plugin.api.filters import make_filter_obj_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
)
from nemo_helix_plugin.schema import PaginationData

# Deployment statuses that block agent deletion.
# "failed" and "deleting" are excluded — they are terminal/in-cleanup and
# do not represent an actively running process the user needs to clean up first.
_BLOCKING_STATUSES = frozenset({"pending", "starting", "running"})

_RUNTIME_PROBE_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger(__name__)

router = APIRouter()

_agent_filter_dep = make_filter_obj_dep(AgentFilter)


@router.post("/agents", response_model=Agent, status_code=201, tags=["Agents"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AgentPerms.CREATE],
)
async def create_agent(
    workspace: str,
    body: CreateAgentRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Agent:
    """Create a new agent from an agent config."""
    config = _validate_agent_config_for_create(body)

    agent = Agent(
        name=body.name,
        workspace=workspace,
        description=body.description,
        config=config,
        config_format=body.config_format,
    )
    try:
        saved = await entity_client.create(agent)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Agent '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except NemoEntityValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create agent '%s'", body.name)
        raise HTTPException(status_code=500, detail="Failed to create agent.") from exc
    return saved


@router.get("/agents", response_model=AgentPage, tags=["Agents"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
    permissions=[AgentPerms.LIST],
)
async def list_agents(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: AgentFilter = Depends(_agent_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentPage:
    """List all agents in the workspace with pagination and filter support."""
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            Agent,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list agents in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list agents.") from exc

    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return AgentPage(
        data=result.data,
        pagination=pagination,
        sort=sort,
        filter=filter,
    )


@router.get("/agents/{name}", response_model=Agent, tags=["Agents"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
    permissions=[AgentPerms.READ],
)
async def get_agent(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Agent:
    """Get a specific agent by name."""
    return await _get_agent_or_404(workspace, name, entity_client)


async def _get_agent_or_404(workspace: str, name: str, entity_client: NemoEntitiesClient) -> Agent:
    try:
        return await entity_client.get(Agent, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get agent '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get agent.") from exc


@router.get(
    "/agents/{name}/status",
    response_model=AgentStatus,
    tags=["Agents"],
    responses={
        404: {"description": "Agent not found"},
        500: {
            "description": "Failed to list deployments for the agent",
            "content": {
                "application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}
            },
        },
    },
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AgentPerms.READ],
)
async def get_agent_status(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentStatus:
    """Report the runtime status of an agent, including its MCP servers."""
    agent = await _get_agent_or_404(workspace, name, entity_client)

    try:
        result = await entity_client.list(AgentDeployment, workspace=workspace)
    except Exception as exc:
        logger.exception("Failed to list deployments for agent '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to list deployments.") from exc

    agent_config = config_from_dict(agent.config)
    deployments = sorted((d for d in result.data if d.agent == name), key=lambda d: d.name)
    deployment_statuses = await asyncio.gather(*(_deployment_status(deployment) for deployment in deployments))
    return AgentStatus(
        agent=name,
        workspace=workspace,
        checked_at=datetime.now(UTC),
        declared_mcp_servers=declared_mcp_servers(agent_config) if agent_config else [],
        deployments=list(deployment_statuses),
    )


async def _deployment_status(deployment: AgentDeployment) -> AgentDeploymentStatus:
    """Read one deployment's MCP status from its runtime, or say why we could not."""
    config = config_from_dict(deployment.config)
    declared = declared_mcp_servers(config) if config else []
    endpoint = get_deployment_endpoint(deployment) or ""
    status = AgentDeploymentStatus(
        deployment=deployment.name,
        deployment_mode=deployment.deployment_mode,
        status=deployment.status,
        endpoint=endpoint,
        error=deployment.error,
        mcp_servers=declared,
    )
    if not is_deployment_routable(deployment):
        return status

    probed, probe_error = await _probe_deployment(endpoint)
    if probed is None:
        status.probe_error = probe_error
        status.mcp_servers = [_unknown(server, probe_error) for server in declared]
        return status
    status.mcp_servers = probed.servers
    return status


async def _probe_deployment(endpoint: str) -> tuple[McpStatusResponse | None, str]:
    """Ask one runtime for its MCP status, or return a short reason it could not be read."""
    url = f"{endpoint.rstrip('/')}/mcp/status"
    try:
        async with httpx.AsyncClient(timeout=_RUNTIME_PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
    except httpx.TimeoutException:
        logger.debug("MCP status check of %s timed out", url, exc_info=True)
        return None, "Runtime did not respond in time."
    except httpx.HTTPError:
        logger.debug("MCP status check of %s failed", url, exc_info=True)
        return None, "Could not reach the runtime."
    if response.status_code == 404:
        return None, "Runtime does not report MCP status."
    if response.status_code >= 400:
        return None, f"Runtime returned HTTP {response.status_code}."
    try:
        return McpStatusResponse.model_validate(response.json()), ""
    except Exception:
        logger.debug("Unreadable MCP status from %s", url, exc_info=True)
        return None, "Runtime returned an unreadable status."


def _unknown(server: McpServerStatus, probe_error: str) -> McpServerStatus:
    """The declared server, marked as un-checked because the runtime did not answer."""
    return server.model_copy(update={"state": "unknown", "detail": probe_error})


@router.delete("/agents/{name}", status_code=204, tags=["Agents"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AgentPerms.DELETE],
)
async def delete_agent(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete an agent by name.

    Returns 409 if any deployments in a live state (pending/starting/running)
    still reference this agent.  Delete or wait for those deployments to finish
    before deleting the agent.
    """
    # Check for live deployments that would be orphaned by this deletion.
    try:
        result = await entity_client.list(AgentDeployment, workspace=workspace)
    except Exception as exc:
        logger.exception("Failed to list deployments before deleting agent '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to check deployments.") from exc

    blocking = [d for d in result.data if d.agent == name and d.status in _BLOCKING_STATUSES]
    if blocking:
        names = ", ".join(d.name for d in blocking)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Agent '{name}' has active deployments that must be removed first: {names}. "
                f"Use DELETE /deployments/{{name}} to remove them."
            ),
        )

    try:
        await entity_client.delete(Agent, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Agent '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the agent and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete agent '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to delete agent.") from exc


def _validate_agent_config_for_create(body: CreateAgentRequest) -> dict[str, Any]:
    try:
        return validate_agent_config(body.config_format, body.config)
    except AgentConfigFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
