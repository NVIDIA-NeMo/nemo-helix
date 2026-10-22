# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent CRUD routes — POST/GET/LIST/DELETE for Agent entities.

All routes are mounted at ``/apis/agents/v2/workspaces/{workspace}/agents``
by the platform (``/apis/agents`` prefix from the platform + ``/v2/workspaces/{workspace}``
prefix from the RouterSpec).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_agents_plugin.agent_config_formats import AgentConfigFormatError, validate_agent_config
from nemo_agents_plugin.api.v2._perms import AgentPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.deployment_routing import (
    get_deployment_endpoint,
    get_deployment_transport,
    is_deployment_routable,
)
from nemo_agents_plugin.entities import NEMO_AGENTS_SPEC_CONFIG_FORMAT, Agent, AgentDeployment
from nemo_agents_plugin.schema import (
    AgentFilter,
    AgentMcpServerTools,
    AgentPage,
    AgentToolsResponse,
    RegisterAgentRequest,
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

# Covers the runtime's per-server discovery timeout, which it applies to every server at once.
_RUNTIME_TOOLS_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

logger = logging.getLogger(__name__)

router = APIRouter()

_agent_filter_dep = make_filter_obj_dep(AgentFilter)


@router.post("/agents", response_model=Agent, status_code=201, tags=["Agents"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AgentPerms.CREATE],
)
async def register_agent(
    workspace: str,
    body: RegisterAgentRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Agent:
    """Register an agent."""
    config = _validate_agent_config_for_register(body)

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
        logger.exception("Failed to register agent '%s'", body.name)
        raise HTTPException(status_code=500, detail="Failed to register agent.") from exc
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
    try:
        agent = await entity_client.get(Agent, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get agent '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get agent.") from exc
    return agent


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


@router.get("/agents/{name}/tools", response_model=AgentToolsResponse, tags=["Agents"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
    permissions=[AgentPerms.READ],
)
async def list_agent_tools(
    workspace: str,
    name: str,
    refresh: bool = Query(default=False, description="Re-list every server instead of using the runtime's cache."),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentToolsResponse:
    """List the tools each of the agent's MCP servers exposes, in one call.

    The agent's first running deployment lists every declared server concurrently and caches the
    result, so this costs one runtime round-trip however many servers the agent declares. Without a
    running deployment, the declared servers come back as ``not_deployed`` with no tools.
    """
    try:
        agent = await entity_client.get(Agent, name=name, workspace=workspace)
        deployments = await entity_client.list(AgentDeployment, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to load agent '%s' for tool listing", name)
        raise HTTPException(status_code=500, detail="Failed to list agent tools.") from exc

    declared = _declared_mcp_servers(agent)
    if not declared:
        return AgentToolsResponse(servers=[])

    deployment = next((d for d in deployments.data if d.agent == name and is_deployment_routable(d)), None)
    if deployment is None:
        return AgentToolsResponse(
            servers=[
                AgentMcpServerTools(name=server, transport=transport, status="not_deployed")
                for server, transport in declared.items()
            ]
        )

    try:
        return await _fetch_runtime_tools(deployment, refresh=refresh)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Deployment '%s' did not report tools", deployment.name, exc_info=True)
        error = (
            "This deployment predates tool listing; redeploy the agent to list its tools."
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404
            else f"Deployment '{deployment.name}' did not report its tools."
        )
        return AgentToolsResponse(
            deployment=deployment.name,
            servers=[
                AgentMcpServerTools(name=server, transport=transport, status="error", error=error)
                for server, transport in declared.items()
            ],
        )


def _declared_mcp_servers(agent: Agent) -> dict[str, str]:
    """Map each MCP server name in an agent spec to its transport; empty for other config formats."""
    if agent.config_format != NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        return {}
    mcp = agent.config.get("mcp")
    servers = mcp.get("servers") if isinstance(mcp, dict) else None
    if not isinstance(servers, dict):
        return {}
    return {
        str(server): str(spec.get("transport", "")) if isinstance(spec, dict) else ""
        for server, spec in servers.items()
    }


async def _fetch_runtime_tools(deployment: AgentDeployment, *, refresh: bool) -> AgentToolsResponse:
    """Ask a running deployment's runtime for its tools; the runtime's view wins over the stored spec."""
    endpoint = get_deployment_endpoint(deployment)
    if endpoint is None:  # pragma: no cover - is_deployment_routable guarantees an endpoint
        raise ValueError(f"Deployment '{deployment.name}' has no routable endpoint.")
    url, headers = get_deployment_transport(deployment).target(f"{endpoint.rstrip('/')}/v1/tools")
    async with httpx.AsyncClient(timeout=_RUNTIME_TOOLS_TIMEOUT) as client:
        response = await client.get(url, headers=headers, params={"refresh": "true"} if refresh else None)
        response.raise_for_status()
    reported = AgentToolsResponse.model_validate(response.json())
    return reported.model_copy(update={"deployment": deployment.name})


def _validate_agent_config_for_register(body: RegisterAgentRequest) -> dict[str, Any]:
    try:
        return validate_agent_config(body.config_format, body.config)
    except AgentConfigFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
