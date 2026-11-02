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

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_agents_plugin.agent_config_formats import AgentConfigFormatError, validate_agent_config
from nemo_agents_plugin.api.v2._perms import AgentPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client, get_files_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.context_schema import check_context_schema
from nemo_agents_plugin.entities import Agent, AgentDeployment
from nemo_agents_plugin.jobs.post_run_actions import validate_post_run_action_kinds
from nemo_agents_plugin.schema import (
    AgentFilter,
    AgentPage,
    RegisterAgentRequest,
)
from nemo_helix_plugin.agents.types import ReferenceConfig
from nemo_helix_plugin.api.parsed_filter import ParsedFilter, make_filter_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
)
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_helix_plugin.refs import FilesetRef, parse_entity_ref
from nemo_helix_plugin.schema import PaginationData

# Deployment statuses that block agent deletion.
# "failed" and "deleting" are excluded — they are terminal/in-cleanup and
# do not represent an actively running process the user needs to clean up first.
_BLOCKING_STATUSES = frozenset({"pending", "starting", "running"})

logger = logging.getLogger(__name__)

router = APIRouter()

_agent_filter_dep = make_filter_dep(AgentFilter)


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
    files_client: AsyncFilesClient = Depends(get_files_client),
) -> Agent:
    """Register an agent."""
    config = _validate_agent_config_for_register(body)
    _validate_run_declarations(body)
    reference = await _resolve_agent_reference(body.reference, workspace=workspace, files_client=files_client)

    agent = Agent(
        name=body.name,
        workspace=workspace,
        description=body.description,
        labels=body.labels,
        config=config,
        config_format=body.config_format,
        context_schema=body.context_schema,
        default_input=body.default_input,
        post_run_actions=body.post_run_actions,
        reference=reference,
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


@router.get(
    "/agents",
    response_model=AgentPage,
    tags=["Agents"],
    openapi_extra=generate_openapi_extra_params(
        filter_schema=AgentFilter,
        filter_description="Filter agents by config_format and labels, e.g. `filter[labels.<key>]=<value>`.",
    ),
)
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
    parsed_filter: ParsedFilter = Depends(_agent_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentPage:
    """List all agents in the workspace with pagination and filter support.

    Filter on a label with ``filter[labels.<key>]=<value>``.
    """
    try:
        result = await entity_client.list(
            Agent,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
        )
    except Exception as exc:
        logger.exception("Failed to list agents in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list agents.") from exc

    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return AgentPage(
        data=result.data,
        pagination=pagination,
        sort=sort,
        filter=parsed_filter.to_response(),
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


async def _resolve_agent_reference(
    reference: ReferenceConfig | None, *, workspace: str, files_client: AsyncFilesClient
) -> ReferenceConfig | None:
    """Check fileset access as the caller and qualify the stored reference."""
    if reference is None or reference.fileset is None:
        return reference
    ref = parse_entity_ref(reference.fileset, default_workspace=workspace)
    qualified = FilesetRef(f"{ref.workspace}/{ref.name}")
    try:
        await files_client.get_fileset(workspace=ref.workspace, name=ref.name)
    except NotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Fileset '{qualified}' not found.") from exc
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=403, detail=f"Access denied to fileset '{qualified}'.") from exc
    return reference.model_copy(update={"fileset": qualified})


def _validate_agent_config_for_register(body: RegisterAgentRequest) -> dict[str, Any]:
    try:
        return validate_agent_config(body.config_format, body.config)
    except AgentConfigFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_run_declarations(body: RegisterAgentRequest) -> None:
    """Check what the agent declares about its runs before it is stored.

    A bad schema or an uninstalled action kind would otherwise surface only
    when a job is created, against an agent that looked valid.
    """
    try:
        if body.context_schema is not None:
            check_context_schema(body.context_schema)
        validate_post_run_action_kinds(body.post_run_actions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
