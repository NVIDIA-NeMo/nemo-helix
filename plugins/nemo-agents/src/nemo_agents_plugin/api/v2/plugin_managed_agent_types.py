# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List the plugin-managed agent types installed on the platform.

Mounted at ``/apis/agents/v2/workspaces/{workspace}/plugin-managed-agent-types``.
What is installed is a property of the platform, but the route is
workspace-scoped because each entry also names the workspace's agent of that
type, which is what a client needs to decide between creating and running one.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from nemo_agents_plugin.api.v2._perms import AgentPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.entities import Agent
from nemo_agents_plugin.plugin_managed_agents import list_plugin_managed_agent_types, type_info
from nemo_helix_plugin.agents.types import PluginManagedAgentTypeInfo
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entities import CLIENT_ERRORS
from nemo_helix_plugin.entity_client import NemoEntitiesClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

router = APIRouter()

# A workspace holds at most one agent per type, so this bounds the lookup well above any real install.
_TYPED_AGENTS_PAGE_SIZE = 100


@router.get(
    "/plugin-managed-agent-types",
    response_model=list[PluginManagedAgentTypeInfo],
    tags=["Plugin-Managed Agent Types"],
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL],
    permissions=[AgentPerms.LIST],
)
async def list_plugin_managed_agent_types_route(
    workspace: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> list[PluginManagedAgentTypeInfo]:
    """List installed plugin-managed agent types, with this workspace's agent of each.

    Create an agent of a type with `POST /agents` and `plugin_managed_agent: {type}`,
    or `nemo agents create <type>`.
    """
    # Off the event loop: loading types imports plugin modules and may scan adapter manifests.
    types = await asyncio.to_thread(list_plugin_managed_agent_types)
    if not types:
        return []
    try:
        agents = await _list_agents_of_types(
            [type_cls.type_id for type_cls in types], workspace=workspace, entity_client=entity_client
        )
    except CLIENT_ERRORS:
        raise
    except Exception as exc:
        logger.exception("Failed to list plugin-managed agents in workspace '%s'", sanitize_for_log(workspace))
        raise HTTPException(status_code=500, detail="Failed to list plugin-managed agent types.") from exc

    # Oldest first, so the agent reported for a type is the one that won any create race.
    agents_by_type: dict[str, str] = {}
    for agent in agents:
        if agent.plugin_managed_agent is not None:
            agents_by_type.setdefault(agent.plugin_managed_agent.type, agent.name)
    return [type_info(type_cls, agent=agents_by_type.get(type_cls.type_id)) for type_cls in types]


async def _list_agents_of_types(
    type_ids: list[str], *, workspace: str, entity_client: NemoEntitiesClient
) -> list[Agent]:
    """List the workspace's agents of *type_ids*, oldest first, following pagination."""
    agents: list[Agent] = []
    page = 1
    while True:
        result = await entity_client.list(
            Agent,
            workspace=workspace,
            page=page,
            page_size=_TYPED_AGENTS_PAGE_SIZE,
            sort="created_at",
            filter_operation=ComparisonOperation(
                operator=FilterOperator.IN, field="data.plugin_managed_agent.type", value=type_ids
            ),
        )
        agents.extend(result.data)
        if result.pagination is None or page >= result.pagination.total_pages:
            return agents
        page += 1
