# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment lifecycle routes — create/list/get/delete AgentDeployment entities.

Creating a deployment:
1. Validates the referenced Agent exists.
2. Deep-copies the agent config and injects the inference-gateway URL.
3. Creates an ``AgentDeployment`` entity with ``status: "pending"``.
4. Returns immediately — the :class:`~nemo_agents_plugin.runner.controller.AgentDeploymentController`
   picks up the pending deployment on the next reconcile cycle.

Deleting a deployment marks it ``deleting``; the controller terminates the
process and removes the entity.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from nemo_agents_plugin.agent_config_formats import AgentConfigFormatError, resolve_agent_config_for_deployment
from nemo_agents_plugin.api.v2._perms import DeploymentPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client, get_files_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.config import AgentsConfig
from nemo_agents_plugin.entities import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    Agent,
    AgentDeployment,
    AgentEnvironmentInline,
    EnvironmentSpecInline,
    is_container_deployment_mode,
)
from nemo_agents_plugin.environment_resolution import (
    EnvironmentResolutionError,
    MergedEnvironment,
    ResolvedEnvironment,
    merge_environment_spec_into_agent_config,
    resolve_environment,
)
from nemo_agents_plugin.runner.deployments_backend import (
    executor_for_mode,
    require_executor_matches_mode,
)
from nemo_agents_plugin.schema import (
    CreateDeploymentRequest,
    DeploymentFilter,
    DeploymentPage,
)
from nemo_agents_plugin.spec_revision import SpecRevision, read_spec_revision
from nemo_platform_plugin.api.filters import make_filter_obj_dep
from nemo_platform_plugin.auth import current_auth_context
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError, NemoEntityNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.schema import PaginationData

logger = logging.getLogger(__name__)

router = APIRouter()

_deployment_filter_dep = make_filter_obj_dep(DeploymentFilter)
_DELETE_MARK_ATTEMPTS = 3


@router.post("/deployments", response_model=AgentDeployment, status_code=201, tags=["Agent Deployments"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[DeploymentPerms.CREATE],
)
async def create_deployment(
    workspace: str,
    body: CreateDeploymentRequest,
    request: Request,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    files_client: AsyncFilesClient = Depends(get_files_client),
) -> AgentDeployment:
    """Create a new deployment for an existing agent.

    The deployment starts in ``pending`` state and is picked up by the
    deployment controller on its next reconcile cycle.
    """
    # 1. Validate the referenced agent exists
    try:
        agent = await entity_client.get(Agent, name=body.agent, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{body.agent}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to look up agent '%s'", body.agent)
        raise HTTPException(status_code=500, detail="Failed to look up agent.") from exc

    # 2. Build deployment name (auto-generate if not provided)
    deployment_name = body.name or f"{body.agent}-{secrets.token_hex(4)}"

    if body.use_image_entrypoint and not is_container_deployment_mode(body.deployment_mode):
        raise HTTPException(
            status_code=400,
            detail="use_image_entrypoint requires deployment_mode 'docker' or 'k8s'.",
        )

    # The controller refuses this too, but only on its next reconcile — by which
    # point a pending deployment exists and the caller has had its 201.
    if is_container_deployment_mode(body.deployment_mode):
        runner_config = AgentsConfig.get().deployments
        try:
            require_executor_matches_mode(executor_for_mode(runner_config, body.deployment_mode), body.deployment_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The runner fails the deployment for this; the answer is already available here.
    if is_container_deployment_mode(body.deployment_mode) and not (
        body.image or AgentsConfig.get().deployments.default_image
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"deployment_mode '{body.deployment_mode}' requires a container image. "
                "Set 'image' on the request, or configure 'deployments.default_image'."
            ),
        )

    # 3. Resolve deployment-time config. NAT workflows need legacy injection;
    # Platform-owned agent configs stay strict and are translated by the runner.
    resolved_config = _resolve_deployment_config(agent, workspace=workspace)

    # 4. Resolve and snapshot the referenced AgentEnvironment. The environment
    # spec is merged into the resolved config (Agent-config-wins precedence); the
    # compute spec and secret-env references are snapshotted for the container
    # backend. Once created, a deployment is not kept in sync with the underlying
    # environment entities.
    resolved_environment = await _resolve_deployment_environment(
        body.environment, workspace=workspace, entity_client=entity_client
    )
    merged = _merge_environment(resolved_config, resolved_environment.environment_spec)

    # 5. Only a Fabric agent stages a spec fileset; a same-named one would mislead.
    spec = SpecRevision()
    if agent.config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        spec = await read_spec_revision(files_client, workspace=workspace, agent_name=body.agent)

    # 6. Create the entity with status "pending"
    deployment = AgentDeployment(
        name=deployment_name,
        workspace=workspace,
        agent=body.agent,
        config=merged.config,
        environment=body.environment,
        compute=resolved_environment.compute_spec,
        sandbox=resolved_environment.sandbox_spec,
        secrets=merged.secrets,
        spec_revision=spec.revision,
        spec_tracked_revision=spec.tracked_revision,
        status="pending",
        deployment_mode=body.deployment_mode,
        image=body.image,
        use_image_entrypoint=body.use_image_entrypoint,
        plugin_deployment=deployment_name if is_container_deployment_mode(body.deployment_mode) else "",
    ).with_auth_context(current_auth_context())
    try:
        saved = await entity_client.create(deployment)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Deployment '{deployment_name}' already exists in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create deployment for agent '%s'", body.agent)
        raise HTTPException(status_code=500, detail="Failed to create deployment.") from exc

    return saved


def _resolve_deployment_config(agent: Agent, *, workspace: str) -> dict[str, Any]:
    try:
        return resolve_agent_config_for_deployment(
            agent.config_format,
            agent.config,
            workspace=workspace,
            agent_name=agent.name,
        )
    except AgentConfigFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _resolve_deployment_environment(
    environment: str | AgentEnvironmentInline | None,
    *,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> ResolvedEnvironment:
    try:
        return await resolve_environment(environment, workspace=workspace, entity_client=entity_client)
    except EnvironmentResolutionError as exc:
        # 422: the request is syntactically valid but references an environment
        # (or one of its specs) that cannot be resolved (e.g. a missing entity).
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _merge_environment(config: dict[str, Any], env_spec: EnvironmentSpecInline | None) -> MergedEnvironment:
    try:
        return merge_environment_spec_into_agent_config(config, env_spec)
    except EnvironmentResolutionError as exc:
        # 422: e.g. a secret env var bound to conflicting references.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/deployments", response_model=DeploymentPage, tags=["Agent Deployments"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[DeploymentPerms.LIST],
)
async def list_deployments(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: DeploymentFilter = Depends(_deployment_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> DeploymentPage:
    """List all deployments in the workspace with pagination and filter support."""
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            AgentDeployment,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list deployments in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list deployments.") from exc

    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return DeploymentPage(
        data=result.data,
        pagination=pagination,
        sort=sort,
        filter=filter,
    )


@router.get("/deployments/{name}", response_model=AgentDeployment, tags=["Agent Deployments"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[DeploymentPerms.READ],
)
async def get_deployment(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentDeployment:
    """Get a deployment by name."""
    try:
        dep = await entity_client.get(AgentDeployment, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get deployment '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get deployment.") from exc
    return dep


@router.delete("/deployments/{name}", status_code=204, tags=["Agent Deployments"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[DeploymentPerms.DELETE],
)
async def delete_deployment(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Stop and remove a deployment.

    Marks the deployment as ``deleting``.  The controller terminates the
    subprocess and removes the entity on the next reconcile cycle.
    """
    for attempt in range(_DELETE_MARK_ATTEMPTS):
        try:
            await _mark_deployment_deleting_once(
                entity_client,
                workspace=workspace,
                name=name,
                retrying=attempt > 0,
            )
            return
        except HTTPException:
            raise
        except NemoEntityConflictError as exc:
            if attempt + 1 >= _DELETE_MARK_ATTEMPTS:
                raise HTTPException(
                    status_code=409,
                    detail=f"Deployment '{name}' is being modified concurrently.",
                ) from exc
            logger.info("Retrying delete for deployment '%s' after concurrent update", name)
        except Exception as exc:
            logger.exception("Failed to mark deployment '%s' as deleting", name)
            raise HTTPException(status_code=500, detail="Failed to update deployment.") from exc


async def _mark_deployment_deleting_once(
    entity_client: NemoEntitiesClient,
    *,
    workspace: str,
    name: str,
    retrying: bool,
) -> None:
    try:
        dep = await entity_client.get(AgentDeployment, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        if retrying:
            logger.info("Deployment '%s' already deleted during delete retry", name)
            return
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to look up deployment '%s' before delete", name)
        raise HTTPException(status_code=500, detail="Failed to look up deployment.") from exc

    if dep.status == "deleting":
        return

    dep.status = "deleting"
    try:
        await entity_client.update(dep)
    except NemoEntityNotFoundError:
        logger.info("Deployment '%s' already deleted before status update", name)
        return
