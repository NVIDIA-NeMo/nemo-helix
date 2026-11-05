# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent image routes — list and remove the images the platform built for a workspace.

Deleting a deployment already removes its image when nothing else uses it; these
routes reclaim what that misses, such as images from failed builds or from
deployments deleted before that cleanup existed.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from nemo_agents_plugin.api.v2._perms import DeploymentPerms
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.images import AgentImageList, list_agent_images, remove_tags
from nemo_agents_plugin.runner.deployments_backend import DeploymentsRunnerBackend
from nemo_agents_plugin.runner.registry import get_runner_registry
from nemo_deployments_plugin.backends.base import DeploymentBackend, ImagesUnsupportedError
from nemo_deployments_plugin.backends.registry import ExecutorNotFoundError
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import NemoEntitiesClient

logger = logging.getLogger(__name__)

router = APIRouter()


async def image_access() -> tuple[DeploymentBackend, NemoEntitiesClient] | None:
    """The host image store and an internal entity client, or None without a docker executor."""
    backend = get_runner_registry().backend_for("docker")
    if not isinstance(backend, DeploymentsRunnerBackend):
        return None
    try:
        store = await backend.image_store()
    except ExecutorNotFoundError:
        logger.debug("No docker executor; agent images are unavailable", exc_info=True)
        return None
    return store, backend.internal_entity_client()


@router.get("/agent-images", tags=["Agent Images"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentPerms.LIST])
async def list_images(
    workspace: str,
    agent: str | None = Query(default=None, description="Only images built for this agent."),
) -> AgentImageList:
    """List the agent images the platform built in this workspace, newest first."""
    access = await image_access()
    if access is None:
        return AgentImageList(data=[])
    try:
        images = await list_agent_images(*access, workspace=workspace)
    except ImagesUnsupportedError:
        return AgentImageList(data=[])
    except Exception as exc:
        logger.exception("Failed to list agent images in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list agent images.") from exc
    matching = [image for image in images if agent is None or image.agent == agent]
    matching.sort(key=lambda image: image.created, reverse=True)
    return AgentImageList(data=matching)


@router.delete("/agent-images/{image_id}", status_code=204, tags=["Agent Images"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentPerms.DELETE])
async def delete_image(workspace: str, image_id: str) -> None:
    """Remove an agent image that no container or deployment uses.

    Only this workspace's tags are removed, so a tag added outside the platform keeps the image.
    """
    access = await image_access()
    not_found = HTTPException(status_code=404, detail=f"Image '{image_id}' not found in workspace '{workspace}'.")
    if access is None:
        raise not_found
    store, entities = access
    try:
        images = await list_agent_images(store, entities, workspace=workspace)
        image = next((i for i in images if i.id == image_id), None)
        if image is None:
            raise not_found
        if image.in_use or not await remove_tags(store, image.tags):
            raise HTTPException(status_code=409, detail=f"Image '{image_id}' is still in use.")
    except HTTPException:
        raise
    except ImagesUnsupportedError as exc:
        raise not_found from exc
    except Exception as exc:
        logger.exception("Failed to remove agent image '%s'", image_id)
        raise HTTPException(status_code=500, detail="Failed to remove agent image.") from exc
