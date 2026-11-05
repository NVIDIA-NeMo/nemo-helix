# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent images the platform built: ownership, usage, and best-effort cleanup.

Only images tagged under ``{TAG_NAMESPACE}/`` are ever touched. That namespace is
where :class:`~nemo_agents_plugin.jobs.package_agent.PackageAgentJob` tags what it
builds; images a user built or pulled themselves live outside it. Removal is by
tag and never forced, so a tag the user added elsewhere keeps the image alive and
Docker refuses to delete one a container (in any state) still runs from.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from nemo_agents_plugin.entities import AgentDeployment
from nemo_deployments_plugin.backends.base import DeploymentBackend, ImageInUseError
from nemo_helix_plugin.entity_client import NemoEntitiesClient

logger = logging.getLogger(__name__)

#: Docker tags are daemon-global but the auth boundary is the workspace, so every
#: image is nested under ``{TAG_NAMESPACE}/{workspace}/``.
TAG_NAMESPACE = "nemo-agents"
_DEPLOYMENT_PAGE_SIZE = 100


def is_platform_built(reference: str, workspace: str) -> bool:
    return reference.startswith(f"{TAG_NAMESPACE}/{workspace}/")


async def list_all_deployments(entities: NemoEntitiesClient) -> list[AgentDeployment]:
    deployments: list[AgentDeployment] = []
    page = 1
    while True:
        result = await entities.list(AgentDeployment, workspace="-", page=page, page_size=_DEPLOYMENT_PAGE_SIZE)
        deployments.extend(result.data)
        if result.pagination is None or page >= result.pagination.total_pages:
            return deployments
        page += 1


def _referenced_images(deployments: Iterable[AgentDeployment]) -> set[str]:
    # A deployment being torn down with its image no longer holds that image.
    return {d.image for d in deployments if d.image and not (d.status == "deleting" and d.remove_image_on_delete)}


async def remove_tags(store: DeploymentBackend, tags: Iterable[str]) -> bool:
    """Untag each of *tags*; the last one removes the image. False when one is still in use."""
    for tag in tags:
        try:
            await store.remove_image(tag)
        except ImageInUseError:
            return False
    return True


async def remove_unused_image(
    store: DeploymentBackend, entities: NemoEntitiesClient, reference: str, workspace: str
) -> bool:
    """Remove *reference* if the platform built it for *workspace* and no container or deployment uses it.

    Best effort: failures are logged and reported as ``False``, never raised.
    """
    if not is_platform_built(reference, workspace):
        return False
    try:
        if reference in _referenced_images(await list_all_deployments(entities)):
            logger.info("Keeping image %s: another deployment still uses it", reference)
            return False
        removed = await remove_tags(store, [reference])
    except Exception:
        logger.warning("Failed to remove image %s", reference, exc_info=True)
        return False
    if removed:
        logger.info("Removed image %s", reference)
    else:
        logger.info("Keeping image %s: a container still uses it", reference)
    return removed
