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


def _is_platform_tag(tag: str, workspace: str) -> bool:
    """The local build tag, or the ``{registry}/nemo-agents/{workspace}/...`` tag a push adds."""
    return is_platform_built(tag, workspace) or f"/{TAG_NAMESPACE}/{workspace}/" in tag


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
    """Remove the image *reference* names if the platform built it for *workspace* and nothing uses it.

    A push adds a ``{registry}/nemo-agents/{workspace}/...`` tag to the same image, so
    every platform tag on it is removed together; removing only one would free no disk.
    The image counts as platform-built only while it still carries the local build tag.

    Best effort: failures are logged and reported as ``False``, never raised.
    """
    if not _is_platform_tag(reference, workspace):
        return False
    try:
        tags = await store.image_tags(reference)
        if not tags or not any(is_platform_built(t, workspace) for t in tags):
            return False
        owned = [t for t in tags if _is_platform_tag(t, workspace)]
        if used := set(owned) & _referenced_images(await list_all_deployments(entities)):
            logger.info("Keeping image %s: another deployment still uses %s", reference, ", ".join(sorted(used)))
            return False
        removed = await remove_tags(store, owned)
    except Exception:
        logger.warning("Failed to remove image %s", reference, exc_info=True)
        return False
    if not removed:
        logger.info("Keeping image %s: a container still uses it", reference)
    elif kept := [t for t in tags if t not in owned]:
        logger.info("Untagged %s; image kept by other tags %s", ", ".join(owned), ", ".join(kept))
    else:
        logger.info("Removed image %s", reference)
    return removed
