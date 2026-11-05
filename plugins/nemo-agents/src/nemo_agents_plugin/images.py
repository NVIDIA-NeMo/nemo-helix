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
from nemo_deployments_plugin.backends.base import DeploymentBackend, ImageInfo, ImageInUseError
from nemo_helix_plugin.entity_client import NemoEntitiesClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Docker tags are daemon-global but the auth boundary is the workspace, so every
#: image is nested under ``{TAG_NAMESPACE}/{workspace}/``.
TAG_NAMESPACE = "nemo-agents"
IMAGE_WORKSPACE_LABEL = "com.nemo.agent.workspace"
IMAGE_AGENT_LABEL = "com.nemo.agent.name"
IMAGE_SPEC_REVISION_LABEL = "com.nemo.agent.spec-revision"
#: Set by the Dockerfile template on every build, including images built before the labels above.
_OCI_TITLE_LABEL = "org.opencontainers.image.title"
_DEPLOYMENT_PAGE_SIZE = 100


class AgentImage(BaseModel):
    """A platform-built agent image on the deployment host."""

    id: str = Field(description="Immutable image ID.")
    tags: list[str] = Field(description="This workspace's tags for the image.")
    agent: str | None = Field(default=None, description="Agent the image was built for, when known.")
    spec_revision: str = Field(
        default="",
        description="Revision of the agent's spec fileset the image was built from. Empty when unknown.",
    )
    size: int = Field(default=0, description="Image size in bytes.")
    created: str = Field(default="", description="When the image was built.")
    in_use: bool = Field(description="A container or a deployment in any workspace still uses the image.")
    deployments: list[str] = Field(
        default_factory=list, description="Deployments in this workspace that reference the image."
    )


class AgentImageList(BaseModel):
    data: list[AgentImage]


def workspace_prefix(workspace: str) -> str:
    return f"{TAG_NAMESPACE}/{workspace}/"


def is_platform_built(reference: str, workspace: str) -> bool:
    return reference.startswith(f"{TAG_NAMESPACE}/{workspace}/")


def _is_platform_tag(tag: str, workspace: str) -> bool:
    """The local build tag, or the ``{registry}/nemo-agents/{workspace}/...`` tag a push adds."""
    return is_platform_built(tag, workspace) or f"/{TAG_NAMESPACE}/{workspace}/" in tag


def build_labels(*, workspace: str, agent: str, spec_revision: str = "") -> dict[str, str]:
    labels = {IMAGE_WORKSPACE_LABEL: workspace, IMAGE_AGENT_LABEL: agent}
    if spec_revision:
        labels[IMAGE_SPEC_REVISION_LABEL] = spec_revision
    return labels


def image_agent(image: ImageInfo, workspace: str) -> str | None:
    """The agent *image* was built for; older builds only carry the agent spec's name."""
    if image.labels.get(IMAGE_WORKSPACE_LABEL) == workspace and image.labels.get(IMAGE_AGENT_LABEL):
        return image.labels[IMAGE_AGENT_LABEL]
    return image.labels.get(_OCI_TITLE_LABEL) or None


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


def _to_agent_image(image: ImageInfo, workspace: str, deployments: list[AgentDeployment]) -> AgentImage:
    owned = [t for t in image.tags if _is_platform_tag(t, workspace)]
    tags = set(owned)
    users = [d for d in deployments if d.image in tags]
    return AgentImage(
        id=image.id,
        tags=owned,
        agent=image_agent(image, workspace),
        spec_revision=image.labels.get(IMAGE_SPEC_REVISION_LABEL, ""),
        size=image.size,
        created=image.created,
        in_use=image.containers > 0 or bool(tags & _referenced_images(users)),
        deployments=sorted(d.name for d in users if d.workspace == workspace),
    )


async def list_agent_images(
    store: DeploymentBackend, entities: NemoEntitiesClient, *, workspace: str
) -> list[AgentImage]:
    images = await store.list_images(repository_prefix=workspace_prefix(workspace))
    if not images:
        return []
    deployments = await list_all_deployments(entities)
    return [_to_agent_image(image, workspace, deployments) for image in images]


async def remove_tags(store: DeploymentBackend, tags: Iterable[str]) -> bool:
    """Untag each of *tags*; the last one removes the image. False when one is still in use."""
    for tag in tags:
        try:
            await store.remove_image(tag)
        except ImageInUseError:
            return False
    return True


async def remove_unused_agent_images(
    store: DeploymentBackend, entities: NemoEntitiesClient, *, workspace: str, agent: str
) -> int:
    """Remove *agent*'s images that nothing uses; returns how many were removed."""
    removed = 0
    for image in await list_agent_images(store, entities, workspace=workspace):
        if image.agent == agent and not image.in_use and await remove_tags(store, image.tags):
            removed += 1
    return removed


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
