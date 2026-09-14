# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the revision an agent's spec fileset is pinned to."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from nemo_agents_plugin.entities import ethos_fileset_name
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError as PluginClientNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class SpecRevision:
    revision: str = ""
    tracked_revision: str = ""


async def read_spec_revision(files_client: AsyncFilesClient | None, *, workspace: str, agent_name: str) -> SpecRevision:
    """Return what the agent's spec fileset is pinned to, or nothing if it cannot be read."""
    if files_client is None:
        return SpecRevision()

    fileset_name = ethos_fileset_name(agent_name)
    try:
        response = await files_client.get_fileset(workspace=workspace, name=fileset_name)
        storage = response.data().storage
    except PluginClientNotFoundError:
        return SpecRevision()
    except Exception:
        logger.warning(
            "Could not read fileset %s/%s for deployment provenance",
            sanitize_for_log(workspace),
            sanitize_for_log(fileset_name),
            exc_info=True,
        )
        return SpecRevision()

    return SpecRevision(revision=storage.pinned_revision, tracked_revision=storage.tracked_revision or "")


def files_client_for(sdk: AsyncNeMoPlatform) -> AsyncFilesClient | None:
    """Adapt the platform SDK, or None — a deployment must not fail for want of provenance."""
    try:
        return client_from_platform(sdk, AsyncFilesClient)
    except Exception:
        logger.warning("Could not build a files client for deployment provenance", exc_info=True)
        return None


async def stage_with_spec_revision(
    files_client: AsyncFilesClient | None,
    *,
    workspace: str,
    agent_name: str,
    stage: Callable[[], Awaitable[T]],
) -> tuple[T, SpecRevision]:
    """Run *stage*, and report the revision the content it staged came from.

    Reading the revision after the download would record a refresh that landed in
    between, so restage once when it moves.
    """
    if files_client is None:
        return await stage(), SpecRevision()

    before = await read_spec_revision(files_client, workspace=workspace, agent_name=agent_name)
    staged = await stage()
    after = await read_spec_revision(files_client, workspace=workspace, agent_name=agent_name)
    if after == before:
        return staged, after

    logger.info(
        "Spec fileset %s/%s moved from %r to %r while staging; staging it again",
        sanitize_for_log(workspace),
        sanitize_for_log(ethos_fileset_name(agent_name)),
        before.revision,
        after.revision,
    )
    staged = await stage()
    settled = await read_spec_revision(files_client, workspace=workspace, agent_name=agent_name)
    if settled != after:
        logger.warning(
            "Spec fileset %s/%s moved again while staging; recording no revision for it",
            sanitize_for_log(workspace),
            sanitize_for_log(ethos_fileset_name(agent_name)),
        )
        return staged, SpecRevision()
    return staged, settled
