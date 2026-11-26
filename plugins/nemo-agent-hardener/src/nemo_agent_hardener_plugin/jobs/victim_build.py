# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a manifest's victim image in the cluster from its staged context, and record it on the manifest.

The builder plugin builds the staged archive with kaniko and publishes it to the deployment's registry; the
digest-pinned reference it returns becomes the manifest's ``victim_image``, as ``manifest set --victim-image``
would set it by hand.
"""

from __future__ import annotations

import hashlib
import re

from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_PROVISIONING,
    CATEGORY_VICTIM_UNAVAILABLE,
    AgentHardenerRunError,
)
from nemo_builder_plugin.builds import HttpImageBuilder, ImageBuilder, archive_source, ensure_build, wait_for_image
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.client_provider import get_async_task_nemo_client
from nemo_helix_plugin.entities import EntityClient, EntityConflictError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient

BUILDER_REMEDIATION = (
    "Enable the builder: set `builder.enabled`, set `platformConfig.builder.registry`, and create the "
    "`builder-registry-username`, `builder-registry-password` and `builder-signing-key` secrets in the workspace."
)

_SET_PREFIX = "ah-victim-"
# The builder names the build's job fileset `job-fileset-builder-<set>-<revision>`, at most 63 characters.
# This leaves room for a four-digit revision.
_MAX_SET_NAME = 63 - len("job-fileset-builder-") - len("-9999")


def victim_build_name(manifest_id: str) -> str:
    """The builder build-set name for *manifest_id*'s victim: ``ah-victim-<manifest id>``, shortened to fit."""
    slug = re.sub(r"[^a-z0-9]+", "-", manifest_id.lower()).strip("-")
    name = f"{_SET_PREFIX}{slug}"
    if slug == manifest_id and len(name) <= _MAX_SET_NAME:
        return name
    # A hash keeps two manifests apart once their ids are cleaned up or cut short.
    digest = hashlib.sha256(manifest_id.encode()).hexdigest()[:8]
    kept = slug[: _MAX_SET_NAME - len(_SET_PREFIX) - len(digest) - 1].rstrip("-")
    return f"{_SET_PREFIX}{kept}-{digest}" if kept else f"{_SET_PREFIX}{digest}"


async def build_in_cluster(*, workspace: str, manifest_id: str, context_ref: str) -> str:
    """Build the victim from the staged archive at *context_ref*, as the job's submitter; return its image."""
    async with get_async_task_nemo_client("agent-hardener") as client:
        return await build_victim_image(
            HttpImageBuilder(client, workspace=workspace),
            EntityClient(AsyncEntitiesClient.from_client(client)),
            workspace=workspace,
            manifest_id=manifest_id,
            context_ref=context_ref,
        )


async def build_victim_image(
    builder: ImageBuilder,
    entities: EntityClient,
    *,
    workspace: str,
    manifest_id: str,
    context_ref: str,
    poll_interval: float = 10.0,
    timeout: float = 60 * 60,
) -> str:
    """Build the staged context, wait for the image, and set it as the manifest's ``victim_image`` once ready."""
    if context_ref.startswith("file://"):
        raise AgentHardenerRunError(
            CATEGORY_PROVISIONING,
            "Building the victim image needs the staging job submitted to the platform, not run locally.",
            remediation="Run `nemo agent-hardener stage --build` against a NeMo Helix deployment with the builder.",
        )
    try:
        submitted = await ensure_build(
            builder, name=victim_build_name(manifest_id), source=archive_source(context_ref), dockerfile=None
        )
    except ConflictError as exc:
        raise AgentHardenerRunError(
            CATEGORY_PROVISIONING,
            f"The builder refused to build the victim image: {exc.detail}",
            remediation=BUILDER_REMEDIATION,
        ) from exc
    job_hint = f"Read the logs of builder job {submitted.job}, fix the victim's build context, then stage again."
    try:
        image = await wait_for_image(builder, submitted.images[0], poll_interval=poll_interval, timeout=timeout)
    except (TimeoutError, LookupError) as exc:
        raise AgentHardenerRunError(CATEGORY_VICTIM_UNAVAILABLE, str(exc), remediation=job_hint) from exc
    if image.status != "ready" or not image.image_ref:
        detail = image.status_detail or "its job ended without publishing the image"
        raise AgentHardenerRunError(
            CATEGORY_VICTIM_UNAVAILABLE,
            f"Building the victim image failed (builder job {submitted.job}): {detail}",
            remediation=job_hint,
        )
    await _record_victim_image(entities, workspace=workspace, manifest_id=manifest_id, image_ref=image.image_ref)
    return image.image_ref


async def _record_victim_image(entities: EntityClient, *, workspace: str, manifest_id: str, image_ref: str) -> None:
    """Set the manifest's ``victim_image``, re-reading it once if another edit landed in between."""
    for attempt in range(2):
        manifest = await entities.get(AgentHardenerManifest, name=manifest_id, workspace=workspace)
        manifest.victim_image = image_ref
        try:
            await entities.update(manifest)
            return
        except EntityConflictError:
            if attempt:
                raise
