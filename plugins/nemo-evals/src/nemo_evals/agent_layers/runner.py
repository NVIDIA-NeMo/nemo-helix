# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent-layer Job's work: make the layer's build context, have Builder build it, and record the result.

The context is the task's environment context (or nothing, for a prebuilt image), the generated Dockerfile, the agent
to install, and the agent installer the Job's first step put on job storage. It goes up as a tar archive in a
fileset of its own. Builder's name for the build is keyed by what the layer holds (the environment's files or image,
the agent install, the installer and the platform), so tasks with the same environment share one build.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tarfile
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from nemo_builder_plugin.identity import normalize_reference
from nemo_builder_plugin.schema import FileSetSource
from nemo_evals.agent_layers.dockerfile import AGENT_DIR, AGENT_FILE, INSTALLER_DIR, LAYER_DOCKERFILE, layer_dockerfile
from nemo_evals.agent_layers.models import AgentLayerEntity, layer_entity_name
from nemo_evals.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.builder import ImageBuilder, adopt_or_build, usable_build
from nemo_evals.image_builds.models import TERMINAL_BUILD_STATUSES, TERMINAL_IMAGE_STATUSES, BuildImage
from nemo_evals.image_builds.planning import MAIN_SERVICE, ImagePlanError, TaskArchive, parse_task_config, plan_images
from nemo_evals.image_builds.runner import apply_builder_image, downloaded_archive, refresh_images
from nemo_evals.revisions import get_revision
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_TIMEOUT_SECONDS = 60 * 60
CONTEXT_ARCHIVE = "context.tar.gz"
CONTEXT_ROOT = "context"
_BUILDER_NAME_PREFIX = "harbor-agent-"


class AgentLayerBuildSpec(BaseModel):
    """Which attempt at which agent layer an agent-layer Job runs."""

    model_config = ConfigDict(extra="forbid")

    workspace: str
    task: str
    revision_digest: str
    agent_key: str
    attempt: int


async def run_agent_layer_build(
    spec: AgentLayerBuildSpec,
    *,
    entities: EntityClient,
    files: AsyncFilesClient,
    builder: ImageBuilder,
    installer: Path,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> AgentLayerEntity:
    """Build one attempt's layer, recording progress on its record; ``installer`` is the installer's tar archive."""
    head = await entities.get(TaskEntity, name=spec.task, workspace=spec.workspace)
    layer = await entities.get(
        AgentLayerEntity,
        name=layer_entity_name(spec.revision_digest, spec.agent_key, spec.attempt),
        workspace=spec.workspace,
        parent=head.id,
    )
    if layer.status in TERMINAL_BUILD_STATUSES:
        return layer
    try:
        revision = await get_revision(entities, TaskRevisionEntity, head, spec.revision_digest)
        if not isinstance(revision.spec, HarborTaskDefinition):
            return await _fail(entities, layer, f"task {layer.task} is not a Harbor task")
        image = await _submit(spec, layer, revision.spec, files=files, builder=builder, installer=installer)
    except ImagePlanError as exc:
        return await _fail(entities, layer, str(exc))
    except Exception as exc:
        logger.exception("Agent layer %s of %s raised", layer.name, layer.task)
        return await _fail(entities, layer, f"agent-layer error: {exc}")
    layer.status = "running"
    layer.image = image
    layer = await entities.update(layer)
    started = clock()
    while layer.image is not None and layer.image.status not in TERMINAL_IMAGE_STATUSES:
        if clock() - started > timeout:
            layer.image.status = "failed"
            layer.image.error = f"no result from Builder within {int(timeout)} seconds"
            break
        await sleep(poll_interval)
        if await refresh_images(builder, [layer.image]):
            layer = await entities.update(layer)
    assert layer.image is not None
    layer.status = "ready" if layer.image.status == "ready" else "failed"
    layer.error = layer.image.error if layer.status == "failed" else None
    logger.info("Agent layer %s of %s finished %s", layer.name, layer.task, layer.status)
    return await entities.update(layer)


async def _submit(
    spec: AgentLayerBuildSpec,
    layer: AgentLayerEntity,
    definition: HarborTaskDefinition,
    *,
    files: AsyncFilesClient,
    builder: ImageBuilder,
    installer: Path,
) -> BuildImage:
    """Use the build of the same layer another task made, or make the layer's context and have Builder build it.

    The build's name comes from the task's plan alone, so a task whose layer another task already built or is
    building uploads nothing: across a taskset, one task per environment does.
    """
    user = parse_task_config(definition.config).agent.user
    with tempfile.TemporaryDirectory(prefix="nemo-agent-layer-") as scratch:
        context = Path(scratch) / CONTEXT_ROOT
        async with downloaded_archive(files, definition.source) as path:
            # The listing alone: a compose task, which ``read_archive`` would extract and parse, gets no layer.
            archive = await asyncio.to_thread(TaskArchive.read, path)
            main = await asyncio.to_thread(_plan_main, definition, archive)
            base_image = str(normalize_reference(main.image)) if main.image is not None else None
            environment = main.context_digest if base_image is None else base_image
            name = layer_builder_name(
                environment=environment or "", agent_key=spec.agent_key, user=user, platform=main.platform
            )
            built = await usable_build(builder, name)
            if built is None:
                environment_dockerfile = await asyncio.to_thread(_stage_environment, main, archive, context)
        if built is None:
            (context / LAYER_DOCKERFILE).write_text(
                layer_dockerfile(environment_dockerfile=environment_dockerfile, base_image=base_image),
                encoding="utf-8",
            )
            (context / AGENT_DIR).mkdir()
            agent = {**layer.agent.model_dump(mode="json"), "user": user}
            (context / AGENT_DIR / AGENT_FILE).write_text(json.dumps(agent), encoding="utf-8")
            await asyncio.to_thread(_unpack, installer, context / INSTALLER_DIR)
            source = await _upload_context(files, spec.workspace, name, Path(scratch))
            built = await adopt_or_build(
                builder, name=name, source=source, dockerfile=LAYER_DOCKERFILE, platform=main.platform
            )
    image = BuildImage(
        type=MAIN_SERVICE,
        source="dockerfile",
        context_path=CONTEXT_ROOT,
        dockerfile=LAYER_DOCKERFILE,
        context_digest=main.context_digest,
        platform=main.platform,
        builder_image=built.name,
        build_job=built.provenance.job.rpartition("/")[2],
    )
    apply_builder_image(image, built)
    return image


def layer_builder_name(*, environment: str, agent_key: str, user: str | int | None, platform: str | None) -> str:
    """Builder build-set name for a layer: keyed by the environment it builds on, the agent install, the task's agent
    user (installs write into that user's home) and the platform."""
    document = {"v": 1, "environment": environment, "agent": agent_key, "user": user, "platform": platform}
    key = json.dumps(document, sort_keys=True)
    return f"{_BUILDER_NAME_PREFIX}{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def _plan_main(definition: HarborTaskDefinition, archive: TaskArchive) -> BuildImage:
    """The task's main image as its image build plans it; a compose task gets no layer."""
    if archive.compose is not None:
        raise ImagePlanError("compose_unsupported", "a task with a compose file isn't given an agent layer")
    images = plan_images(definition.config, archive, allowed_registries=[])
    return next(image for image in images if image.type == MAIN_SERVICE)


def _stage_environment(main: BuildImage, archive: TaskArchive, context: Path) -> str | None:
    """Copy the main image's build context into ``context``; returns its Dockerfile, or None for a prebuilt image."""
    context.mkdir(parents=True)
    if main.source == "prebuilt":
        return None
    assert main.context_path is not None and main.dockerfile is not None and archive.path is not None
    prefix = f"{main.context_path}/"
    with tarfile.open(archive.path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.name.startswith(prefix) or not (member.isfile() or member.isdir() or member.issym()):
                continue
            member.name = member.name.removeprefix(prefix)
            tar.extract(member, context, filter="data")
    return (context / main.dockerfile).read_text(encoding="utf-8")


def _unpack(installer: Path, target: Path) -> None:
    """The installer archive holds ``nhx-agent-installer/``; unpack its contents into ``target``."""
    with tarfile.open(installer, "r:*") as tar:
        for member in tar.getmembers():
            top, _, rest = member.name.partition("/")
            if top != INSTALLER_DIR or not rest:
                continue
            member.name = rest
            tar.extract(member, target, filter="tar")


async def _upload_context(files: AsyncFilesClient, workspace: str, name: str, scratch: Path) -> FileSetSource:
    """Upload the context as a tar archive in a fileset named for the build; returns Builder's source for it."""
    archive = scratch / CONTEXT_ARCHIVE
    await asyncio.to_thread(_pack, scratch / CONTEXT_ROOT, archive)
    await files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=name), exist_ok=True)
    with archive.open("rb") as stream:
        content = stream.read()
    await files.upload_file(workspace=workspace, name=name, path=CONTEXT_ARCHIVE, content=content)
    return FileSetSource(fileset=f"{workspace}/{name}", archive=CONTEXT_ARCHIVE, context_path=CONTEXT_ROOT)


def _pack(context: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(context, arcname=CONTEXT_ROOT)


async def _fail(entities: EntityClient, layer: AgentLayerEntity, error: str) -> AgentLayerEntity:
    layer.status = "failed"
    layer.error = error
    logger.info("Agent layer %s of %s failed: %s", layer.name, layer.task, error)
    return await entities.update(layer)
