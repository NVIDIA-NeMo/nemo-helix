# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry one image build of a task revision from ``pending`` to ``ready`` or ``failed``."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from nemo_builder_plugin.identity import normalize_reference
from nemo_builder_plugin.schema import FileSetSource, ImageSource
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskDefinition
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.harbor.archive import MAX_ARCHIVE_BYTES
from nemo_evals.harbor.archive_io import download_verified_async
from nemo_evals.image_builds.builder import (
    ENDED_JOB_STATUSES,
    ImageBuilder,
    archive_source,
    builder_name,
    copy_builder_name,
    ensure_build,
)
from nemo_evals.image_builds.models import (
    TERMINAL_BUILD_STATUSES,
    TERMINAL_IMAGE_STATUSES,
    BuildImage,
    TaskImageBuildEntity,
    build_entity_name,
)
from nemo_evals.image_builds.planning import ImagePlanError, TaskArchive, plan_images
from nemo_evals.revisions import get_revision
from nemo_helix_plugin.client.errors import InternalServerError, NemoClientError, NemoTransportError
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60


class TaskImageBuildSpec(BaseModel):
    """Which build of which task revision a task-builder Job runs."""

    model_config = ConfigDict(extra="forbid")

    workspace: str
    task: str
    revision_digest: str
    attempt: int


async def run_task_image_build(
    spec: TaskImageBuildSpec,
    *,
    entities: EntityClient,
    files: AsyncFilesClient,
    builder: ImageBuilder,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> TaskImageBuildEntity:
    """Plan, submit and follow every image of one build, recording progress on its record."""
    head = await entities.get(TaskEntity, name=spec.task, workspace=spec.workspace)
    # Any revision with this digest will do: they hold the same contents.
    revision = await get_revision(entities, TaskRevisionEntity, head, spec.revision_digest)
    build = await entities.get(
        TaskImageBuildEntity,
        name=build_entity_name(spec.revision_digest, spec.attempt),
        workspace=spec.workspace,
        parent=head.id,
    )
    if build.status in TERMINAL_BUILD_STATUSES:
        return build
    try:
        return await _build(
            build,
            revision,
            entities=entities,
            files=files,
            builder=builder,
            poll_interval=poll_interval,
            timeout=timeout,
            sleep=sleep,
            clock=clock,
        )
    except Exception as exc:
        logger.exception("Image build %s of %s raised", build.name, build.task)
        current = await entities.get(TaskImageBuildEntity, name=build.name, workspace=build.workspace, parent=head.id)
        if current.status in TERMINAL_BUILD_STATUSES:
            return current
        return await _fail(entities, current, f"task-builder error: {exc}")


async def _build(
    build: TaskImageBuildEntity,
    revision: TaskRevisionEntity,
    *,
    entities: EntityClient,
    files: AsyncFilesClient,
    builder: ImageBuilder,
    poll_interval: float,
    timeout: float,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> TaskImageBuildEntity:
    if not isinstance(revision.spec, HarborTaskDefinition):
        return await _fail(entities, build, f"task {build.task} is not a Harbor task")
    definition = revision.spec

    try:
        archive = await read_task_archive(files, definition.source)
        images = plan_images(definition.config, archive, allowed_registries=build.allowed_registries)
    except ImagePlanError as exc:
        return await _fail(entities, build, str(exc))

    build.status = "running"
    build.images = images
    for image in images:
        await _submit(builder, image, definition)
    build = await entities.update(build)

    started = clock()
    while not _settled(build.images):
        if clock() - started > timeout:
            for image in build.images:
                if image.status not in TERMINAL_IMAGE_STATUSES:
                    image.status = "failed"
                    image.error = f"no result from Builder within {int(timeout)} seconds"
            break
        await sleep(poll_interval)
        if await _refresh(builder, build.images):
            build = await entities.update(build)

    failed = [image for image in build.images if image.status == "failed"]
    build.status = "failed" if failed else "ready"
    build.error = f"{failed[0].role}: {failed[0].error}" if failed else None
    logger.info("Image build %s of %s finished %s", build.name, build.task, build.status)
    return await entities.update(build)


def _settled(images: list[BuildImage]) -> bool:
    return all(image.status in TERMINAL_IMAGE_STATUSES for image in images) or any(
        image.status == "failed" for image in images
    )


async def read_task_archive(files: AsyncFilesClient, source: HarborArchiveSource) -> TaskArchive:
    """Download a task archive, check its digest, and list its files."""
    with tempfile.TemporaryDirectory(prefix="task-image-build-") as scratch:
        path = Path(scratch) / "task_archive"
        with path.open("wb") as output:
            await download_verified_async(
                files,
                source.fileset_ref,
                output,
                limit=MAX_ARCHIVE_BYTES,
                expected_digest=source.files_hash,
            )
        return await asyncio.to_thread(TaskArchive.read, path)


async def _submit(builder: ImageBuilder, image: BuildImage, definition: HarborTaskDefinition) -> None:
    source: FileSetSource | ImageSource
    dockerfile: str | None
    if image.image is not None:
        reference = str(normalize_reference(image.image))
        name = copy_builder_name(reference)
        source, dockerfile = ImageSource(image=reference), None
    else:
        assert image.context_path is not None and image.dockerfile is not None
        name = builder_name(
            archive=definition.source.fileset_ref,
            archive_digest=definition.source.files_hash,
            context_path=image.context_path,
            dockerfile=image.dockerfile,
        )
        source, dockerfile = archive_source(definition.source.fileset_ref, image.context_path), image.dockerfile
    try:
        submitted = await ensure_build(builder, name=name, source=source, dockerfile=dockerfile)
    except NemoClientError as exc:
        image.status = "failed"
        image.error = f"Builder refused the build: {exc}"
        return
    builder_image = submitted.images[0]
    image.builder_image = builder_image.name
    image.build_job = submitted.job
    _apply(image, builder_image.status, builder_image.image_ref, builder_image.status_detail)


async def _refresh(builder: ImageBuilder, images: list[BuildImage]) -> bool:
    """Read each unfinished image's Builder record; True when any image changed."""
    changed = False
    for image in images:
        if image.status in TERMINAL_IMAGE_STATUSES or image.builder_image is None:
            continue
        try:
            current = await builder.get_image(image.builder_image)
            if current is not None and current.status == "pending" and image.build_job is not None:
                status = await builder.job_status(image.build_job)
                if status in ENDED_JOB_STATUSES or status is None:
                    # Builder never fails a row whose build failed: read it once more, after its job ended.
                    current = await builder.get_image(image.builder_image)
                    if current is not None and current.status == "pending":
                        image.status = "failed"
                        image.error = f"Builder job {image.build_job} ended ({status or 'gone'}) without the image"
                        changed = True
                        continue
        except (NemoTransportError, InternalServerError) as exc:
            logger.warning("Builder unavailable reading %s; retrying: %s", image.builder_image, exc)
            continue
        except NemoClientError as exc:
            image.status = "failed"
            image.error = f"could not read the Builder image record: {exc}"
            changed = True
            continue
        if current is None:
            image.status = "failed"
            image.error = f"Builder image record {image.builder_image} no longer exists"
            changed = True
            continue
        before = image.status
        _apply(image, current.status, current.image_ref, current.status_detail)
        changed = changed or image.status != before
    return changed


def _apply(image: BuildImage, status: str, image_ref: str | None, detail: str | None) -> None:
    if status == "ready":
        image.status = "ready"
        image.image_ref = image_ref
    elif status == "failed":
        image.status = "failed"
        image.error = detail or "the Builder build failed"
    else:
        image.status = "building"


async def _fail(entities: EntityClient, build: TaskImageBuildEntity, error: str) -> TaskImageBuildEntity:
    build.status = "failed"
    build.error = error
    logger.info("Image build %s of %s failed: %s", build.name, build.task, error)
    return await entities.update(build)
