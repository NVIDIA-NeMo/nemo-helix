# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task-builder Job: builds every container image of one task revision through NeMo Builder."""

from __future__ import annotations

import asyncio

from nemo_evals.image_builds.builder import HttpImageBuilder
from nemo_evals.image_builds.models import TASK_IMAGE_BUILD_JOB_SOURCE
from nemo_evals.image_builds.runner import TaskImageBuildSpec, run_task_image_build
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
)
from nemo_helix_plugin.jobs.image import get_qualified_image
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

TASK_IMAGE_BUILD_MODULE = "nemo_evals.tasks.task_image_build"

TASK_IMAGE_BUILD_RESOURCES = ResourcesSpec(
    requests=ResourcesRequestsSpec(cpu="50m", memory="128Mi"),
    limits=ResourcesLimitsSpec(cpu="500m", memory="512Mi"),
)


class TaskImageBuildJob(NemoJob):
    """Build the container images of one task revision."""

    name = "task-image-build"
    description = "Build the container images of a task revision with NeMo Builder."
    container = "nhx-tasks"
    spec_schema = TaskImageBuildSpec

    def run(self, config: dict, *, ctx: JobContext, async_client: AsyncNemoClient) -> dict:
        """Run the build to a ready or failed record."""
        spec = TaskImageBuildSpec.model_validate(config)
        build = asyncio.run(
            run_task_image_build(
                spec,
                entities=EntityClient(AsyncEntitiesClient.from_client(async_client)),
                files=AsyncFilesClient.from_client(async_client),
                builder=HttpImageBuilder(async_client, workspace=spec.workspace),
            )
        )
        return {"status": "completed" if build.status == "ready" else "failed", "build_status": build.status}


def task_image_build_job_request(
    spec: TaskImageBuildSpec, *, name: str, profile: str = "default"
) -> CreateHelixJobRequest:
    """The Jobs request that starts the task-builder Job ``name`` for ``spec``."""
    step = HelixJobStep(
        name="build-images",
        executor=CPUExecutionProviderSpec(
            profile=profile,
            provider="cpu",
            container=ContainerSpec(
                image=get_qualified_image("nhx-tasks"), entrypoint=["python", "-m"], command=[TASK_IMAGE_BUILD_MODULE]
            ),
            resources=TASK_IMAGE_BUILD_RESOURCES,
        ),
        config=spec.model_dump(mode="json"),
    )
    return CreateHelixJobRequest(
        name=name,
        description=f"Container image build {spec.attempt} of {spec.workspace}/{spec.task}",
        source=TASK_IMAGE_BUILD_JOB_SOURCE,
        spec=spec.model_dump(mode="json"),
        platform_spec=HelixJobSpec(steps=[step]),
        custom_fields={"task_image_build": _build_key(spec)},
    )


def is_job_for(custom_fields: dict | None, spec: TaskImageBuildSpec) -> bool:
    """Whether a Job's custom fields say it runs ``spec``'s build."""
    return (custom_fields or {}).get("task_image_build") == _build_key(spec)


def _build_key(spec: TaskImageBuildSpec) -> str:
    # ``by-contents`` as in ``build_job_name``: a Job started for a build kept under a revision isn't this build's.
    return f"by-contents:{spec.workspace}/{spec.task}#{spec.revision_digest}/{spec.attempt}"
