# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job API endpoints for hello world service."""

from fastapi import APIRouter
from nemo_platform_plugin.client.adapter import AsyncPlatformClient
from nemo_platform_plugin.entities import EntityClient
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    PlatformJobSpec,
    PlatformJobStep,
    job_route_factory,
)
from nemo_platform_plugin.jobs.image import get_qualified_image
from nmp.hello_world.api.v2.jobs.schemas import HelloWorldJobConfig


def compile_hello_world_job(
    workspace: str,
    original_spec: HelloWorldJobConfig,
    transformed_spec: HelloWorldJobConfig,
    entity_client: EntityClient,
    job_name: str | None,
    sdk: AsyncPlatformClient,
) -> PlatformJobSpec:
    """Compile a hello world job config into a platform job spec.

    Args:
        workspace: The workspace for this job.
        original_spec: The user-provided input specification.
        transformed_spec: The spec after applying the input-to-output transformer.
            Since no transformer is configured for hello-world, original_spec
            and transformed_spec are identical.
        entity_client: Entity client for lookups.
        job_name: The resolved job name (user-provided or auto-generated).
        sdk: Platform client for accessing secrets, files, and models with user context.
    """
    return PlatformJobSpec(
        steps=[
            PlatformJobStep(
                name="hello-world",
                executor=CPUExecutionProviderSpec(
                    provider="cpu",
                    profile="default",
                    container=ContainerSpec(
                        image=get_qualified_image("nmp-cpu-tasks"),
                        entrypoint=["nemo-platform"],
                        command=[
                            "run",
                            "task",
                            "--task",
                            "nmp.hello_world.tasks.hello_world",
                        ],
                    ),
                ),
                config=transformed_spec.model_dump(),
            )
        ]
    )


jobs_router = job_route_factory(
    service_name="hello-world",
    job_type="HelloWorld",
    job_input=HelloWorldJobConfig,
    platform_job_config_compiler=compile_hello_world_job,
)

# Create a router for jobs (prefix will be added at service level)
router = APIRouter()
router.include_router(jobs_router)
