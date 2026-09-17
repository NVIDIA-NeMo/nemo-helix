# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform Job for one scaled-evals task-image build attempt."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar

from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
    ExecutorSpec,
    PlatformJobSpec,
    PlatformJobStep,
    ResourcesSpec,
    SubprocessExecutionProviderSpec,
)
from nemo_scaled_evals_plugin.jobs.naming import task_image_build_job_name
from nemo_scaled_evals_plugin.jobs.specs import TaskImageBuildSpec
from pydantic import BaseModel
from scaled_evals.api.build.queue_worker import TaskBuildWorker
from scaled_evals.api.repositories.build_repository import TaskBuildJob as BackendTaskBuildJob
from scaled_evals.api.settings import settings


class TaskImageBuildJob(NemoJob):
    """Build or resolve one immutable task image."""

    name: ClassVar[str] = "task-image-build"
    description: ClassVar[str] = "Build or resolve a scaled-evals task image."
    spec_schema: ClassVar[type[BaseModel]] = TaskImageBuildSpec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> PlatformJobSpec:
        """Compile a task-image build into one CPU container step."""
        del workspace, entity_client, job_name, async_sdk
        canonical = TaskImageBuildSpec.model_validate(spec)
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="task-image-build",
                    executor=resolve_executor(
                        options,
                        profile=profile or "default",
                        module="nemo_scaled_evals_plugin.tasks.task_image_build",
                    ),
                    environment=resolve_secret_environment(),
                    config=canonical.model_dump(mode="json"),
                )
            ]
        )

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute the frozen backend operation and persist its terminal state."""
        spec = TaskImageBuildSpec.model_validate(config)
        worker = TaskBuildWorker(
            worker_id=task_image_build_job_name(
                spec.task_id,
                spec.revision,
                spec.build_attempt,
            )
        )
        completed = worker.run(
            BackendTaskBuildJob(
                task_id=spec.task_id,
                revision=spec.revision,
                backend=spec.backend,
                payload=spec.payload,
                credentials={},
                object_key=spec.object_key,
                attempt=spec.build_attempt,
            )
        )
        return {"status": "completed" if completed else "failed"}


def resolve_application_image(options: dict[str, Any] | None) -> str | None:
    """Resolve the application image from submission or deployment configuration."""
    configured = (options or {}).get("scaled_evals")
    if isinstance(configured, Mapping):
        image = configured.get("application_image")
        if isinstance(image, str) and image.strip():
            return image.strip()
    return os.getenv("SCALED_EVALS_APPLICATION_IMAGE") or os.getenv("API_IMAGE")


def resolve_executor(
    options: dict[str, Any] | None,
    *,
    profile: str,
    module: str,
    resources: ResourcesSpec | None = None,
) -> ExecutorSpec:
    """Resolve the configured Platform Jobs execution provider."""
    configured = (options or {}).get("scaled_evals")
    provider = configured.get("provider") if isinstance(configured, Mapping) else None
    if provider == "subprocess":
        return SubprocessExecutionProviderSpec(
            provider="subprocess",
            profile=profile,
            command=["python", "-m", module],
        )
    executor = CPUExecutionProviderSpec(
        provider="cpu",
        profile=profile,
        container=ContainerSpec(
            image=resolve_application_image(options),
            entrypoint=["python", "-m"],
            command=[module],
        ),
    )
    return executor if resources is None else executor.model_copy(update={"resources": resources})


def resolve_secret_environment() -> list[EnvironmentVariable] | None:
    """Reference deployment secrets needed by isolated Platform Job containers."""
    refs = {
        "PGPASSWORD": settings.platform_jobs_postgres_password_secret,
        "CREDENTIALS_ENCRYPTION_KEY": settings.platform_jobs_credentials_encryption_key_secret,
        "TASK_IMAGE_REGISTRY_AUTH_JSON": settings.platform_jobs_registry_auth_secret,
    }
    environment = [
        EnvironmentVariable(name=name, from_secret=EnvironmentVariableFromSecret(name=secret))
        for name, secret in refs.items()
        if secret
    ]
    return environment or None
