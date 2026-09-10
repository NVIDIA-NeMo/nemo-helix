# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe Synthesizer generate job."""

from __future__ import annotations

import logging
from typing import Any, ClassVar
from urllib.parse import urlparse

from filesets import FilesetPathError, parse_fileset_ref
from nemo_platform_plugin.client.adapter import AsyncPlatformClient, client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
    FileResultSerializer,
    GPUExecutionProviderSpec,
    PlatformJobResultRoute,
    PlatformJobSpec,
    PlatformJobStep,
    PydanticResultSerializer,
    ResourcesLimitsSpec,
    ResourcesRequestsSpec,
    ResourcesSpec,
)
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_platform_plugin.jobs.image import get_qualified_image
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_safe_synthesizer.config.external_results import SafeSynthesizerSummary
from nemo_safe_synthesizer_plugin.config import config as plugin_config
from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerJobConfig, parse_pretrained_model_job_ref
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RESULT_ROUTES: list[PlatformJobResultRoute] = [
    PlatformJobResultRoute(
        name="summary",
        serializer=PydanticResultSerializer(model=SafeSynthesizerSummary),
    ),
    PlatformJobResultRoute(
        name="synthetic-data",
        serializer=FileResultSerializer(),
    ),
    PlatformJobResultRoute(
        name="evaluation-report",
        serializer=FileResultSerializer(),
    ),
    PlatformJobResultRoute(
        name="adapter",
        serializer=FileResultSerializer(),
    ),
]


class GenerateJob(NemoJob):
    """Submit a Safe Synthesizer generation job to the platform."""

    name: ClassVar[str] = "generate"
    description: ClassVar[str] = "Generate synthetic data using Safe Synthesizer."
    spec_schema: ClassVar[type[BaseModel] | None] = SafeSynthesizerJobConfig
    generate_legacy_verbs: ClassVar[bool] = False
    # Preserve the existing /jobs URL instead of /jobs/generate.
    job_collection_path: ClassVar[str | None] = "/jobs"

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncPlatformClient,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        assert isinstance(spec, SafeSynthesizerJobConfig)
        if options:
            raise PlatformJobCompilationError("Safe Synthesizer does not support submit options.")

        steps = []

        try:
            ds_workspace, fileset_name, _ = parse_fileset_ref(spec.data_source, workspace_fallback=workspace)
        except FilesetPathError as e:
            raise PlatformJobCompilationError(f"Invalid data_source format: {spec.data_source!r}") from e
        files = client_from_platform(async_sdk, AsyncFilesClient)
        try:
            await files.get_fileset(name=fileset_name, workspace=ds_workspace)
        except ClientNotFoundError as e:
            raise PlatformJobCompilationError(
                f"Could not find fileset {fileset_name!r} in workspace {ds_workspace!r}"
            ) from e
        except ClientPermissionDeniedError as e:
            raise PermissionError(f"Access denied to fileset {fileset_name!r} in workspace {ds_workspace!r}") from e

        environment: list[EnvironmentVariable] = [
            EnvironmentVariable(name="DATA_SOURCE", value=spec.data_source),
        ]

        classify_model_provider = None
        if spec.config.replace_pii:
            classify_model_provider = spec.config.replace_pii.globals.classify.classify_model_provider
        if classify_model_provider:
            parts = classify_model_provider.split("/", 1)
            if len(parts) != 2:
                raise PlatformJobCompilationError(
                    f"Invalid classify_model_provider format: '{classify_model_provider}'. "
                    "Expected 'workspace/provider_name' format."
                )
            provider_workspace, provider_name = parts
            models = client_from_platform(async_sdk, AsyncModelsClient)
            try:
                provider = (await models.get_provider(name=provider_name, workspace=provider_workspace)).data()
            except ClientNotFoundError as e:
                raise PlatformJobCompilationError(
                    f"Could not find model provider {provider_name!r} in workspace {provider_workspace!r}"
                ) from e
            except ClientPermissionDeniedError as e:
                raise PlatformJobCompilationError(
                    f"Failed to retrieve model provider {classify_model_provider!r}: Access denied to workspace {provider_workspace!r}"
                ) from e
            nim_endpoint_url = models.get_provider_route_openai_url(provider)
            parsed_url = urlparse(nim_endpoint_url)
            environment.append(EnvironmentVariable(name="CLASSIFY_LLM_ENDPOINT_PATH", value=parsed_url.path))
            logger.info("Configured NIM endpoint URL: %s (provider: %s)", nim_endpoint_url, classify_model_provider)

        if spec.hf_token_secret:
            environment.append(
                EnvironmentVariable(
                    name="HF_TOKEN", from_secret=EnvironmentVariableFromSecret(name=spec.hf_token_secret)
                )
            )

        if spec.pretrained_model_job:
            model_workspace, model_job = parse_pretrained_model_job_ref(
                spec.pretrained_model_job, workspace_fallback=workspace
            )
            try:
                jobs_client = client_from_platform(async_sdk, AsyncJobsClient)
                await jobs_client.get_job_result(name="adapter", job=model_job, workspace=model_workspace)
            except ClientNotFoundError as e:
                raise PlatformJobCompilationError(
                    f"Could not find adapter result for NSS job {model_workspace}/{model_job!r}"
                ) from e
            except ClientPermissionDeniedError as e:
                raise PlatformJobCompilationError(
                    f"Failed to retrieve adapter result for NSS job {model_workspace}/{model_job!r}: "
                    f"access denied to workspace {model_workspace!r}"
                ) from e

        if spec.config:
            steps.append(_create_job_step(spec, environment, profile=profile))

        if not steps:
            raise PlatformJobCompilationError("No steps to run")
        return PlatformJobSpec(steps=steps)

    def run(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError("Safe Synthesizer does not support local execution.")


def _create_job_step(
    spec: SafeSynthesizerJobConfig,
    environment: list[EnvironmentVariable],
    *,
    profile: str | None = None,
) -> PlatformJobStep:
    resources = ResourcesSpec(
        limits=ResourcesLimitsSpec(
            memory=plugin_config.default_job_resource_memory_limit,
            cpu=plugin_config.default_job_resource_cpu_limit,
        ),
        requests=ResourcesRequestsSpec(
            memory=plugin_config.default_job_resource_memory_request,
            cpu=plugin_config.default_job_resource_cpu_request,
        ),
    )
    return PlatformJobStep(
        name="safe-synthesizer",
        executor=GPUExecutionProviderSpec(
            provider="gpu",
            profile=profile or plugin_config.job_executor_profile,
            container=ContainerSpec(
                image=plugin_config.container_image_ref or get_qualified_image(plugin_config.container_image),
                entrypoint=plugin_config.entrypoint,
            ),
            resources=resources,
        ),
        config=_task_job_config(spec),
        environment=environment,
    )


def _task_job_config(spec: SafeSynthesizerJobConfig) -> dict[str, Any]:
    dumped = spec.model_dump()
    if spec.pretrained_model_job:
        training = dumped.get("config", {}).get("training")
        if isinstance(training, dict):
            training.pop("pretrained_model", None)
    return dumped
