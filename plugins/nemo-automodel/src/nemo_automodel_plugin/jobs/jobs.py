# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel training job (NemoJob).

Shared scaffold (``to_spec`` + the Docker-runtime guard) lives in
:class:`nmp.customization_common.contributor.jobs.BaseSubmitJob`; ``compile`` stays here
because it validates for training and resolves the execution profile from the
schema (automodel-specific).
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from nemo_automodel_plugin.config import get_config
from nemo_automodel_plugin.schema import AutomodelJobInput, AutomodelJobOutput, ValidationError
from nemo_automodel_plugin.transform import transform_input_to_output
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nemo_platform_plugin.jobs.docker import validate_gpu_available_for_docker
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nmp.automodel.compile import platform_job_config_compiler
from nmp.customization_common.contributor.jobs import BaseSubmitJob, require_container_runtime
from nmp.customization_common.service.platform_client import (
    AsyncCustomizationPlatformClients,
    async_customization_platform_clients_from_platform,
)
from pydantic import BaseModel


class AutomodelJob(BaseSubmitJob[AutomodelJobInput, AutomodelJobOutput]):
    """GPU Automodel fine-tuning job under the customization router."""

    name: ClassVar[str] = "automodel.jobs"
    description: ClassVar[str] = "Automodel SFT, retrieval, and knowledge-distillation training jobs."
    job_collection_path: ClassVar[str | None] = "/automodel/jobs"
    input_spec_schema: ClassVar[type[AutomodelJobInput] | None] = AutomodelJobInput
    spec_schema: ClassVar[type[AutomodelJobOutput] | None] = AutomodelJobOutput
    runtime_label: ClassVar[str] = "Automodel"

    @classmethod
    def _job_input_schema(cls) -> type[AutomodelJobInput]:
        return AutomodelJobInput

    @classmethod
    async def _transform(
        cls,
        job_input: AutomodelJobInput,
        workspace: str,
        platform: AsyncCustomizationPlatformClients,
    ) -> AutomodelJobOutput:
        return await transform_input_to_output(job_input, workspace, platform)

    @classmethod
    async def compile(
        cls,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNeMoPlatform,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        del entity_client, options
        platform = async_customization_platform_clients_from_platform(async_sdk)
        canonical = (
            spec if isinstance(spec, AutomodelJobOutput) else AutomodelJobOutput.model_validate(spec.model_dump())
        )
        # Multi-node jobs compile to a gpu_distributed (Volcano) executor, which
        # only exists on Kubernetes; gate here so docker platforms fail fast.
        # Probe is sync (≤5s); keep it off the event loop.
        await asyncio.to_thread(
            require_container_runtime,
            cls.runtime_label,
            num_nodes=canonical.parallelism.num_nodes,
        )
        try:
            canonical.validate_for_training()
        except ValidationError as e:
            raise PlatformJobCompilationError(str(e)) from e

        plugin_config = get_config()
        execution_profile = (
            canonical.training.execution_profile or profile or plugin_config.default_training_execution_profile
        )

        platform_spec = await platform_job_config_compiler(
            canonical,
            workspace,
            platform,
            job_name=job_name,
            profile=execution_profile,
        )

        validate_gpu_available_for_docker(platform_spec)
        return platform_spec
