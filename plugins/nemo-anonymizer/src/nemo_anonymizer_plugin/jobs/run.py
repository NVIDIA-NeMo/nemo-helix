# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anonymizer run job — submitted to the cpu-tasks container."""

from __future__ import annotations

from typing import ClassVar, cast

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidConfigError
from data_designer_nemo.errors import NDDInvalidConfigError
from nemo_anonymizer_plugin.app.context import create_anonymizer_context, require_model_configs_for_execution
from nemo_anonymizer_plugin.app.errors import AnonymizerInvalidConfigError
from nemo_anonymizer_plugin.app.model_configs import (
    build_model_configs_yaml,
    validate_selected_models_have_model_configs,
)
from nemo_anonymizer_plugin.app.task_config import (
    AnonymizerRequest,
    AnonymizerStepConfig,
)
from nemo_anonymizer_plugin.tasks.anonymizer.run import run_step_config
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    PlatformJobSpec,
    PlatformJobStep,
)
from nemo_platform_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_platform_plugin.jobs.image import get_qualified_image
from pydantic import BaseModel


class RunJob(NemoJob):
    name: ClassVar[str] = "run"
    description: ClassVar[str] = "Anonymize a dataset of records"
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False

    input_spec_schema = AnonymizerRequest
    spec_schema = AnonymizerStepConfig

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,  # AnonymizerRequest
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform,
        is_local: bool,
    ) -> AnonymizerStepConfig:
        del entity_client, is_local
        input_spec = cast(AnonymizerRequest, input_spec)
        anon_ctx = create_anonymizer_context(async_sdk, workspace)

        try:
            cls._validate_anonymizer_config(input_spec.config)
            anon_ctx.validate_input_reference(input_spec.data)
            validate_selected_models_have_model_configs(
                model_configs=input_spec.model_configs,
                selected_models=input_spec.selected_models,
            )
            model_configs = require_model_configs_for_execution(input_spec.model_configs)

            dd_providers = await anon_ctx.make_model_providers(model_configs)
            yaml_body = build_model_configs_yaml(
                model_configs=model_configs,
                selected_models=input_spec.selected_models,
            )
        except (AnonymizerInvalidConfigError, NDDInvalidConfigError) as e:
            raise PlatformJobCompilationError(str(e)) from e

        return AnonymizerStepConfig(
            request=input_spec,
            model_configs_yaml=yaml_body,
            dd_model_providers=[p.model_dump(mode="json") for p in dd_providers or []],
        )

    @classmethod
    def _validate_anonymizer_config(cls, config: AnonymizerConfig) -> None:
        # ``Anonymizer.validate_config`` cross-checks the user-supplied config
        # against the model selection (e.g. that a ``Substitute`` strategy has
        # a ``replacement_generator`` model defined). Doing it here gives the
        # caller a synchronous 422 instead of an async job failure.
        try:
            Anonymizer().validate_config(config)
        except InvalidConfigError as e:
            raise AnonymizerInvalidConfigError(str(e)) from e

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,  # AnonymizerStepConfig
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNeMoPlatform,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="anonymizer-job",
                    executor=CPUExecutionProviderSpec(
                        profile=profile or "default",
                        provider="cpu",
                        container=ContainerSpec(
                            image=get_qualified_image("nmp-cpu-tasks"),
                            entrypoint=["python", "-m"],
                            command=["nemo_anonymizer_plugin.tasks.anonymizer"],
                        ),
                    ),
                    config=spec.model_dump(),
                    environment=_make_env_vars(),
                )
            ],
        )

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        sdk: NeMoPlatform,
    ) -> dict:
        step_config = AnonymizerStepConfig.model_validate(config)
        return {"exit_code": run_step_config(step_config, ctx=ctx, sdk=sdk)}


def _make_env_vars() -> list[EnvironmentVariable]:
    return [
        EnvironmentVariable(
            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            value=DEFAULT_JOB_STORAGE_PATH,
        ),
    ]
