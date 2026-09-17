# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar, cast

from nemo_data_designer_plugin.jobs.retrieval_generate import RetrievalGenerateJob
from nemo_data_designer_plugin.jobs.retrieval_prepare import RetrievalPrepareJob
from nemo_data_designer_plugin.jobs.retrieval_spec import (
    RetrievalPrepareStepConfig,
    RetrievalRunJobConfig,
)
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from pydantic import BaseModel


class RetrievalRunJob(NemoJob):
    name: ClassVar[str] = "retrieval-run"
    description: ClassVar[str] = "Chain retrieval generate then prepare as a multi-step jobs-service workflow."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False

    input_spec_schema = RetrievalRunJobConfig
    spec_schema = RetrievalRunJobConfig

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNeMoPlatform,
        is_local: bool,
    ) -> BaseModel:
        run = cast(RetrievalRunJobConfig, input_spec)
        # Resolve the generate spec so unresolvable IGW providers fail at submit
        # rather than after the first step is already scheduled.
        await RetrievalGenerateJob.to_spec(
            run.generate,
            workspace=workspace,
            entity_client=entity_client,
            async_sdk=async_sdk,
            is_local=False,
        )
        prepare_cfg = run.prepare
        if prepare_cfg.sdg_input is None and prepare_cfg.train_input_file is None:
            prepare_cfg = prepare_cfg.model_copy(update={"sdg_input": "stage0_sdg"})
        prepare_step = cast(
            RetrievalPrepareStepConfig,
            await RetrievalPrepareJob.to_spec(
                prepare_cfg,
                workspace=workspace,
                entity_client=entity_client,
                async_sdk=async_sdk,
                is_local=False,
            ),
        )
        return RetrievalRunJobConfig(generate=run.generate, prepare=prepare_step.job_config)

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
        spec = cast(RetrievalRunJobConfig, spec)
        generate_step = await RetrievalGenerateJob.to_spec(
            spec.generate,
            workspace=workspace,
            entity_client=entity_client,
            async_sdk=async_sdk,
            is_local=False,
        )
        generate_job = await RetrievalGenerateJob.compile(
            workspace=workspace,
            spec=generate_step,
            entity_client=entity_client,
            job_name=job_name,
            async_sdk=async_sdk,
            profile=profile,
            options=options,
        )
        prepare_input = spec.prepare
        if prepare_input.sdg_input is None and prepare_input.train_input_file is None:
            prepare_input = prepare_input.model_copy(update={"sdg_input": "stage0_sdg"})
        prepare_step = cast(
            RetrievalPrepareStepConfig,
            await RetrievalPrepareJob.to_spec(
                prepare_input,
                workspace=workspace,
                entity_client=entity_client,
                async_sdk=async_sdk,
                is_local=False,
            ),
        )
        prepare_job = await RetrievalPrepareJob.compile(
            workspace=workspace,
            spec=prepare_step,
            entity_client=entity_client,
            job_name=job_name,
            async_sdk=async_sdk,
            profile=profile,
            options=options,
        )
        return PlatformJobSpec(steps=[*generate_job["steps"], *prepare_job["steps"]])

    def run(self, config: dict, *, ctx: JobContext, sdk: NeMoPlatform) -> dict:
        del config, ctx, sdk
        raise NotImplementedError("retrieval-run is remote-only; compile emits generate and prepare steps.")
