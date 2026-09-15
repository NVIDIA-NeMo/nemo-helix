# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar, cast

from data_designer_nemo.context import create_validation_context
from nemo_data_designer_plugin.jobs.retrieval_common import retrieval_step, work_dir
from nemo_data_designer_plugin.jobs.retrieval_spec import RetrievalGenerateJobConfig, RetrievalGenerateStepConfig
from nemo_data_designer_plugin.retrieval.corpus import hf_token_from_env, materialize_corpus
from nemo_data_designer_plugin.retrieval.providers import build_retrieval_model_configs, resolve_retrieval_providers
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from pydantic import BaseModel


class RetrievalGenerateJob(NemoJob):
    name: ClassVar[str] = "retrieval-generate"
    description: ClassVar[str] = "Generate retrieval Q&A JSONL from a document corpus (Nemotron Stage 0)."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False

    input_spec_schema = RetrievalGenerateJobConfig
    spec_schema = RetrievalGenerateStepConfig

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        workspace: str,
        entity_client: object,
        async_sdk: object,
        is_local: bool,
    ) -> BaseModel:
        async_sdk = cast(AsyncNeMoPlatform, async_sdk)
        job_config = cast(RetrievalGenerateJobConfig, input_spec)
        dd_ctx = create_validation_context(async_sdk, workspace)
        model_configs = build_retrieval_model_configs(
            provider=job_config.provider,
            chat_provider=job_config.chat_provider,
            embed_provider=job_config.embed_provider,
            artifact_extraction_model=job_config.artifact_extraction_model,
            qa_generation_model=job_config.qa_generation_model,
            quality_judge_model=job_config.quality_judge_model,
            embed_model=job_config.embed_model,
        )
        model_providers = await resolve_retrieval_providers(dd_ctx, model_configs)
        return RetrievalGenerateStepConfig(
            job_config=job_config,
            model_providers=model_providers,
            chat_provider_name=job_config.chat_provider or job_config.provider,
            embed_provider_name=job_config.embed_provider or job_config.provider,
        )

    @classmethod
    async def compile(
        cls,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        spec = cast(RetrievalGenerateStepConfig, spec)
        return PlatformJobSpec(
            steps=[
                await retrieval_step(
                    "retrieval-generate",
                    "nemo_data_designer_plugin.jobs.retrieval_generate",
                    spec,
                    profile=profile,
                    async_sdk=async_sdk,
                    hf_token_secret=spec.job_config.hf_token_secret,
                )
            ]
        )

    def run(self, config: dict, ctx: JobContext, sdk: NeMoPlatform) -> dict:
        from nemo_data_designer_plugin.retrieval.generation import build_generation_run_config, execute_generation

        step = RetrievalGenerateStepConfig.model_validate(config)
        job = step.job_config
        output_dir = work_dir(ctx, "stage0_sdg")
        artifact_path = output_dir / "artifacts"
        corpus_dir = materialize_corpus(
            job.corpus,
            dest=ctx.storage.ephemeral / "corpus",
            sdk=sdk,
            workspace=ctx.workspace,
            hf_token=hf_token_from_env(),
        )
        run_config = build_generation_run_config(
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            artifact_path=artifact_path,
            dataset_name=job.dataset_name or job.corpus_id,
            chat_provider_name=step.chat_provider_name,
            embed_provider_name=step.embed_provider_name,
            model_providers=step.model_providers,
            file_extensions=job.file_extensions,
            min_text_length=job.min_text_length,
            sentences_per_chunk=job.sentences_per_chunk,
            num_sections=job.num_sections,
            num_files=job.num_files,
            max_artifacts_per_type=job.max_artifacts_per_type,
            num_pairs=job.num_pairs,
            query_counts=job.query_counts,
            min_hops=job.min_hops,
            max_hops=job.max_hops,
            reasoning_counts=job.reasoning_counts,
            min_complexity=job.min_complexity,
            similarity_threshold=job.similarity_threshold,
            buffer_size=job.buffer_size,
            resume=job.resume,
            num_records=job.num_records,
            artifact_extraction_model=job.artifact_extraction_model,
            qa_generation_model=job.qa_generation_model,
            quality_judge_model=job.quality_judge_model,
            embed_model=job.embed_model,
        )
        result = execute_generation(run_config)
        artifacts = ctx.results.save(name="artifacts", local_path=output_dir)
        return {
            "exit_code": 0,
            "workspace": ctx.workspace,
            "dataset_name": getattr(result, "dataset_name", job.dataset_name or job.corpus_id),
            "num_records": getattr(result, "num_records", None),
            "results": {"artifacts": artifacts.model_dump()},
        }


if __name__ == "__main__":
    from nemo_data_designer_plugin.jobs.retrieval_bridge import run_job_module

    raise SystemExit(run_job_module(RetrievalGenerateJob, RetrievalGenerateStepConfig))
