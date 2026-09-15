# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar, Literal

from data_designer_nemo.context import create_validation_context
from data_designer_nemo.errors import NDDInvalidConfigError
from nemo_data_designer_plugin.jobs.retrieval_spec import RetrievalPreviewSpec
from nemo_data_designer_plugin.retrieval.corpus import materialize_corpus
from nemo_data_designer_plugin.retrieval.providers import build_retrieval_model_configs, resolve_retrieval_providers
from nemo_data_designer_plugin.retrieval.secrets import resolve_hf_token
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.errors import NemoClientError
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.functions.frames import Done, Error
from pydantic import BaseModel


class RetrievalPreviewFrame(BaseModel):
    kind: Literal["retrieval_preview"] = "retrieval_preview"
    num_seed_records: int
    num_preview_records: int


async def _resolve_hf_token_or_raise(
    async_sdk: AsyncNeMoPlatform,
    hf_token_secret: str | None,
    workspace: str,
) -> str | None:
    """Resolve the spec's secret reference, reporting a bad reference as invalid config."""
    try:
        return await resolve_hf_token(async_sdk, hf_token_secret, workspace)
    except NemoClientError as exc:
        raise NDDInvalidConfigError(f"Could not access secret {hf_token_secret!r} named by hf_token_secret.") from exc


class RetrievalPreviewFunction(NemoFunction[RetrievalPreviewSpec]):
    name: ClassVar[str] = "retrieval-preview"
    description: ClassVar[str] = "Preview retrieval SDG generation without publishing a full job fileset."
    spec_schema: ClassVar[type[RetrievalPreviewSpec]] = RetrievalPreviewSpec
    generate_legacy_verbs: ClassVar[bool] = False

    async def run(
        self,
        spec: RetrievalPreviewSpec,
        ctx: FunctionContext,
        sdk: NeMoPlatform,
        async_sdk: AsyncNeMoPlatform,
        is_local: bool = False,
    ) -> AsyncIterator[BaseModel]:
        job = spec.generate
        validation_ctx = create_validation_context(async_sdk, ctx.workspace)
        model_configs = build_retrieval_model_configs(
            provider=job.provider,
            chat_provider=job.chat_provider,
            embed_provider=job.embed_provider,
            artifact_extraction_model=job.artifact_extraction_model,
            qa_generation_model=job.qa_generation_model,
            quality_judge_model=job.quality_judge_model,
            embed_model=job.embed_model,
        )
        # These can raise NDDInvalidConfigError or NDDInternalError, which map to
        # appropriate error-coded responses rather than returning a 200 with a
        # response stream that only holds an Error frame
        model_providers = await resolve_retrieval_providers(validation_ctx, model_configs)
        hf_token = await _resolve_hf_token_or_raise(async_sdk, job.hf_token_secret, ctx.workspace)

        def run_preview() -> BaseModel:
            from nemo_data_designer_plugin.retrieval.generation import (
                build_generation_run_config,
                execute_generation,
            )

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                corpus_dir = materialize_corpus(
                    job.corpus,
                    dest=tmp_path / "corpus",
                    sdk=sdk,
                    workspace=ctx.workspace,
                    hf_token=hf_token,
                    allow_local_path=is_local,
                )
                run_config = build_generation_run_config(
                    corpus_dir=corpus_dir,
                    output_dir=tmp_path / "out",
                    artifact_path=tmp_path / "artifacts",
                    dataset_name=job.dataset_name or job.corpus_id,
                    chat_provider_name=job.chat_provider or job.provider,
                    embed_provider_name=job.embed_provider or job.provider,
                    model_providers=model_providers,
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
                    num_records=spec.num_records,
                    artifact_extraction_model=job.artifact_extraction_model,
                    qa_generation_model=job.qa_generation_model,
                    quality_judge_model=job.quality_judge_model,
                    embed_model=job.embed_model,
                )
                result = execute_generation(run_config, preview=True, num_records=spec.num_records)
                return RetrievalPreviewFrame(
                    num_seed_records=getattr(result, "num_seed_records", 0),
                    num_preview_records=getattr(result, "num_preview_records", spec.num_records),
                )

        try:
            yield await asyncio.to_thread(run_preview)
        except Exception as exc:
            yield Error(message=str(exc), details={"type": type(exc).__name__})
            return
        yield Done()
