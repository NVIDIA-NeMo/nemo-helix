# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import data_designer.config as dd
from data_designer_nemo.context import DataDesignerValidationContext
from data_designer_nemo.errors import NDDInvalidConfigError


def build_retrieval_model_configs(
    provider: str,
    artifact_extraction_model: str,
    qa_generation_model: str,
    quality_judge_model: str,
    embed_model: str,
    chat_provider: str | None = None,
    embed_provider: str | None = None,
) -> list[dd.ModelConfig]:
    """Build IGW-facing model configs for the four retrieval SDG roles."""
    chat_provider = chat_provider or provider
    embed_provider = embed_provider or provider
    chat_params = dd.ChatCompletionInferenceParams()
    embed_params = dd.EmbeddingInferenceParams()
    return [
        dd.ModelConfig(
            alias="retrieval-artifact-extraction",
            model=artifact_extraction_model,
            provider=chat_provider,
            inference_parameters=chat_params,
        ),
        dd.ModelConfig(
            alias="retrieval-qa-generation",
            model=qa_generation_model,
            provider=chat_provider,
            inference_parameters=chat_params,
        ),
        dd.ModelConfig(
            alias="retrieval-quality-judge",
            model=quality_judge_model,
            provider=chat_provider,
            inference_parameters=chat_params,
        ),
        dd.ModelConfig(
            alias="retrieval-embed",
            model=embed_model,
            provider=embed_provider,
            inference_parameters=embed_params,
        ),
    ]


async def resolve_retrieval_providers(
    dd_ctx: DataDesignerValidationContext,
    model_configs: list[dd.ModelConfig],
) -> list[dd.ModelProvider]:
    """Resolve Inference Gateway providers for retrieval model configs."""
    providers = await dd_ctx.get_model_providers(model_configs)
    if not providers:
        raise NDDInvalidConfigError("No Inference Gateway model providers resolved for retrieval SDG.")
    return providers
