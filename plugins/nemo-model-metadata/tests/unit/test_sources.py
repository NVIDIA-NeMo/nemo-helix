# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelModality
from nemo_model_metadata.sources.provider import parse_provider_model


def test_parse_provider_model_reads_openrouter_fields():
    metadata = parse_provider_model(
        {
            "id": "meta-llama/llama-3.1-8b-instruct",
            "name": "Meta: Llama 3.1 8B Instruct",
            "description": "Meta's latest class of model",
            "context_length": 131072,
            "architecture": {"input_modalities": ["text", "file"], "output_modalities": ["text"]},
            "top_provider": {"max_completion_tokens": 16384},
            "supported_parameters": ["tools", "tool_choice", "response_format", "temperature"],
        }
    )

    assert metadata == ModelMetadata(
        display_name="Meta: Llama 3.1 8B Instruct",
        description="Meta's latest class of model",
        capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT],
        input_modalities=[ModelModality.TEXT, ModelModality.PDF],
        output_modalities=[ModelModality.TEXT],
        context_length=131072,
        max_output_tokens=16384,
    )


def test_parse_provider_model_reads_vllm_max_model_len():
    metadata = parse_provider_model({"id": "Qwen/Qwen2.5-7B-Instruct", "owned_by": "vllm", "max_model_len": 32768})

    assert metadata == ModelMetadata(context_length=32768)


def test_parse_provider_model_returns_none_for_bare_openai_entry():
    assert parse_provider_model({"id": "gpt-4o", "object": "model", "owned_by": "openai"}) is None
