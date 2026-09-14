# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tests.utils import assert_matches_type
from nemo_platform import NeMoPlatform, AsyncNeMoPlatform
from nemo_platform._utils import parse_datetime
from nemo_platform.pagination import SyncDefaultPagination, AsyncDefaultPagination
from nemo_platform.types.models import (
    ModelEntity,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestModels:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoPlatform) -> None:
        model = client.models.create(
            workspace="workspace",
            name="llama-3.1-8b",
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoPlatform) -> None:
        model = client.models.create(
            workspace="workspace",
            name="llama-3.1-8b",
            api_endpoint={
                "api_key": "api_key",
                "format": "format",
                "model_id": "model_id",
                "url": "https://example.com",
            },
            backend_format="OPENAI_CHAT",
            base_model="base_model",
            custom_fields={"foo": "bar"},
            description="description",
            fileset="fileset",
            finetuning_type="lora_merged",
            model_providers=["string"],
            ownership={"foo": "bar"},
            project="project",
            prompt={
                "icl_few_shot_examples": "icl_few_shot_examples",
                "inference_params": {
                    "max_completion_tokens": 1,
                    "max_tokens": 1,
                    "model": "model",
                    "stop": ["string"],
                    "temperature": 0,
                    "top_p": 0,
                },
                "system_prompt": "system_prompt",
                "system_prompt_template": "system_prompt_template",
            },
            spec={
                "base_num_parameters": 0,
                "checkpoint_model_name": "checkpoint_model_name",
                "family": "family",
                "ffn_hidden_size": 0,
                "gated_mlp": True,
                "hidden_size": 0,
                "num_attention_heads": 0,
                "num_kv_heads": 0,
                "num_layers": 0,
                "precision": "precision",
                "tied_embeddings": True,
                "vocab_size": 0,
                "chat_template": "chat_template",
                "context_size": 0,
                "head_type": "causal_lm",
                "is_chat": True,
                "is_embedding_model": True,
                "linear_layers": [
                    {
                        "in_features": 0,
                        "name": "name",
                        "out_features": 0,
                    }
                ],
                "mamba_config": {
                    "is_hybrid": True,
                    "num_mamba_layers": 0,
                    "conv_kernel": 0,
                    "num_attention_layers": 0,
                    "num_mlp_layers": 0,
                    "state_size": 0,
                },
                "minimum_gpus_all_weights": 0,
                "minimum_gpus_lora": 0,
                "moe_config": {
                    "num_expert_layers": 0,
                    "num_experts": 0,
                    "num_experts_per_tok": 0,
                    "expert_ffn_size": 0,
                    "num_shared_experts": 0,
                },
                "num_virtual_tokens": 0,
                "sliding_window_config": {"window_size": 0},
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            trust_remote_code=True,
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoPlatform) -> None:
        response = client.models.with_raw_response.create(
            workspace="workspace",
            name="llama-3.1-8b",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoPlatform) -> None:
        with client.models.with_streaming_response.create(
            workspace="workspace",
            name="llama-3.1-8b",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(ModelEntity, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.with_raw_response.create(
                workspace="",
                name="llama-3.1-8b",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoPlatform) -> None:
        model = client.models.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve_with_all_params(self, client: NeMoPlatform) -> None:
        model = client.models.retrieve(
            name="name",
            workspace="workspace",
            verbose=True,
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoPlatform) -> None:
        response = client.models.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoPlatform) -> None:
        with client.models.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(ModelEntity, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.models.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update(self, client: NeMoPlatform) -> None:
        model = client.models.update(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_with_all_params(self, client: NeMoPlatform) -> None:
        model = client.models.update(
            name="name",
            workspace="workspace",
            verbose=True,
            api_endpoint={
                "api_key": "api_key",
                "format": "format",
                "model_id": "model_id",
                "url": "https://example.com",
            },
            backend_format="OPENAI_CHAT",
            base_model="base_model",
            custom_fields={"foo": "bar"},
            description="description",
            fileset="fileset",
            finetuning_type="lora_merged",
            model_providers=["string"],
            ownership={"foo": "bar"},
            prompt={
                "icl_few_shot_examples": "icl_few_shot_examples",
                "inference_params": {
                    "max_completion_tokens": 1,
                    "max_tokens": 1,
                    "model": "model",
                    "stop": ["string"],
                    "temperature": 0,
                    "top_p": 0,
                },
                "system_prompt": "system_prompt",
                "system_prompt_template": "system_prompt_template",
            },
            spec={
                "base_num_parameters": 0,
                "checkpoint_model_name": "checkpoint_model_name",
                "family": "family",
                "ffn_hidden_size": 0,
                "gated_mlp": True,
                "hidden_size": 0,
                "num_attention_heads": 0,
                "num_kv_heads": 0,
                "num_layers": 0,
                "precision": "precision",
                "tied_embeddings": True,
                "vocab_size": 0,
                "chat_template": "chat_template",
                "context_size": 0,
                "head_type": "causal_lm",
                "is_chat": True,
                "is_embedding_model": True,
                "linear_layers": [
                    {
                        "in_features": 0,
                        "name": "name",
                        "out_features": 0,
                    }
                ],
                "mamba_config": {
                    "is_hybrid": True,
                    "num_mamba_layers": 0,
                    "conv_kernel": 0,
                    "num_attention_layers": 0,
                    "num_mlp_layers": 0,
                    "state_size": 0,
                },
                "minimum_gpus_all_weights": 0,
                "minimum_gpus_lora": 0,
                "moe_config": {
                    "num_expert_layers": 0,
                    "num_experts": 0,
                    "num_experts_per_tok": 0,
                    "expert_ffn_size": 0,
                    "num_shared_experts": 0,
                },
                "num_virtual_tokens": 0,
                "sliding_window_config": {"window_size": 0},
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            trust_remote_code=True,
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update(self, client: NeMoPlatform) -> None:
        response = client.models.with_raw_response.update(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update(self, client: NeMoPlatform) -> None:
        with client.models.with_streaming_response.update(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(ModelEntity, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.with_raw_response.update(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.models.with_raw_response.update(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoPlatform) -> None:
        model = client.models.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[ModelEntity], model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoPlatform) -> None:
        model = client.models.list(
            workspace="workspace",
            filter={
                "adapters": {"finetuning_type": "lora_merged"},
                "base_model": {
                    "name": {
                        "eq": "$eq",
                        "in_": ["string"],
                        "like": "$like",
                        "nin": ["string"],
                    }
                },
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "description": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "fileset": True,
                "finetuning_type": "lora_merged",
                "lora_enabled": True,
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "prompt": True,
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=0,
            page_size=0,
            sort="name",
            verbose=True,
        )
        assert_matches_type(SyncDefaultPagination[ModelEntity], model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoPlatform) -> None:
        response = client.models.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(SyncDefaultPagination[ModelEntity], model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoPlatform) -> None:
        with client.models.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(SyncDefaultPagination[ModelEntity], model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoPlatform) -> None:
        model = client.models.delete(
            name="name",
            workspace="workspace",
        )
        assert model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoPlatform) -> None:
        response = client.models.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoPlatform) -> None:
        with client.models.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert model is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.models.with_raw_response.delete(
                name="",
                workspace="workspace",
            )


class TestAsyncModels:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.create(
            workspace="workspace",
            name="llama-3.1-8b",
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.create(
            workspace="workspace",
            name="llama-3.1-8b",
            api_endpoint={
                "api_key": "api_key",
                "format": "format",
                "model_id": "model_id",
                "url": "https://example.com",
            },
            backend_format="OPENAI_CHAT",
            base_model="base_model",
            custom_fields={"foo": "bar"},
            description="description",
            fileset="fileset",
            finetuning_type="lora_merged",
            model_providers=["string"],
            ownership={"foo": "bar"},
            project="project",
            prompt={
                "icl_few_shot_examples": "icl_few_shot_examples",
                "inference_params": {
                    "max_completion_tokens": 1,
                    "max_tokens": 1,
                    "model": "model",
                    "stop": ["string"],
                    "temperature": 0,
                    "top_p": 0,
                },
                "system_prompt": "system_prompt",
                "system_prompt_template": "system_prompt_template",
            },
            spec={
                "base_num_parameters": 0,
                "checkpoint_model_name": "checkpoint_model_name",
                "family": "family",
                "ffn_hidden_size": 0,
                "gated_mlp": True,
                "hidden_size": 0,
                "num_attention_heads": 0,
                "num_kv_heads": 0,
                "num_layers": 0,
                "precision": "precision",
                "tied_embeddings": True,
                "vocab_size": 0,
                "chat_template": "chat_template",
                "context_size": 0,
                "head_type": "causal_lm",
                "is_chat": True,
                "is_embedding_model": True,
                "linear_layers": [
                    {
                        "in_features": 0,
                        "name": "name",
                        "out_features": 0,
                    }
                ],
                "mamba_config": {
                    "is_hybrid": True,
                    "num_mamba_layers": 0,
                    "conv_kernel": 0,
                    "num_attention_layers": 0,
                    "num_mlp_layers": 0,
                    "state_size": 0,
                },
                "minimum_gpus_all_weights": 0,
                "minimum_gpus_lora": 0,
                "moe_config": {
                    "num_expert_layers": 0,
                    "num_experts": 0,
                    "num_experts_per_tok": 0,
                    "expert_ffn_size": 0,
                    "num_shared_experts": 0,
                },
                "num_virtual_tokens": 0,
                "sliding_window_config": {"window_size": 0},
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            trust_remote_code=True,
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.models.with_raw_response.create(
            workspace="workspace",
            name="llama-3.1-8b",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.models.with_streaming_response.create(
            workspace="workspace",
            name="llama-3.1-8b",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(ModelEntity, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.with_raw_response.create(
                workspace="",
                name="llama-3.1-8b",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve_with_all_params(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.retrieve(
            name="name",
            workspace="workspace",
            verbose=True,
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.models.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.models.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(ModelEntity, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.models.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.update(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.update(
            name="name",
            workspace="workspace",
            verbose=True,
            api_endpoint={
                "api_key": "api_key",
                "format": "format",
                "model_id": "model_id",
                "url": "https://example.com",
            },
            backend_format="OPENAI_CHAT",
            base_model="base_model",
            custom_fields={"foo": "bar"},
            description="description",
            fileset="fileset",
            finetuning_type="lora_merged",
            model_providers=["string"],
            ownership={"foo": "bar"},
            prompt={
                "icl_few_shot_examples": "icl_few_shot_examples",
                "inference_params": {
                    "max_completion_tokens": 1,
                    "max_tokens": 1,
                    "model": "model",
                    "stop": ["string"],
                    "temperature": 0,
                    "top_p": 0,
                },
                "system_prompt": "system_prompt",
                "system_prompt_template": "system_prompt_template",
            },
            spec={
                "base_num_parameters": 0,
                "checkpoint_model_name": "checkpoint_model_name",
                "family": "family",
                "ffn_hidden_size": 0,
                "gated_mlp": True,
                "hidden_size": 0,
                "num_attention_heads": 0,
                "num_kv_heads": 0,
                "num_layers": 0,
                "precision": "precision",
                "tied_embeddings": True,
                "vocab_size": 0,
                "chat_template": "chat_template",
                "context_size": 0,
                "head_type": "causal_lm",
                "is_chat": True,
                "is_embedding_model": True,
                "linear_layers": [
                    {
                        "in_features": 0,
                        "name": "name",
                        "out_features": 0,
                    }
                ],
                "mamba_config": {
                    "is_hybrid": True,
                    "num_mamba_layers": 0,
                    "conv_kernel": 0,
                    "num_attention_layers": 0,
                    "num_mlp_layers": 0,
                    "state_size": 0,
                },
                "minimum_gpus_all_weights": 0,
                "minimum_gpus_lora": 0,
                "moe_config": {
                    "num_expert_layers": 0,
                    "num_experts": 0,
                    "num_experts_per_tok": 0,
                    "expert_ffn_size": 0,
                    "num_shared_experts": 0,
                },
                "num_virtual_tokens": 0,
                "sliding_window_config": {"window_size": 0},
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            trust_remote_code=True,
        )
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.models.with_raw_response.update(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(ModelEntity, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.models.with_streaming_response.update(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(ModelEntity, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.with_raw_response.update(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.models.with_raw_response.update(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[ModelEntity], model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.list(
            workspace="workspace",
            filter={
                "adapters": {"finetuning_type": "lora_merged"},
                "base_model": {
                    "name": {
                        "eq": "$eq",
                        "in_": ["string"],
                        "like": "$like",
                        "nin": ["string"],
                    }
                },
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "description": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "fileset": True,
                "finetuning_type": "lora_merged",
                "lora_enabled": True,
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "prompt": True,
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=0,
            page_size=0,
            sort="name",
            verbose=True,
        )
        assert_matches_type(AsyncDefaultPagination[ModelEntity], model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.models.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(AsyncDefaultPagination[ModelEntity], model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.models.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(AsyncDefaultPagination[ModelEntity], model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoPlatform) -> None:
        model = await async_client.models.delete(
            name="name",
            workspace="workspace",
        )
        assert model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.models.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.models.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert model is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.models.with_raw_response.delete(
                name="",
                workspace="workspace",
            )
