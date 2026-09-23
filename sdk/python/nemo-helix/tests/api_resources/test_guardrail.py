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
from nemo_helix import NeMoHelix, AsyncNeMoHelix
from nemo_helix.types.guardrail import GuardrailCheckResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestGuardrail:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_check(self, client: NeMoHelix) -> None:
        guardrail = client.guardrail.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                }
            ],
            model="model",
        )
        assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_check_with_all_params(self, client: NeMoHelix) -> None:
        guardrail = client.guardrail.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                    "name": "name",
                }
            ],
            model="model",
            frequency_penalty=-2,
            function_call="string",
            guardrails={
                "config": "string",
                "config_id": "config_id",
                "config_ids": ["string"],
                "context": {"foo": "bar"},
                "options": {
                    "llm_output": True,
                    "llm_params": {"foo": "bar"},
                    "log": {
                        "activated_rails": True,
                        "colang_history": True,
                        "internal_events": True,
                        "llm_calls": True,
                        "stats": True,
                    },
                    "output_vars": True,
                    "rails": {
                        "dialog": True,
                        "input": True,
                        "output": True,
                        "retrieval": True,
                    },
                },
                "return_choice": True,
                "state": {"foo": "bar"},
                "stream": True,
            },
            ignore_eos=True,
            logit_bias={"foo": 0},
            logprobs=True,
            max_completion_tokens=1,
            max_tokens=1,
            n=1,
            presence_penalty=-2,
            reasoning_effort="reasoning_effort",
            response_format={"foo": "bar"},
            seed=0,
            stop="string",
            stream=True,
            stream_options={"foo": True},
            temperature=0,
            tool_choice="string",
            tools=[{"foo": "bar"}],
            top_logprobs=0,
            top_p=0,
            user="user",
            vision=True,
        )
        assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_check(self, client: NeMoHelix) -> None:
        response = client.guardrail.with_raw_response.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                }
            ],
            model="model",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        guardrail = response.parse()
        assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_check(self, client: NeMoHelix) -> None:
        with client.guardrail.with_streaming_response.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                }
            ],
            model="model",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            guardrail = response.parse()
            assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_check(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.guardrail.with_raw_response.check(
                workspace="",
                messages=[
                    {
                        "content": "content",
                        "role": "system",
                    }
                ],
                model="model",
            )


class TestAsyncGuardrail:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_check(self, async_client: AsyncNeMoHelix) -> None:
        guardrail = await async_client.guardrail.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                }
            ],
            model="model",
        )
        assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_check_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        guardrail = await async_client.guardrail.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                    "name": "name",
                }
            ],
            model="model",
            frequency_penalty=-2,
            function_call="string",
            guardrails={
                "config": "string",
                "config_id": "config_id",
                "config_ids": ["string"],
                "context": {"foo": "bar"},
                "options": {
                    "llm_output": True,
                    "llm_params": {"foo": "bar"},
                    "log": {
                        "activated_rails": True,
                        "colang_history": True,
                        "internal_events": True,
                        "llm_calls": True,
                        "stats": True,
                    },
                    "output_vars": True,
                    "rails": {
                        "dialog": True,
                        "input": True,
                        "output": True,
                        "retrieval": True,
                    },
                },
                "return_choice": True,
                "state": {"foo": "bar"},
                "stream": True,
            },
            ignore_eos=True,
            logit_bias={"foo": 0},
            logprobs=True,
            max_completion_tokens=1,
            max_tokens=1,
            n=1,
            presence_penalty=-2,
            reasoning_effort="reasoning_effort",
            response_format={"foo": "bar"},
            seed=0,
            stop="string",
            stream=True,
            stream_options={"foo": True},
            temperature=0,
            tool_choice="string",
            tools=[{"foo": "bar"}],
            top_logprobs=0,
            top_p=0,
            user="user",
            vision=True,
        )
        assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_check(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.guardrail.with_raw_response.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                }
            ],
            model="model",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        guardrail = await response.parse()
        assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_check(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.guardrail.with_streaming_response.check(
            workspace="workspace",
            messages=[
                {
                    "content": "content",
                    "role": "system",
                }
            ],
            model="model",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            guardrail = await response.parse()
            assert_matches_type(GuardrailCheckResponse, guardrail, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_check(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.guardrail.with_raw_response.check(
                workspace="",
                messages=[
                    {
                        "content": "content",
                        "role": "system",
                    }
                ],
                model="model",
            )
