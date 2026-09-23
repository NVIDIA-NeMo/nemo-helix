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
from nemo_helix.types.inference.gateway.openai.v1 import OpenAIModelResp, OpenAIListModelsResp

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestModels:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        model = client.inference.gateway.openai.v1.models.list(
            workspace="workspace",
        )
        assert_matches_type(OpenAIListModelsResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.openai.v1.models.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(OpenAIListModelsResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.inference.gateway.openai.v1.models.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(OpenAIListModelsResp, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.openai.v1.models.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get(self, client: NeMoHelix) -> None:
        model = client.inference.gateway.openai.v1.models.get(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(OpenAIModelResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_get(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.openai.v1.models.with_raw_response.get(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(OpenAIModelResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_get(self, client: NeMoHelix) -> None:
        with client.inference.gateway.openai.v1.models.with_streaming_response.get(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(OpenAIModelResp, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_get(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.openai.v1.models.with_raw_response.get(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.gateway.openai.v1.models.with_raw_response.get(
                name="",
                workspace="workspace",
            )


class TestAsyncModels:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        model = await async_client.inference.gateway.openai.v1.models.list(
            workspace="workspace",
        )
        assert_matches_type(OpenAIListModelsResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.openai.v1.models.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(OpenAIListModelsResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.openai.v1.models.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(OpenAIListModelsResp, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.openai.v1.models.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get(self, async_client: AsyncNeMoHelix) -> None:
        model = await async_client.inference.gateway.openai.v1.models.get(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(OpenAIModelResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_get(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.openai.v1.models.with_raw_response.get(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(OpenAIModelResp, model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_get(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.openai.v1.models.with_streaming_response.get(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(OpenAIModelResp, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_get(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.openai.v1.models.with_raw_response.get(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.gateway.openai.v1.models.with_raw_response.get(
                name="",
                workspace="workspace",
            )
