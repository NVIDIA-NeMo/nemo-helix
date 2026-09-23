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
from nemo_helix.types.inference.gateway import (
    ProviderPutResponse,
    ProviderPostResponse,
    ProviderPatchResponse,
    ProviderReadyResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestProvider:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.delete(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.provider.with_raw_response.delete(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.inference.gateway.provider.with_streaming_response.delete(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(object, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.provider.with_raw_response.delete(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.gateway.provider.with_raw_response.delete(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            client.inference.gateway.provider.with_raw_response.delete(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.get(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_get(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.provider.with_raw_response.get(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_get(self, client: NeMoHelix) -> None:
        with client.inference.gateway.provider.with_streaming_response.get(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(object, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_get(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.provider.with_raw_response.get(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.gateway.provider.with_raw_response.get(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            client.inference.gateway.provider.with_raw_response.get(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(ProviderPatchResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch_with_all_params(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
            body={"foo": "bar"},
        )
        assert_matches_type(ProviderPatchResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_patch(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.provider.with_raw_response.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ProviderPatchResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_patch(self, client: NeMoHelix) -> None:
        with client.inference.gateway.provider.with_streaming_response.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ProviderPatchResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_patch(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.provider.with_raw_response.patch(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.gateway.provider.with_raw_response.patch(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            client.inference.gateway.provider.with_raw_response.patch(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_post(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(ProviderPostResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_post_with_all_params(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
            body={"foo": "bar"},
        )
        assert_matches_type(ProviderPostResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_post(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.provider.with_raw_response.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ProviderPostResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_post(self, client: NeMoHelix) -> None:
        with client.inference.gateway.provider.with_streaming_response.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ProviderPostResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_post(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.provider.with_raw_response.post(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.gateway.provider.with_raw_response.post(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            client.inference.gateway.provider.with_raw_response.post(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_put(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(ProviderPutResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_put_with_all_params(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
            body={"foo": "bar"},
        )
        assert_matches_type(ProviderPutResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_put(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.provider.with_raw_response.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ProviderPutResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_put(self, client: NeMoHelix) -> None:
        with client.inference.gateway.provider.with_streaming_response.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ProviderPutResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_put(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.provider.with_raw_response.put(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.gateway.provider.with_raw_response.put(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            client.inference.gateway.provider.with_raw_response.put(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_ready(self, client: NeMoHelix) -> None:
        provider = client.inference.gateway.provider.ready(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ProviderReadyResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_ready(self, client: NeMoHelix) -> None:
        response = client.inference.gateway.provider.with_raw_response.ready(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ProviderReadyResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_ready(self, client: NeMoHelix) -> None:
        with client.inference.gateway.provider.with_streaming_response.ready(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ProviderReadyResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_ready(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.gateway.provider.with_raw_response.ready(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.gateway.provider.with_raw_response.ready(
                name="",
                workspace="workspace",
            )


class TestAsyncProvider:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.delete(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.provider.with_raw_response.delete(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.provider.with_streaming_response.delete(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(object, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.delete(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.delete(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.delete(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.get(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_get(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.provider.with_raw_response.get(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(object, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_get(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.provider.with_streaming_response.get(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(object, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_get(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.get(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.get(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.get(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(ProviderPatchResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
            body={"foo": "bar"},
        )
        assert_matches_type(ProviderPatchResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.provider.with_raw_response.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ProviderPatchResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.provider.with_streaming_response.patch(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ProviderPatchResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_patch(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.patch(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.patch(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.patch(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_post(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(ProviderPostResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_post_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
            body={"foo": "bar"},
        )
        assert_matches_type(ProviderPostResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_post(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.provider.with_raw_response.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ProviderPostResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_post(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.provider.with_streaming_response.post(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ProviderPostResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_post(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.post(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.post(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.post(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_put(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(ProviderPutResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_put_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
            body={"foo": "bar"},
        )
        assert_matches_type(ProviderPutResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_put(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.provider.with_raw_response.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ProviderPutResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_put(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.provider.with_streaming_response.put(
            trailing_uri="trailing_uri",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ProviderPutResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_put(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.put(
                trailing_uri="trailing_uri",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.put(
                trailing_uri="trailing_uri",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `trailing_uri` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.put(
                trailing_uri="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_ready(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.gateway.provider.ready(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ProviderReadyResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_ready(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.gateway.provider.with_raw_response.ready(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ProviderReadyResponse, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_ready(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.gateway.provider.with_streaming_response.ready(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ProviderReadyResponse, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_ready(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.ready(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.gateway.provider.with_raw_response.ready(
                name="",
                workspace="workspace",
            )
