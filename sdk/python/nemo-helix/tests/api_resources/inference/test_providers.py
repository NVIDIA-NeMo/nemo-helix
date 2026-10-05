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
from nemo_helix._utils import parse_datetime
from nemo_helix.pagination import SyncDefaultPagination, AsyncDefaultPagination
from nemo_helix.types.inference import (
    ModelProvider,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestProviders:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
            api_key_secret_name="api_key_secret_name",
            auth_header_format="auth_header_format",
            default_extra_body={"foo": "bar"},
            default_extra_headers={"foo": "string"},
            description="description",
            enabled_models=["string"],
            model_deployment_id="model_deployment_id",
            project="project",
            required_extra_body={"foo": "bar"},
            required_extra_headers={"foo": "string"},
            status="UNKNOWN",
            status_message="status_message",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.inference.providers.with_raw_response.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.inference.providers.with_streaming_response.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.providers.with_raw_response.create(
                workspace="",
                host_url="host_url",
                name="my-nim-provider",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoHelix) -> None:
        response = client.inference.providers.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoHelix) -> None:
        with client.inference.providers.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.providers.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.providers.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_with_all_params(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
            api_key_secret_name="api_key_secret_name",
            auth_header_format="auth_header_format",
            default_extra_body={"foo": "bar"},
            default_extra_headers={"foo": "string"},
            description="description",
            enabled_models=["string"],
            model_deployment_id="model_deployment_id",
            project="project",
            required_extra_body={"foo": "bar"},
            required_extra_headers={"foo": "string"},
            status="UNKNOWN",
            status_message="status_message",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update(self, client: NeMoHelix) -> None:
        response = client.inference.providers.with_raw_response.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update(self, client: NeMoHelix) -> None:
        with client.inference.providers.with_streaming_response.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.providers.with_raw_response.update(
                name="name",
                workspace="",
                host_url="host_url",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.providers.with_raw_response.update(
                name="",
                workspace="workspace",
                host_url="host_url",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[ModelProvider], provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.list(
            workspace="workspace",
            filter={
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
                "host_url": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "model_deployment_id": "model_deployment_id",
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "status": "UNKNOWN",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=0,
            page_size=0,
            sort="name",
        )
        assert_matches_type(SyncDefaultPagination[ModelProvider], provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.inference.providers.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(SyncDefaultPagination[ModelProvider], provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.inference.providers.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(SyncDefaultPagination[ModelProvider], provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.providers.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.delete(
            name="name",
            workspace="workspace",
        )
        assert provider is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.inference.providers.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert provider is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.inference.providers.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert provider is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.providers.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.providers.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_status(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.update_status(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_status_with_all_params(self, client: NeMoHelix) -> None:
        provider = client.inference.providers.update_status(
            name="name",
            workspace="workspace",
            model_deployment_id="model_deployment_id",
            served_models=[
                {
                    "model_entity_id": "model_entity_id",
                    "served_model_name": "served_model_name",
                }
            ],
            status="UNKNOWN",
            status_message="status_message",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update_status(self, client: NeMoHelix) -> None:
        response = client.inference.providers.with_raw_response.update_status(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update_status(self, client: NeMoHelix) -> None:
        with client.inference.providers.with_streaming_response.update_status(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update_status(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.providers.with_raw_response.update_status(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.providers.with_raw_response.update_status(
                name="",
                workspace="workspace",
            )


class TestAsyncProviders:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
            api_key_secret_name="api_key_secret_name",
            auth_header_format="auth_header_format",
            default_extra_body={"foo": "bar"},
            default_extra_headers={"foo": "string"},
            description="description",
            enabled_models=["string"],
            model_deployment_id="model_deployment_id",
            project="project",
            required_extra_body={"foo": "bar"},
            required_extra_headers={"foo": "string"},
            status="UNKNOWN",
            status_message="status_message",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.providers.with_raw_response.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.providers.with_streaming_response.create(
            workspace="workspace",
            host_url="host_url",
            name="my-nim-provider",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.providers.with_raw_response.create(
                workspace="",
                host_url="host_url",
                name="my-nim-provider",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.providers.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.providers.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.providers.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.providers.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
            api_key_secret_name="api_key_secret_name",
            auth_header_format="auth_header_format",
            default_extra_body={"foo": "bar"},
            default_extra_headers={"foo": "string"},
            description="description",
            enabled_models=["string"],
            model_deployment_id="model_deployment_id",
            project="project",
            required_extra_body={"foo": "bar"},
            required_extra_headers={"foo": "string"},
            status="UNKNOWN",
            status_message="status_message",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.providers.with_raw_response.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.providers.with_streaming_response.update(
            name="name",
            workspace="workspace",
            host_url="host_url",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.providers.with_raw_response.update(
                name="name",
                workspace="",
                host_url="host_url",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.providers.with_raw_response.update(
                name="",
                workspace="workspace",
                host_url="host_url",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[ModelProvider], provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.list(
            workspace="workspace",
            filter={
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
                "host_url": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "model_deployment_id": "model_deployment_id",
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "status": "UNKNOWN",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=0,
            page_size=0,
            sort="name",
        )
        assert_matches_type(AsyncDefaultPagination[ModelProvider], provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.providers.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(AsyncDefaultPagination[ModelProvider], provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.providers.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(AsyncDefaultPagination[ModelProvider], provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.providers.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.delete(
            name="name",
            workspace="workspace",
        )
        assert provider is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.providers.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert provider is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.providers.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert provider is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.providers.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.providers.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_status(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.update_status(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_status_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        provider = await async_client.inference.providers.update_status(
            name="name",
            workspace="workspace",
            model_deployment_id="model_deployment_id",
            served_models=[
                {
                    "model_entity_id": "model_entity_id",
                    "served_model_name": "served_model_name",
                }
            ],
            status="UNKNOWN",
            status_message="status_message",
        )
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update_status(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.providers.with_raw_response.update_status(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        provider = await response.parse()
        assert_matches_type(ModelProvider, provider, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update_status(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.providers.with_streaming_response.update_status(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            provider = await response.parse()
            assert_matches_type(ModelProvider, provider, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update_status(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.providers.with_raw_response.update_status(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.providers.with_raw_response.update_status(
                name="",
                workspace="workspace",
            )
