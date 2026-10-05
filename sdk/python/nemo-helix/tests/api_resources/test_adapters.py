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
from nemo_helix.types.models import Adapter

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestAdapters:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        adapter = client.adapters.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        adapter = client.adapters.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
            description="description",
            enabled=True,
            lora_config={
                "rank": 0,
                "alpha": 0,
            },
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.adapters.with_raw_response.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.adapters.with_streaming_response.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.adapters.with_raw_response.create(
                workspace="",
                fileset="fileset",
                finetuning_type="lora_merged",
                model="llama-3-8b-instruct",
                name="lora-adapter-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoHelix) -> None:
        adapter = client.adapters.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoHelix) -> None:
        response = client.adapters.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoHelix) -> None:
        with client.adapters.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.adapters.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.adapters.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        adapter = client.adapters.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[Adapter], adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        adapter = client.adapters.list(
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
                "enabled": True,
                "fileset": "fileset",
                "finetuning_type": "lora_merged",
                "model": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
            },
            page=0,
            page_size=1,
            sort="sort",
        )
        assert_matches_type(SyncDefaultPagination[Adapter], adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.adapters.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert_matches_type(SyncDefaultPagination[Adapter], adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.adapters.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert_matches_type(SyncDefaultPagination[Adapter], adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.adapters.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        adapter = client.adapters.delete(
            name="name",
            workspace="workspace",
        )
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.adapters.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.adapters.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert adapter is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.adapters.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.adapters.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch(self, client: NeMoHelix) -> None:
        adapter = client.adapters.patch(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch_with_all_params(self, client: NeMoHelix) -> None:
        adapter = client.adapters.patch(
            name="name",
            workspace="workspace",
            description="description",
            enabled=True,
            fileset="fileset",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_patch(self, client: NeMoHelix) -> None:
        response = client.adapters.with_raw_response.patch(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_patch(self, client: NeMoHelix) -> None:
        with client.adapters.with_streaming_response.patch(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_patch(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.adapters.with_raw_response.patch(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.adapters.with_raw_response.patch(
                name="",
                workspace="workspace",
            )


class TestAsyncAdapters:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
            description="description",
            enabled=True,
            lora_config={
                "rank": 0,
                "alpha": 0,
            },
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.adapters.with_raw_response.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.adapters.with_streaming_response.create(
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            model="llama-3-8b-instruct",
            name="lora-adapter-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.adapters.with_raw_response.create(
                workspace="",
                fileset="fileset",
                finetuning_type="lora_merged",
                model="llama-3-8b-instruct",
                name="lora-adapter-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.adapters.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.adapters.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.adapters.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.adapters.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[Adapter], adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.list(
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
                "enabled": True,
                "fileset": "fileset",
                "finetuning_type": "lora_merged",
                "model": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
            },
            page=0,
            page_size=1,
            sort="sort",
        )
        assert_matches_type(AsyncDefaultPagination[Adapter], adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.adapters.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert_matches_type(AsyncDefaultPagination[Adapter], adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.adapters.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert_matches_type(AsyncDefaultPagination[Adapter], adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.adapters.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.delete(
            name="name",
            workspace="workspace",
        )
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.adapters.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.adapters.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert adapter is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.adapters.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.adapters.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.patch(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.adapters.patch(
            name="name",
            workspace="workspace",
            description="description",
            enabled=True,
            fileset="fileset",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.adapters.with_raw_response.patch(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.adapters.with_streaming_response.patch(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_patch(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.adapters.with_raw_response.patch(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.adapters.with_raw_response.patch(
                name="",
                workspace="workspace",
            )
