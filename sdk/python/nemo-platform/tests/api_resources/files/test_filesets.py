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
from nemo_platform.types.files import (
    Fileset,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestFilesets:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.create(
            workspace="workspace",
            name="training-data-v1",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.create(
            workspace="workspace",
            name="training-data-v1",
            cache=True,
            custom_fields={"foo": "bar"},
            description="description",
            metadata={
                "dataset": {
                    "schema": {"foo": "bar"},
                    "schema_defs": {"foo": {"foo": "bar"}},
                    "schemas_by_path": {"foo": {"foo": "bar"}},
                },
                "environment": {
                    "format": "format",
                    "name": "name",
                    "adapter_agent": "adapter_agent",
                    "config_paths": ["string"],
                    "hub_id": "hub_id",
                    "vf_env_id": "vf_env_id",
                },
                "model": {
                    "tool_calling": {
                        "auto_tool_choice": True,
                        "chat_template": "chat_template",
                        "tool_call_parser": "tool_call_parser",
                        "tool_call_plugin": "tool_call_plugin",
                    }
                },
            },
            project="project",
            purpose="dataset",
            storage={
                "path": "path",
                "read_chunk_size": 0,
                "type": "local",
                "write_buffer_size": 0,
            },
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoPlatform) -> None:
        response = client.files.filesets.with_raw_response.create(
            workspace="workspace",
            name="training-data-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoPlatform) -> None:
        with client.files.filesets.with_streaming_response.create(
            workspace="workspace",
            name="training-data-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.filesets.with_raw_response.create(
                workspace="",
                name="training-data-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoPlatform) -> None:
        response = client.files.filesets.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoPlatform) -> None:
        with client.files.filesets.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.filesets.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.filesets.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.update(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_with_all_params(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.update(
            name="name",
            workspace="workspace",
            custom_fields={"foo": "bar"},
            description="description",
            metadata={
                "dataset": {
                    "schema": {"foo": "bar"},
                    "schema_defs": {"foo": {"foo": "bar"}},
                    "schemas_by_path": {"foo": {"foo": "bar"}},
                },
                "environment": {
                    "format": "format",
                    "name": "name",
                    "adapter_agent": "adapter_agent",
                    "config_paths": ["string"],
                    "hub_id": "hub_id",
                    "vf_env_id": "vf_env_id",
                },
                "model": {
                    "tool_calling": {
                        "auto_tool_choice": True,
                        "chat_template": "chat_template",
                        "tool_call_parser": "tool_call_parser",
                        "tool_call_plugin": "tool_call_plugin",
                    }
                },
            },
            project="project",
            purpose="dataset",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update(self, client: NeMoPlatform) -> None:
        response = client.files.filesets.with_raw_response.update(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update(self, client: NeMoPlatform) -> None:
        with client.files.filesets.with_streaming_response.update(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.filesets.with_raw_response.update(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.filesets.with_raw_response.update(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[Fileset], fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.list(
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
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "purpose": "dataset",
                "storage_type": "local",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
            },
            page=1,
            page_size=1,
            sort="created_at",
        )
        assert_matches_type(SyncDefaultPagination[Fileset], fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoPlatform) -> None:
        response = client.files.filesets.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = response.parse()
        assert_matches_type(SyncDefaultPagination[Fileset], fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoPlatform) -> None:
        with client.files.filesets.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = response.parse()
            assert_matches_type(SyncDefaultPagination[Fileset], fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.filesets.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.delete(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoPlatform) -> None:
        response = client.files.filesets.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoPlatform) -> None:
        with client.files.filesets.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.filesets.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.filesets.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_refresh(self, client: NeMoPlatform) -> None:
        fileset = client.files.filesets.refresh(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_refresh(self, client: NeMoPlatform) -> None:
        response = client.files.filesets.with_raw_response.refresh(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_refresh(self, client: NeMoPlatform) -> None:
        with client.files.filesets.with_streaming_response.refresh(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_refresh(self, client: NeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.filesets.with_raw_response.refresh(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.filesets.with_raw_response.refresh(
                name="",
                workspace="workspace",
            )


class TestAsyncFilesets:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.create(
            workspace="workspace",
            name="training-data-v1",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.create(
            workspace="workspace",
            name="training-data-v1",
            cache=True,
            custom_fields={"foo": "bar"},
            description="description",
            metadata={
                "dataset": {
                    "schema": {"foo": "bar"},
                    "schema_defs": {"foo": {"foo": "bar"}},
                    "schemas_by_path": {"foo": {"foo": "bar"}},
                },
                "environment": {
                    "format": "format",
                    "name": "name",
                    "adapter_agent": "adapter_agent",
                    "config_paths": ["string"],
                    "hub_id": "hub_id",
                    "vf_env_id": "vf_env_id",
                },
                "model": {
                    "tool_calling": {
                        "auto_tool_choice": True,
                        "chat_template": "chat_template",
                        "tool_call_parser": "tool_call_parser",
                        "tool_call_plugin": "tool_call_plugin",
                    }
                },
            },
            project="project",
            purpose="dataset",
            storage={
                "path": "path",
                "read_chunk_size": 0,
                "type": "local",
                "write_buffer_size": 0,
            },
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.files.filesets.with_raw_response.create(
            workspace="workspace",
            name="training-data-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = await response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.files.filesets.with_streaming_response.create(
            workspace="workspace",
            name="training-data-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = await response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.filesets.with_raw_response.create(
                workspace="",
                name="training-data-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.files.filesets.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = await response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.files.filesets.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = await response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.filesets.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.filesets.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.update(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.update(
            name="name",
            workspace="workspace",
            custom_fields={"foo": "bar"},
            description="description",
            metadata={
                "dataset": {
                    "schema": {"foo": "bar"},
                    "schema_defs": {"foo": {"foo": "bar"}},
                    "schemas_by_path": {"foo": {"foo": "bar"}},
                },
                "environment": {
                    "format": "format",
                    "name": "name",
                    "adapter_agent": "adapter_agent",
                    "config_paths": ["string"],
                    "hub_id": "hub_id",
                    "vf_env_id": "vf_env_id",
                },
                "model": {
                    "tool_calling": {
                        "auto_tool_choice": True,
                        "chat_template": "chat_template",
                        "tool_call_parser": "tool_call_parser",
                        "tool_call_plugin": "tool_call_plugin",
                    }
                },
            },
            project="project",
            purpose="dataset",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.files.filesets.with_raw_response.update(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = await response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.files.filesets.with_streaming_response.update(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = await response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.filesets.with_raw_response.update(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.filesets.with_raw_response.update(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[Fileset], fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.list(
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
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "purpose": "dataset",
                "storage_type": "local",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
            },
            page=1,
            page_size=1,
            sort="created_at",
        )
        assert_matches_type(AsyncDefaultPagination[Fileset], fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.files.filesets.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = await response.parse()
        assert_matches_type(AsyncDefaultPagination[Fileset], fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.files.filesets.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = await response.parse()
            assert_matches_type(AsyncDefaultPagination[Fileset], fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.filesets.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.delete(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.files.filesets.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = await response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.files.filesets.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = await response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.filesets.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.filesets.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_refresh(self, async_client: AsyncNeMoPlatform) -> None:
        fileset = await async_client.files.filesets.refresh(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_refresh(self, async_client: AsyncNeMoPlatform) -> None:
        response = await async_client.files.filesets.with_raw_response.refresh(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        fileset = await response.parse()
        assert_matches_type(Fileset, fileset, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_refresh(self, async_client: AsyncNeMoPlatform) -> None:
        async with async_client.files.filesets.with_streaming_response.refresh(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            fileset = await response.parse()
            assert_matches_type(Fileset, fileset, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_refresh(self, async_client: AsyncNeMoPlatform) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.filesets.with_raw_response.refresh(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.filesets.with_raw_response.refresh(
                name="",
                workspace="workspace",
            )
