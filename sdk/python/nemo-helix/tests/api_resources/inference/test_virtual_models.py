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
    VirtualModel,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestVirtualModels:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.create(
            workspace="workspace",
            name="name",
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.create(
            workspace="workspace",
            name="name",
            autoprovisioned=True,
            default_model_entity="default_model_entity",
            models=[
                {
                    "model": "model",
                    "backend_format": "OPENAI_CHAT",
                }
            ],
            override_proxy="override_proxy",
            post_response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            request_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.inference.virtual_models.with_raw_response.create(
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = response.parse()
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.inference.virtual_models.with_streaming_response.create(
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = response.parse()
            assert_matches_type(VirtualModel, virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.virtual_models.with_raw_response.create(
                workspace="",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoHelix) -> None:
        response = client.inference.virtual_models.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = response.parse()
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoHelix) -> None:
        with client.inference.virtual_models.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = response.parse()
            assert_matches_type(VirtualModel, virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.virtual_models.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.virtual_models.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.list(
            workspace="workspace",
            exclude_autoprovisioned=True,
            filter={
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "default_model_entity": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "guardrail_config": "guardrail_config",
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=1,
            page_size=1,
            sort="sort",
        )
        assert_matches_type(SyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.inference.virtual_models.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = response.parse()
        assert_matches_type(SyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.inference.virtual_models.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = response.parse()
            assert_matches_type(SyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.virtual_models.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.delete(
            name="name",
            workspace="workspace",
        )
        assert virtual_model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete_with_all_params(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.delete(
            name="name",
            workspace="workspace",
            expected_db_version=0,
        )
        assert virtual_model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.inference.virtual_models.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = response.parse()
        assert virtual_model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.inference.virtual_models.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = response.parse()
            assert virtual_model is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.virtual_models.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.virtual_models.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.patch(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch_with_all_params(self, client: NeMoHelix) -> None:
        virtual_model = client.inference.virtual_models.patch(
            name="name",
            workspace="workspace",
            autoprovisioned=True,
            default_model_entity="default_model_entity",
            models=[
                {
                    "model": "model",
                    "backend_format": "OPENAI_CHAT",
                }
            ],
            override_proxy="override_proxy",
            post_response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            request_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_patch(self, client: NeMoHelix) -> None:
        response = client.inference.virtual_models.with_raw_response.patch(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = response.parse()
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_patch(self, client: NeMoHelix) -> None:
        with client.inference.virtual_models.with_streaming_response.patch(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = response.parse()
            assert_matches_type(VirtualModel, virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_patch(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.virtual_models.with_raw_response.patch(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.virtual_models.with_raw_response.patch(
                name="",
                workspace="workspace",
            )


class TestAsyncVirtualModels:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.create(
            workspace="workspace",
            name="name",
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.create(
            workspace="workspace",
            name="name",
            autoprovisioned=True,
            default_model_entity="default_model_entity",
            models=[
                {
                    "model": "model",
                    "backend_format": "OPENAI_CHAT",
                }
            ],
            override_proxy="override_proxy",
            post_response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            request_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.virtual_models.with_raw_response.create(
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = await response.parse()
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.virtual_models.with_streaming_response.create(
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = await response.parse()
            assert_matches_type(VirtualModel, virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.create(
                workspace="",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.virtual_models.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = await response.parse()
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.virtual_models.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = await response.parse()
            assert_matches_type(VirtualModel, virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.list(
            workspace="workspace",
            exclude_autoprovisioned=True,
            filter={
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "default_model_entity": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "guardrail_config": "guardrail_config",
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=1,
            page_size=1,
            sort="sort",
        )
        assert_matches_type(AsyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.virtual_models.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = await response.parse()
        assert_matches_type(AsyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.virtual_models.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = await response.parse()
            assert_matches_type(AsyncDefaultPagination[VirtualModel], virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.delete(
            name="name",
            workspace="workspace",
        )
        assert virtual_model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.delete(
            name="name",
            workspace="workspace",
            expected_db_version=0,
        )
        assert virtual_model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.virtual_models.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = await response.parse()
        assert virtual_model is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.virtual_models.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = await response.parse()
            assert virtual_model is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.patch(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        virtual_model = await async_client.inference.virtual_models.patch(
            name="name",
            workspace="workspace",
            autoprovisioned=True,
            default_model_entity="default_model_entity",
            models=[
                {
                    "model": "model",
                    "backend_format": "OPENAI_CHAT",
                }
            ],
            override_proxy="override_proxy",
            post_response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            request_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
            response_middleware=[
                {
                    "config_type": "config_type",
                    "name": "name",
                    "config": {"foo": "bar"},
                    "config_id": "config_id",
                }
            ],
        )
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.virtual_models.with_raw_response.patch(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        virtual_model = await response.parse()
        assert_matches_type(VirtualModel, virtual_model, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.virtual_models.with_streaming_response.patch(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            virtual_model = await response.parse()
            assert_matches_type(VirtualModel, virtual_model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_patch(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.patch(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.virtual_models.with_raw_response.patch(
                name="",
                workspace="workspace",
            )
