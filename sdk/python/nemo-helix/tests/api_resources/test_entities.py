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
from nemo_helix.pagination import SyncDefaultPagination, AsyncDefaultPagination
from nemo_helix.types.shared import DeleteResponse
from nemo_helix.types.entities import (
    Entity,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestEntities:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        entity = client.entities.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        entity = client.entities.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
            name="my-config",
            parent="parent",
            project="project",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.entities.with_raw_response.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.entities.with_streaming_response.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.entities.with_raw_response.create(
                entity_type="entity_type",
                workspace="",
                data={"foo": "bar"},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            client.entities.with_raw_response.create(
                entity_type="",
                workspace="workspace",
                data={"foo": "bar"},
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        entity = client.entities.list(
            entity_type="entity_type",
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[Entity], entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        entity = client.entities.list(
            entity_type="entity_type",
            workspace="workspace",
            count_by="count_by",
            filter="filter",
            page=1,
            page_size=1,
            sort="-created_at",
        )
        assert_matches_type(SyncDefaultPagination[Entity], entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.entities.with_raw_response.list(
            entity_type="entity_type",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = response.parse()
        assert_matches_type(SyncDefaultPagination[Entity], entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.entities.with_streaming_response.list(
            entity_type="entity_type",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = response.parse()
            assert_matches_type(SyncDefaultPagination[Entity], entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.entities.with_raw_response.list(
                entity_type="entity_type",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            client.entities.with_raw_response.list(
                entity_type="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete_entity_by_name(self, client: NeMoHelix) -> None:
        entity = client.entities.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )
        assert_matches_type(DeleteResponse, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete_entity_by_name_with_all_params(self, client: NeMoHelix) -> None:
        entity = client.entities.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            expected_db_version=0,
            parent="parent",
        )
        assert_matches_type(DeleteResponse, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete_entity_by_name(self, client: NeMoHelix) -> None:
        response = client.entities.with_raw_response.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = response.parse()
        assert_matches_type(DeleteResponse, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete_entity_by_name(self, client: NeMoHelix) -> None:
        with client.entities.with_streaming_response.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = response.parse()
            assert_matches_type(DeleteResponse, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete_entity_by_name(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.entities.with_raw_response.delete_entity_by_name(
                name="name",
                workspace="",
                entity_type="entity_type",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            client.entities.with_raw_response.delete_entity_by_name(
                name="name",
                workspace="workspace",
                entity_type="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.entities.with_raw_response.delete_entity_by_name(
                name="",
                workspace="workspace",
                entity_type="entity_type",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get_entity_by_id(self, client: NeMoHelix) -> None:
        entity = client.entities.get_entity_by_id(
            "id",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_get_entity_by_id(self, client: NeMoHelix) -> None:
        response = client.entities.with_raw_response.get_entity_by_id(
            "id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_get_entity_by_id(self, client: NeMoHelix) -> None:
        with client.entities.with_streaming_response.get_entity_by_id(
            "id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_get_entity_by_id(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            client.entities.with_raw_response.get_entity_by_id(
                "",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get_entity_by_name(self, client: NeMoHelix) -> None:
        entity = client.entities.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get_entity_by_name_with_all_params(self, client: NeMoHelix) -> None:
        entity = client.entities.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            parent="parent",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_get_entity_by_name(self, client: NeMoHelix) -> None:
        response = client.entities.with_raw_response.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_get_entity_by_name(self, client: NeMoHelix) -> None:
        with client.entities.with_streaming_response.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_get_entity_by_name(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.entities.with_raw_response.get_entity_by_name(
                name="name",
                workspace="",
                entity_type="entity_type",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            client.entities.with_raw_response.get_entity_by_name(
                name="name",
                workspace="workspace",
                entity_type="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.entities.with_raw_response.get_entity_by_name(
                name="",
                workspace="workspace",
                entity_type="entity_type",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_entity_by_name(self, client: NeMoHelix) -> None:
        entity = client.entities.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_entity_by_name_with_all_params(self, client: NeMoHelix) -> None:
        entity = client.entities.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
            parent="parent",
            expected_db_version=0,
            new_name="my-config",
            project="project",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update_entity_by_name(self, client: NeMoHelix) -> None:
        response = client.entities.with_raw_response.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update_entity_by_name(self, client: NeMoHelix) -> None:
        with client.entities.with_streaming_response.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update_entity_by_name(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.entities.with_raw_response.update_entity_by_name(
                name="name",
                workspace="",
                entity_type="entity_type",
                data={"foo": "bar"},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            client.entities.with_raw_response.update_entity_by_name(
                name="name",
                workspace="workspace",
                entity_type="",
                data={"foo": "bar"},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.entities.with_raw_response.update_entity_by_name(
                name="",
                workspace="workspace",
                entity_type="entity_type",
                data={"foo": "bar"},
            )


class TestAsyncEntities:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
            name="my-config",
            parent="parent",
            project="project",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.entities.with_raw_response.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = await response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.entities.with_streaming_response.create(
            entity_type="entity_type",
            workspace="workspace",
            data={"foo": "bar"},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = await response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.entities.with_raw_response.create(
                entity_type="entity_type",
                workspace="",
                data={"foo": "bar"},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            await async_client.entities.with_raw_response.create(
                entity_type="",
                workspace="workspace",
                data={"foo": "bar"},
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.list(
            entity_type="entity_type",
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[Entity], entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.list(
            entity_type="entity_type",
            workspace="workspace",
            count_by="count_by",
            filter="filter",
            page=1,
            page_size=1,
            sort="-created_at",
        )
        assert_matches_type(AsyncDefaultPagination[Entity], entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.entities.with_raw_response.list(
            entity_type="entity_type",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = await response.parse()
        assert_matches_type(AsyncDefaultPagination[Entity], entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.entities.with_streaming_response.list(
            entity_type="entity_type",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = await response.parse()
            assert_matches_type(AsyncDefaultPagination[Entity], entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.entities.with_raw_response.list(
                entity_type="entity_type",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            await async_client.entities.with_raw_response.list(
                entity_type="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )
        assert_matches_type(DeleteResponse, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete_entity_by_name_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            expected_db_version=0,
            parent="parent",
        )
        assert_matches_type(DeleteResponse, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.entities.with_raw_response.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = await response.parse()
        assert_matches_type(DeleteResponse, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.entities.with_streaming_response.delete_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = await response.parse()
            assert_matches_type(DeleteResponse, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.entities.with_raw_response.delete_entity_by_name(
                name="name",
                workspace="",
                entity_type="entity_type",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            await async_client.entities.with_raw_response.delete_entity_by_name(
                name="name",
                workspace="workspace",
                entity_type="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.entities.with_raw_response.delete_entity_by_name(
                name="",
                workspace="workspace",
                entity_type="entity_type",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get_entity_by_id(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.get_entity_by_id(
            "id",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_get_entity_by_id(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.entities.with_raw_response.get_entity_by_id(
            "id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = await response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_get_entity_by_id(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.entities.with_streaming_response.get_entity_by_id(
            "id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = await response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_get_entity_by_id(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            await async_client.entities.with_raw_response.get_entity_by_id(
                "",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get_entity_by_name_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            parent="parent",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_get_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.entities.with_raw_response.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = await response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_get_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.entities.with_streaming_response.get_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = await response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_get_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.entities.with_raw_response.get_entity_by_name(
                name="name",
                workspace="",
                entity_type="entity_type",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            await async_client.entities.with_raw_response.get_entity_by_name(
                name="name",
                workspace="workspace",
                entity_type="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.entities.with_raw_response.get_entity_by_name(
                name="",
                workspace="workspace",
                entity_type="entity_type",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_entity_by_name_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        entity = await async_client.entities.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
            parent="parent",
            expected_db_version=0,
            new_name="my-config",
            project="project",
        )
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.entities.with_raw_response.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        entity = await response.parse()
        assert_matches_type(Entity, entity, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.entities.with_streaming_response.update_entity_by_name(
            name="name",
            workspace="workspace",
            entity_type="entity_type",
            data={"foo": "bar"},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            entity = await response.parse()
            assert_matches_type(Entity, entity, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update_entity_by_name(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.entities.with_raw_response.update_entity_by_name(
                name="name",
                workspace="",
                entity_type="entity_type",
                data={"foo": "bar"},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `entity_type` but received ''"):
            await async_client.entities.with_raw_response.update_entity_by_name(
                name="name",
                workspace="workspace",
                entity_type="",
                data={"foo": "bar"},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.entities.with_raw_response.update_entity_by_name(
                name="",
                workspace="workspace",
                entity_type="entity_type",
                data={"foo": "bar"},
            )
