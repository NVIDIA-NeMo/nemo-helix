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
from nemo_helix.types.shared import DeleteResponse
from nemo_helix.types.workspaces import (
    WorkspaceMember,
    WorkspaceMemberListResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestMembers:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        member = client.workspaces.members.create(
            workspace="workspace",
            principal="user@example.com",
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        member = client.workspaces.members.create(
            workspace="workspace",
            principal="user@example.com",
            wait_role_propagation=True,
            roles=["Editor"],
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.workspaces.members.with_raw_response.create(
            workspace="workspace",
            principal="user@example.com",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = response.parse()
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.workspaces.members.with_streaming_response.create(
            workspace="workspace",
            principal="user@example.com",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = response.parse()
            assert_matches_type(WorkspaceMember, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.workspaces.members.with_raw_response.create(
                workspace="",
                principal="user@example.com",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update(self, client: NeMoHelix) -> None:
        member = client.workspaces.members.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_with_all_params(self, client: NeMoHelix) -> None:
        member = client.workspaces.members.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
            wait_role_propagation=True,
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update(self, client: NeMoHelix) -> None:
        response = client.workspaces.members.with_raw_response.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = response.parse()
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update(self, client: NeMoHelix) -> None:
        with client.workspaces.members.with_streaming_response.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = response.parse()
            assert_matches_type(WorkspaceMember, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.workspaces.members.with_raw_response.update(
                principal_id="principal_id",
                workspace="",
                roles=["Viewer"],
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `principal_id` but received ''"):
            client.workspaces.members.with_raw_response.update(
                principal_id="",
                workspace="workspace",
                roles=["Viewer"],
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        member = client.workspaces.members.list(
            workspace="workspace",
        )
        assert_matches_type(WorkspaceMemberListResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.workspaces.members.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = response.parse()
        assert_matches_type(WorkspaceMemberListResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.workspaces.members.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = response.parse()
            assert_matches_type(WorkspaceMemberListResponse, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.workspaces.members.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        member = client.workspaces.members.delete(
            principal_id="principal_id",
            workspace="workspace",
        )
        assert_matches_type(DeleteResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete_with_all_params(self, client: NeMoHelix) -> None:
        member = client.workspaces.members.delete(
            principal_id="principal_id",
            workspace="workspace",
            wait_role_propagation=True,
        )
        assert_matches_type(DeleteResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.workspaces.members.with_raw_response.delete(
            principal_id="principal_id",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = response.parse()
        assert_matches_type(DeleteResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.workspaces.members.with_streaming_response.delete(
            principal_id="principal_id",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = response.parse()
            assert_matches_type(DeleteResponse, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.workspaces.members.with_raw_response.delete(
                principal_id="principal_id",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `principal_id` but received ''"):
            client.workspaces.members.with_raw_response.delete(
                principal_id="",
                workspace="workspace",
            )


class TestAsyncMembers:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        member = await async_client.workspaces.members.create(
            workspace="workspace",
            principal="user@example.com",
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        member = await async_client.workspaces.members.create(
            workspace="workspace",
            principal="user@example.com",
            wait_role_propagation=True,
            roles=["Editor"],
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.workspaces.members.with_raw_response.create(
            workspace="workspace",
            principal="user@example.com",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = await response.parse()
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.workspaces.members.with_streaming_response.create(
            workspace="workspace",
            principal="user@example.com",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = await response.parse()
            assert_matches_type(WorkspaceMember, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.workspaces.members.with_raw_response.create(
                workspace="",
                principal="user@example.com",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update(self, async_client: AsyncNeMoHelix) -> None:
        member = await async_client.workspaces.members.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        member = await async_client.workspaces.members.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
            wait_role_propagation=True,
        )
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.workspaces.members.with_raw_response.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = await response.parse()
        assert_matches_type(WorkspaceMember, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.workspaces.members.with_streaming_response.update(
            principal_id="principal_id",
            workspace="workspace",
            roles=["Viewer"],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = await response.parse()
            assert_matches_type(WorkspaceMember, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.workspaces.members.with_raw_response.update(
                principal_id="principal_id",
                workspace="",
                roles=["Viewer"],
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `principal_id` but received ''"):
            await async_client.workspaces.members.with_raw_response.update(
                principal_id="",
                workspace="workspace",
                roles=["Viewer"],
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        member = await async_client.workspaces.members.list(
            workspace="workspace",
        )
        assert_matches_type(WorkspaceMemberListResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.workspaces.members.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = await response.parse()
        assert_matches_type(WorkspaceMemberListResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.workspaces.members.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = await response.parse()
            assert_matches_type(WorkspaceMemberListResponse, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.workspaces.members.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        member = await async_client.workspaces.members.delete(
            principal_id="principal_id",
            workspace="workspace",
        )
        assert_matches_type(DeleteResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        member = await async_client.workspaces.members.delete(
            principal_id="principal_id",
            workspace="workspace",
            wait_role_propagation=True,
        )
        assert_matches_type(DeleteResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.workspaces.members.with_raw_response.delete(
            principal_id="principal_id",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        member = await response.parse()
        assert_matches_type(DeleteResponse, member, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.workspaces.members.with_streaming_response.delete(
            principal_id="principal_id",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            member = await response.parse()
            assert_matches_type(DeleteResponse, member, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.workspaces.members.with_raw_response.delete(
                principal_id="principal_id",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `principal_id` but received ''"):
            await async_client.workspaces.members.with_raw_response.delete(
                principal_id="",
                workspace="workspace",
            )
