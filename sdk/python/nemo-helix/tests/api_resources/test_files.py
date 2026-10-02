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

import httpx
import pytest
from respx import MockRouter

from tests.utils import assert_matches_type
from nemo_helix import NeMoHelix, AsyncNeMoHelix
from nemo_helix._response import (
    BinaryAPIResponse,
    AsyncBinaryAPIResponse,
    StreamedBinaryAPIResponse,
    AsyncStreamedBinaryAPIResponse,
)
from nemo_helix.types.files import (
    FilesetFile,
    ListFilesetFilesResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestFiles:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete_file(self, client: NeMoHelix) -> None:
        file = client.files._delete_file(
            path="path",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete_file(self, client: NeMoHelix) -> None:
        response = client.files.with_raw_response._delete_file(
            path="path",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = response.parse()
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete_file(self, client: NeMoHelix) -> None:
        with client.files.with_streaming_response._delete_file(
            path="path",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = response.parse()
            assert_matches_type(FilesetFile, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete_file(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.with_raw_response._delete_file(
                path="path",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.with_raw_response._delete_file(
                path="path",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path` but received ''"):
            client.files.with_raw_response._delete_file(
                path="",
                workspace="workspace",
                name="name",
            )

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    def test_method_download_file(self, client: NeMoHelix, respx_mock: MockRouter) -> None:
        respx_mock.get("/apis/files/v2/workspaces/workspace/filesets/name/-/path").mock(
            return_value=httpx.Response(200, json={"foo": "bar"})
        )
        file = client.files._download_file(
            path="path",
            workspace="workspace",
            name="name",
        )
        assert file.is_closed
        assert file.json() == {"foo": "bar"}
        assert cast(Any, file.is_closed) is True
        assert isinstance(file, BinaryAPIResponse)

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    def test_raw_response_download_file(self, client: NeMoHelix, respx_mock: MockRouter) -> None:
        respx_mock.get("/apis/files/v2/workspaces/workspace/filesets/name/-/path").mock(
            return_value=httpx.Response(200, json={"foo": "bar"})
        )

        file = client.files.with_raw_response._download_file(
            path="path",
            workspace="workspace",
            name="name",
        )

        assert file.is_closed is True
        assert file.http_request.headers.get("X-Stainless-Lang") == "python"
        assert file.json() == {"foo": "bar"}
        assert isinstance(file, BinaryAPIResponse)

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    def test_streaming_response_download_file(self, client: NeMoHelix, respx_mock: MockRouter) -> None:
        respx_mock.get("/apis/files/v2/workspaces/workspace/filesets/name/-/path").mock(
            return_value=httpx.Response(200, json={"foo": "bar"})
        )
        with client.files.with_streaming_response._download_file(
            path="path",
            workspace="workspace",
            name="name",
        ) as file:
            assert not file.is_closed
            assert file.http_request.headers.get("X-Stainless-Lang") == "python"

            assert file.json() == {"foo": "bar"}
            assert cast(Any, file.is_closed) is True
            assert isinstance(file, StreamedBinaryAPIResponse)

        assert cast(Any, file.is_closed) is True

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    def test_path_params_download_file(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.with_raw_response._download_file(
                path="path",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.with_raw_response._download_file(
                path="path",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path` but received ''"):
            client.files.with_raw_response._download_file(
                path="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_files(self, client: NeMoHelix) -> None:
        file = client.files._list_files(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_files_with_all_params(self, client: NeMoHelix) -> None:
        file = client.files._list_files(
            name="name",
            workspace="workspace",
            include_cache_status=True,
            path="path",
        )
        assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list_files(self, client: NeMoHelix) -> None:
        response = client.files.with_raw_response._list_files(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = response.parse()
        assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list_files(self, client: NeMoHelix) -> None:
        with client.files.with_streaming_response._list_files(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = response.parse()
            assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list_files(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.with_raw_response._list_files(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.with_raw_response._list_files(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_upload_file(self, client: NeMoHelix) -> None:
        file = client.files._upload_file(
            path="path",
            body=b"Example data",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_upload_file(self, client: NeMoHelix) -> None:
        response = client.files.with_raw_response._upload_file(
            path="path",
            body=b"Example data",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = response.parse()
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_upload_file(self, client: NeMoHelix) -> None:
        with client.files.with_streaming_response._upload_file(
            path="path",
            body=b"Example data",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = response.parse()
            assert_matches_type(FilesetFile, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_upload_file(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.files.with_raw_response._upload_file(
                path="path",
                body=b"Example data",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.files.with_raw_response._upload_file(
                path="path",
                body=b"Example data",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path` but received ''"):
            client.files.with_raw_response._upload_file(
                path="",
                body=b"Example data",
                workspace="workspace",
                name="name",
            )


class TestAsyncFiles:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete_file(self, async_client: AsyncNeMoHelix) -> None:
        file = await async_client.files._delete_file(
            path="path",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete_file(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.files.with_raw_response._delete_file(
            path="path",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = await response.parse()
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete_file(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.files.with_streaming_response._delete_file(
            path="path",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = await response.parse()
            assert_matches_type(FilesetFile, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete_file(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.with_raw_response._delete_file(
                path="path",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.with_raw_response._delete_file(
                path="path",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path` but received ''"):
            await async_client.files.with_raw_response._delete_file(
                path="",
                workspace="workspace",
                name="name",
            )

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    async def test_method_download_file(self, async_client: AsyncNeMoHelix, respx_mock: MockRouter) -> None:
        respx_mock.get("/apis/files/v2/workspaces/workspace/filesets/name/-/path").mock(
            return_value=httpx.Response(200, json={"foo": "bar"})
        )
        file = await async_client.files._download_file(
            path="path",
            workspace="workspace",
            name="name",
        )
        assert file.is_closed
        assert await file.json() == {"foo": "bar"}
        assert cast(Any, file.is_closed) is True
        assert isinstance(file, AsyncBinaryAPIResponse)

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    async def test_raw_response_download_file(self, async_client: AsyncNeMoHelix, respx_mock: MockRouter) -> None:
        respx_mock.get("/apis/files/v2/workspaces/workspace/filesets/name/-/path").mock(
            return_value=httpx.Response(200, json={"foo": "bar"})
        )

        file = await async_client.files.with_raw_response._download_file(
            path="path",
            workspace="workspace",
            name="name",
        )

        assert file.is_closed is True
        assert file.http_request.headers.get("X-Stainless-Lang") == "python"
        assert await file.json() == {"foo": "bar"}
        assert isinstance(file, AsyncBinaryAPIResponse)

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    async def test_streaming_response_download_file(
        self, async_client: AsyncNeMoHelix, respx_mock: MockRouter
    ) -> None:
        respx_mock.get("/apis/files/v2/workspaces/workspace/filesets/name/-/path").mock(
            return_value=httpx.Response(200, json={"foo": "bar"})
        )
        async with async_client.files.with_streaming_response._download_file(
            path="path",
            workspace="workspace",
            name="name",
        ) as file:
            assert not file.is_closed
            assert file.http_request.headers.get("X-Stainless-Lang") == "python"

            assert await file.json() == {"foo": "bar"}
            assert cast(Any, file.is_closed) is True
            assert isinstance(file, AsyncStreamedBinaryAPIResponse)

        assert cast(Any, file.is_closed) is True

    @parametrize
    @pytest.mark.respx(base_url=base_url)
    async def test_path_params_download_file(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.with_raw_response._download_file(
                path="path",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.with_raw_response._download_file(
                path="path",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path` but received ''"):
            await async_client.files.with_raw_response._download_file(
                path="",
                workspace="workspace",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_files(self, async_client: AsyncNeMoHelix) -> None:
        file = await async_client.files._list_files(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_files_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        file = await async_client.files._list_files(
            name="name",
            workspace="workspace",
            include_cache_status=True,
            path="path",
        )
        assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list_files(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.files.with_raw_response._list_files(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = await response.parse()
        assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list_files(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.files.with_streaming_response._list_files(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = await response.parse()
            assert_matches_type(ListFilesetFilesResponse, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list_files(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.with_raw_response._list_files(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.with_raw_response._list_files(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_upload_file(self, async_client: AsyncNeMoHelix) -> None:
        file = await async_client.files._upload_file(
            path="path",
            body=b"Example data",
            workspace="workspace",
            name="name",
        )
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_upload_file(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.files.with_raw_response._upload_file(
            path="path",
            body=b"Example data",
            workspace="workspace",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        file = await response.parse()
        assert_matches_type(FilesetFile, file, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_upload_file(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.files.with_streaming_response._upload_file(
            path="path",
            body=b"Example data",
            workspace="workspace",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            file = await response.parse()
            assert_matches_type(FilesetFile, file, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_upload_file(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.files.with_raw_response._upload_file(
                path="path",
                body=b"Example data",
                workspace="",
                name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.files.with_raw_response._upload_file(
                path="path",
                body=b"Example data",
                workspace="workspace",
                name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path` but received ''"):
            await async_client.files.with_raw_response._upload_file(
                path="",
                body=b"Example data",
                workspace="workspace",
                name="name",
            )
