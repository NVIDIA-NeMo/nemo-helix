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
from nemo_helix.types.intake.spans import SpanGroup

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestGroups:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        group = client.intake.spans.groups.list(
            workspace="workspace",
            by="by",
        )
        assert_matches_type(SyncDefaultPagination[SpanGroup], group, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        group = client.intake.spans.groups.list(
            workspace="workspace",
            by="by",
            filter={
                "agent_id": "agent_id",
                "agent_name": "agent_name",
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "kind": "LLM",
                "model": "model",
                "parent_span_id": "parent_span_id",
                "project": "project",
                "provider": "provider",
                "session_id": "session_id",
                "source": "source",
                "started_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "status": "success",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
                "tool_name": "tool_name",
                "trace_id": "trace_id",
            },
            page=1,
            page_size=1,
            sort="span_count",
        )
        assert_matches_type(SyncDefaultPagination[SpanGroup], group, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.intake.spans.groups.with_raw_response.list(
            workspace="workspace",
            by="by",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        group = response.parse()
        assert_matches_type(SyncDefaultPagination[SpanGroup], group, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.intake.spans.groups.with_streaming_response.list(
            workspace="workspace",
            by="by",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            group = response.parse()
            assert_matches_type(SyncDefaultPagination[SpanGroup], group, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.spans.groups.with_raw_response.list(
                workspace="",
                by="by",
            )


class TestAsyncGroups:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        group = await async_client.intake.spans.groups.list(
            workspace="workspace",
            by="by",
        )
        assert_matches_type(AsyncDefaultPagination[SpanGroup], group, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        group = await async_client.intake.spans.groups.list(
            workspace="workspace",
            by="by",
            filter={
                "agent_id": "agent_id",
                "agent_name": "agent_name",
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "kind": "LLM",
                "model": "model",
                "parent_span_id": "parent_span_id",
                "project": "project",
                "provider": "provider",
                "session_id": "session_id",
                "source": "source",
                "started_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "status": "success",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
                "tool_name": "tool_name",
                "trace_id": "trace_id",
            },
            page=1,
            page_size=1,
            sort="span_count",
        )
        assert_matches_type(AsyncDefaultPagination[SpanGroup], group, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.spans.groups.with_raw_response.list(
            workspace="workspace",
            by="by",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        group = await response.parse()
        assert_matches_type(AsyncDefaultPagination[SpanGroup], group, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.spans.groups.with_streaming_response.list(
            workspace="workspace",
            by="by",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            group = await response.parse()
            assert_matches_type(AsyncDefaultPagination[SpanGroup], group, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.spans.groups.with_raw_response.list(
                workspace="",
                by="by",
            )
