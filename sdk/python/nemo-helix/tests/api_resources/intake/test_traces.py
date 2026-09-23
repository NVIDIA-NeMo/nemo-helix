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
from nemo_helix.types.intake import (
    Trace,
    TraceMetrics,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestTraces:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoHelix) -> None:
        trace = client.intake.traces.retrieve(
            id="id",
            workspace="workspace",
        )
        assert_matches_type(Trace, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve_with_all_params(self, client: NeMoHelix) -> None:
        trace = client.intake.traces.retrieve(
            id="id",
            workspace="workspace",
            mode="summary",
        )
        assert_matches_type(Trace, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoHelix) -> None:
        response = client.intake.traces.with_raw_response.retrieve(
            id="id",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        trace = response.parse()
        assert_matches_type(Trace, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoHelix) -> None:
        with client.intake.traces.with_streaming_response.retrieve(
            id="id",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            trace = response.parse()
            assert_matches_type(Trace, trace, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.traces.with_raw_response.retrieve(
                id="id",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            client.intake.traces.with_raw_response.retrieve(
                id="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        trace = client.intake.traces.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[Trace], trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        trace = client.intake.traces.list(
            workspace="workspace",
            filter={
                "id": "id",
                "agent_name": "agent_name",
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "session_id": "session_id",
                "started_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "status": "success",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
            },
            mode="summary",
            page=1,
            page_size=1,
            sort="started_at",
        )
        assert_matches_type(SyncDefaultPagination[Trace], trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.intake.traces.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        trace = response.parse()
        assert_matches_type(SyncDefaultPagination[Trace], trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.intake.traces.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            trace = response.parse()
            assert_matches_type(SyncDefaultPagination[Trace], trace, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.traces.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get_metrics(self, client: NeMoHelix) -> None:
        trace = client.intake.traces.get_metrics(
            workspace="workspace",
        )
        assert_matches_type(TraceMetrics, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_get_metrics_with_all_params(self, client: NeMoHelix) -> None:
        trace = client.intake.traces.get_metrics(
            workspace="workspace",
            bucket="total",
            filter={
                "id": "id",
                "agent_name": "agent_name",
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "session_id": "session_id",
                "started_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "status": "success",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
            },
            timezone="timezone",
        )
        assert_matches_type(TraceMetrics, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_get_metrics(self, client: NeMoHelix) -> None:
        response = client.intake.traces.with_raw_response.get_metrics(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        trace = response.parse()
        assert_matches_type(TraceMetrics, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_get_metrics(self, client: NeMoHelix) -> None:
        with client.intake.traces.with_streaming_response.get_metrics(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            trace = response.parse()
            assert_matches_type(TraceMetrics, trace, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_get_metrics(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.traces.with_raw_response.get_metrics(
                workspace="",
            )


class TestAsyncTraces:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        trace = await async_client.intake.traces.retrieve(
            id="id",
            workspace="workspace",
        )
        assert_matches_type(Trace, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        trace = await async_client.intake.traces.retrieve(
            id="id",
            workspace="workspace",
            mode="summary",
        )
        assert_matches_type(Trace, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.traces.with_raw_response.retrieve(
            id="id",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        trace = await response.parse()
        assert_matches_type(Trace, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.traces.with_streaming_response.retrieve(
            id="id",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            trace = await response.parse()
            assert_matches_type(Trace, trace, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.traces.with_raw_response.retrieve(
                id="id",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `id` but received ''"):
            await async_client.intake.traces.with_raw_response.retrieve(
                id="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        trace = await async_client.intake.traces.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[Trace], trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        trace = await async_client.intake.traces.list(
            workspace="workspace",
            filter={
                "id": "id",
                "agent_name": "agent_name",
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "session_id": "session_id",
                "started_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "status": "success",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
            },
            mode="summary",
            page=1,
            page_size=1,
            sort="started_at",
        )
        assert_matches_type(AsyncDefaultPagination[Trace], trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.traces.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        trace = await response.parse()
        assert_matches_type(AsyncDefaultPagination[Trace], trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.traces.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            trace = await response.parse()
            assert_matches_type(AsyncDefaultPagination[Trace], trace, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.traces.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get_metrics(self, async_client: AsyncNeMoHelix) -> None:
        trace = await async_client.intake.traces.get_metrics(
            workspace="workspace",
        )
        assert_matches_type(TraceMetrics, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_get_metrics_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        trace = await async_client.intake.traces.get_metrics(
            workspace="workspace",
            bucket="total",
            filter={
                "id": "id",
                "agent_name": "agent_name",
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "session_id": "session_id",
                "started_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "status": "success",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
            },
            timezone="timezone",
        )
        assert_matches_type(TraceMetrics, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_get_metrics(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.traces.with_raw_response.get_metrics(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        trace = await response.parse()
        assert_matches_type(TraceMetrics, trace, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_get_metrics(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.traces.with_streaming_response.get_metrics(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            trace = await response.parse()
            assert_matches_type(TraceMetrics, trace, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_get_metrics(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.traces.with_raw_response.get_metrics(
                workspace="",
            )
