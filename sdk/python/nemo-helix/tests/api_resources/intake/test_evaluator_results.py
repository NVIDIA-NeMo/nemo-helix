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
    EvaluatorResult,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestEvaluatorResults:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        evaluator_result = client.intake.evaluator_results.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
        )
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        evaluator_result = client.intake.evaluator_results.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
            comment="comment",
            string_value="string_value",
            value=0,
        )
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.intake.evaluator_results.with_raw_response.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluator_result = response.parse()
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.intake.evaluator_results.with_streaming_response.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluator_result = response.parse()
            assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.evaluator_results.with_raw_response.create(
                workspace="",
                data_type="NUMERIC",
                name="name",
                session_id="session_id",
                span_id="span_id",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoHelix) -> None:
        evaluator_result = client.intake.evaluator_results.retrieve(
            evaluator_result_id="evaluator_result_id",
            workspace="workspace",
        )
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoHelix) -> None:
        response = client.intake.evaluator_results.with_raw_response.retrieve(
            evaluator_result_id="evaluator_result_id",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluator_result = response.parse()
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoHelix) -> None:
        with client.intake.evaluator_results.with_streaming_response.retrieve(
            evaluator_result_id="evaluator_result_id",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluator_result = response.parse()
            assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.evaluator_results.with_raw_response.retrieve(
                evaluator_result_id="evaluator_result_id",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `evaluator_result_id` but received ''"):
            client.intake.evaluator_results.with_raw_response.retrieve(
                evaluator_result_id="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        evaluator_result = client.intake.evaluator_results.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        evaluator_result = client.intake.evaluator_results.list(
            workspace="workspace",
            filter={
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "created_by": "created_by",
                "data_type": "NUMERIC",
                "name": "name",
                "session_id": "session_id",
                "span_id": "span_id",
                "value": {
                    "gte": 0,
                    "lte": 0,
                },
            },
            page=1,
            page_size=1,
            sort="created_at",
        )
        assert_matches_type(SyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.intake.evaluator_results.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluator_result = response.parse()
        assert_matches_type(SyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.intake.evaluator_results.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluator_result = response.parse()
            assert_matches_type(SyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.evaluator_results.with_raw_response.list(
                workspace="",
            )


class TestAsyncEvaluatorResults:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        evaluator_result = await async_client.intake.evaluator_results.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
        )
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        evaluator_result = await async_client.intake.evaluator_results.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
            comment="comment",
            string_value="string_value",
            value=0,
        )
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.evaluator_results.with_raw_response.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluator_result = await response.parse()
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.evaluator_results.with_streaming_response.create(
            workspace="workspace",
            data_type="NUMERIC",
            name="name",
            session_id="session_id",
            span_id="span_id",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluator_result = await response.parse()
            assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.evaluator_results.with_raw_response.create(
                workspace="",
                data_type="NUMERIC",
                name="name",
                session_id="session_id",
                span_id="span_id",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        evaluator_result = await async_client.intake.evaluator_results.retrieve(
            evaluator_result_id="evaluator_result_id",
            workspace="workspace",
        )
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.evaluator_results.with_raw_response.retrieve(
            evaluator_result_id="evaluator_result_id",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluator_result = await response.parse()
        assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.evaluator_results.with_streaming_response.retrieve(
            evaluator_result_id="evaluator_result_id",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluator_result = await response.parse()
            assert_matches_type(EvaluatorResult, evaluator_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.evaluator_results.with_raw_response.retrieve(
                evaluator_result_id="evaluator_result_id",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `evaluator_result_id` but received ''"):
            await async_client.intake.evaluator_results.with_raw_response.retrieve(
                evaluator_result_id="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        evaluator_result = await async_client.intake.evaluator_results.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        evaluator_result = await async_client.intake.evaluator_results.list(
            workspace="workspace",
            filter={
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "created_by": "created_by",
                "data_type": "NUMERIC",
                "name": "name",
                "session_id": "session_id",
                "span_id": "span_id",
                "value": {
                    "gte": 0,
                    "lte": 0,
                },
            },
            page=1,
            page_size=1,
            sort="created_at",
        )
        assert_matches_type(AsyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.evaluator_results.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluator_result = await response.parse()
        assert_matches_type(AsyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.evaluator_results.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluator_result = await response.parse()
            assert_matches_type(AsyncDefaultPagination[EvaluatorResult], evaluator_result, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.evaluator_results.with_raw_response.list(
                workspace="",
            )
