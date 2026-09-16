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
from nemo_helix.types.evaluations import (
    EvaluationResponse,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestEvaluations:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
            dataset_version="dataset_version",
            description="description",
            experiment_group_id="experiment_group_id",
            experiment_ids=["string"],
            metadata={"foo": "string"},
            parent_evaluation_id="parent_evaluation_id",
            root_cause="root_cause",
            source_link="https://example.com",
            status="status",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.create(
                workspace="",
                dataset_name="dataset_name",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.evaluations.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_with_all_params(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
            dataset_version="dataset_version",
            description="description",
            experiment_group_id="experiment_group_id",
            experiment_ids=["string"],
            metadata={"foo": "string"},
            parent_evaluation_id="parent_evaluation_id",
            root_cause="root_cause",
            source_link="https://example.com",
            status="status",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.update(
                path_name="name",
                workspace="",
                dataset_name="dataset_name",
                body_name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path_name` but received ''"):
            client.evaluations.with_raw_response.update(
                path_name="",
                workspace="workspace",
                dataset_name="dataset_name",
                body_name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.list(
            workspace="workspace",
            filter={
                "agent_name": "agent_name",
                "agent_version": "agent_version",
                "cost_usd": {
                    "count": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "mean": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "median": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p90": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p95": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p99": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "sum": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                },
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "created_by": "created_by",
                "dataset_name": "dataset_name",
                "dataset_version": "dataset_version",
                "evaluators": {
                    "foo": {
                        "count": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "mean": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "median": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "p90": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "p95": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "p99": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "sum": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                    }
                },
                "experiment_group_id": "experiment_group_id",
                "experiment_id": "experiment_id",
                "is_deleted": True,
                "is_pinned": True,
                "latency_ms": {
                    "count": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "mean": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "median": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p90": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p95": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p99": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "sum": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                },
                "metadata": {"foo": "string"},
                "model_name": "model_name",
                "name": "name",
                "run_count": {
                    "eq": 0,
                    "gt": 0,
                    "gte": 0,
                    "lt": 0,
                    "lte": 0,
                },
                "tokens": {
                    "count": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "mean": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "median": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p90": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p95": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p99": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "sum": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                },
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
            },
            page=1,
            page_size=1,
            sort="sort",
        )
        assert_matches_type(SyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert_matches_type(SyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert_matches_type(SyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.delete(
            name="name",
            workspace="workspace",
        )
        assert evaluation is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert evaluation is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert evaluation is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.evaluations.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.patch(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_patch_with_all_params(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.patch(
            name="name",
            workspace="workspace",
            description="description",
            experiment_ids=["string"],
            metadata={"foo": "string"},
            parent_evaluation_id="parent_evaluation_id",
            root_cause="root_cause",
            source_link="https://example.com",
            status="status",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_patch(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.patch(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_patch(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.patch(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_patch(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.patch(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.evaluations.with_raw_response.patch(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_pin(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.pin(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_pin(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.pin(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_pin(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.pin(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_pin(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.pin(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.evaluations.with_raw_response.pin(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_unpin(self, client: NeMoHelix) -> None:
        evaluation = client.evaluations.unpin(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_unpin(self, client: NeMoHelix) -> None:
        response = client.evaluations.with_raw_response.unpin(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_unpin(self, client: NeMoHelix) -> None:
        with client.evaluations.with_streaming_response.unpin(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_unpin(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.evaluations.with_raw_response.unpin(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.evaluations.with_raw_response.unpin(
                name="",
                workspace="workspace",
            )


class TestAsyncEvaluations:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
            dataset_version="dataset_version",
            description="description",
            experiment_group_id="experiment_group_id",
            experiment_ids=["string"],
            metadata={"foo": "string"},
            parent_evaluation_id="parent_evaluation_id",
            root_cause="root_cause",
            source_link="https://example.com",
            status="status",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.create(
            workspace="workspace",
            dataset_name="dataset_name",
            name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.create(
                workspace="",
                dataset_name="dataset_name",
                name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.evaluations.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
            dataset_version="dataset_version",
            description="description",
            experiment_group_id="experiment_group_id",
            experiment_ids=["string"],
            metadata={"foo": "string"},
            parent_evaluation_id="parent_evaluation_id",
            root_cause="root_cause",
            source_link="https://example.com",
            status="status",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.update(
            path_name="name",
            workspace="workspace",
            dataset_name="dataset_name",
            body_name="name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.update(
                path_name="name",
                workspace="",
                dataset_name="dataset_name",
                body_name="name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `path_name` but received ''"):
            await async_client.evaluations.with_raw_response.update(
                path_name="",
                workspace="workspace",
                dataset_name="dataset_name",
                body_name="name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.list(
            workspace="workspace",
            filter={
                "agent_name": "agent_name",
                "agent_version": "agent_version",
                "cost_usd": {
                    "count": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "mean": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "median": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p90": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p95": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p99": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "sum": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                },
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "created_by": "created_by",
                "dataset_name": "dataset_name",
                "dataset_version": "dataset_version",
                "evaluators": {
                    "foo": {
                        "count": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "mean": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "median": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "p90": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "p95": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "p99": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                        "sum": {
                            "eq": 0,
                            "gt": 0,
                            "gte": 0,
                            "lt": 0,
                            "lte": 0,
                        },
                    }
                },
                "experiment_group_id": "experiment_group_id",
                "experiment_id": "experiment_id",
                "is_deleted": True,
                "is_pinned": True,
                "latency_ms": {
                    "count": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "mean": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "median": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p90": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p95": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p99": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "sum": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                },
                "metadata": {"foo": "string"},
                "model_name": "model_name",
                "name": "name",
                "run_count": {
                    "eq": 0,
                    "gt": 0,
                    "gte": 0,
                    "lt": 0,
                    "lte": 0,
                },
                "tokens": {
                    "count": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "mean": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "median": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p90": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p95": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "p99": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                    "sum": {
                        "eq": 0,
                        "gt": 0,
                        "gte": 0,
                        "lt": 0,
                        "lte": 0,
                    },
                },
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
            },
            page=1,
            page_size=1,
            sort="sort",
        )
        assert_matches_type(AsyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert_matches_type(AsyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert_matches_type(AsyncDefaultPagination[EvaluationResponse], evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.delete(
            name="name",
            workspace="workspace",
        )
        assert evaluation is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert evaluation is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert evaluation is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.evaluations.with_raw_response.delete(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.patch(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_patch_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.patch(
            name="name",
            workspace="workspace",
            description="description",
            experiment_ids=["string"],
            metadata={"foo": "string"},
            parent_evaluation_id="parent_evaluation_id",
            root_cause="root_cause",
            source_link="https://example.com",
            status="status",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.patch(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_patch(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.patch(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_patch(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.patch(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.evaluations.with_raw_response.patch(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_pin(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.pin(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_pin(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.pin(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_pin(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.pin(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_pin(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.pin(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.evaluations.with_raw_response.pin(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_unpin(self, async_client: AsyncNeMoHelix) -> None:
        evaluation = await async_client.evaluations.unpin(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_unpin(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.evaluations.with_raw_response.unpin(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        evaluation = await response.parse()
        assert_matches_type(EvaluationResponse, evaluation, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_unpin(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.evaluations.with_streaming_response.unpin(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            evaluation = await response.parse()
            assert_matches_type(EvaluationResponse, evaluation, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_unpin(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.evaluations.with_raw_response.unpin(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.evaluations.with_raw_response.unpin(
                name="",
                workspace="workspace",
            )
