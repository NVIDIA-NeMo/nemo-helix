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

from nemo_helix import NeMoHelix, AsyncNeMoHelix
from nemo_helix._utils import parse_datetime

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestAtif:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        atif = client.intake.ingest.atif.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
            },
            schema_version="ATIF-v1.0",
        )
        assert atif is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        atif = client.intake.ingest.atif.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
                "extra": {"foo": "bar"},
                "model_name": "model_name",
                "tool_definitions": [{"foo": "bar"}],
            },
            schema_version="ATIF-v1.0",
            continued_trajectory_ref="continued_trajectory_ref",
            evaluation_context={
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
            },
            extra={"foo": "bar"},
            final_metrics={
                "extra": {"foo": "bar"},
                "total_cached_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0,
                "total_prompt_tokens": 0,
                "total_steps": 0,
            },
            notes="notes",
            session_id="session_id",
            steps=[
                {
                    "source": "system",
                    "step_id": 1,
                    "extra": {"foo": "bar"},
                    "is_copied_context": True,
                    "llm_call_count": 0,
                    "message": "string",
                    "observation": {
                        "results": [
                            {
                                "content": "string",
                                "extra": {"foo": "bar"},
                                "source_call_id": "source_call_id",
                                "subagent_trajectory_ref": [
                                    {
                                        "extra": {"foo": "bar"},
                                        "session_id": "session_id",
                                        "trajectory_id": "trajectory_id",
                                        "trajectory_path": "trajectory_path",
                                    }
                                ],
                            }
                        ]
                    },
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            subagent_trajectories=[
                {
                    "agent": {
                        "name": "name",
                        "version": "version",
                        "extra": {"foo": "bar"},
                        "model_name": "model_name",
                        "tool_definitions": [{"foo": "bar"}],
                    },
                    "continued_trajectory_ref": "continued_trajectory_ref",
                    "evaluation_context": {
                        "evaluation_id": "evaluation_id",
                        "evaluation_name": "evaluation_name",
                        "test_case_id": "test_case_id",
                        "test_case_name": "test_case_name",
                    },
                    "extra": {"foo": "bar"},
                    "final_metrics": {
                        "extra": {"foo": "bar"},
                        "total_cached_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_cost_usd": 0,
                        "total_prompt_tokens": 0,
                        "total_steps": 0,
                    },
                    "notes": "notes",
                    "schema_version": "ATIF-v1.0",
                    "session_id": "session_id",
                    "steps": [
                        {
                            "source": "system",
                            "step_id": 1,
                            "extra": {"foo": "bar"},
                            "is_copied_context": True,
                            "llm_call_count": 0,
                            "message": "string",
                            "observation": {
                                "results": [
                                    {
                                        "content": "string",
                                        "extra": {"foo": "bar"},
                                        "source_call_id": "source_call_id",
                                        "subagent_trajectory_ref": [
                                            {
                                                "extra": {"foo": "bar"},
                                                "session_id": "session_id",
                                                "trajectory_id": "trajectory_id",
                                                "trajectory_path": "trajectory_path",
                                            }
                                        ],
                                    }
                                ]
                            },
                            "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                        }
                    ],
                    "subagent_trajectories": [],
                    "trajectory_id": "trajectory_id",
                }
            ],
            trajectory_id="trajectory_id",
        )
        assert atif is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.intake.ingest.atif.with_raw_response.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
            },
            schema_version="ATIF-v1.0",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        atif = response.parse()
        assert atif is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.intake.ingest.atif.with_streaming_response.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
            },
            schema_version="ATIF-v1.0",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            atif = response.parse()
            assert atif is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.ingest.atif.with_raw_response.create(
                workspace="",
                agent={
                    "name": "name",
                    "version": "version",
                },
                schema_version="ATIF-v1.0",
            )


class TestAsyncAtif:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        atif = await async_client.intake.ingest.atif.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
            },
            schema_version="ATIF-v1.0",
        )
        assert atif is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        atif = await async_client.intake.ingest.atif.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
                "extra": {"foo": "bar"},
                "model_name": "model_name",
                "tool_definitions": [{"foo": "bar"}],
            },
            schema_version="ATIF-v1.0",
            continued_trajectory_ref="continued_trajectory_ref",
            evaluation_context={
                "evaluation_id": "evaluation_id",
                "evaluation_name": "evaluation_name",
                "test_case_id": "test_case_id",
                "test_case_name": "test_case_name",
            },
            extra={"foo": "bar"},
            final_metrics={
                "extra": {"foo": "bar"},
                "total_cached_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0,
                "total_prompt_tokens": 0,
                "total_steps": 0,
            },
            notes="notes",
            session_id="session_id",
            steps=[
                {
                    "source": "system",
                    "step_id": 1,
                    "extra": {"foo": "bar"},
                    "is_copied_context": True,
                    "llm_call_count": 0,
                    "message": "string",
                    "observation": {
                        "results": [
                            {
                                "content": "string",
                                "extra": {"foo": "bar"},
                                "source_call_id": "source_call_id",
                                "subagent_trajectory_ref": [
                                    {
                                        "extra": {"foo": "bar"},
                                        "session_id": "session_id",
                                        "trajectory_id": "trajectory_id",
                                        "trajectory_path": "trajectory_path",
                                    }
                                ],
                            }
                        ]
                    },
                    "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                }
            ],
            subagent_trajectories=[
                {
                    "agent": {
                        "name": "name",
                        "version": "version",
                        "extra": {"foo": "bar"},
                        "model_name": "model_name",
                        "tool_definitions": [{"foo": "bar"}],
                    },
                    "continued_trajectory_ref": "continued_trajectory_ref",
                    "evaluation_context": {
                        "evaluation_id": "evaluation_id",
                        "evaluation_name": "evaluation_name",
                        "test_case_id": "test_case_id",
                        "test_case_name": "test_case_name",
                    },
                    "extra": {"foo": "bar"},
                    "final_metrics": {
                        "extra": {"foo": "bar"},
                        "total_cached_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_cost_usd": 0,
                        "total_prompt_tokens": 0,
                        "total_steps": 0,
                    },
                    "notes": "notes",
                    "schema_version": "ATIF-v1.0",
                    "session_id": "session_id",
                    "steps": [
                        {
                            "source": "system",
                            "step_id": 1,
                            "extra": {"foo": "bar"},
                            "is_copied_context": True,
                            "llm_call_count": 0,
                            "message": "string",
                            "observation": {
                                "results": [
                                    {
                                        "content": "string",
                                        "extra": {"foo": "bar"},
                                        "source_call_id": "source_call_id",
                                        "subagent_trajectory_ref": [
                                            {
                                                "extra": {"foo": "bar"},
                                                "session_id": "session_id",
                                                "trajectory_id": "trajectory_id",
                                                "trajectory_path": "trajectory_path",
                                            }
                                        ],
                                    }
                                ]
                            },
                            "timestamp": parse_datetime("2019-12-27T18:11:19.117Z"),
                        }
                    ],
                    "subagent_trajectories": [],
                    "trajectory_id": "trajectory_id",
                }
            ],
            trajectory_id="trajectory_id",
        )
        assert atif is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.ingest.atif.with_raw_response.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
            },
            schema_version="ATIF-v1.0",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        atif = await response.parse()
        assert atif is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.ingest.atif.with_streaming_response.create(
            workspace="workspace",
            agent={
                "name": "name",
                "version": "version",
            },
            schema_version="ATIF-v1.0",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            atif = await response.parse()
            assert atif is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.ingest.atif.with_raw_response.create(
                workspace="",
                agent={
                    "name": "name",
                    "version": "version",
                },
                schema_version="ATIF-v1.0",
            )
