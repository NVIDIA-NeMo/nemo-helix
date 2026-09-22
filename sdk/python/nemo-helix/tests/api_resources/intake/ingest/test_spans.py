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


class TestSpans:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        span = client.intake.ingest.spans.create(
            workspace="workspace",
            source="source",
            spans=[
                {
                    "span_id": "x",
                    "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "trace_id": "x",
                }
            ],
        )
        assert span is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.intake.ingest.spans.with_raw_response.create(
            workspace="workspace",
            source="source",
            spans=[
                {
                    "span_id": "x",
                    "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "trace_id": "x",
                }
            ],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        span = response.parse()
        assert span is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.intake.ingest.spans.with_streaming_response.create(
            workspace="workspace",
            source="source",
            spans=[
                {
                    "span_id": "x",
                    "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "trace_id": "x",
                }
            ],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            span = response.parse()
            assert span is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.intake.ingest.spans.with_raw_response.create(
                workspace="",
                source="source",
                spans=[
                    {
                        "span_id": "x",
                        "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                        "trace_id": "x",
                    }
                ],
            )


class TestAsyncSpans:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        span = await async_client.intake.ingest.spans.create(
            workspace="workspace",
            source="source",
            spans=[
                {
                    "span_id": "x",
                    "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "trace_id": "x",
                }
            ],
        )
        assert span is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.intake.ingest.spans.with_raw_response.create(
            workspace="workspace",
            source="source",
            spans=[
                {
                    "span_id": "x",
                    "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "trace_id": "x",
                }
            ],
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        span = await response.parse()
        assert span is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.intake.ingest.spans.with_streaming_response.create(
            workspace="workspace",
            source="source",
            spans=[
                {
                    "span_id": "x",
                    "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "trace_id": "x",
                }
            ],
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            span = await response.parse()
            assert span is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.intake.ingest.spans.with_raw_response.create(
                workspace="",
                source="source",
                spans=[
                    {
                        "span_id": "x",
                        "started_at": parse_datetime("2019-12-27T18:11:19.117Z"),
                        "trace_id": "x",
                    }
                ],
            )
