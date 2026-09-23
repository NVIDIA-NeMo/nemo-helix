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
from nemo_helix.types.models import Adapter

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestAdapters:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        adapter = client.models.adapters.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        adapter = client.models.adapters.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
            description="description",
            enabled=True,
            lora_config={
                "rank": 0,
                "alpha": 0,
            },
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.models.adapters.with_raw_response.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.models.adapters.with_streaming_response.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.adapters.with_raw_response.create(
                model_name="model_name",
                workspace="",
                fileset="fileset",
                finetuning_type="lora_merged",
                name="lora-adapter-v1",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `model_name` but received ''"):
            client.models.adapters.with_raw_response.create(
                model_name="",
                workspace="workspace",
                fileset="fileset",
                finetuning_type="lora_merged",
                name="lora-adapter-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update(self, client: NeMoHelix) -> None:
        adapter = client.models.adapters.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_with_all_params(self, client: NeMoHelix) -> None:
        adapter = client.models.adapters.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
            description="description",
            enabled=True,
            fileset="fileset",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update(self, client: NeMoHelix) -> None:
        response = client.models.adapters.with_raw_response.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update(self, client: NeMoHelix) -> None:
        with client.models.adapters.with_streaming_response.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.adapters.with_raw_response.update(
                adapter="adapter",
                workspace="",
                model_name="model_name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `model_name` but received ''"):
            client.models.adapters.with_raw_response.update(
                adapter="adapter",
                workspace="workspace",
                model_name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `adapter` but received ''"):
            client.models.adapters.with_raw_response.update(
                adapter="",
                workspace="workspace",
                model_name="model_name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        adapter = client.models.adapters.delete(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.models.adapters.with_raw_response.delete(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = response.parse()
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.models.adapters.with_streaming_response.delete(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = response.parse()
            assert adapter is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.models.adapters.with_raw_response.delete(
                adapter="adapter",
                workspace="",
                model_name="model_name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `model_name` but received ''"):
            client.models.adapters.with_raw_response.delete(
                adapter="adapter",
                workspace="workspace",
                model_name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `adapter` but received ''"):
            client.models.adapters.with_raw_response.delete(
                adapter="",
                workspace="workspace",
                model_name="model_name",
            )


class TestAsyncAdapters:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.models.adapters.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.models.adapters.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
            description="description",
            enabled=True,
            lora_config={
                "rank": 0,
                "alpha": 0,
            },
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.models.adapters.with_raw_response.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.models.adapters.with_streaming_response.create(
            model_name="model_name",
            workspace="workspace",
            fileset="fileset",
            finetuning_type="lora_merged",
            name="lora-adapter-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.adapters.with_raw_response.create(
                model_name="model_name",
                workspace="",
                fileset="fileset",
                finetuning_type="lora_merged",
                name="lora-adapter-v1",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `model_name` but received ''"):
            await async_client.models.adapters.with_raw_response.create(
                model_name="",
                workspace="workspace",
                fileset="fileset",
                finetuning_type="lora_merged",
                name="lora-adapter-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.models.adapters.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.models.adapters.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
            description="description",
            enabled=True,
            fileset="fileset",
        )
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.models.adapters.with_raw_response.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert_matches_type(Adapter, adapter, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.models.adapters.with_streaming_response.update(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert_matches_type(Adapter, adapter, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.adapters.with_raw_response.update(
                adapter="adapter",
                workspace="",
                model_name="model_name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `model_name` but received ''"):
            await async_client.models.adapters.with_raw_response.update(
                adapter="adapter",
                workspace="workspace",
                model_name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `adapter` but received ''"):
            await async_client.models.adapters.with_raw_response.update(
                adapter="",
                workspace="workspace",
                model_name="model_name",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        adapter = await async_client.models.adapters.delete(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.models.adapters.with_raw_response.delete(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        adapter = await response.parse()
        assert adapter is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.models.adapters.with_streaming_response.delete(
            adapter="adapter",
            workspace="workspace",
            model_name="model_name",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            adapter = await response.parse()
            assert adapter is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.models.adapters.with_raw_response.delete(
                adapter="adapter",
                workspace="",
                model_name="model_name",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `model_name` but received ''"):
            await async_client.models.adapters.with_raw_response.delete(
                adapter="adapter",
                workspace="workspace",
                model_name="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `adapter` but received ''"):
            await async_client.models.adapters.with_raw_response.delete(
                adapter="",
                workspace="workspace",
                model_name="model_name",
            )
