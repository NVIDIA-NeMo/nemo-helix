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
from nemo_helix.types.inference import (
    ModelDeploymentConfig,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestDeploymentConfigs:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.create(
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
            name="nim-config-v1",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.create(
            workspace="workspace",
            engine="nim",
            executor_config={
                "gpu": 0,
                "additional_args": ["string"],
                "additional_envs": {"foo": "string"},
                "disk_size": "disk_size",
                "health_check_path": "health_check_path",
                "image_name": "image_name",
                "image_tag": "image_tag",
                "k8s_nim_operator_config": {
                    "node_selector": {"foo": "string"},
                    "resources": {"foo": "bar"},
                    "startup_probe_grace_seconds": 1,
                    "tolerations": [{"foo": "bar"}],
                },
                "override_config": {"foo": "bar"},
                "run_as_group": 0,
                "run_as_user": 0,
            },
            model_spec={
                "chat_template": "chat_template",
                "lora_enabled": True,
                "model_name": "model_name",
                "model_namespace": "model_namespace",
                "model_revision": "model_revision",
                "model_type": "llm",
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            name="nim-config-v1",
            description="description",
            model_entity_id="model_entity_id",
            project="project",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: NeMoHelix) -> None:
        response = client.inference.deployment_configs.with_raw_response.create(
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
            name="nim-config-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = response.parse()
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: NeMoHelix) -> None:
        with client.inference.deployment_configs.with_streaming_response.create(
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
            name="nim-config-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = response.parse()
            assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_create(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.deployment_configs.with_raw_response.create(
                workspace="",
                engine="nim",
                executor_config={"gpu": 0},
                model_spec={},
                name="nim-config-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: NeMoHelix) -> None:
        response = client.inference.deployment_configs.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = response.parse()
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: NeMoHelix) -> None:
        with client.inference.deployment_configs.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = response.parse()
            assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_retrieve(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.deployment_configs.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.deployment_configs.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_update_with_all_params(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={
                "gpu": 0,
                "additional_args": ["string"],
                "additional_envs": {"foo": "string"},
                "disk_size": "disk_size",
                "health_check_path": "health_check_path",
                "image_name": "image_name",
                "image_tag": "image_tag",
                "k8s_nim_operator_config": {
                    "node_selector": {"foo": "string"},
                    "resources": {"foo": "bar"},
                    "startup_probe_grace_seconds": 1,
                    "tolerations": [{"foo": "bar"}],
                },
                "override_config": {"foo": "bar"},
                "run_as_group": 0,
                "run_as_user": 0,
            },
            model_spec={
                "chat_template": "chat_template",
                "lora_enabled": True,
                "model_name": "model_name",
                "model_namespace": "model_namespace",
                "model_revision": "model_revision",
                "model_type": "llm",
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            description="description",
            model_entity_id="model_entity_id",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_update(self, client: NeMoHelix) -> None:
        response = client.inference.deployment_configs.with_raw_response.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = response.parse()
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_update(self, client: NeMoHelix) -> None:
        with client.inference.deployment_configs.with_streaming_response.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = response.parse()
            assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_update(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.deployment_configs.with_raw_response.update(
                name="name",
                workspace="",
                engine="nim",
                executor_config={"gpu": 0},
                model_spec={},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.deployment_configs.with_raw_response.update(
                name="",
                workspace="workspace",
                engine="nim",
                executor_config={"gpu": 0},
                model_spec={},
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.list(
            workspace="workspace",
        )
        assert_matches_type(SyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_list_with_all_params(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.list(
            workspace="workspace",
            filter={
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "description": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "model_entity_id": "model_entity_id",
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=0,
            page_size=0,
            sort="sort",
        )
        assert_matches_type(SyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_list(self, client: NeMoHelix) -> None:
        response = client.inference.deployment_configs.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = response.parse()
        assert_matches_type(SyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_list(self, client: NeMoHelix) -> None:
        with client.inference.deployment_configs.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = response.parse()
            assert_matches_type(SyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_list(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.deployment_configs.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_method_delete(self, client: NeMoHelix) -> None:
        deployment_config = client.inference.deployment_configs.delete(
            name="name",
            workspace="workspace",
        )
        assert deployment_config is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_raw_response_delete(self, client: NeMoHelix) -> None:
        response = client.inference.deployment_configs.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = response.parse()
        assert deployment_config is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_streaming_response_delete(self, client: NeMoHelix) -> None:
        with client.inference.deployment_configs.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = response.parse()
            assert deployment_config is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    def test_path_params_delete(self, client: NeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            client.inference.deployment_configs.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            client.inference.deployment_configs.with_raw_response.delete(
                name="",
                workspace="workspace",
            )


class TestAsyncDeploymentConfigs:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.create(
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
            name="nim-config-v1",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.create(
            workspace="workspace",
            engine="nim",
            executor_config={
                "gpu": 0,
                "additional_args": ["string"],
                "additional_envs": {"foo": "string"},
                "disk_size": "disk_size",
                "health_check_path": "health_check_path",
                "image_name": "image_name",
                "image_tag": "image_tag",
                "k8s_nim_operator_config": {
                    "node_selector": {"foo": "string"},
                    "resources": {"foo": "bar"},
                    "startup_probe_grace_seconds": 1,
                    "tolerations": [{"foo": "bar"}],
                },
                "override_config": {"foo": "bar"},
                "run_as_group": 0,
                "run_as_user": 0,
            },
            model_spec={
                "chat_template": "chat_template",
                "lora_enabled": True,
                "model_name": "model_name",
                "model_namespace": "model_namespace",
                "model_revision": "model_revision",
                "model_type": "llm",
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            name="nim-config-v1",
            description="description",
            model_entity_id="model_entity_id",
            project="project",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.deployment_configs.with_raw_response.create(
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
            name="nim-config-v1",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = await response.parse()
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.deployment_configs.with_streaming_response.create(
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
            name="nim-config-v1",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = await response.parse()
            assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_create(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.create(
                workspace="",
                engine="nim",
                executor_config={"gpu": 0},
                model_spec={},
                name="nim-config-v1",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.retrieve(
            name="name",
            workspace="workspace",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.deployment_configs.with_raw_response.retrieve(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = await response.parse()
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.deployment_configs.with_streaming_response.retrieve(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = await response.parse()
            assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_retrieve(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.retrieve(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.retrieve(
                name="",
                workspace="workspace",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_update_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={
                "gpu": 0,
                "additional_args": ["string"],
                "additional_envs": {"foo": "string"},
                "disk_size": "disk_size",
                "health_check_path": "health_check_path",
                "image_name": "image_name",
                "image_tag": "image_tag",
                "k8s_nim_operator_config": {
                    "node_selector": {"foo": "string"},
                    "resources": {"foo": "bar"},
                    "startup_probe_grace_seconds": 1,
                    "tolerations": [{"foo": "bar"}],
                },
                "override_config": {"foo": "bar"},
                "run_as_group": 0,
                "run_as_user": 0,
            },
            model_spec={
                "chat_template": "chat_template",
                "lora_enabled": True,
                "model_name": "model_name",
                "model_namespace": "model_namespace",
                "model_revision": "model_revision",
                "model_type": "llm",
                "tool_call_config": {
                    "auto_tool_choice": True,
                    "tool_call_parser": "tool_call_parser",
                    "tool_call_plugin": "tool_call_plugin",
                },
            },
            description="description",
            model_entity_id="model_entity_id",
        )
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_update(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.deployment_configs.with_raw_response.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = await response.parse()
        assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_update(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.deployment_configs.with_streaming_response.update(
            name="name",
            workspace="workspace",
            engine="nim",
            executor_config={"gpu": 0},
            model_spec={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = await response.parse()
            assert_matches_type(ModelDeploymentConfig, deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_update(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.update(
                name="name",
                workspace="",
                engine="nim",
                executor_config={"gpu": 0},
                model_spec={},
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.update(
                name="",
                workspace="workspace",
                engine="nim",
                executor_config={"gpu": 0},
                model_spec={},
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.list(
            workspace="workspace",
        )
        assert_matches_type(AsyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_list_with_all_params(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.list(
            workspace="workspace",
            filter={
                "created_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "description": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "model_entity_id": "model_entity_id",
                "name": {
                    "eq": "$eq",
                    "in_": ["string"],
                    "like": "$like",
                    "nin": ["string"],
                },
                "project": "project",
                "updated_at": {
                    "gte": parse_datetime("2019-12-27T18:11:19.117Z"),
                    "lte": parse_datetime("2019-12-27T18:11:19.117Z"),
                },
                "workspace": "workspace",
            },
            page=0,
            page_size=0,
            sort="sort",
        )
        assert_matches_type(AsyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_list(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.deployment_configs.with_raw_response.list(
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = await response.parse()
        assert_matches_type(AsyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_list(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.deployment_configs.with_streaming_response.list(
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = await response.parse()
            assert_matches_type(AsyncDefaultPagination[ModelDeploymentConfig], deployment_config, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_list(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.list(
                workspace="",
            )

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_method_delete(self, async_client: AsyncNeMoHelix) -> None:
        deployment_config = await async_client.inference.deployment_configs.delete(
            name="name",
            workspace="workspace",
        )
        assert deployment_config is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_raw_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        response = await async_client.inference.deployment_configs.with_raw_response.delete(
            name="name",
            workspace="workspace",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        deployment_config = await response.parse()
        assert deployment_config is None

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_streaming_response_delete(self, async_client: AsyncNeMoHelix) -> None:
        async with async_client.inference.deployment_configs.with_streaming_response.delete(
            name="name",
            workspace="workspace",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            deployment_config = await response.parse()
            assert deployment_config is None

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Mock server tests are disabled")
    @parametrize
    async def test_path_params_delete(self, async_client: AsyncNeMoHelix) -> None:
        with pytest.raises(ValueError, match=r"Expected a non-empty value for `workspace` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.delete(
                name="name",
                workspace="",
            )

        with pytest.raises(ValueError, match=r"Expected a non-empty value for `name` but received ''"):
            await async_client.inference.deployment_configs.with_raw_response.delete(
                name="",
                workspace="workspace",
            )
