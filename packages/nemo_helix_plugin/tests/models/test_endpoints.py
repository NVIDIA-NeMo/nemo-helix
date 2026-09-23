# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Models service endpoint definitions (PreparedRequest shape)."""

from __future__ import annotations

import json
from typing import get_origin

from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.models import endpoints
from nemo_helix_plugin.models.types import (
    Adapter,
    ContainerExecutorConfig,
    CreateAdapterRequest,
    CreateModelAdapterRequest,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    CreateModelProviderRequest,
    CreatePromptRequest,
    Engine,
    FinetuningType,
    ModelDeployment,
    ModelDeploymentConfig,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    ModelEntity,
    ModelProvider,
    ModelProviderStatus,
    Prompt,
    UpdateAdapterRequest,
    UpdateModelDeploymentConfigRequest,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
    UpdateModelEntityRequest,
    UpdateModelProviderStatusRequest,
    UpdatePromptRequest,
    UpsertModelProviderRequest,
)

_PREFIX = "/apis/models/v2/workspaces/{workspace}"


def _json_body(prepared: PreparedRequest) -> dict:
    """Decode a prepared request's JSON body (asserting it is present bytes)."""
    assert isinstance(prepared.content, bytes)
    return json.loads(prepared.content)


# ---------------------------------------------------------------------------
# Model entities
# ---------------------------------------------------------------------------


def test_create_model() -> None:
    prepared = endpoints.create_model(workspace="default", body=CreateModelEntityRequest(name="llama"))
    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == _PREFIX + "/models"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content_type == "application/json"
    assert prepared.response_type is ModelEntity
    # exist_ok wiring: conflict resolver prebuilt, but not requested by default.
    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.method == "GET"
    assert prepared.on_conflict_get.path_params == {"workspace": "default", "name": "llama"}


def test_create_model_excludes_unset() -> None:
    prepared = endpoints.create_model(workspace="w", body=CreateModelEntityRequest(name="m"))
    assert _json_body(prepared) == {"name": "m"}


def test_create_model_workspace_optional() -> None:
    prepared = endpoints.create_model(body=CreateModelEntityRequest(name="m"))
    assert prepared.path_params == {}


def test_list_models_paginated_with_query() -> None:
    prepared = endpoints.list_models(
        workspace="default", query_params={"page": 2, "sort": "-created_at", "verbose": True}
    )
    assert prepared.method == "GET"
    assert prepared.path_template == _PREFIX + "/models"
    assert get_origin(prepared.response_type) is Paginated
    assert prepared.query_params == {"page": 2, "sort": "-created_at", "verbose": True}


def test_get_model_with_verbose() -> None:
    prepared = endpoints.get_model(workspace="default", name="m", query_params={"verbose": True})
    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "name": "m"}
    assert prepared.query_params == {"verbose": True}
    assert prepared.response_type is ModelEntity


def test_update_model_patch_with_verbose() -> None:
    prepared = endpoints.update_model(
        workspace="default", name="m", body=UpdateModelEntityRequest(description="d"), query_params={"verbose": False}
    )
    assert prepared.method == "PATCH"
    assert prepared.path_params == {"workspace": "default", "name": "m"}
    assert _json_body(prepared) == {"description": "d"}
    assert prepared.query_params == {"verbose": False}


def test_delete_model_returns_none() -> None:
    prepared = endpoints.delete_model(workspace="default", name="m")
    assert prepared.method == "DELETE"
    assert prepared.content is None
    assert prepared.response_type is None


# ---------------------------------------------------------------------------
# Adapters (nested + top-level)
# ---------------------------------------------------------------------------


def test_create_model_adapter_nested_path() -> None:
    prepared = endpoints.create_model_adapter(
        workspace="w",
        model_name="base",
        body=CreateModelAdapterRequest(name="a", fileset="w/fs", finetuning_type=FinetuningType.LORA),
    )
    assert prepared.method == "POST"
    assert prepared.path_template == _PREFIX + "/models/{model_name}/adapters"
    assert prepared.path_params == {"workspace": "w", "model_name": "base"}
    assert prepared.response_type is Adapter


def test_update_model_adapter_path() -> None:
    prepared = endpoints.update_model_adapter(
        workspace="w", model_name="base", adapter="a", body=UpdateAdapterRequest(enabled=False)
    )
    assert prepared.method == "PATCH"
    assert prepared.path_params == {"workspace": "w", "model_name": "base", "adapter": "a"}
    assert _json_body(prepared) == {"enabled": False}


def test_delete_model_adapter_path() -> None:
    prepared = endpoints.delete_model_adapter(workspace="w", model_name="base", adapter="a")
    assert prepared.method == "DELETE"
    assert prepared.path_template == _PREFIX + "/models/{model_name}/adapters/{adapter}"
    assert prepared.response_type is None


def test_create_adapter_top_level_conflict_resolver() -> None:
    body = CreateAdapterRequest(name="a", fileset="w/fs", finetuning_type=FinetuningType.LORA, model="ws/base")
    prepared = endpoints.create_adapter(workspace="w", body=body)
    assert prepared.path_template == _PREFIX + "/adapters"
    assert prepared.response_type is Adapter
    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.path_params == {"workspace": "w", "name": "a"}


def test_list_adapters_paginated() -> None:
    prepared = endpoints.list_adapters(workspace="w", query_params={"filter": "name:a"})
    assert get_origin(prepared.response_type) is Paginated
    assert prepared.query_params == {"filter": "name:a"}


def test_get_and_delete_adapter() -> None:
    assert endpoints.get_adapter(workspace="w", name="a").response_type is Adapter
    assert endpoints.delete_adapter(workspace="w", name="a").method == "DELETE"


# ---------------------------------------------------------------------------
# Model providers
# ---------------------------------------------------------------------------


def test_create_provider() -> None:
    prepared = endpoints.create_provider(workspace="w", body=CreateModelProviderRequest(name="p", host_url="http://x"))
    assert prepared.method == "POST"
    assert prepared.path_template == _PREFIX + "/providers"
    assert prepared.response_type is ModelProvider
    assert prepared.on_conflict_get is not None


def test_upsert_provider_is_put() -> None:
    prepared = endpoints.upsert_provider(workspace="w", name="p", body=UpsertModelProviderRequest(host_url="http://x"))
    assert prepared.method == "PUT"
    assert prepared.path_template == _PREFIX + "/providers/{name}"
    assert prepared.response_type is ModelProvider


def test_update_provider_status_is_put_status_path() -> None:
    prepared = endpoints.update_provider_status(
        workspace="w", name="p", body=UpdateModelProviderStatusRequest(status=ModelProviderStatus.READY)
    )
    assert prepared.method == "PUT"
    assert prepared.path_template == _PREFIX + "/providers/{name}/status"


def test_list_get_delete_provider() -> None:
    assert get_origin(endpoints.list_providers(workspace="w").response_type) is Paginated
    assert endpoints.get_provider(workspace="w", name="p").response_type is ModelProvider
    assert endpoints.delete_provider(workspace="w", name="p").response_type is None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_prompt_crud_paths() -> None:
    assert endpoints.create_prompt(workspace="w", body=CreatePromptRequest(name="p")).method == "POST"
    assert endpoints.update_prompt(workspace="w", name="p", body=UpdatePromptRequest()).method == "PUT"
    assert endpoints.get_prompt(workspace="w", name="p").response_type is Prompt
    assert get_origin(endpoints.list_prompts(workspace="w").response_type) is Paginated
    assert endpoints.delete_prompt(workspace="w", name="p").response_type is None


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------


def _create_deployment_body() -> CreateModelDeploymentRequest:
    return CreateModelDeploymentRequest(name="d", config="cfg")


def test_create_deployment() -> None:
    prepared = endpoints.create_deployment(workspace="w", body=_create_deployment_body())
    assert prepared.method == "POST"
    assert prepared.path_template == _PREFIX + "/deployments"
    assert prepared.response_type is ModelDeployment


def test_update_deployment_is_post_name_path() -> None:
    prepared = endpoints.update_deployment(workspace="w", name="d", body=UpdateModelDeploymentRequest(config="cfg"))
    assert prepared.method == "POST"
    assert prepared.path_template == _PREFIX + "/deployments/{name}"


def test_update_deployment_status_with_version_query() -> None:
    prepared = endpoints.update_deployment_status(
        workspace="w",
        name="d",
        body=UpdateModelDeploymentStatusRequest(status=ModelDeploymentStatus.READY),
        query_params={"version": "2"},
    )
    assert prepared.method == "POST"
    assert prepared.path_template == _PREFIX + "/deployments/{name}/status"
    assert prepared.query_params == {"version": "2"}


def test_deployment_versions_and_models() -> None:
    assert endpoints.list_deployment_versions(workspace="w", name="d").response_type == list[ModelDeployment]
    assert endpoints.get_deployment_version(workspace="w", deployment="d", name="2").response_type is ModelDeployment
    models_ep = endpoints.get_deployment_models(workspace="w", name="d")
    assert models_ep.path_template == _PREFIX + "/deployments/{name}/models"


def test_delete_deployment_and_version_return_none() -> None:
    assert endpoints.delete_deployment(workspace="w", name="d").response_type is None
    assert (
        endpoints.delete_deployment_version(workspace="w", deployment="d", name="2").path_template
        == _PREFIX + "/deployments/{deployment}/versions/{name}"
    )


# ---------------------------------------------------------------------------
# Deployment configs
# ---------------------------------------------------------------------------


def _create_config_body() -> CreateModelDeploymentConfigRequest:
    return CreateModelDeploymentConfigRequest(
        name="cfg",
        engine=Engine.VLLM,
        model_spec=ModelDeploymentConfigModelSpec(model_name="llama"),
        executor_config=ContainerExecutorConfig(gpu=1),
    )


def test_deployment_config_crud_paths() -> None:
    create = endpoints.create_deployment_config(workspace="w", body=_create_config_body())
    assert create.method == "POST"
    assert create.path_template == _PREFIX + "/deployment-configs"
    assert create.response_type is ModelDeploymentConfig
    assert create.on_conflict_get is not None

    update = endpoints.update_deployment_config(
        workspace="w",
        name="cfg",
        body=UpdateModelDeploymentConfigRequest(
            engine=Engine.VLLM,
            model_spec=ModelDeploymentConfigModelSpec(model_name="llama"),
            executor_config=ContainerExecutorConfig(gpu=1),
        ),
    )
    assert update.method == "POST"
    assert update.path_template == _PREFIX + "/deployment-configs/{name}"

    assert (
        endpoints.list_deployment_config_versions(workspace="w", name="cfg").response_type
        == list[ModelDeploymentConfig]
    )
    assert (
        endpoints.get_deployment_config_version(workspace="w", config="cfg", name="1").response_type
        is ModelDeploymentConfig
    )
    assert endpoints.delete_deployment_config(workspace="w", name="cfg").response_type is None
    assert (
        endpoints.delete_deployment_config_version(workspace="w", config="cfg", name="1").path_template
        == _PREFIX + "/deployment-configs/{config}/versions/{name}"
    )
