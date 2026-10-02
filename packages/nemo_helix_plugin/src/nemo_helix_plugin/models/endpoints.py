# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Models service.

These are the single source of truth for the HTTP contract. All paths include
the ``/apis/models`` gateway prefix. Although the Stainless SDK grouped some of
these under ``sdk.inference.*`` (deployments, providers, prompts), every route
is served by the Models service under ``/apis/models/v2/...``.

The service exposes six resource groups:
- model entities (``/models``) and their nested adapters (``/models/{m}/adapters``),
- top-level adapters (``/adapters``),
- model providers (``/providers``),
- prompts (``/prompts``),
- model deployments (``/deployments``) with immutable versioning, and
- model deployment configs (``/deployment-configs``) with immutable versioning.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from nemo_helix_plugin.client.endpoint import delete, get, patch, post, put
from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.models.types import (
    Adapter,
    CreateAdapterRequest,
    CreateModelAdapterRequest,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    CreateModelProviderRequest,
    CreatePromptRequest,
    GetModelQueryParams,
    ListAdaptersQueryParams,
    ListDeploymentConfigsQueryParams,
    ListDeploymentsQueryParams,
    ListModelsQueryParams,
    ListPromptsQueryParams,
    ListProvidersQueryParams,
    ModelDeployment,
    ModelDeploymentConfig,
    ModelEntity,
    ModelProvider,
    Prompt,
    UpdateAdapterRequest,
    UpdateDeploymentStatusQueryParams,
    UpdateModelDeploymentConfigRequest,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
    UpdateModelEntityRequest,
    UpdateModelProviderStatusRequest,
    UpdatePromptRequest,
    UpsertModelProviderRequest,
)

_MODELS = "/apis/models/v2/workspaces/{workspace}"


# ---------------------------------------------------------------------------
# Model entities
# ---------------------------------------------------------------------------


@get(_MODELS + "/models/{name}")
@abstractmethod
def get_model(
    *, workspace: str | None = None, name: str, query_params: GetModelQueryParams | None = None
) -> ModelEntity: ...


@get(_MODELS + "/models")
@abstractmethod
def list_models(
    *, workspace: str | None = None, query_params: ListModelsQueryParams | None = None
) -> Paginated[ModelEntity]: ...


def _get_model_on_conflict(body: CreateModelEntityRequest, workspace: str | None) -> PreparedRequest[ModelEntity]:
    """Retrieve request replayed when ``create_model(exist_ok=True)`` 409s."""
    return get_model(name=body.name, workspace=workspace)


@post(_MODELS + "/models", get_on_conflict=_get_model_on_conflict)
@abstractmethod
def create_model(
    *, workspace: str | None = None, body: CreateModelEntityRequest, exist_ok: bool = False
) -> ModelEntity: ...


@patch(_MODELS + "/models/{name}")
@abstractmethod
def update_model(
    *,
    workspace: str | None = None,
    name: str,
    body: UpdateModelEntityRequest,
    query_params: GetModelQueryParams | None = None,
) -> ModelEntity: ...


@delete(_MODELS + "/models/{name}")
@abstractmethod
def delete_model(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Nested adapters (base model in the path)
# ---------------------------------------------------------------------------


@post(_MODELS + "/models/{model_name}/adapters")
@abstractmethod
def create_model_adapter(
    *, workspace: str | None = None, model_name: str, body: CreateModelAdapterRequest
) -> Adapter: ...


@patch(_MODELS + "/models/{model_name}/adapters/{adapter}")
@abstractmethod
def update_model_adapter(
    *, workspace: str | None = None, model_name: str, adapter: str, body: UpdateAdapterRequest
) -> Adapter: ...


@delete(_MODELS + "/models/{model_name}/adapters/{adapter}")
@abstractmethod
def delete_model_adapter(*, workspace: str | None = None, model_name: str, adapter: str) -> None: ...


# ---------------------------------------------------------------------------
# Top-level adapters
# ---------------------------------------------------------------------------


@get(_MODELS + "/adapters/{name}")
@abstractmethod
def get_adapter(*, workspace: str | None = None, name: str) -> Adapter: ...


@get(_MODELS + "/adapters")
@abstractmethod
def list_adapters(
    *, workspace: str | None = None, query_params: ListAdaptersQueryParams | None = None
) -> Paginated[Adapter]: ...


def _get_adapter_on_conflict(body: CreateAdapterRequest, workspace: str | None) -> PreparedRequest[Adapter]:
    return get_adapter(name=body.name, workspace=workspace)


@post(_MODELS + "/adapters", get_on_conflict=_get_adapter_on_conflict)
@abstractmethod
def create_adapter(*, workspace: str | None = None, body: CreateAdapterRequest, exist_ok: bool = False) -> Adapter: ...


@patch(_MODELS + "/adapters/{name}")
@abstractmethod
def update_adapter(*, workspace: str | None = None, name: str, body: UpdateAdapterRequest) -> Adapter: ...


@delete(_MODELS + "/adapters/{name}")
@abstractmethod
def delete_adapter(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Model providers
# ---------------------------------------------------------------------------


@get(_MODELS + "/providers/{name}")
@abstractmethod
def get_provider(*, workspace: str | None = None, name: str) -> ModelProvider: ...


@get(_MODELS + "/providers")
@abstractmethod
def list_providers(
    *, workspace: str | None = None, query_params: ListProvidersQueryParams | None = None
) -> Paginated[ModelProvider]: ...


def _get_provider_on_conflict(
    body: CreateModelProviderRequest, workspace: str | None
) -> PreparedRequest[ModelProvider]:
    return get_provider(name=body.name, workspace=workspace)


@post(_MODELS + "/providers", get_on_conflict=_get_provider_on_conflict)
@abstractmethod
def create_provider(
    *, workspace: str | None = None, body: CreateModelProviderRequest, exist_ok: bool = False
) -> ModelProvider: ...


@put(_MODELS + "/providers/{name}")
@abstractmethod
def upsert_provider(*, workspace: str | None = None, name: str, body: UpsertModelProviderRequest) -> ModelProvider: ...


@put(_MODELS + "/providers/{name}/status")
@abstractmethod
def update_provider_status(
    *, workspace: str | None = None, name: str, body: UpdateModelProviderStatusRequest
) -> ModelProvider: ...


@delete(_MODELS + "/providers/{name}")
@abstractmethod
def delete_provider(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@get(_MODELS + "/prompts/{name}")
@abstractmethod
def get_prompt(*, workspace: str | None = None, name: str) -> Prompt: ...


@get(_MODELS + "/prompts")
@abstractmethod
def list_prompts(
    *, workspace: str | None = None, query_params: ListPromptsQueryParams | None = None
) -> Paginated[Prompt]: ...


def _get_prompt_on_conflict(body: CreatePromptRequest, workspace: str | None) -> PreparedRequest[Prompt]:
    return get_prompt(name=body.name, workspace=workspace)


@post(_MODELS + "/prompts", get_on_conflict=_get_prompt_on_conflict)
@abstractmethod
def create_prompt(*, workspace: str | None = None, body: CreatePromptRequest, exist_ok: bool = False) -> Prompt: ...


@put(_MODELS + "/prompts/{name}")
@abstractmethod
def update_prompt(*, workspace: str | None = None, name: str, body: UpdatePromptRequest) -> Prompt: ...


@delete(_MODELS + "/prompts/{name}")
@abstractmethod
def delete_prompt(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Model deployments
# ---------------------------------------------------------------------------


@get(_MODELS + "/deployments/{name}")
@abstractmethod
def get_deployment(*, workspace: str | None = None, name: str) -> ModelDeployment: ...


@get(_MODELS + "/deployments")
@abstractmethod
def list_deployments(
    *, workspace: str | None = None, query_params: ListDeploymentsQueryParams | None = None
) -> Paginated[ModelDeployment]: ...


@get(_MODELS + "/deployments/{name}/models")
@abstractmethod
def get_deployment_models(*, workspace: str | None = None, name: str) -> dict[str, Any]: ...


@get(_MODELS + "/deployments/{name}/versions")
@abstractmethod
def list_deployment_versions(*, workspace: str | None = None, name: str) -> list[ModelDeployment]: ...


@get(_MODELS + "/deployments/{deployment}/versions/{name}")
@abstractmethod
def get_deployment_version(*, workspace: str | None = None, deployment: str, name: str) -> ModelDeployment: ...


def _get_deployment_on_conflict(
    body: CreateModelDeploymentRequest, workspace: str | None
) -> PreparedRequest[ModelDeployment]:
    return get_deployment(name=body.name, workspace=workspace)


@post(_MODELS + "/deployments", get_on_conflict=_get_deployment_on_conflict)
@abstractmethod
def create_deployment(
    *, workspace: str | None = None, body: CreateModelDeploymentRequest, exist_ok: bool = False
) -> ModelDeployment: ...


@post(_MODELS + "/deployments/{name}")
@abstractmethod
def update_deployment(
    *, workspace: str | None = None, name: str, body: UpdateModelDeploymentRequest
) -> ModelDeployment: ...


@post(_MODELS + "/deployments/{name}/status")
@abstractmethod
def update_deployment_status(
    *,
    workspace: str | None = None,
    name: str,
    body: UpdateModelDeploymentStatusRequest,
    query_params: UpdateDeploymentStatusQueryParams | None = None,
) -> ModelDeployment: ...


@delete(_MODELS + "/deployments/{name}")
@abstractmethod
def delete_deployment(*, workspace: str | None = None, name: str) -> None:
    """Delete a deployment.

    Returns 202 Accepted when teardown is asynchronous (the deployment enters
    DELETING while infrastructure is torn down) or 204 No Content when the
    delete is synchronous (already hard-deleted). Both are success and the typed
    body is ``None``; read ``resp.http_response.status_code`` to distinguish them.
    """


@delete(_MODELS + "/deployments/{deployment}/versions/{name}")
@abstractmethod
def delete_deployment_version(*, workspace: str | None = None, deployment: str, name: str) -> None:
    """Delete a single deployment version (202 async / 204 synchronous; body ``None``).

    See :func:`delete_deployment` for the 202-vs-204 distinction.
    """


# ---------------------------------------------------------------------------
# Model deployment configs
# ---------------------------------------------------------------------------


@get(_MODELS + "/deployment-configs/{name}")
@abstractmethod
def get_deployment_config(*, workspace: str | None = None, name: str) -> ModelDeploymentConfig: ...


@get(_MODELS + "/deployment-configs")
@abstractmethod
def list_deployment_configs(
    *, workspace: str | None = None, query_params: ListDeploymentConfigsQueryParams | None = None
) -> Paginated[ModelDeploymentConfig]: ...


@get(_MODELS + "/deployment-configs/{name}/versions")
@abstractmethod
def list_deployment_config_versions(*, workspace: str | None = None, name: str) -> list[ModelDeploymentConfig]: ...


@get(_MODELS + "/deployment-configs/{config}/versions/{name}")
@abstractmethod
def get_deployment_config_version(*, workspace: str | None = None, config: str, name: str) -> ModelDeploymentConfig: ...


def _get_deployment_config_on_conflict(
    body: CreateModelDeploymentConfigRequest, workspace: str | None
) -> PreparedRequest[ModelDeploymentConfig]:
    return get_deployment_config(name=body.name, workspace=workspace)


@post(_MODELS + "/deployment-configs", get_on_conflict=_get_deployment_config_on_conflict)
@abstractmethod
def create_deployment_config(
    *, workspace: str | None = None, body: CreateModelDeploymentConfigRequest, exist_ok: bool = False
) -> ModelDeploymentConfig: ...


@post(_MODELS + "/deployment-configs/{name}")
@abstractmethod
def update_deployment_config(
    *, workspace: str | None = None, name: str, body: UpdateModelDeploymentConfigRequest
) -> ModelDeploymentConfig: ...


@delete(_MODELS + "/deployment-configs/{name}")
@abstractmethod
def delete_deployment_config(*, workspace: str | None = None, name: str) -> None: ...


@delete(_MODELS + "/deployment-configs/{config}/versions/{name}")
@abstractmethod
def delete_deployment_config_version(*, workspace: str | None = None, config: str, name: str) -> None: ...
