# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Annotated

from aiohttp import ClientSession
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from nemo_platform_plugin.client.client import AsyncNemoClient
from nmp.common.service.dependencies import get_nemo_client
from nmp.core.inference_gateway.api.authz import (
    OPENAI_EXEC_PERMISSION,
    enforce_delegated_workspace_access,
)
from nmp.core.inference_gateway.api.dependencies import (
    global_http_client,
    global_middleware_registry,
    global_model_cache,
    global_virtual_model_cache,
)
from nmp.core.inference_gateway.api.errors import (
    raise_virtual_model_not_found,
)
from nmp.core.inference_gateway.api.middleware_registry import (
    MiddlewareRegistry,
)
from nmp.core.inference_gateway.api.mock_provider import (
    handle_mock_request,
    is_mock_request,
)
from nmp.core.inference_gateway.api.model_cache import ModelCache
from nmp.core.inference_gateway.api.proxy import (
    PROXY_OPENAPI_EXTRA,
    virtual_model_proxy,
)
from nmp.core.inference_gateway.api.validation import validate_entity_name, validate_model_entity_name
from nmp.core.inference_gateway.api.virtual_model_cache import VirtualModelCache
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ParseOpenAIModelError(Exception):
    """Exception raised when parsing an inference gateway OpenAI model identifier fails."""


def parse_igw_openai_model(igw_openai_model: str) -> tuple[str, str]:
    """Parse an inference gateway OpenAI model identifier into its components.

    The model identifier format is: ``workspace/model_entity_name``.

    Split is on the first ``/`` only, so the returned ``model_entity_name`` may itself contain
    ``/`` — this is intentional for two cases:

    - **LoRA composite ids**: ``{base}&adapters/{adapter_workspace}/{adapter_name}`` — the
      ``ModelCache`` uses the same ``split("/", 1)`` rule so the returned tuple is a valid cache key.
    - **Legacy backwards-compat**: old callers sometimes appended
      ``/{served_model_name}``; since ``served_model_name`` is resolved from the cache, any extra
      trailing segments are simply kept as part of ``model_entity_name`` and either hit or miss the
      cache on their own merits.

    Callers that assume the returned ``model_entity_name`` matches the entity-store
    ``NAME_PATTERN`` (plain alphanumeric + ``-._+@``) must first handle the composite LoRA shape;
    use :func:`validate_model_entity_name` for that.

    Args:
        igw_openai_model: Model identifier in format workspace/model_entity_name

    Returns:
        Tuple of (workspace, model_entity_name)

    Raises:
        ParseOpenAIModelError: If the model identifier doesn't contain at least 2 parts
    """
    parts = igw_openai_model.split("/", 1)
    if len(parts) < 2:
        raise ParseOpenAIModelError(
            f"Failed to parse workspace and model_entity_name from '{igw_openai_model}'. "
            f"Expected format: workspace/model_entity_name"
        )
    return parts[0], parts[1]


class OpenAIModelResp(BaseModel):
    """Duplicated structure for an OpenAI /v1/models individual model response."""

    id: str
    owned_by: str
    object: str = "model"
    created: int = 0


class OpenAIListModelsResp(BaseModel):
    """Duplicated structure for an OpenAI /v1/models response."""

    data: list[OpenAIModelResp]
    object: str = "list"


@router.get(
    "/v2/workspaces/{workspace}/openai/-/v1/models",
    summary="OpenAI List Models",
    response_description="List models request to OpenAI-compatible endpoint",
    operation_id="openai_proxy_list_models",
    status_code=status.HTTP_200_OK,
)
async def openai_get_models(
    workspace: str,
    virtual_model_cache: Annotated[VirtualModelCache, Depends(global_virtual_model_cache)],
) -> OpenAIListModelsResp:
    """
    This endpoint lists the routable VirtualModels in the requested workspace and
    returns them in OpenAI's list models format. Each model ID is the VirtualModel
    identifier in format workspace/name. This includes both autoprovisioned
    VirtualModels (one per served model entity) and custom VirtualModels, keeping
    the catalog in agreement with the inference proxy, which also resolves
    VirtualModels scoped to the request workspace.
    """
    validate_entity_name(workspace, field_name="workspace")
    await enforce_delegated_workspace_access(workspace, OPENAI_EXEC_PERMISSION)

    all_oai_models: list[OpenAIModelResp] = []

    for (vm_workspace, vm_name), _ in virtual_model_cache.virtual_model_map.items():
        if vm_workspace != workspace:
            continue
        all_oai_models.append(OpenAIModelResp(id=f"{vm_workspace}/{vm_name}", owned_by=vm_workspace))

    return OpenAIListModelsResp(data=all_oai_models)


@router.get(
    "/v2/workspaces/{workspace}/openai/-/v1/models/{name:path}",
    summary="OpenAI Get Model",
    response_description="Get model request to OpenAI-compatible endpoint",
    operation_id="openai_proxy_get_model",
    status_code=status.HTTP_200_OK,
)
async def openai_get_model(
    workspace: str,
    name: str,
    virtual_model_cache: Annotated[VirtualModelCache, Depends(global_virtual_model_cache)],
) -> OpenAIModelResp:
    """
    Retrieve information about a specific OpenAI-compatible model.
    Workspace is always taken from the URL path; name may be the VirtualModel
    name or workspace/name (workspace prefix is ignored). Resolves against
    routable VirtualModels, including custom ones, so this route agrees with
    the list route and the inference proxy.
    """
    model_name = name.removeprefix(f"{workspace}/")

    validate_entity_name(workspace, field_name="workspace")
    validate_model_entity_name(model_name, field_name="model")
    await enforce_delegated_workspace_access(workspace, OPENAI_EXEC_PERMISSION)
    if virtual_model_cache.get(workspace, model_name) is None:
        raise_virtual_model_not_found(workspace, model_name)

    return OpenAIModelResp(
        id=f"{workspace}/{model_name}",
        owned_by=workspace,
    )


# Use individual decorators with explicit ordering to ensure deterministic OpenAPI generation
@router.get(
    "/v2/workspaces/{workspace}/openai/-/{trailing_uri:path}",
    summary="OpenAI Inference Proxy GET",
    response_description="Proxy GET request to OpenAI-compatible endpoint",
    operation_id="openai_proxy_get",
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/v2/workspaces/{workspace}/openai/-/{trailing_uri:path}",
    summary="OpenAI Inference Proxy POST",
    response_description="Proxy POST request to OpenAI-compatible endpoint",
    operation_id="openai_proxy_post",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
@router.put(
    "/v2/workspaces/{workspace}/openai/-/{trailing_uri:path}",
    summary="OpenAI Inference Proxy PUT",
    response_description="Proxy PUT request to OpenAI-compatible endpoint",
    operation_id="openai_proxy_put",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
@router.delete(
    "/v2/workspaces/{workspace}/openai/-/{trailing_uri:path}",
    summary="OpenAI Inference Proxy DELETE",
    response_description="Proxy DELETE request to OpenAI-compatible endpoint",
    operation_id="openai_proxy_delete",
    status_code=status.HTTP_200_OK,
)
@router.patch(
    "/v2/workspaces/{workspace}/openai/-/{trailing_uri:path}",
    summary="OpenAI Inference Proxy PATCH",
    response_description="Proxy PATCH request to OpenAI-compatible endpoint",
    operation_id="openai_proxy_patch",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
async def openai_proxy(
    request: Request,
    workspace: str,
    trailing_uri: str,
    http_client: Annotated[ClientSession, Depends(global_http_client)],
    nemo_client: Annotated[AsyncNemoClient, Depends(get_nemo_client)],
    model_cache: Annotated[ModelCache, Depends(global_model_cache)],
    virtual_model_cache: Annotated[VirtualModelCache, Depends(global_virtual_model_cache)],
    registry: Annotated[MiddlewareRegistry, Depends(global_middleware_registry)],
) -> Response:
    """
    Proxy requests to OpenAI-compatible inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's
    provider reconciler auto-creates an implicit `autoprovisioned` VirtualModel
    for every served model entity (named after the entity, with
    `default_model_entity` set to the entity ref) so this is the typical case;
    operators can also create custom VirtualModels for routing, plugin chains,
    LoRA escape-hatches, etc. Requests for which no VirtualModel can be found
    return `404`.
    """
    # Scope delegated (on-behalf-of) service-principal calls to the target
    # workspace before any routing — including the mock short-circuit — so a
    # delegated workload cannot reach a workspace its creator cannot.
    validate_entity_name(workspace, field_name="workspace")
    await enforce_delegated_workspace_access(workspace, OPENAI_EXEC_PERMISSION)

    # If mock mode enabled and request has explicit mock response, skip model lookup
    if is_mock_request(request):
        return await handle_mock_request(request=request, trailing_uri=trailing_uri)

    try:
        json_body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Invalid request body: {exc!s}") from exc
    try:
        body_model = json_body["model"]
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail=f"Could not extract model from openai request: {exc!s}"
        ) from exc

    # Always use workspace from the request path (no cross-workspace routing).
    # If body contains "workspace/model", use only the model name part.
    model_name = body_model.removeprefix(f"{workspace}/")

    validate_model_entity_name(model_name, field_name="model")

    virtual_model = virtual_model_cache.get(workspace, model_name)
    logger.debug(
        "openai_proxy: workspace=%s model_name=%s body_model=%s vm_hit=%s",
        workspace,
        model_name,
        body_model,
        virtual_model is not None,
    )

    if virtual_model is None:
        raise_virtual_model_not_found(workspace, model_name)

    return await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=model_name,
        virtual_model=virtual_model,
        trailing_uri=trailing_uri,
        json_body=json_body,
        http_client=http_client,
        model_cache=model_cache,
        registry=registry,
        request_nemo_client=nemo_client,
    )
