# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

from aiohttp import ClientSession
from fastapi import APIRouter, Depends, Request, Response, status
from nemo_platform_plugin.client.client import AsyncNemoClient
from nmp.common.service.dependencies import get_nemo_client
from nmp.core.inference_gateway.api.authz import (
    MODEL_EXEC_PERMISSION,
    enforce_delegated_workspace_access,
)
from nmp.core.inference_gateway.api.dependencies import (
    global_http_client,
    global_middleware_registry,
    global_model_cache,
    global_virtual_model_cache,
)
from nmp.core.inference_gateway.api.errors import raise_virtual_model_not_found
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

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/v2/workspaces/{workspace}/model/{name:path}/-/{trailing_uri:path}",
    summary="Model Inference Proxy GET",
    response_description="Proxy GET request to model entity inference endpoint",
    operation_id="gateway_proxy_get",
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/v2/workspaces/{workspace}/model/{name:path}/-/{trailing_uri:path}",
    summary="Model Inference Proxy POST",
    response_description="Proxy POST request to model entity inference endpoint",
    operation_id="gateway_proxy_post",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
@router.put(
    "/v2/workspaces/{workspace}/model/{name:path}/-/{trailing_uri:path}",
    summary="Model Inference Proxy PUT",
    response_description="Proxy PUT request to model entity inference endpoint",
    operation_id="gateway_proxy_put",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
@router.delete(
    "/v2/workspaces/{workspace}/model/{name:path}/-/{trailing_uri:path}",
    summary="Model Inference Proxy DELETE",
    response_description="Proxy DELETE request to model entity inference endpoint",
    operation_id="gateway_proxy_delete",
    status_code=status.HTTP_200_OK,
)
@router.patch(
    "/v2/workspaces/{workspace}/model/{name:path}/-/{trailing_uri:path}",
    summary="Model Inference Proxy PATCH",
    response_description="Proxy PATCH request to model entity inference endpoint",
    operation_id="gateway_proxy_patch",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
async def model_entity_proxy(
    request: Request,
    workspace: str,
    name: str,
    trailing_uri: str,
    http_client: Annotated[ClientSession, Depends(global_http_client)],
    nemo_client: Annotated[AsyncNemoClient, Depends(get_nemo_client)],
    model_cache: Annotated[ModelCache, Depends(global_model_cache)],
    virtual_model_cache: Annotated[VirtualModelCache, Depends(global_virtual_model_cache)],
    registry: Annotated[MiddlewareRegistry, Depends(global_middleware_registry)],
) -> Response:
    """
    Proxy requests to model entity inference endpoints.

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
    await enforce_delegated_workspace_access(workspace, MODEL_EXEC_PERMISSION)

    # If mock mode enabled and request has explicit mock response, skip model lookup
    if is_mock_request(request):
        return await handle_mock_request(request=request, trailing_uri=trailing_uri)

    # ``name`` may be a composite LoRA model_entity_name like
    # ``base&adapters/{adapter_ws}/{adapter_name}``; ``validate_model_entity_name``
    # accepts that shape (per-segment NAME_PATTERN) while still rejecting bare
    # invalid names.
    validate_model_entity_name(name, field_name="name")
    logger.info(f"Model entity proxy request: {workspace}/{name}/-/{trailing_uri}")

    virtual_model = virtual_model_cache.get(workspace, name)

    if virtual_model is None:
        raise_virtual_model_not_found(workspace, name)

    # Parse body before handing off — bodyless requests (e.g. GET) get an
    # empty dict; virtual_model_proxy handles the rest.
    try:
        json_body: dict = await request.json()
    except Exception:
        json_body = {}

    return await virtual_model_proxy(
        request=request,
        workspace=workspace,
        vm_name=name,
        virtual_model=virtual_model,
        trailing_uri=trailing_uri,
        json_body=json_body,
        http_client=http_client,
        model_cache=model_cache,
        registry=registry,
        request_nemo_client=nemo_client,
    )
