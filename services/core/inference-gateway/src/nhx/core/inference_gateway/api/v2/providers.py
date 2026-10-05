# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

from aiohttp import ClientSession
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from nhx.core.inference_gateway.api.authz import (
    PROVIDER_EXEC_PERMISSION,
    PROVIDER_READ_PERMISSION,
    enforce_delegated_workspace_access,
)
from nhx.core.inference_gateway.api.dependencies import global_http_client, global_model_cache
from nhx.core.inference_gateway.api.errors import raise_unresolved_provider_secret
from nhx.core.inference_gateway.api.mock_provider import (
    handle_mock_request,
    is_mock_provider,
    is_mock_request,
)
from nhx.core.inference_gateway.api.model_cache import ModelCache
from nhx.core.inference_gateway.api.proxy import PROXY_OPENAPI_EXTRA, build_next_request, proxy_request
from nhx.core.inference_gateway.api.validation import validate_workspace_and_name

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/v2/workspaces/{workspace}/provider/{name}/ready",
    summary="Check Provider Readiness",
    response_description="Check if the gateway can route to a provider",
    operation_id="provider_ready",
    status_code=status.HTTP_200_OK,
)
async def provider_ready(
    workspace: str,
    name: str,
    model_cache: Annotated[ModelCache, Depends(global_model_cache)],
) -> dict:
    """
    Check if a model provider is registered in the gateway's cache.

    This is a lightweight endpoint that only checks the gateway's internal state,
    without making any requests to the actual provider backend. Use this to verify
    the gateway is ready to route requests to a provider after deployment.

    Returns:
        200 OK with provider info if the provider is registered
        404 Not Found if the provider is not yet in the gateway's cache
    """
    validate_workspace_and_name(workspace, name)
    await enforce_delegated_workspace_access(workspace, PROVIDER_READ_PERMISSION)
    logger.info(f"Provider ready check: {workspace}/{name}")

    model_info = model_cache.get_from_provider(workspace, name)
    if model_info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Model provider not found for {workspace}/{name}")

    if model_info.model_provider.status != "READY":
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Provider not ready for {workspace}/{name}",
        )

    return {
        "ready": True,
        "provider": f"{workspace}/{name}",
        "host_url": model_info.model_provider.host_url,
    }


@router.get(
    "/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri:path}",
    summary="Provider Inference Proxy GET",
    response_description="Proxy GET request to provider inference endpoint",
    operation_id="provider_proxy_get",
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri:path}",
    summary="Provider Inference Proxy POST",
    response_description="Proxy POST request to provider inference endpoint",
    operation_id="provider_proxy_post",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
@router.put(
    "/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri:path}",
    summary="Provider Inference Proxy PUT",
    response_description="Proxy PUT request to provider inference endpoint",
    operation_id="provider_proxy_put",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
@router.delete(
    "/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri:path}",
    summary="Provider Inference Proxy DELETE",
    response_description="Proxy DELETE request to provider inference endpoint",
    operation_id="provider_proxy_delete",
    status_code=status.HTTP_200_OK,
)
@router.patch(
    "/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri:path}",
    summary="Provider Inference Proxy PATCH",
    response_description="Proxy PATCH request to provider inference endpoint",
    operation_id="provider_proxy_patch",
    status_code=status.HTTP_200_OK,
    openapi_extra=PROXY_OPENAPI_EXTRA,
)
async def provider_proxy(
    request: Request,
    workspace: str,
    name: str,
    trailing_uri: str,
    http_client: Annotated[ClientSession, Depends(global_http_client)],
    model_cache: Annotated[ModelCache, Depends(global_model_cache)],
) -> Response:
    """
    Proxy requests to provider inference endpoints.
    """
    # Scope delegated (on-behalf-of) service-principal calls to the target
    # workspace before any routing — including the mock short-circuit — so a
    # delegated workload cannot reach a workspace its creator cannot.
    validate_workspace_and_name(workspace, name)
    await enforce_delegated_workspace_access(workspace, PROVIDER_EXEC_PERMISSION)

    # If mock mode enabled and request has explicit mock response, skip provider lookup
    if is_mock_request(request):
        return await handle_mock_request(request=request, trailing_uri=trailing_uri)

    model_info = model_cache.get_from_provider(workspace, name)
    if model_info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Model provider not found for {workspace}/{name}")

    if model_info.model_provider.api_key_secret_name and not model_info.secret_value:
        raise_unresolved_provider_secret(workspace, name)

    # Check if this specific provider is a mock provider based on name prefix
    if is_mock_provider(name):
        return await handle_mock_request(
            request=request,
            trailing_uri=trailing_uri,
            default_extra_headers=model_info.model_provider.default_extra_headers,
        )

    next_request_info = await build_next_request(
        request,
        host_url=model_info.model_provider.host_url,
        trailing_uri=trailing_uri,
        auth_token=model_info.secret_value,
        auth_header_format=model_info.model_provider.auth_header_format,
        default_extra_body=model_info.model_provider.default_extra_body,
        default_extra_headers=model_info.model_provider.default_extra_headers,
        required_extra_body=model_info.model_provider.required_extra_body,
        required_extra_headers=model_info.model_provider.required_extra_headers,
    )

    return await proxy_request(http_client, next_request_info)
