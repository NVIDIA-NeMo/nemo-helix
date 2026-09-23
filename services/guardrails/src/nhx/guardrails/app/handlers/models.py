# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import httpx
from fastapi import HTTPException, status
from nemo_helix import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
)
from nhx.guardrails.app.constants import X_MODEL_AUTHORIZATION_HEADER

logger = logging.getLogger(__name__)


def handle_authentication_error(e: AuthenticationError):
    message = f"Failed to get models from NIM due to authentication error. Status code: {e.status_code}"
    logger.error(
        f"{message}. Error: {str(e)}",
        extra={"metadata": {"status_code": e.status_code, "url": str(e.request.url)}},
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"{message}. Provide a valid authentication token in the {X_MODEL_AUTHORIZATION_HEADER} header.",
    ) from e


def handle_authorization_error(e: PermissionDeniedError):
    message = f"Failed to get models from NIM due to authorization error. Status code: {e.status_code}"
    logger.error(
        f"{message}. Error: {str(e)}",
        extra={"metadata": {"status_code": e.status_code, "url": str(e.request.url)}},
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"{message}. Provide a valid authentication token in the {X_MODEL_AUTHORIZATION_HEADER} header.",
    ) from e


def handle_timeout_error(e: APITimeoutError | APIConnectionError):
    message = "Failed to get models from NIM due to timeout."
    logger.error(
        f"{message} Error: {str(e)}",
        extra={"metadata": {"timeout_settings": e.request.extensions.get("timeout"), "url": str(e.request.url)}},
    )
    raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=message) from e


def handle_status_code_error(e: APIStatusError):
    message = f"Failed to get models from NIM. NIM responded with status code: {e.status_code}"
    logger.error(
        f"{message}. Error: {str(e)}",
        extra={"metadata": {"status_code": e.status_code, "response_body": e.body, "url": str(e.request.url)}},
    )
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message) from e


def handle_unexpected_error(e: Exception, inference_base_url: httpx.URL, model_id: Optional[str] = None):
    metadata = {
        "nim_endpoint_url": str(inference_base_url.join("/v1")),
    }
    if model_id:
        metadata["model_id"] = model_id

    logger.exception(
        f"Failed to get models from NIM due to unexpected error. Error type: {type(e).__name__}. Error: {str(e)}",
        extra={"metadata": metadata},
    )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected server error occurred."
    ) from e
