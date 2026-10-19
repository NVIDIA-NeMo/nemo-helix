# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI exception handlers for typed client and entity client exceptions.

When a service makes an internal typed-client call to another service and receives
an HTTP error, the client raises ``NemoHTTPError`` (NotFoundError, ConflictError, etc.).
These handlers convert those exceptions back to proper HTTP responses.

This also handles entity client exceptions (EntityNotFoundError, EntityConflictError)
which wrap client exceptions with additional context.

This prevents "Exception in ASGI application" errors for expected HTTP error responses
from service-to-service calls.

Usage:
    from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers

    # In your service's FastAPI app setup:
    register_sdk_exception_handlers(app)
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nemo_helix_plugin.client.errors import NemoHTTPError
from nhx.common.entities.client import (
    EntityConflictError,
    EntityNotFoundError,
    EntityStoreError,
    EntityValidationError,
)

logger = logging.getLogger(__name__)


def _scrub_crlf(value: str) -> str:
    """Strip CR/LF so user-controlled request fields cannot forge log lines."""
    return value.replace("\r", " ").replace("\n", " ")


# Map entity store exceptions to HTTP status codes
ENTITY_ERROR_STATUS_CODES: dict[type[EntityStoreError], int] = {
    EntityNotFoundError: 404,
    EntityConflictError: 409,
    EntityValidationError: 422,
}


async def nemo_client_error_handler(request: Request, exc: NemoHTTPError) -> JSONResponse:
    """Convert NemoClient HTTP exceptions back to HTTP responses.

    The typed ``NemoClient`` raises ``NemoHTTPError`` (and subclasses) on
    non-2xx service-to-service responses; convert them to proper HTTP
    responses with the same status code and detail.
    """
    logger.debug(
        "Converting NemoClient exception to HTTP response: %s %s -> %d",
        _scrub_crlf(request.method),
        _scrub_crlf(request.url.path),
        exc.status_code,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


async def entity_store_error_handler(request: Request, exc: EntityStoreError) -> JSONResponse:
    """Convert EntityStoreError exceptions to HTTP responses.

    Maps entity store exceptions to appropriate HTTP status codes:
    - EntityNotFoundError -> 404
    - EntityConflictError -> 409 (entity already exists on create, or version mismatch on update)
    - EntityValidationError -> 422
    - Other EntityStoreError -> 500

    Args:
        request: The FastAPI request object
        exc: The EntityStoreError exception

    Returns:
        JSONResponse with the appropriate status code
    """
    # Look up status code for specific exception type
    status_code = ENTITY_ERROR_STATUS_CODES.get(type(exc), 500)

    logger.debug(
        "Converting %s to %d: %s %s",
        type(exc).__name__,
        status_code,
        _scrub_crlf(request.method),
        _scrub_crlf(request.url.path),
    )

    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc)},
    )


def register_sdk_exception_handlers(app: FastAPI) -> None:
    """Register typed client and entity client exception handlers on a FastAPI app.

    This registers handlers for:
    - NemoHTTPError: Base class for typed client HTTP errors (NotFoundError, ConflictError, etc.)
    - EntityStoreError: Base class for entity client errors (EntityNotFoundError, etc.)

    Args:
        app: The FastAPI application to register handlers on
    """
    # Handlers are annotated with the specific exception subtype they handle;
    # Starlette's stub types the callback against the base ``Exception``, so ty
    # flags the narrower signature. The handlers are correct at runtime.
    app.add_exception_handler(NemoHTTPError, nemo_client_error_handler)  # ty: ignore[invalid-argument-type]
    app.add_exception_handler(EntityStoreError, entity_store_error_handler)  # ty: ignore[invalid-argument-type]


__all__ = [
    "nemo_client_error_handler",
    "entity_store_error_handler",
    "register_sdk_exception_handlers",
]
