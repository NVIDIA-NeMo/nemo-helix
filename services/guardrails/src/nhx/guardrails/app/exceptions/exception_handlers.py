# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from nemoguardrails.exceptions import InvalidRailsConfigurationError, LLMCallException
from nemoguardrails.rails.llm.llmrails import ModelInitializationError
from nhx.guardrails.app.common.utils import clean_llm_call_error, clean_model_initialization_error
from openai import APIError, AuthenticationError, RateLimitError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .application_exceptions import CustomHTTPException, GuardrailConfigurationNotFoundError

logger = logging.getLogger(__name__)
DEFAULT_404_DETAIL_MSG = "The requested resource was not found."


def _request_has_image_urls(messages: object) -> bool:
    """Return True if any message in `messages` contains an image URL."""
    if not isinstance(messages, list):
        return False
    for msg in messages:
        content = msg.get("content", []) if isinstance(msg, dict) else []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


async def llm_call_exception_handler(request: Request, exc: LLMCallException):
    has_image_urls = False
    try:
        body = await request.json()
        has_image_urls = _request_has_image_urls(body.get("messages", []))
    except Exception:
        logger.debug("Failed to parse request body to detect image URLs", exc_info=True)

    inner_ex = exc.inner_exception
    inner_ex_message = clean_llm_call_error(str(inner_ex), has_image_urls=has_image_urls)

    # We expect `exc.detail` to contain the model and endpoint URL where the error occurred.
    context = exc.detail
    # Construct our API error message with the context, if available.
    error_message = f"{context}: {inner_ex_message}" if context else inner_ex_message

    logger.error(f"LLMCallException occurred: {error_message}", exc_info=True)

    if isinstance(inner_ex, HTTPException):
        # For FastAPI HTTPException, surface the inner exception's detail in our API response
        detail = inner_ex.detail
        error_message = f"{context}: {detail}" if context else detail
        return JSONResponse(
            status_code=inner_ex.status_code,
            content={"detail": error_message},
        )
    elif isinstance(inner_ex, AuthenticationError):
        # Handle OpenAI AuthenticationError explicitly to ensure 401 is returned
        return JSONResponse(
            status_code=401,
            content={"detail": error_message},
        )
    elif isinstance(inner_ex, RateLimitError):
        # Handle OpenAI RateLimitError explicitly to ensure 429 is returned
        return JSONResponse(
            status_code=429,
            content={"detail": error_message},
        )
    elif hasattr(inner_ex, "status_code"):
        return JSONResponse(
            status_code=inner_ex.status_code,
            content={"detail": error_message},
        )
    elif isinstance(inner_ex, APIError):
        status_code = inner_ex.code or 500

        return JSONResponse(
            status_code=int(status_code),
            content={"detail": error_message},
        )
    elif hasattr(inner_ex, "body") and isinstance(inner_ex.body, dict):
        body = inner_ex.body
        error_message = body.get("message", str(inner_ex))
        status_code = body.get("code", 500)

        if not isinstance(status_code, int):
            status_code = 500

        return JSONResponse(
            status_code=status_code,
            content={"detail": error_message},
        )

    # Otherwise, we return a generic 500 Internal Server Error.
    return JSONResponse(
        status_code=500,
        content={"detail": error_message},
    )


async def config_not_found_error_handler(request: Request, exc: GuardrailConfigurationNotFoundError):
    logger.error(f"GuardrailConfigurationNotFoundError occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


async def model_initialization_error_handler(request: Request, exc: ModelInitializationError):
    logger.error(f"ModelInitializationError occurred: {exc}", exc_info=True)
    error_message = clean_model_initialization_error(str(exc))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": error_message},
    )


async def invalid_rails_configuration_error_handler(request: Request, exc: InvalidRailsConfigurationError):
    logger.error(f"InvalidRailsConfigurationError occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(exc)},
    )


async def custom_exception_handler(request: Request, exc: CustomHTTPException):
    logger.error(f"CustomHTTPException occurred: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message},
    )


async def authentication_error_handler(request: Request, exc: AuthenticationError):
    logger.error(f"AuthenticationError occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=int(exc.code) or 401,
        content={"detail": exc.message},
    )


async def rate_limit_error_handler(request: Request, exc: RateLimitError):
    logger.error(f"RateLimitError occurred: {exc}", exc_info=True)
    status_code = 429
    if exc.code:
        status_code = int(exc.code)

    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.message},
    )


async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    logger.debug(f"404 Not Found: {request.url.path}")

    detail = getattr(exc, "detail", DEFAULT_404_DETAIL_MSG)
    if detail == "Not Found":
        detail = DEFAULT_404_DETAIL_MSG

    return JSONResponse(
        status_code=404,
        content={
            "detail": detail,
            "path": request.url.path,
            "method": request.method,
        },
    )


def _format_field_path(loc: tuple) -> str:
    """Given a model's field where an error was thrown (for example: (body, data)),
    formats the field tuple into a readable path string."""
    return ".".join(str(x) for x in loc) if loc else ""


def _format_validation_message(message: str, loc: tuple, include_prefix: bool = True) -> str:
    """Format a validation error message with the field path where the error was thrown, and an
    optional prefix."""
    field_path = _format_field_path(loc)

    if field_path and include_prefix:
        return f"Validation error at {field_path}: {message}"
    elif field_path:
        return f"{field_path}: {message}"
    else:
        return message


async def validation_error_handler(request: Request, exc: RequestValidationError | ValidationError):
    """
    Handles validation errors and transforms the error object into a user-friendly response.

    This handles both:
    - `RequestValidationError`: raised by FastAPI during request body parsing
    - `ValidationError`: raised when manually constructing Pydantic models
    """
    logger.error(f"Validation error occurred: {exc}", exc_info=True)

    errors = exc.errors()

    # If an error was thrown by a custom validator, extract it from `ctx.error`
    for error in errors:
        ctx = error.get("ctx", {})
        if "error" in ctx:
            actual_error = ctx["error"]
            detail = _format_validation_message(str(actual_error), error.get("loc", ()))

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": detail},
            )

    # If a custom error wasn't thrown, extract the Pydantic validation error(s)
    if len(errors) == 1:
        error = errors[0]
        detail = _format_validation_message(error.get("msg", "Validation error"), error.get("loc", ()))
    else:
        formatted_errors = [
            _format_validation_message(error.get("msg", "Validation error"), error.get("loc", ()), include_prefix=False)
            for error in errors
        ]
        detail = "; ".join(formatted_errors)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": detail},
    )
