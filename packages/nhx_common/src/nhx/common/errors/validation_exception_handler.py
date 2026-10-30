# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A request-validation handler that always answers 422.

FastAPI's default handler echoes each invalid value back in the error's ``input``. Python's JSON parser accepts
``NaN``, ``Infinity`` and lone surrogates, and a non-JSON body arrives as raw bytes that may not be UTF-8, but a
response can't encode any of them, so a request carrying one got 500. This handler keeps the default body and drops
``input`` and ``ctx`` only when that body can't be encoded.
"""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

# The only fields that hold request values as Python objects. pydantic-core builds type, loc and msg as UTF-8 strings:
# it replaces a lone surrogate in a loc key with U+FFFD, and a validator message holding one fails validation itself.
_UNENCODABLE_FIELDS = ("input", "ctx")


async def request_validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer 422 with FastAPI's default body, minus the fields that can't be encoded when there are any."""
    if not isinstance(exc, RequestValidationError):
        raise exc
    # Encoding fails on non-UTF-8 bytes (UnicodeDecodeError); rendering fails on NaN or a lone surrogate.
    try:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": jsonable_encoder(exc.errors())}
        )
    except (ValueError, UnicodeEncodeError):
        errors = [
            {field: value for field, value in error.items() if field not in _UNENCODABLE_FIELDS}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": jsonable_encoder(errors)}
        )


def register_validation_exception_handler(app: FastAPI) -> None:
    """Answer every request-validation error on ``app`` with 422."""
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)


__all__ = ["register_validation_exception_handler", "request_validation_error_handler"]
