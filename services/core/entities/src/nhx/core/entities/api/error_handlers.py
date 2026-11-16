# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP error policy shared by the deployed and standalone Entities apps."""

from fastapi import FastAPI, Request, status
from nhx.core.entities.app.repository.exceptions import InvalidEntityDataError
from starlette.responses import JSONResponse


async def invalid_entity_data_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return the repository's payload-free validation message as a client error."""
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


def register_entity_exception_handlers(app: FastAPI) -> None:
    """Map only unsafe entity data, leaving unrelated ValueErrors unchanged."""
    app.add_exception_handler(InvalidEntityDataError, invalid_entity_data_handler)
