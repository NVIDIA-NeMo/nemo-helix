# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from nhx.common.version import OPENAPI_SPEC_VERSION
from nhx.core.entities.api.v2.entities import router as entities_router
from nhx.core.entities.api.v2.projects import router as projects_router
from nhx.core.entities.api.v2.workspaces import router as workspaces_router
from starlette import status
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

ENTITIES_ROUTER_NAME = "Entities"
PROJECTS_ROUTER_NAME = "Projects"
WORKSPACES_ROUTER_NAME = "Workspaces"

tags_metadata = [
    {"name": ENTITIES_ROUTER_NAME, "description": "Operations related to entities."},
    {"name": PROJECTS_ROUTER_NAME, "description": "Operations related to projects."},
    {"name": WORKSPACES_ROUTER_NAME, "description": "Operations related to workspaces."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Initializing entities service...")
    logger.info("Initialized entities service")
    yield
    logger.info("Entities service cleanup completed")


app = FastAPI(
    title="NeMo Helix Entities Microservice",
    description="Generic entity storage service with schema-agnostic design.",
    openapi_tags=tags_metadata,
    version=OPENAPI_SPEC_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_exception_handler(request: Request, ex: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(ex)},
    )


app.include_router(entities_router, tags=[ENTITIES_ROUTER_NAME])
app.include_router(projects_router, tags=[PROJECTS_ROUTER_NAME])
app.include_router(workspaces_router, tags=[WORKSPACES_ROUTER_NAME])
