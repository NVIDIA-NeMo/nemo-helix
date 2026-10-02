# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Models service implementation."""

import logging
from typing import ClassVar, List

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from nhx.common.service import RouterConfig, Service
from nhx.core.models.config import ModelsConfig
from starlette import status
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class ModelsService(Service[ModelsConfig]):
    """Models service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities", "auth", "secrets", "files"]

    def __init__(self):
        """Initialize the models service."""
        super().__init__(name="models", module_name="nhx.core.models")

    @property
    def title(self) -> str:
        return "NeMo Helix Models Microservice"

    @property
    def description(self) -> str:
        return "Service for model configuration, scaling, and deployment."

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the models service."""
        from nhx.core.models.api.v2 import adapters, deployment_configs, deployments, models, prompts, providers

        return [
            RouterConfig(
                models.router,
                tag="Models",
                description="Operations related to model entities.",
            ),
            RouterConfig(
                adapters.router,
                tag="Adapters",
                description="CRUD for adapter (LoRA, etc.) entities.",
            ),
            RouterConfig(
                deployment_configs.router,
                tag="ModelDeploymentConfigs",
                description="Operations related to model deployment configurations.",
            ),
            RouterConfig(
                deployments.router,
                tag="ModelDeployments",
                description="Operations related to model deployments.",
            ),
            RouterConfig(
                providers.router,
                tag="ModelProviders",
                description="Operations related to model providers.",
            ),
            RouterConfig(
                prompts.router,
                tag="Prompts",
                description="CRUD operations for reusable chat prompt entities.",
            ),
        ]

    def configure_app(self) -> None:
        """Configure exception handlers for the app."""

        @self.app.exception_handler(RequestValidationError)
        async def validation_error_exception_handler(request: Request, ex: RequestValidationError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )
