# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails service implementation."""

import logging
import os
from typing import ClassVar, List

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.rails.llm.llmrails import ModelInitializationError
from nhx.common.service import RouterConfig, Service
from nhx.guardrails.api.v2.checks import endpoints as checks
from nhx.guardrails.api.v2.configs import endpoints as configs
from nhx.guardrails.app.exceptions import CustomHTTPException, LLMCallException
from nhx.guardrails.app.exceptions.application_exceptions import GuardrailConfigurationNotFoundError
from nhx.guardrails.app.exceptions.exception_handlers import (
    config_not_found_error_handler,
    custom_404_handler,
    custom_exception_handler,
    invalid_rails_configuration_error_handler,
    llm_call_exception_handler,
    model_initialization_error_handler,
    validation_error_handler,
)
from nhx.guardrails.app.patches import apply_langchain_patch
from nhx.guardrails.config import GuardrailsServiceConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class GuardrailsService(Service[GuardrailsServiceConfig]):
    """NeMo Guardrails service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities", "auth", "secrets", "files", "models"]

    def __init__(self) -> None:
        """Initialize the guardrails service."""
        super().__init__(name="guardrails", module_name="nhx.guardrails")

        # Apply langchain patches at service init
        apply_langchain_patch()

    @property
    def title(self) -> str:
        return "NeMo GuardRails Microservice"

    @property
    def description(self) -> str:
        return "NeMo Guardrails allows you to add programmable guardrails to LLM endpoints."

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the guardrails service."""
        return [
            RouterConfig(configs.router, tag="Guardrails", description="Guardrail configuration endpoints"),
            RouterConfig(checks.router, tag="Guardrails", description="Guardrail check endpoints"),
        ]

    async def on_startup(self) -> None:
        """Initialize on startup."""
        # Ensure NVIDIA_BASE_URL is set for langchain-nvidia-ai-endpoints
        # TODO(v2): CONFIG
        if "NIM_ENDPOINT_URL" in os.environ and "NVIDIA_BASE_URL" not in os.environ:
            from nhx.guardrails.app.constants import FALLBACK_DEFAULT_ENDPOINT_URL

            inference_base_url = os.environ.get("NIM_ENDPOINT_URL", FALLBACK_DEFAULT_ENDPOINT_URL)
            os.environ["NVIDIA_BASE_URL"] = inference_base_url
            logger.info(f"Set NVIDIA_BASE_URL to: {inference_base_url}")

    def configure_app(self, app: FastAPI) -> None:
        """Configure additional app settings after creation."""
        # Add middlewares
        self._configure_middlewares(app)

        # Register exception handlers
        self._register_exception_handlers(app)

    def _configure_middlewares(self, app: FastAPI) -> None:
        """Configure request middlewares."""
        from nhx.guardrails.app.middlewares import (
            add_request_id_header,
            capture_trace_id,
        )

        app.middleware("http")(add_request_id_header)
        app.middleware("http")(capture_trace_id)

    def _register_exception_handlers(self, app: FastAPI) -> None:
        """Register custom exception handlers."""
        app.add_exception_handler(GuardrailConfigurationNotFoundError, config_not_found_error_handler)
        app.add_exception_handler(CustomHTTPException, custom_exception_handler)
        app.add_exception_handler(LLMCallException, llm_call_exception_handler)
        app.add_exception_handler(ModelInitializationError, model_initialization_error_handler)
        app.add_exception_handler(InvalidRailsConfigurationError, invalid_rails_configuration_error_handler)
        app.add_exception_handler(RequestValidationError, validation_error_handler)
        app.add_exception_handler(ValidationError, validation_error_handler)
        app.add_exception_handler(404, custom_404_handler)
