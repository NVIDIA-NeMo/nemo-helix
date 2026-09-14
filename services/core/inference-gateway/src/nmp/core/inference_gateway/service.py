# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway service implementation."""

import asyncio
import logging
from typing import ClassVar, List, Optional, Set

import aiohttp
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from nemo_platform import AsyncNeMoPlatform
from nmp.common.service import RouterConfig, Service
from starlette import status
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class InferenceGatewayService(Service):
    """Inference Gateway service for NeMo Platform."""

    dependencies: ClassVar[List[str]] = ["entities", "auth", "secrets", "models"]

    def __init__(self):
        """Initialize the inference gateway service."""
        super().__init__(name="inference-gateway", module_name="nmp.core.inference_gateway")
        self._http_client: Optional[aiohttp.ClientSession] = None
        self._background_tasks: Set[asyncio.Task] = set()
        self._middleware_registry = None

    @property
    def title(self) -> str:
        return "NeMo Platform Inference Gateway Microservice"

    @property
    def description(self) -> str:
        return "Service for proxying inference requests to various providers and gateways."

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the inference gateway service."""
        from nmp.core.inference_gateway.api.v2 import models, openai, providers, virtual_models

        return [
            RouterConfig(
                providers.router,
                tag="Inference Gateway",
                description="Operations related to inference request proxying.",
            ),
            RouterConfig(
                models.router,
                tag="Inference Gateway",
                description="Operations related to inference request proxying.",
            ),
            RouterConfig(
                openai.router,
                tag="Inference Gateway",
                description="Operations related to inference request proxying.",
            ),
            RouterConfig(
                virtual_models.router,
                tag="Virtual Models",
                description="CRUD operations for VirtualModel inference routing entities.",
            ),
        ]

    def configure_app(self) -> None:
        """Configure exception handlers for the app."""
        super().configure_app()

        @self.app.exception_handler(RequestValidationError)
        async def validation_error_exception_handler(request: Request, ex: RequestValidationError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

    async def on_startup(self) -> None:
        """Initialize resources on startup."""
        from nmp.core.inference_gateway.api.dependencies import (
            set_global_http_client,
            set_global_middleware_registry,
            set_global_model_cache,
            set_global_virtual_model_cache,
        )
        from nmp.core.inference_gateway.api.middleware_registry import load_middleware_plugins
        from nmp.core.inference_gateway.api.model_cache import (
            ModelCache,
            debug_model_provider_getter,
            model_entity_getter_from_sdk,
            model_provider_getter_from_sdk,
            refresh_model_cache,
            refresh_model_cache_task,
        )
        from nmp.core.inference_gateway.api.virtual_model_cache import VirtualModelCache
        from nmp.core.inference_gateway.config import config as inference_gateway_config

        sdk = self.dependency_provider.get_sdk_client(as_service="inference-gateway")

        def plugin_sdk_factory(plugin_name: str) -> AsyncNeMoPlatform:
            return self.dependency_provider.get_sdk_client(as_service=plugin_name)

        # Initialize caches
        model_cache = set_global_model_cache(ModelCache(secret_value_ttl=inference_gateway_config.secrets_ttl_sec))
        virtual_model_cache = set_global_virtual_model_cache(VirtualModelCache())

        # Discover and load inference middleware plugins
        middleware_registry = set_global_middleware_registry(
            await load_middleware_plugins(model_cache, virtual_model_cache, plugin_sdk_factory=plugin_sdk_factory)
        )
        self._middleware_registry = middleware_registry

        if debug_model_providers := inference_gateway_config.debug_model_providers:
            await refresh_model_cache(
                model_cache=model_cache,
                model_provider_getter=debug_model_provider_getter(debug_model_providers),
                model_entity_getter=model_entity_getter_from_sdk(sdk),
                secrets_sdk=sdk,
                virtual_model_cache=virtual_model_cache,
                middleware_registry=middleware_registry,
            )
            logger.debug("Initialized model provider cache with debug providers")
        else:
            # Use the SDK to refresh the model_providers
            model_provider_getter = model_provider_getter_from_sdk(sdk)
            model_entity_getter = model_entity_getter_from_sdk(sdk)

            # Start background refresh task if configured — refreshes both caches each cycle
            if sleep_duration_s := inference_gateway_config.refresh_model_cache_interval_sec:
                self._background_tasks.add(
                    asyncio.create_task(
                        refresh_model_cache_task(
                            model_cache=model_cache,
                            model_provider_getter=model_provider_getter,
                            secrets_sdk=sdk,
                            model_entity_getter=model_entity_getter,
                            sleep_duration_s=sleep_duration_s,
                            virtual_model_cache=virtual_model_cache,
                            middleware_registry=middleware_registry,
                        )
                    )
                )
                logger.debug(f"Started background model cache refresh task (interval: {sleep_duration_s}s)")

        # Initialize HTTP client.
        #
        # - DummyCookieJar: this client session will be used across different
        #   domains, so we don't want to mix cookies between providers.
        # - auto_decompress=False: direct proxy responses pass through
        #   transparently. The inference-middleware request overrides this
        #   setting so aiohttp decodes before JSON or SSE parsing.
        # - skip_auto_headers=["Accept-Encoding"]: pass through the client's
        #   encoding preferences instead of advertising our own.
        # - Explicit TCPConnector: aiohttp's default ClientSession connector
        #   caps at limit=100 total outbound connections, which silently
        #   bottlenecks IGW under concurrent streaming load (every request
        #   above 100 in-flight queues for a slot). The values below mirror
        #   the pattern used by `services/core/files/.../http_session.py`.
        #   Per-host limit prevents one slow provider from starving the rest.
        self._http_client = set_global_http_client(
            aiohttp.ClientSession(
                cookie_jar=aiohttp.DummyCookieJar(),
                auto_decompress=False,
                skip_auto_headers=["Accept-Encoding"],
                connector=aiohttp.TCPConnector(
                    limit=500,
                    limit_per_host=100,
                    ttl_dns_cache=300,
                ),
            )
        )
        logger.debug("Initialized HTTP client for inference-gateway service")

    async def on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""

        for task in self._background_tasks:
            task.cancel()
        logger.debug("Cancelled background tasks for inference-gateway service")

        if self._middleware_registry is not None:
            await self._middleware_registry.shutdown()
            logger.debug("Shut down inference middleware plugins")

        if self._http_client is not None:
            await self._http_client.close()
            logger.debug("Closed HTTP client for inference-gateway service")
        await super().on_shutdown()
