# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Files service implementation."""

import logging
from typing import ClassVar, List

import sniffio
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from nhx.common.service import RouterConfig, Service
from nhx.core.files.api.v2.filesets import endpoints as filesets
from nhx.core.files.api.v2.hf import endpoints as hf
from nhx.core.files.api.v2.otlp import endpoints as otlp
from nhx.core.files.app.backends import storage_impl_factory
from nhx.core.files.app.http_session import close_http_session
from nhx.core.files.config import FilesConfig
from starlette import status
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class FilesService(Service[FilesConfig]):
    """Files service for NeMo Helix."""

    dependencies: ClassVar[List[str]] = ["entities", "auth", "secrets"]

    def __init__(self):
        """Initialize the files service."""
        super().__init__(name="files", module_name="nhx.core.files")

    def get_routers(self) -> List[RouterConfig]:
        """Return routers for the files service."""
        return [
            RouterConfig(filesets.router, tag="Files", description="File management endpoints"),
            RouterConfig(
                hf.router,
                tag="HuggingFace",
                description="HuggingFace Hub compatible endpoints for dataset/model access",
            ),
            RouterConfig(
                otlp.router,
                tag="OTLP",
                description="OTLP ingestion endpoints for filesets",
            ),
        ]

    async def on_startup(self) -> None:
        """Initialize storage backend on startup."""
        # Short-circuit sniffio's async library detection.
        # This eliminates getpid() syscalls on every anyio checkpoint by telling
        # sniffio we're using asyncio without it needing to detect it each time.
        sniffio.current_async_library_cvar.set("asyncio")

        if self._service_config is None:
            logger.warning("FilesService started without config - storage validation skipped")
            return

        # Initialize storage backend
        impl = storage_impl_factory(self._service_config.default_storage_config)
        await impl.validate_storage()

    async def on_shutdown(self) -> None:
        """Clean up resources on shutdown."""
        await close_http_session()
        await super().on_shutdown()

    def configure_app(self) -> None:
        """Configure the FastAPI app with custom exception handlers."""
        super().configure_app()

        @self.app.exception_handler(RequestValidationError)
        async def validation_error_exception_handler(request: Request, ex: RequestValidationError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )
