# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anonymizer plugin service implementation."""

from __future__ import annotations

from typing import ClassVar

from anonymizer.interface.errors import AnonymizerError, InvalidConfigError
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from fastapi import Request
from nemo_anonymizer_plugin.app.errors import AnonymizerInternalError, AnonymizerInvalidConfigError
from nemo_platform_plugin.service import NemoService, RouterSpec
from pydantic import ValidationError
from starlette import status
from starlette.responses import JSONResponse


class AnonymizerService(NemoService):
    """Anonymizer service for NeMo Platform."""

    name: ClassVar[str] = "anonymizer"
    dependencies: ClassVar[list[str]] = [
        "entities",
        "auth",
        "jobs",
        "secrets",
        "files",
        "inference-gateway",
    ]

    def get_routers(self) -> list[RouterSpec]:
        from nemo_anonymizer_plugin.app import entity_labels
        from nemo_anonymizer_plugin.functions.preview import PreviewFunction
        from nemo_anonymizer_plugin.jobs.run import RunJob
        from nemo_platform_plugin.authz import AuthzScope
        from nemo_platform_plugin.functions.routes import add_function_routes
        from nemo_platform_plugin.jobs.routes import add_job_routes

        scope = AuthzScope("anonymizer")
        return [
            RouterSpec(
                add_function_routes(
                    PreviewFunction,
                    authz=scope,
                    permission_description="Preview an Anonymizer config",
                ),
                prefix="/v2/workspaces/{workspace}",
                tag="Anonymizer",
                description="Streaming preview of an Anonymizer config.",
            ),
            RouterSpec(
                add_job_routes(RunJob, authz=scope),
                prefix="/v2/workspaces/{workspace}",
                tag="Anonymizer",
                description="Job endpoints",
            ),
            RouterSpec(
                entity_labels.router,
                prefix="/v2/workspaces/{workspace}",
                tag="Anonymizer",
                description="List the default Anonymizer entity labels.",
            ),
        ]

    async def on_startup(self) -> None:
        from nemo_anonymizer_plugin.functions._preview_logs import attach_preview_handler

        attach_preview_handler()

    def get_exception_handlers(self) -> dict:
        async def validation_error_handler(_request: Request, ex: ValidationError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        async def type_error_handler(_request: Request, ex: TypeError):
            # The upstream `ReplaceMethod` union resolves `kind` through a callable
            # `Discriminator` that raises a bare `TypeError` on a bad/missing `kind`
            # instead of a pydantic `ValidationError` (ASTD-328). Any caller that
            # still reaches the upstream discriminator directly — bypassing the
            # `kind`-aware wire schema in `task_config.py` — gets a 422 here too.
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        async def anonymizer_invalid_config_handler(_request: Request, ex: InvalidConfigError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        async def anonymizer_error_handler(_request: Request, ex: AnonymizerError):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(ex)},
            )

        async def plugin_internal_error_handler(_request: Request, ex: AnonymizerInternalError):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(ex)},
            )

        async def plugin_invalid_config_handler(_request: Request, ex: AnonymizerInvalidConfigError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        async def ndd_internal_error_handler(_request: Request, ex: NDDInternalError):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(ex)},
            )

        async def ndd_invalid_config_handler(_request: Request, ex: NDDInvalidConfigError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        return {
            ValidationError: validation_error_handler,
            TypeError: type_error_handler,
            InvalidConfigError: anonymizer_invalid_config_handler,
            AnonymizerError: anonymizer_error_handler,
            AnonymizerInternalError: plugin_internal_error_handler,
            AnonymizerInvalidConfigError: plugin_invalid_config_handler,
            NDDInternalError: ndd_internal_error_handler,
            NDDInvalidConfigError: ndd_invalid_config_handler,
        }
