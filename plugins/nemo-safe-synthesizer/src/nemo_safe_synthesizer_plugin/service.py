# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe Synthesizer plugin service implementation."""

from __future__ import annotations

from typing import ClassVar

from fastapi import Request
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import ExceptionHandler, NemoService, RouterSpec
from pydantic import ValidationError
from starlette import status
from starlette.responses import JSONResponse

_SERVICE_NAME = "safe-synthesizer"
_AUTHZ = AuthzScope(_SERVICE_NAME)


class SafeSynthesizerService(NemoService):
    """Safe Synthesizer service exposed as an NHX plugin."""

    name: ClassVar[str] = _SERVICE_NAME
    dependencies: ClassVar[list[str]] = ["entities", "auth", "jobs", "secrets", "files"]

    def get_routers(self) -> list[RouterSpec]:
        from nemo_safe_synthesizer_plugin.jobs.generate import RESULT_ROUTES, GenerateJob

        return [
            RouterSpec(
                add_job_routes(
                    GenerateJob,
                    service_name=_SERVICE_NAME,
                    job_result_routes=RESULT_ROUTES,
                    authz=_AUTHZ,
                ),
                prefix="/v2/workspaces/{workspace}",
                tag="Safe Synthesizer",
                description="Job endpoints",
            )
        ]

    def get_exception_handlers(self) -> dict[type[Exception], ExceptionHandler]:
        async def validation_error_handler(_request: Request, ex: Exception):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        return {ValidationError: validation_error_handler}
