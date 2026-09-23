# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer plugin service implementation."""

from __future__ import annotations

from typing import ClassVar

from data_designer.engine.errors import DataDesignerRuntimeError
from data_designer.errors import DataDesignerError
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from fastapi import Request
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import ValidationError
from starlette import status
from starlette.responses import JSONResponse


class DataDesignerService(NemoService):
    """Data Designer service for NeMo Helix."""

    name: ClassVar[str] = "data-designer"
    dependencies: ClassVar[list[str]] = ["entities", "auth", "jobs", "secrets", "files", "inference-gateway"]

    def get_routers(self) -> list[RouterSpec]:
        from nemo_data_designer_plugin.config import get_config
        from nemo_data_designer_plugin.functions.preview import PreviewFunction
        from nemo_data_designer_plugin.functions.retrieval_preview import RetrievalPreviewFunction
        from nemo_data_designer_plugin.jobs.create import CreateJob
        from nemo_data_designer_plugin.jobs.retrieval_generate import RetrievalGenerateJob
        from nemo_data_designer_plugin.jobs.retrieval_prepare import RetrievalPrepareJob
        from nemo_data_designer_plugin.jobs.retrieval_run import RetrievalRunJob
        from nemo_helix_plugin.authz import AuthzScope
        from nemo_helix_plugin.functions.routes import add_function_routes
        from nemo_helix_plugin.jobs.routes import add_job_routes

        scope = AuthzScope("data-designer")
        prefix = "/v2/workspaces/{workspace}"
        job_profile = get_config().job_executor_profile
        return [
            RouterSpec(
                add_function_routes(
                    PreviewFunction,
                    authz=scope,
                    permission_description="Preview a Data Designer config",
                ),
                prefix=prefix,
                tag="Data Designer",
                description="Streaming preview of a Data Designer config.",
            ),
            RouterSpec(
                add_function_routes(
                    RetrievalPreviewFunction,
                    authz=scope,
                    permission_description="Preview retrieval synthetic data generation",
                ),
                prefix=prefix,
                tag="Data Designer",
                description="Streaming preview of retrieval synthetic data generation.",
            ),
            RouterSpec(
                add_job_routes(CreateJob, authz=scope),
                prefix=prefix,
                tag="Data Designer",
                description="Job endpoints",
            ),
            RouterSpec(
                add_job_routes(RetrievalGenerateJob, default_profile=job_profile, authz=scope),
                prefix=prefix,
                tag="Data Designer",
                description="Retrieval synthetic data generation job endpoints.",
            ),
            RouterSpec(
                add_job_routes(RetrievalPrepareJob, default_profile=job_profile, authz=scope),
                prefix=prefix,
                tag="Data Designer",
                description="Retrieval dataset preparation job endpoints.",
            ),
            RouterSpec(
                add_job_routes(RetrievalRunJob, default_profile=job_profile, authz=scope),
                prefix=prefix,
                tag="Data Designer",
                description="Retrieval generation and preparation job endpoints.",
            ),
        ]

    def get_exception_handlers(self) -> dict:
        async def validation_error_handler(_request: Request, ex: ValidationError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        async def data_designer_error_handler(_request: Request, ex: DataDesignerError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        async def data_designer_runtime_error_handler(_request: Request, ex: DataDesignerRuntimeError):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(ex)},
            )

        async def ndd_internal_error_handler(_request: Request, ex: NDDInternalError):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": str(ex)},
            )

        async def ndd_bad_request_error_handler(_request: Request, ex: NDDInvalidConfigError):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": str(ex)},
            )

        return {
            ValidationError: validation_error_handler,
            DataDesignerError: data_designer_error_handler,
            DataDesignerRuntimeError: data_designer_runtime_error_handler,
            NDDInternalError: ndd_internal_error_handler,
            NDDInvalidConfigError: ndd_bad_request_error_handler,
        }
