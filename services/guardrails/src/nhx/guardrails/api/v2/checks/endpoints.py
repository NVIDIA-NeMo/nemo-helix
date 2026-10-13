# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

from fastapi import APIRouter, Request, status
from nhx.guardrails.api.dependencies import RailsServiceDep
from nhx.guardrails.app.handlers.checks import CheckRequestHandler
from nhx.guardrails.entities.values.check import (
    GuardrailCheckRequest,
    GuardrailCheckResponse,
)

router = APIRouter()


def responses_dict(additional: Optional[Dict] = None):
    """Helper for standard status responses and additional ones."""
    return {
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid Request Body",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Validation Error",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal Server Error",
        },
        **(additional or {}),
    }


@router.post(
    "/v2/workspaces/{workspace}/checks",
    summary="Guardrail check request",
    responses=responses_dict(),
    response_model=GuardrailCheckResponse,
    response_model_exclude_none=True,
)
async def check(
    workspace: str,
    request_body: GuardrailCheckRequest,
    request: Request,
    rails_service: RailsServiceDep,
):
    """Chat completion for the provided conversation."""

    check_request_handler = CheckRequestHandler(
        rails_service=rails_service,
        request=request,
        request_body=request_body,
        response_model=GuardrailCheckResponse,
        workspace=workspace,
    )

    result = await check_request_handler.handle_request()
    return result
