# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read routes for persisted eval results.

Two collections under ``/apis/evaluator/v2/workspaces/{workspace}``:

* ``/agent-eval-results`` — :class:`AgentEvalResultEntity` (from ``AgentEvalJob``)
* ``/eval-results``       — :class:`EvaluateResultEntity` (from ``EvaluateJob``)

They're distinct entity types, so each collection has its own concretely-typed routes (the generated
SDK then sees the real result type, not an abstract base). Both support filtering by traits — target,
dataset, timestamps — mirroring the legacy ``job_result_routes``.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_evaluator.api.dependencies import get_result_service
from nemo_evaluator.api.schemas import AgentEvalResult, DataFilter, EvaluateResult
from nemo_evaluator.api.service.result_service import ResultService
from nemo_evaluator.authz import scope
from nemo_helix_plugin.api.parsed_filter import ParsedFilter, make_filter_dep
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.schema import DatetimeFilter, Page

logger = logging.getLogger(__name__)


class ResultPerms(PermissionSet, namespace="evaluator"):
    """Permissions for the read-only eval-result collections (results are job-produced, not created via API)."""

    LIST = perm("List stored eval results", suffix="results.list")
    READ = perm("Read a stored eval result", suffix="results.read")
    DELETE = perm("Delete a stored eval result", suffix="results.delete")


class ResultSort(StrEnum):
    """Sort fields for result queries (``-`` prefix sorts descending)."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


_PAGE = Query(default=1, ge=1, description="Page number.")
_PAGE_SIZE = Query(default=100, ge=1, le=1000, description="Page size.")
_SORT = Query(
    default=ResultSort.CREATED_AT_DESC,
    description="Sort field; prefix with '-' for descending.",
)


class ResultFilter(DataFilter):
    """Traits shared by both result collections (used directly for agent-eval results)."""

    workspace: str | None = None
    name: str | None = None
    job_id: str | None = None
    target_kind: str | None = None
    target_name: str | None = None
    created_at: DatetimeFilter | None = None
    updated_at: DatetimeFilter | None = None


class EvaluateResultFilter(ResultFilter):
    """Adds row-eval's referenceable-input trait."""

    dataset_ref: str | None = None


agent_eval_results_router = APIRouter()
evaluate_results_router = APIRouter()


# --- agent-eval results ------------------------------------------------------


@agent_eval_results_router.get(
    "/agent-eval-results",
    summary="List Agent Eval Results By Workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[AgentEvalResult],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=ResultFilter,
        filter_description="Filter by workspace, name, target, and timestamps.",
    ),
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ResultPerms.LIST])
async def list_agent_eval_results(
    workspace: str,
    page: int = _PAGE,
    page_size: int = _PAGE_SIZE,
    sort: ResultSort = _SORT,
    parsed_filter: ParsedFilter = Depends(make_filter_dep(ResultFilter)),
    service: ResultService = Depends(get_result_service),
) -> Page[AgentEvalResult]:
    """List agent-evaluation result records for a workspace."""
    parsed_filter.remove("workspace")
    try:
        return await service.list_agent_eval_results(
            workspace=workspace, page=page, page_size=page_size, sort=sort, filter_operation=parsed_filter.operation
        )
    except Exception:
        logger.exception(f"Failed to list agent-eval results for workspace {sanitize_for_log(workspace)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@agent_eval_results_router.get(
    "/agent-eval-results/{name}",
    summary="Get Agent Eval Result",
    status_code=status.HTTP_200_OK,
    response_model=AgentEvalResult,
    response_model_exclude_none=True,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Result not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ResultPerms.READ])
async def get_agent_eval_result(
    workspace: str,
    name: str,
    service: ResultService = Depends(get_result_service),
) -> AgentEvalResult:
    """Get an agent-evaluation result record by workspace and name."""
    try:
        result = await service.get_agent_eval_result(workspace, name)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result not found: {workspace}/{name}")
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Failed to get agent-eval result {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@agent_eval_results_router.delete(
    "/agent-eval-results/{name}",
    summary="Delete Agent Eval Result",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Result not found"}},
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ResultPerms.DELETE])
async def delete_agent_eval_result(
    workspace: str,
    name: str,
    service: ResultService = Depends(get_result_service),
) -> None:
    """Delete an agent-evaluation result record by workspace and name."""
    try:
        if not await service.delete_agent_eval_result(workspace, name):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result not found: {workspace}/{name}")
        return None
    except HTTPException:
        raise
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Result was modified by another request: {workspace}/{name}. Refresh and try again.",
        ) from exc
    except Exception:
        logger.exception(f"Failed to delete agent-eval result {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# --- (row) eval results ------------------------------------------------------


@evaluate_results_router.get(
    "/eval-results",
    summary="List Eval Results By Workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[EvaluateResult],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=EvaluateResultFilter,
        filter_description="Filter by workspace, name, target, dataset_ref, and timestamps.",
    ),
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ResultPerms.LIST])
async def list_eval_results(
    workspace: str,
    page: int = _PAGE,
    page_size: int = _PAGE_SIZE,
    sort: ResultSort = _SORT,
    parsed_filter: ParsedFilter = Depends(make_filter_dep(EvaluateResultFilter)),
    service: ResultService = Depends(get_result_service),
) -> Page[EvaluateResult]:
    """List (row) evaluation result records for a workspace."""
    parsed_filter.remove("workspace")
    try:
        return await service.list_eval_results(
            workspace=workspace, page=page, page_size=page_size, sort=sort, filter_operation=parsed_filter.operation
        )
    except Exception:
        logger.exception(f"Failed to list eval results for workspace {sanitize_for_log(workspace)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@evaluate_results_router.get(
    "/eval-results/{name}",
    summary="Get Eval Result",
    status_code=status.HTTP_200_OK,
    response_model=EvaluateResult,
    response_model_exclude_none=True,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Result not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ResultPerms.READ])
async def get_eval_result(
    workspace: str,
    name: str,
    service: ResultService = Depends(get_result_service),
) -> EvaluateResult:
    """Get a (row) evaluation result record by workspace and name."""
    try:
        result = await service.get_eval_result(workspace, name)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result not found: {workspace}/{name}")
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Failed to get eval result {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@evaluate_results_router.delete(
    "/eval-results/{name}",
    summary="Delete Eval Result",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Result not found"}},
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ResultPerms.DELETE])
async def delete_eval_result(
    workspace: str,
    name: str,
    service: ResultService = Depends(get_result_service),
) -> None:
    """Delete a (row) evaluation result record by workspace and name."""
    try:
        if not await service.delete_eval_result(workspace, name):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result not found: {workspace}/{name}")
        return None
    except HTTPException:
        raise
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Result was modified by another request: {workspace}/{name}. Refresh and try again.",
        ) from exc
    except Exception:
        logger.exception(f"Failed to delete eval result {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
