# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create, list, get, and delete endpoints for Evaluations and Experiments.

Entity-store writes use ``EntityClient`` directly. Evaluation reads use an
application service that composes Entities with ClickHouse-backed rollups and
sessions. PUT updates only mutable fields; an Evaluation's identity and the
dataset/agent it ran against are fixed.
"""

from __future__ import annotations

import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, NamedTuple, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.routing import APIRoute
from nhx.common.api.common import Page, PaginationData
from nhx.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.intake.api.v2.experiments.dependencies import (
    DenormalizerDep,
    EntityClientDep,
    EvaluationReadServiceDep,
)
from nhx.intake.api.v2.experiments.schemas import (
    EvaluationFilter,
    EvaluationPatchRequest,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationSessionFilter,
    EvaluationSessionMode,
    EvaluationSessionResponse,
    EvaluatorAggregate,
    ExperimentFilter,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentUpdateRequest,
)

# The API/Studio expose this as an "Evaluation", but it is still stored as the Experiment entity
# (entity rename + data migration deferred — see entities/experiments.py). Alias it to the name this
# layer uses; only the entity's own field names (e.g. parent_experiment_id) reference Experiment directly.
from nhx.intake.entities.experiments import Experiment as Evaluation
from nhx.intake.entities.experiments import ExperimentGroup
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.experiments.read_service import (
    EvaluationNotFoundError,
    EvaluationRead,
    EvaluationReadLimitExceededError,
    EvaluationTelemetryUnavailableError,
    InvalidEvaluationSessionStatusError,
)
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.repository.evaluation_rollup import EvaluationRollup, ScoreRollup
from nhx.intake.repository.evaluation_session import MetricSortTooLargeError
from nhx.intake.spans.api.dependencies import require_workspace_access, validate_list_query_params
from nhx.intake.spans.domain import SpanStatus
from nhx.intake.spans.storage import make_pagination

logger = logging.getLogger(__name__)


router = APIRouter(dependencies=[Depends(require_workspace_access)])

EXPERIMENTS_TAG = "Experiments"
EVALUATIONS_TAG = "Evaluations"

ExperimentSortField = Literal["-created_at", "created_at", "-updated_at", "updated_at", "-name", "name"]

# The evaluations list is sorted in the application layer (compute-on-read) so a single request can
# sort by a ClickHouse rollup metric, not just entity columns. `sort` is therefore a free string,
# validated against these: an entity column, run_count, or a `<metric>.<stat>` rollup path.
_ENTITY_SORT_FIELDS = frozenset({"name", "created_at", "updated_at", "pinned_at"})
# Sessions are sorted in ClickHouse (ORDER BY before LIMIT/OFFSET) so sort composes
# correctly with pagination. These are the allowed field names; each maps to an SQL
# expression in the ClickHouse repository.
_SESSION_SORT_FIELDS = frozenset(
    {
        "test_case_name",
        "test_case_id",
        "started_at",
        "ended_at",
        "latency_ms",
        "cost_total_usd",
        "status",
        "tokens",
    }
)
_METRIC_STATS = frozenset({"sum", "mean", "median", "p90", "p95", "p99", "count"})
# Per-group evaluation fetch bound for the in-memory merge. Groups are expected to hold at most
# hundreds; a query that selects more than this is rejected rather than sorted on a partial set — the
# trigger to denormalize metrics into an entity-store-sortable column instead.
_MAX_GROUP_EVALUATIONS = 1000

EntityT = TypeVar("EntityT", Evaluation, ExperimentGroup)

ExperimentFilterDep = Annotated[ParsedFilter, Depends(make_filter_dep(ExperimentFilter))]
EvaluationFilterDep = Annotated[ParsedFilter, Depends(make_filter_dep(EvaluationFilter))]
EvaluationSessionFilterDep = Annotated[ParsedFilter, Depends(make_filter_dep(EvaluationSessionFilter))]


@router.post(
    "/v2/workspaces/{workspace}/experiments",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=[EXPERIMENTS_TAG],
    responses={409: {"description": "Experiment already exists"}},
)
async def create_experiment(
    workspace: str,
    body: ExperimentRequest,
    entity_client: EntityClientDep,
) -> ExperimentResponse:
    _validate_default_sort(body.default_sort)
    entity = ExperimentGroup(
        workspace=workspace,
        name=body.name,
        description=body.description,
        insight_id=body.insight_id,
        summary=body.summary,
        metadata=body.metadata,
        default_sort=body.default_sort,
        pareto=body.pareto,
        column_layout=body.column_layout,
        is_favorite=body.is_favorite,
        show_evaluations_over_time=body.show_evaluations_over_time,
    )
    try:
        created = await entity_client.create(entity)
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment '{workspace}/{body.name}' already exists.",
        ) from e
    return ExperimentResponse.from_entity(created)


@router.get(
    "/v2/workspaces/{workspace}/experiments",
    response_model=Page[ExperimentResponse],
    tags=[EXPERIMENTS_TAG],
    openapi_extra=generate_openapi_extra_params(
        filter_schema=ExperimentFilter,
        filter_description=(
            "Filter experiments by name, insight_id, is_favorite, show_evaluations_over_time, "
            "baseline_evaluation_name, is_deleted, or a metadata key/value "
            "(filter[metadata.<key>]=<value>). "
            "Pass is_deleted=true to return only soft-deleted experiments; omit to see only live ones."
        ),
    ),
)
async def list_experiments(
    workspace: str,
    request: Request,
    entity_client: EntityClientDep,
    parsed: ExperimentFilterDep,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    sort: ExperimentSortField = Query(default="-created_at", description="Sort field; prefix with '-' for descending."),
) -> Page[ExperimentResponse]:
    validate_list_query_params(request)
    _apply_is_deleted_filter(parsed)
    _apply_default_false_boolean_filter(parsed, "is_favorite")
    _apply_default_false_boolean_filter(parsed, "show_evaluations_over_time")
    result = await entity_client.list(
        ExperimentGroup,
        workspace=workspace,
        filter_operation=parsed.operation,
        sort=sort,
        page=page,
        page_size=page_size,
    )
    responses = [ExperimentResponse.from_entity(e) for e in result.data]
    counts = await _count_live_evaluations_by_group(
        entity_client, workspace=workspace, group_ids=[g.id for g in result.data]
    )
    for response in responses:
        response.evaluation_count = counts.get(response.id, 0)
    return Page(
        data=responses,
        pagination=PaginationData(**result.pagination.model_dump()),
        sort=sort,
        filter=parsed.to_response(),
    )


@router.get(
    "/v2/workspaces/{workspace}/experiments/{name}",
    response_model=ExperimentResponse,
    tags=[EXPERIMENTS_TAG],
    responses={404: {"description": "Experiment not found"}},
)
async def get_experiment(
    workspace: str,
    name: str,
    entity_client: EntityClientDep,
) -> ExperimentResponse:
    entity = await _get_or_404(
        entity_client,
        ExperimentGroup,
        workspace=workspace,
        name=name,
        label="Experiment",
    )
    _reject_if_deleted(entity, workspace=workspace, name=name, label="Experiment")
    response = ExperimentResponse.from_entity(entity)
    response.evaluation_count = await _count_live_evaluations_in_group(
        entity_client, workspace=workspace, group_id=entity.id
    )
    return response


@router.put(
    "/v2/workspaces/{workspace}/experiments/{name}",
    response_model=ExperimentResponse,
    tags=[EXPERIMENTS_TAG],
    responses={
        400: {"description": "The baseline Evaluation does not exist or is not a live member"},
        404: {"description": "Experiment not found"},
        409: {"description": "Attempt to rename the experiment"},
    },
)
async def update_experiment(
    workspace: str,
    name: str,
    body: ExperimentUpdateRequest,
    entity_client: EntityClientDep,
) -> ExperimentResponse:
    existing = await _get_or_404(
        entity_client,
        ExperimentGroup,
        workspace=workspace,
        name=name,
        label="Experiment",
    )
    _reject_if_deleted(existing, workspace=workspace, name=name, label="Experiment")
    if body.name != name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot rename an experiment; the name is its identity.",
        )
    _validate_default_sort(body.default_sort)
    # Guarded so a partial update cannot blank the fields it omits; an explicit null still clears.
    if "description" in body.model_fields_set:
        existing.description = body.description
    if "insight_id" in body.model_fields_set:
        existing.insight_id = body.insight_id
    if "summary" in body.model_fields_set:
        existing.summary = body.summary
    if "metadata" in body.model_fields_set:
        existing.metadata = body.metadata
    if "default_sort" in body.model_fields_set:
        existing.default_sort = body.default_sort
    # Only overwrite the saved axes when the client actually sent them; an omitted `pareto` (older
    # clients) must not silently reset customized axes to the cost/latency default.
    if body.pareto is not None:
        existing.pareto = body.pareto
    if body.column_layout is not None:
        existing.column_layout = body.column_layout
    # Preserve values written by newer clients when an older client sends a full update without
    # fields it does not know about.
    if "is_favorite" in body.model_fields_set:
        existing.is_favorite = body.is_favorite
    if "show_evaluations_over_time" in body.model_fields_set:
        existing.show_evaluations_over_time = body.show_evaluations_over_time
    if "baseline_evaluation_name" in body.model_fields_set:
        await _validate_baseline_evaluation(
            entity_client,
            workspace=workspace,
            group_id=existing.id,
            evaluation_name=body.baseline_evaluation_name,
        )
        existing.baseline_evaluation_name = body.baseline_evaluation_name
    updated = await entity_client.update(existing)
    response = ExperimentResponse.from_entity(updated)
    response.evaluation_count = await _count_live_evaluations_in_group(
        entity_client, workspace=workspace, group_id=updated.id
    )
    return response


@router.delete(
    "/v2/workspaces/{workspace}/experiments/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=[EXPERIMENTS_TAG],
    responses={404: {"description": "Experiment not found"}},
)
async def delete_experiment(
    workspace: str,
    name: str,
    entity_client: EntityClientDep,
) -> None:
    # Soft delete: flip ``is_deleted`` and rename the row so the original name is free for reuse.
    # The unique index on (workspace, entity_type, name) doesn't read into the JSON data column,
    # so renaming on delete is what lets a new group/evaluation claim the same name later.
    group = await _get_or_404(
        entity_client,
        ExperimentGroup,
        workspace=workspace,
        name=name,
        label="Experiment",
    )
    _reject_if_deleted(group, workspace=workspace, name=name, label="Experiment")

    # Reference-counted cascade, sequential — one update per member. Linear in group size, fine for
    # now. A member whose *sole* membership was this group is soft-deleted; a member also in another
    # group just drops this membership and survives there. Both outcomes remove the member from the
    # live-membership filter below (soft-delete renames + flips is_deleted; membership removal drops
    # the group id from experiment_ids and leaves no legacy scalar), so re-fetching page 1 keeps
    # returning the next unprocessed batch until nothing is left. No fixed cap on group size.
    live_children_filter = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            _group_membership_filter(group.id),
            LogicalOperation(
                operator=FilterOperator.NOT,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field="data.is_deleted", value=True),
                ],
            ),
        ],
    )
    while True:
        page = await entity_client.list(
            Evaluation,
            workspace=workspace,
            filter_operation=live_children_filter,
            page=1,
            page_size=100,
        )
        if not page.data:
            break
        for child in page.data:
            remaining = [gid for gid in child.experiment_ids if gid != group.id]
            if remaining:
                # Shared with another group — drop only this membership; the member lives on there.
                child.experiment_ids = remaining
                await entity_client.update(child)
            else:
                # This group was the member's sole membership — cascade the delete.
                await _soft_delete(entity_client, child)
    await _soft_delete(entity_client, group)


@router.post(
    "/v2/workspaces/{workspace}/evaluations",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=[EVALUATIONS_TAG],
    responses={409: {"description": "Evaluation already exists"}},
)
async def create_evaluation(
    workspace: str,
    body: EvaluationRequest,
    entity_client: EntityClientDep,
) -> EvaluationResponse:
    await _validate_groups_exist(entity_client, group_ids=body.experiment_ids)
    await _validate_parent_evaluation_exists(entity_client, parent_evaluation_id=body.parent_evaluation_id)
    entity = Evaluation(
        workspace=workspace,
        name=body.name,
        experiment_ids=body.experiment_ids,
        dataset_name=body.dataset_name,
        dataset_version=body.dataset_version,
        source_link=body.source_link,
        metadata=body.metadata,
        description=body.description,
        parent_experiment_id=body.parent_evaluation_id,
        status=body.status,
        root_cause=body.root_cause,
    )
    try:
        created = await entity_client.create(entity)
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Evaluation '{workspace}/{body.name}' already exists.",
        ) from e
    return EvaluationResponse.from_entity(created)


@router.get(
    "/v2/workspaces/{workspace}/evaluations",
    response_model=Page[EvaluationResponse],
    tags=[EVALUATIONS_TAG],
    responses={
        400: {"description": "Unsupported sort or filter field"},
        413: {"description": "Too many evaluations selected to sort in one request"},
        503: {"description": "Telemetry store unavailable for a metric-based sort or filter"},
    },
    openapi_extra=generate_openapi_extra_params(
        filter_schema=EvaluationFilter,
        filter_description=(
            "Filter evaluations by name, experiment_id (experiment group membership; "
            "experiment_group_id is a deprecated alias), "
            "dataset_name, dataset_version, created_by, created_at, or updated_at. "
            "Pass is_deleted=true to return only soft-deleted evaluations; omit to see only live ones. "
            "Pass is_pinned=true (or false) to filter by pinned state; omit to return both. "
            "Filter by a metadata key/value: filter[metadata.<key>]=<value>. "
            "Filter by a rollup metric with numeric range operators ($gte/$lte/$gt/$lt/$eq): "
            "filter[run_count][$gte]=5, filter[cost_usd.mean][$lte]=0.5, "
            "filter[latency_ms.p95][$lte]=1000, filter[tokens.mean][$lte]=5000, "
            "or filter[evaluators.<name>.mean][$gte]=0.8."
        ),
    ),
)
async def list_evaluations(
    workspace: str,
    request: Request,
    read_service: EvaluationReadServiceDep,
    parsed: EvaluationFilterDep,
    denormalizer: DenormalizerDep,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    sort: str | None = Query(
        default=None,
        description=(
            "Comma-separated list of fields to sort by, applied in order (the first field dominates); "
            "prefix any field with '-' for descending — e.g. '-evaluators.reward.mean,cost_usd.mean'. "
            "Each field is an evaluation attribute (name, created_at, updated_at, pinned_at) or an "
            "aggregate metric: run_count, test_case_count, cost_usd.<stat>, latency_ms.<stat>, "
            "tokens.<stat>, or evaluators.<name>.<stat>, "
            "where <stat> is one of mean, median, p90, p95, p99, sum, count. When omitted, defaults to "
            "-created_at with pinned evaluations first."
        ),
    ),
) -> Page[EvaluationResponse]:
    validate_list_query_params(request)
    _apply_is_deleted_filter(parsed)
    _apply_is_pinned_filter(parsed)
    # `sort` is a comma-separated list of fields, each optionally '-'-prefixed for descending, applied
    # in order (the first field dominates) — e.g. `-evaluators.reward.mean,cost_usd.mean`. When omitted,
    # fall back to -created_at with pinned evaluations floated to the top.
    if sort is not None:
        sort_keys, explicit_metric_sort = _parse_sort_keys(sort)
        pinned_first = False
    else:
        sort_keys = [("created_at", True)]
        pinned_first = True
        explicit_metric_sort = False
    # Rollup-metric predicates live in ClickHouse, not the entity store, so they can't be pushed to
    # Postgres. Split them out of the filter tree: only the entity predicates go to entity_client.list;
    # the metric ones are applied in memory after hydration. parsed (the full user filter) is left
    # intact so the response still echoes it.
    entity_operation, metric_predicates = _extract_metric_predicates(parsed.operation)
    # Translate the exposed `experiment_group_id` filter into a membership match over `experiment_ids`
    # (plus the legacy scalar), so listing a group returns every evaluation that belongs to it.
    entity_operation = _rewrite_group_filter(entity_operation)
    # Turn the scalar agent_name/agent_version/model_name params into $contains matches on the
    # denormalized list facets, so a workspace-wide list can filter by name against the entity store.
    entity_operation = _rewrite_facet_filters(entity_operation)
    # Compute-on-read: fetch the whole (entity-filtered) group, hydrate every rollup, then filter, sort,
    # and paginate in memory so a single request can sort/filter by a ClickHouse metric that lives
    # outside the entity store. Bounded to hundreds of evaluations per group (see _MAX_GROUP_EVALUATIONS).
    try:
        result = await read_service.list_evaluations(
            workspace=workspace,
            filter_operation=entity_operation,
            limit=_MAX_GROUP_EVALUATIONS,
        )
    except EvaluationReadLimitExceededError as exc:
        # The whole filtered set is sorted in memory; anything past the fetch cap can't be sorted, so a
        # returned page would be silently incomplete. Fail loudly and tell the caller how to scope the
        # query instead (or denormalize rollup metrics for entity-store sorting once groups grow this big).
        logger.warning(
            "Evaluation list selected %d evaluations, over the %d-row in-memory sort cap; refusing "
            "to return a partially sorted result.",
            exc.selected,
            exc.limit,
        )
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"This query selects {exc.selected} evaluations, exceeding the maximum of "
                f"{exc.limit} that can be sorted in one request. Narrow the result with a "
                "filter (e.g. experiment_group_id)."
            ),
        ) from exc
    _enqueue_stale_denormalization(denormalizer, workspace=workspace, reads=result.evaluations)
    responses = [_to_evaluation_response(evaluation) for evaluation in result.evaluations]
    # A metric-backed sort or filter is meaningless without rollups: if hydration was skipped (ClickHouse
    # disabled or down) every metric value would be unset, so a metric sort would silently collapse to
    # name order and a metric filter would drop everything. Reject the request instead of returning a
    # misleading 200. Entity-column sorts/filters still work and an empty group still hydrates fine.
    # An explicit metric sort or metric filter genuinely can't be served without rollups → 503. A
    # default sort degrades gracefully instead: its metric values come back unset, so the appended
    # -created_at key orders the list (the documented fallback), no error.
    if not result.rollups_available and (explicit_metric_sort or metric_predicates):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot sort or filter evaluations by a rollup metric: the telemetry store is unavailable.",
        )
    if metric_predicates:
        responses = [r for r in responses if _matches_metric_predicates(r, metric_predicates)]
    ordered = _sort_evaluations(responses, keys=sort_keys, pinned_first=pinned_first)
    start = (page - 1) * page_size
    page_items = ordered[start : start + page_size]
    return Page(
        data=page_items,
        pagination=make_pagination(
            page=page, page_size=page_size, current_page_size=len(page_items), total_results=len(ordered)
        ),
        sort=sort,
        filter=parsed.to_response(),
    )


@router.get(
    "/v2/workspaces/{workspace}/evaluations/{name}",
    response_model=EvaluationResponse,
    tags=[EVALUATIONS_TAG],
    responses={404: {"description": "Evaluation not found"}},
)
async def get_evaluation(
    workspace: str,
    name: str,
    read_service: EvaluationReadServiceDep,
    denormalizer: DenormalizerDep,
) -> EvaluationResponse:
    try:
        evaluation = await read_service.get_evaluation(workspace=workspace, name=name)
    except EvaluationNotFoundError as exc:
        raise _evaluation_not_found_http_error(exc) from exc
    _enqueue_stale_denormalization(denormalizer, workspace=workspace, reads=[evaluation])
    return _to_evaluation_response(evaluation)


# Identity and the dataset it was run against are fixed for the life of an
# Evaluation (see the ingest invariants); changing them means it's a different
# Evaluation. PUT may only edit group membership, source link, description, metadata.
_IMMUTABLE_EVALUATION_FIELDS = ("name", "dataset_name", "dataset_version")


@router.put(
    "/v2/workspaces/{workspace}/evaluations/{name}",
    response_model=EvaluationResponse,
    tags=[EVALUATIONS_TAG],
    responses={
        404: {"description": "Evaluation not found"},
        409: {"description": "Attempt to change an immutable field, or remove a baseline membership"},
    },
)
async def update_evaluation(
    workspace: str,
    name: str,
    body: EvaluationRequest,
    entity_client: EntityClientDep,
    read_service: EvaluationReadServiceDep,
) -> EvaluationResponse:
    existing = await _get_or_404(
        entity_client,
        Evaluation,
        workspace=workspace,
        name=name,
        label="Evaluation",
    )
    _reject_if_deleted(existing, workspace=workspace, name=name, label="Evaluation")
    new_group_ids = [gid for gid in body.experiment_ids if gid not in existing.experiment_ids]
    if new_group_ids:
        await _validate_groups_exist(entity_client, group_ids=new_group_ids)
    await _validate_parent_evaluation_exists(entity_client, parent_evaluation_id=body.parent_evaluation_id)

    changed = [f for f in _IMMUTABLE_EVALUATION_FIELDS if getattr(body, f) != getattr(existing, f)]
    if changed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot change immutable field(s) {changed} on an existing evaluation; "
                "create a new evaluation instead."
            ),
        )

    await _validate_baseline_memberships_preserved(
        entity_client,
        evaluation=existing,
        desired_group_ids=body.experiment_ids,
    )

    existing.experiment_ids = body.experiment_ids
    existing.source_link = body.source_link
    existing.metadata = body.metadata
    existing.description = body.description
    existing.parent_experiment_id = body.parent_evaluation_id
    existing.status = body.status
    existing.root_cause = body.root_cause
    updated = await entity_client.update(existing)
    return await _evaluation_response_with_rollup(read_service, workspace=workspace, evaluation=updated)


@router.patch(
    "/v2/workspaces/{workspace}/evaluations/{name}",
    response_model=EvaluationResponse,
    tags=[EVALUATIONS_TAG],
    responses={
        400: {"description": "A referenced ExperimentGroup does not exist, or experiment_ids is empty"},
        404: {"description": "Evaluation not found"},
        409: {"description": "Attempt to remove a baseline Evaluation from its Experiment"},
    },
)
async def patch_evaluation(
    workspace: str,
    name: str,
    body: EvaluationPatchRequest,
    entity_client: EntityClientDep,
    read_service: EvaluationReadServiceDep,
) -> EvaluationResponse:
    """Partially update an evaluation: only fields present in the request are changed.

    The common case is curating an evaluation into another ExperimentGroup — PATCH with the merged
    ``experiment_ids``. Membership is replaced (not appended), so send the full desired set; any new
    group must exist and the set must be non-empty (an evaluation always belongs to >=1 group). Omitted
    fields are left untouched (unlike the full-body PUT, which overwrites them).
    """
    existing = await _get_or_404(entity_client, Evaluation, workspace=workspace, name=name, label="Evaluation")
    _reject_if_deleted(existing, workspace=workspace, name=name, label="Evaluation")

    fields_set = body.model_fields_set
    desired_group_ids = existing.experiment_ids
    if "experiment_ids" in fields_set:
        if not body.experiment_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="experiment_ids must be non-empty: an evaluation must belong to at least one group.",
            )
        new_group_ids = [gid for gid in body.experiment_ids if gid not in existing.experiment_ids]
        if new_group_ids:
            await _validate_groups_exist(entity_client, group_ids=new_group_ids)
        # Membership is a set — store deduped so a duplicated id can't inflate a group's count.
        desired_group_ids = list(dict.fromkeys(body.experiment_ids))
    await _validate_baseline_memberships_preserved(
        entity_client,
        evaluation=existing,
        desired_group_ids=desired_group_ids,
    )
    existing.experiment_ids = desired_group_ids
    if "parent_evaluation_id" in fields_set:
        await _validate_parent_evaluation_exists(entity_client, parent_evaluation_id=body.parent_evaluation_id)
        existing.parent_experiment_id = body.parent_evaluation_id
    if "source_link" in fields_set:
        existing.source_link = body.source_link
    if "metadata" in fields_set:
        # Entity metadata is a non-null dict; a client clearing it (explicit null) resets to empty.
        existing.metadata = body.metadata or {}
    if "description" in fields_set:
        existing.description = body.description
    if "status" in fields_set:
        existing.status = body.status
    if "root_cause" in fields_set:
        existing.root_cause = body.root_cause

    updated = await entity_client.update(existing)
    return await _evaluation_response_with_rollup(read_service, workspace=workspace, evaluation=updated)


@router.delete(
    "/v2/workspaces/{workspace}/evaluations/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=[EVALUATIONS_TAG],
    responses={
        404: {"description": "Evaluation not found"},
        409: {"description": "Evaluation is selected as an Experiment's baseline"},
    },
)
async def delete_evaluation(
    workspace: str,
    name: str,
    entity_client: EntityClientDep,
) -> None:
    entity = await _get_or_404(
        entity_client,
        Evaluation,
        workspace=workspace,
        name=name,
        label="Evaluation",
    )
    _reject_if_deleted(entity, workspace=workspace, name=name, label="Evaluation")
    await _validate_baseline_memberships_preserved(
        entity_client,
        evaluation=entity,
        desired_group_ids=[],
    )
    await _soft_delete(entity_client, entity)


@router.post(
    "/v2/workspaces/{workspace}/evaluations/{name}/pin",
    response_model=EvaluationResponse,
    tags=[EVALUATIONS_TAG],
    responses={404: {"description": "Evaluation not found"}},
)
async def pin_evaluation(
    workspace: str,
    name: str,
    entity_client: EntityClientDep,
    read_service: EvaluationReadServiceDep,
) -> EvaluationResponse:
    """Pin an evaluation to the top of the list (workspace-shared).

    Re-pinning an already-pinned evaluation refreshes ``pinned_at`` to the current timestamp,
    which is intentional (most-recently-pinned sorts first).
    """
    entity = await _get_or_404(
        entity_client,
        Evaluation,
        workspace=workspace,
        name=name,
        label="Evaluation",
    )
    _reject_if_deleted(entity, workspace=workspace, name=name, label="Evaluation")
    entity.pinned_at = datetime.now(timezone.utc)
    updated = await entity_client.update(entity)
    return await _evaluation_response_with_rollup(read_service, workspace=workspace, evaluation=updated)


@router.delete(
    "/v2/workspaces/{workspace}/evaluations/{name}/pin",
    response_model=EvaluationResponse,
    tags=[EVALUATIONS_TAG],
    responses={404: {"description": "Evaluation not found"}},
)
async def unpin_evaluation(
    workspace: str,
    name: str,
    entity_client: EntityClientDep,
    read_service: EvaluationReadServiceDep,
) -> EvaluationResponse:
    """Unpin an evaluation. Idempotent: unpinning an already-unpinned evaluation is a no-op."""
    entity = await _get_or_404(
        entity_client,
        Evaluation,
        workspace=workspace,
        name=name,
        label="Evaluation",
    )
    _reject_if_deleted(entity, workspace=workspace, name=name, label="Evaluation")
    entity.pinned_at = None
    updated = await entity_client.update(entity)
    return await _evaluation_response_with_rollup(read_service, workspace=workspace, evaluation=updated)


@router.get(
    "/v2/workspaces/{workspace}/evaluations/{name}/sessions",
    response_model=Page[EvaluationSessionResponse],
    responses={
        400: {"description": "Invalid filter value, unsupported sort field, or empty sort"},
        404: {"description": "Evaluation not found"},
        413: {"description": "Too many sessions to sort by cost or tokens"},
        503: {"description": "ClickHouse unavailable"},
    },
    openapi_extra=generate_openapi_extra_params(
        filter_schema=EvaluationSessionFilter,
        filter_description="Filter sessions by test_case_name and status.",
    ),
)
async def list_evaluation_sessions(
    workspace: str,
    name: str,
    request: Request,
    read_service: EvaluationReadServiceDep,
    parsed: EvaluationSessionFilterDep,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    mode: EvaluationSessionMode = Query(
        default="detailed",
        description=(
            "Response mode. summary omits root-span input and output; preview includes both truncated to "
            "300 characters; detailed returns full root-span payloads."
        ),
    ),
    sort: str | None = Query(
        default=None,
        description=(
            "Comma-separated list of fields to sort by, applied in order (the first field dominates); "
            "prefix a field with '-' for descending — e.g. '-cost_total_usd,latency_ms'. Fields: "
            "test_case_name, started_at, ended_at, latency_ms, status, cost_total_usd, tokens. When omitted, "
            "sessions are ordered by started_at ascending."
        ),
    ),
) -> Page[EvaluationSessionResponse]:
    validate_list_query_params(request, additional_params={"mode"})
    sort_keys = _parse_session_sort_keys(sort) if sort is not None else None
    test_case_name = _session_test_case_name(parsed)
    status_raw: str | None = parsed.extract("status")
    try:
        result = await read_service.list_sessions(
            workspace=workspace,
            evaluation_name=name,
            status=status_raw,
            test_case_name=test_case_name,
            page=page,
            page_size=page_size,
            mode=mode,
            sort_keys=sort_keys,
        )
    except EvaluationNotFoundError as exc:
        raise _evaluation_not_found_http_error(exc) from exc
    except InvalidEvaluationSessionStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{exc.value}'. Valid values: {[s.value for s in SpanStatus]}",
        ) from exc
    except MetricSortTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"This query selects {exc.total} sessions, exceeding the maximum of "
                f"{exc.limit} that can be sorted by cost or tokens in one request. "
                "Narrow the result with a filter (e.g. filter[status]=success) or sort by a "
                "different field (started_at, latency_ms, status, test_case_name)."
            ),
        ) from exc
    except EvaluationTelemetryUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=CLICKHOUSE_UNAVAILABLE_MESSAGE,
        ) from exc
    data = [EvaluationSessionResponse.from_row(row, mode=mode) for row in result.rows]
    return Page(
        data=data,
        pagination=make_pagination(
            page=page,
            page_size=page_size,
            current_page_size=len(data),
            total_results=result.total,
        ),
        filter=parsed.to_response(),
    )


async def _get_or_404(
    entity_client: EntityClient,
    entity_type: type[EntityT],
    *,
    workspace: str,
    name: str,
    label: str,
) -> EntityT:
    try:
        return await entity_client.get(entity_type, name=name, workspace=workspace)
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} '{workspace}/{name}' not found.",
        ) from e


def _reject_if_deleted(
    entity: Evaluation | ExperimentGroup,
    *,
    workspace: str,
    name: str,
    label: str,
) -> None:
    """Treat soft-deleted entities as 404 for callers that didn't explicitly opt in."""
    if entity.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} '{workspace}/{name}' not found.",
        )


_DELETED_MARKER = "-deleted-"
_DELETED_RAND_BYTES = 3  # 6 lowercase hex chars; combined with millis is enough to avoid collisions.
_NAME_MAX_LEN = 63  # matches entity-store NAME_PATTERN length cap.


def _deleted_name(original: str) -> str:
    """Mangle a soft-deleted entity's name so the original is free for reuse.

    The DB unique index on (workspace, entity_type, name) doesn't see into the JSON
    ``data`` column, so the row needs a different ``name`` after soft-delete. The
    suffix is lowercase-only to satisfy NAME_PATTERN (``[a-z0-9\\-@.+_]``).
    """
    # Base36 of unix milliseconds is ~8 chars today and sortable; add 6 hex chars of
    # randomness so concurrent deletes can't collide within the same millisecond.
    ts = _to_base36(int(time.time() * 1000))
    rand = secrets.token_hex(_DELETED_RAND_BYTES)
    suffix = f"{_DELETED_MARKER}{ts}{rand}"
    head_budget = _NAME_MAX_LEN - len(suffix)
    head = original[:head_budget].rstrip("-") if len(original) > head_budget else original
    return f"{head}{suffix}"


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    while value:
        value, rem = divmod(value, 36)
        out.append(digits[rem])
    return "".join(reversed(out))


async def _soft_delete(entity_client: EntityClient, entity: Evaluation | ExperimentGroup) -> None:
    """Flip ``is_deleted`` and rename the entity in a single update."""
    original_name = entity.name
    entity.is_deleted = True
    entity.name = _deleted_name(original_name)
    await entity_client.update(entity, original_name=original_name)


async def _count_live_evaluations_in_group(entity_client: EntityClient, *, workspace: str, group_id: str) -> int:
    """Return the number of non-soft-deleted evaluations in a single group.

    Fetches via ``list(page_size=1)`` so the response carries only ``pagination.total_results``.
    Used by single-group endpoints (GET, PUT). List endpoints should use the bulk variant.
    """
    result = await entity_client.list(
        Evaluation,
        workspace=workspace,
        filter_operation=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                _group_membership_filter(group_id),
                LogicalOperation(
                    operator=FilterOperator.NOT,
                    operations=[
                        ComparisonOperation(operator=FilterOperator.EQ, field="data.is_deleted", value=True),
                    ],
                ),
            ],
        ),
        page=1,
        page_size=1,
    )
    return result.pagination.total_results


async def _count_live_evaluations_by_group(
    entity_client: EntityClient, *, workspace: str, group_ids: list[str]
) -> dict[str, int]:
    """Bulk-count non-soft-deleted evaluations for many groups in one (paginated) query.

    Issues a single ``IN``-filter list against the entity store and tallies per group_id
    client-side. Replaces N parallel ``_count_live_evaluations_in_group`` calls on the
    group-list endpoint so the request shape is 1-to-1 with the entity store rather than
    1-to-N (which is fragile under web-server concurrency).

    Returns a ``{group_id: count}`` map covering every requested group_id, with ``0`` for
    groups that have no live evaluations.
    """
    counts: dict[str, int] = {group_id: 0 for group_id in group_ids}
    if not group_ids:
        return counts
    page = 1
    # Aligned with ``EntityClient.list``'s max — paginates when a workspace's total live
    # evaluation count across the requested groups exceeds this.
    page_size = 1000
    filter_operation = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            LogicalOperation(
                operator=FilterOperator.OR,
                operations=[_group_membership_filter(gid) for gid in dict.fromkeys(group_ids)],
            ),
            LogicalOperation(
                operator=FilterOperator.NOT,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field="data.is_deleted", value=True),
                ],
            ),
        ],
    )
    while True:
        result = await entity_client.list(
            Evaluation,
            workspace=workspace,
            filter_operation=filter_operation,
            page=page,
            page_size=page_size,
        )
        # An evaluation may belong to several of the requested groups; count it once per membership.
        for evaluation in result.data:
            for gid in evaluation.experiment_ids:
                if gid in counts:
                    counts[gid] += 1
        if page >= result.pagination.total_pages:
            break
        page += 1
    return counts


async def _validate_parent_evaluation_exists(entity_client: EntityClient, *, parent_evaluation_id: str | None) -> None:
    """Reject with 400 if ``parent_evaluation_id`` is set but doesn't reference an existing evaluation."""
    if parent_evaluation_id is None:
        return
    try:
        await entity_client.get_by_id(Evaluation, entity_id=parent_evaluation_id)
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"parent_evaluation_id '{parent_evaluation_id}' does not reference an existing evaluation.",
        ) from e


async def _validate_baseline_evaluation(
    entity_client: EntityClient,
    *,
    workspace: str,
    group_id: str,
    evaluation_name: str | None,
) -> None:
    """Require a selected baseline to be a live Evaluation belonging to the Experiment."""
    if evaluation_name is None:
        return
    try:
        evaluation = await entity_client.get(Evaluation, name=evaluation_name, workspace=workspace)
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"baseline_evaluation_name '{evaluation_name}' does not reference an existing Evaluation.",
        ) from e
    if evaluation.is_deleted or group_id not in evaluation.experiment_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Evaluation '{evaluation_name}' must be a live member of the Experiment before it can be "
                "selected as its baseline."
            ),
        )


async def _validate_baseline_memberships_preserved(
    entity_client: EntityClient,
    *,
    evaluation: Evaluation,
    desired_group_ids: list[str],
) -> None:
    """Reject removing an Evaluation from any live Experiment that selects it as baseline."""
    removed_group_ids = set(evaluation.experiment_ids).difference(desired_group_ids)
    for group_id in removed_group_ids:
        try:
            group = await entity_client.get_by_id(ExperimentGroup, entity_id=group_id)
        except EntityNotFoundError:
            continue
        if not group.is_deleted and group.baseline_evaluation_name == evaluation.name:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Evaluation '{evaluation.name}' is the baseline for Experiment '{group.name}'. "
                    "Clear that Experiment's baseline_evaluation_name before removing the membership or "
                    "deleting the Evaluation."
                ),
            )


async def _validate_group_exists(entity_client: EntityClient, *, group_id: str) -> None:
    """Reject the request with 400 if the referenced ExperimentGroup doesn't exist or is deleted."""
    try:
        group = await entity_client.get_by_id(ExperimentGroup, entity_id=group_id)
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"ExperimentGroup '{group_id}' must be created before an Evaluation can reference it."),
        ) from e
    if group.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"ExperimentGroup '{group_id}' has been deleted and can no longer accept new Evaluations.",
        )


async def _validate_groups_exist(entity_client: EntityClient, *, group_ids: list[str]) -> None:
    """Reject with 400 if any referenced ExperimentGroup doesn't exist or is deleted (deduped)."""
    for group_id in dict.fromkeys(group_ids):
        await _validate_group_exists(entity_client, group_id=group_id)


def _group_membership_filter(group_id: str) -> LogicalOperation:
    """Match evaluations that belong to ``group_id`` across both membership representations.

    New rows store membership as the ``experiment_ids`` list (matched with ``$contains``); legacy
    rows that predate many-to-many still store a scalar ``experiment_group_id`` and are matched by
    equality until they are next rewritten. The OR keeps un-migrated rows queryable, no migration.
    """
    return LogicalOperation(
        operator=FilterOperator.OR,
        operations=[
            ComparisonOperation(operator=FilterOperator.CONTAINS, field="data.experiment_ids", value=group_id),
            ComparisonOperation(operator=FilterOperator.EQ, field="data.experiment_group_id", value=group_id),
        ],
    )


def _rewrite_group_filter(operation: FilterOperation | None) -> FilterOperation | None:
    """Rewrite a group-membership equality in a parsed filter into a membership match.

    Both the canonical ``experiment_id`` filter param and its deprecated ``experiment_group_id`` alias
    mean "belongs to this group"; with many-to-many membership that spans the ``experiment_ids`` list
    and the legacy scalar, so both route through the same ``_group_membership_filter``.
    """
    if operation is None:
        return None
    if isinstance(operation, ComparisonOperation):
        if (
            operation.field in ("data.experiment_id", "data.experiment_group_id")
            and operation.operator == FilterOperator.EQ
        ):
            return _group_membership_filter(operation.value)
        return operation
    if isinstance(operation, LogicalOperation):
        return LogicalOperation(
            operator=operation.operator,
            operations=[
                rewritten for op in operation.operations if (rewritten := _rewrite_group_filter(op)) is not None
            ],
        )
    return operation


# Denormalized list facets on the Evaluation entity whose scalar filter param means "list contains value".
_FACET_CONTAINS_FIELDS = frozenset({"data.agent_names", "data.agent_versions", "data.model_names"})


def _rewrite_facet_filters(operation: FilterOperation | None) -> FilterOperation | None:
    """Rewrite an equality on a denormalized name-facet field into a ``$contains`` membership match.

    The user-facing params (``agent_name``/``agent_version``/``model_name``) are scalars that parse to an
    equality, but each is stored as a list of distinct observed names (``agent_names``/``agent_versions``/
    ``model_names``). "Matches this name" therefore means "the list contains it", mirroring how
    ``experiment_id`` matches membership in ``experiment_ids``.
    """
    if operation is None:
        return None
    if isinstance(operation, ComparisonOperation):
        if operation.field in _FACET_CONTAINS_FIELDS and operation.operator == FilterOperator.EQ:
            return ComparisonOperation(operator=FilterOperator.CONTAINS, field=operation.field, value=operation.value)
        return operation
    if isinstance(operation, LogicalOperation):
        return LogicalOperation(
            operator=operation.operator,
            operations=[
                rewritten for op in operation.operations if (rewritten := _rewrite_facet_filters(op)) is not None
            ],
        )
    return operation


def _apply_is_deleted_filter(parsed: ParsedFilter) -> None:
    """Append an ``is_deleted`` clause so list endpoints hide soft-deleted rows by default.

    If the caller passes ``filter[is_deleted]=true``, only soft-deleted rows are returned.
    Anything else (no filter, or ``filter[is_deleted]=false``) returns only live rows.
    """
    # Bracket-style filters arrive as strings (``filter[is_deleted]=true``); JSON-style filters
    # arrive as booleans. Normalize both before deciding which branch to take.
    raw_value = parsed.remove("is_deleted")
    if isinstance(raw_value, bool):
        want_deleted = raw_value
    elif isinstance(raw_value, str):
        want_deleted = raw_value.strip().lower() in ("true", "1", "yes")
    else:
        want_deleted = False
    if want_deleted:
        parsed.and_with(
            ComparisonOperation(operator=FilterOperator.EQ, field="data.is_deleted", value=True),
        )
        return
    parsed.and_with(
        LogicalOperation(
            operator=FilterOperator.NOT,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="data.is_deleted", value=True),
            ],
        ),
    )


def _apply_is_pinned_filter(parsed: ParsedFilter) -> None:
    """Translate the user-facing ``is_pinned`` boolean into a clause on ``data.pinned_at``.

    The entity stores ``pinned_at: datetime | None``; "pinned" maps to "not null".

    - ``filter[is_pinned]=true``  → ``data.pinned_at IS NOT NULL`` (only pinned rows).
    - ``filter[is_pinned]=false`` → ``data.pinned_at IS NULL`` (only unpinned rows).
    - Omitted                     → no filter clause; both pinned and unpinned are returned.
    """
    raw_value = parsed.remove("is_pinned")
    if isinstance(raw_value, bool):
        want_pinned: bool | None = raw_value
    elif isinstance(raw_value, str):
        want_pinned = raw_value.strip().lower() in ("true", "1", "yes")
    else:
        want_pinned = None
    if want_pinned is None:
        return
    null_clause = ComparisonOperation(operator=FilterOperator.EQ, field="data.pinned_at", value=None)
    if want_pinned:
        parsed.and_with(LogicalOperation(operator=FilterOperator.NOT, operations=[null_clause]))
    else:
        parsed.and_with(null_clause)


def _apply_default_false_boolean_filter(parsed: ParsedFilter, field: str) -> None:
    """Filter a schema-on-read boolean while treating legacy rows with no stored value as false."""
    raw_value = parsed.remove(field)
    if isinstance(raw_value, bool):
        wanted: bool | None = raw_value
    elif isinstance(raw_value, str):
        wanted = raw_value.strip().lower() in ("true", "1", "yes")
    else:
        wanted = None
    if wanted is None:
        return

    entity_field = f"data.{field}"
    true_clause = ComparisonOperation(operator=FilterOperator.EQ, field=entity_field, value=True)
    if wanted:
        parsed.and_with(true_clause)
    else:
        parsed.and_with(
            LogicalOperation(
                operator=FilterOperator.OR,
                operations=[
                    ComparisonOperation(operator=FilterOperator.EQ, field=entity_field, value=False),
                    ComparisonOperation(operator=FilterOperator.EQ, field=entity_field, value=None),
                ],
            ),
        )


# Metric heads whose dotted sub-paths address a ClickHouse rollup (not an entity column). Declared as
# self-mapping namespaces on EvaluationFilter so paths survive filter validation untranslated.
_METRIC_NAMESPACES = frozenset({"cost_usd", "latency_ms", "tokens", "evaluators"})
_NUMERIC_FILTER_OPERATORS = frozenset(
    {FilterOperator.GTE, FilterOperator.LTE, FilterOperator.GT, FilterOperator.LT, FilterOperator.EQ}
)


class _MetricPredicate(NamedTuple):
    field: str
    operator: FilterOperator
    threshold: float


def _is_valid_metric_path(field: str) -> bool:
    """True if `field` is a rollup-metric path: run_count, test_case_count, <metric>.<stat>, or
    evaluators.<name>.<stat>."""
    if field in ("run_count", "test_case_count"):
        return True
    head, _, rest = field.partition(".")
    if head in ("cost_usd", "latency_ms", "tokens"):
        return rest in _METRIC_STATS
    if head == "evaluators":
        # Evaluator names can contain dots (e.g. "harbor.verifier"); the stat is the last segment.
        name, _, stat = rest.rpartition(".")
        return bool(name) and stat in _METRIC_STATS
    return False


def _validate_sort_field(field: str) -> None:
    """Reject a sort field that isn't an entity column or a known rollup-metric path."""
    if field in _ENTITY_SORT_FIELDS or _is_valid_metric_path(field):
        return
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported sort field: {field}")


def _parse_sort_keys(sort: str) -> tuple[list[tuple[str, bool]], bool]:
    """Parse the comma-separated ``sort`` param into an ordered list of ``(field, descending)`` keys.

    Each field may be '-'-prefixed for descending; the keys are applied in order (the first field
    dominates). Returns the keys plus whether *any* key is a rollup metric — a metric-backed sort
    requires hydrated rollups, so this drives the 503-if-unavailable guard. Raises 400 if a field is
    unsupported or the list is empty.
    """
    sort_keys: list[tuple[str, bool]] = []
    explicit_metric_sort = False
    for token in sort.split(","):
        field_token = token.strip()
        if not field_token:
            continue
        descending = field_token.startswith("-")
        sort_field = field_token[1:] if descending else field_token
        _validate_sort_field(sort_field)
        sort_keys.append((sort_field, descending))
        explicit_metric_sort = explicit_metric_sort or sort_field not in _ENTITY_SORT_FIELDS
    if not sort_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'sort' parameter must contain at least one field.",
        )
    return sort_keys, explicit_metric_sort


def _parse_session_sort_keys(sort: str) -> list[tuple[str, bool]]:
    """Parse the comma-separated ``sort`` param into an ordered list of ``(field, descending)`` keys.

    Each field may be '-'-prefixed for descending; the keys are applied in order (the first field
    dominates). Returns the keys. Raises 400 if a field is unsupported or the list is empty.
    """
    sort_keys: list[tuple[str, bool]] = []
    for token in sort.split(","):
        field_token = token.strip()
        if not field_token:
            continue
        descending = field_token.startswith("-")
        sort_field = field_token[1:] if descending else field_token
        if sort_field not in _SESSION_SORT_FIELDS:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported sort field: {sort_field}")
        if sort_field == "test_case_id":
            sort_field = "test_case_name"
        sort_keys.append((sort_field, descending))
    if not sort_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'sort' parameter must contain at least one field.",
        )
    return sort_keys


def _session_test_case_name(parsed: ParsedFilter) -> str | None:
    test_case_name: str | None = parsed.extract("test_case_name")
    deprecated_test_case_id: str | None = parsed.extract("test_case_id")
    if test_case_name is not None and deprecated_test_case_id is not None and test_case_name != deprecated_test_case_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conflicting evaluation session filters for test_case_name",
        )
    return test_case_name if test_case_name is not None else deprecated_test_case_id


def _is_metric_field(field: str) -> bool:
    """True if `field` is *intended* as a rollup metric (by head), valid path or not.

    Looser than ``_is_valid_metric_path``: classifies e.g. ``cost_usd.bogus`` as a metric so it gets
    extracted and rejected with a 400 rather than forwarded to the entity store. Entity fields (already
    translated to ``data.*`` by the filter dep) never match.
    """
    return field in ("run_count", "test_case_count") or field.split(".", 1)[0] in _METRIC_NAMESPACES


def _operation_references_metric(operation: FilterOperation | None) -> bool:
    if isinstance(operation, ComparisonOperation):
        return _is_metric_field(operation.field)
    if isinstance(operation, LogicalOperation):
        return any(_operation_references_metric(child) for child in operation.operations)
    return False


def _validated_metric_predicate(operation: ComparisonOperation) -> _MetricPredicate:
    field = operation.field
    if not _is_valid_metric_path(field):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported metric filter field: {field}")
    if operation.operator not in _NUMERIC_FILTER_OPERATORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Metric filter '{field}' supports only numeric operators ($gte/$lte/$gt/$lt/$eq).",
        )
    try:
        threshold = float(operation.value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Metric filter '{field}' requires a numeric value, got {operation.value!r}.",
        ) from exc
    return _MetricPredicate(field=field, operator=operation.operator, threshold=threshold)


def _extract_metric_predicates(
    operation: FilterOperation | None,
) -> tuple[FilterOperation | None, list[_MetricPredicate]]:
    """Split rollup-metric comparisons out of the filter tree.

    Returns ``(entity_operation, metric_predicates)``: the entity operation is forwarded to the entity
    store, the metric predicates are applied in memory after hydration. Metric filters must be AND-ed
    (at any nesting depth) with entity filters; a metric field under OR/NOT raises 400, since we can't
    evaluate half a boolean tree in SQL and half in the application layer. Nested ANDs are flattened by
    recursion, so a metric comparison inside a sub-AND is accepted.
    """
    if operation is None:
        return None, []
    if isinstance(operation, ComparisonOperation):
        if _is_metric_field(operation.field):
            return None, [_validated_metric_predicate(operation)]
        return operation, []
    if isinstance(operation, LogicalOperation):
        if operation.operator != FilterOperator.AND:
            if _operation_references_metric(operation):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Metric filters can only be combined with AND, not OR/NOT.",
                )
            return operation, []
        entity_ops: list[FilterOperation] = []
        metric_predicates: list[_MetricPredicate] = []
        for child in operation.operations:
            # Recurse so metric comparisons nested inside sub-ANDs are extracted too (and OR/NOT
            # children that reference a metric still raise inside this call).
            child_entity, child_metrics = _extract_metric_predicates(child)
            if child_entity is not None:
                entity_ops.append(child_entity)
            metric_predicates.extend(child_metrics)
        if not entity_ops:
            return None, metric_predicates
        if len(entity_ops) == 1:
            return entity_ops[0], metric_predicates
        return LogicalOperation(operator=FilterOperator.AND, operations=entity_ops), metric_predicates
    return operation, []


def _matches_metric_predicates(response: EvaluationResponse, predicates: list[_MetricPredicate]) -> bool:
    """True if the response satisfies every metric predicate. A missing metric never matches."""
    for predicate in predicates:
        value = _evaluation_sort_value(response, predicate.field)
        if value is None or not _compare_metric(value, predicate.operator, predicate.threshold):
            return False
    return True


def _compare_metric(value: float, operator: FilterOperator, threshold: float) -> bool:
    if operator == FilterOperator.GTE:
        return value >= threshold
    if operator == FilterOperator.LTE:
        return value <= threshold
    if operator == FilterOperator.GT:
        return value > threshold
    if operator == FilterOperator.LT:
        return value < threshold
    return value == threshold  # EQ


def _evaluation_sort_value(response: EvaluationResponse, field: str) -> Any:
    """Value for `field` on a hydrated response, or None when the metric is absent (sorts last)."""
    if field in _ENTITY_SORT_FIELDS:
        return getattr(response, field)
    if field == "run_count":
        return response.run_count
    if field == "test_case_count":
        return response.test_case_count
    head, _, rest = field.partition(".")
    if head == "cost_usd":
        return getattr(response.cost_usd, rest, None) if response.cost_usd is not None else None
    if head == "latency_ms":
        return getattr(response.latency_ms, rest, None) if response.latency_ms is not None else None
    if head == "tokens":
        return getattr(response.tokens, rest, None) if response.tokens is not None else None
    name, _, stat = rest.rpartition(".")  # head == "evaluators"
    score = (response.aggregate_scores or {}).get(name)
    return getattr(score, stat, None) if score is not None else None


def _validate_default_sort(default_sort: str | None) -> None:
    """Reject a default sort whose field(s) the evaluations list can't sort by.

    The value is a ``sort``-param string — a comma-separated list of fields, each with an optional
    leading '-' for descending (e.g. ``-evaluators.reward.mean,cost_usd.mean``); every field must
    satisfy the same rule as the list ``sort`` query param.
    """
    if default_sort is None:
        return
    fields = [token.strip() for token in default_sort.split(",") if token.strip()]
    if not fields:
        # An empty or all-blank string (``""``, ``","``) is not a usable sort; reject it here rather
        # than persist a value the list ``sort`` param would later 400 on.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'default_sort' parameter must contain at least one field.",
        )
    for field_token in fields:
        field = field_token[1:] if field_token.startswith("-") else field_token
        _validate_sort_field(field)


def _sort_evaluations(
    responses: list[EvaluationResponse],
    *,
    keys: list[tuple[str, bool]],
    pinned_first: bool = False,
) -> list[EvaluationResponse]:
    """Sort by an ordered list of ``(field, descending)`` keys.

    Missing values sort last per key; ties break by name. Keys are applied from lowest to highest
    priority via successive stable sorts, so the first key dominates. With ``pinned_first``, pinned
    evaluations float to the top while preserving key order within the pinned and unpinned groups.
    """
    ordered = sorted(responses, key=lambda r: r.name)  # stable base tiebreak
    for field, descending in reversed(keys):
        present = [r for r in ordered if _evaluation_sort_value(r, field) is not None]
        missing = [r for r in ordered if _evaluation_sort_value(r, field) is None]
        present.sort(key=lambda r, f=field: _evaluation_sort_value(r, f), reverse=descending)
        ordered = present + missing
    if pinned_first:
        # Stable: pinned (False sorts before True) float up, key order preserved within each group.
        ordered = sorted(ordered, key=lambda r: r.pinned_at is None)
    return ordered


def _to_evaluation_response(evaluation: EvaluationRead) -> EvaluationResponse:
    response = EvaluationResponse.from_entity(evaluation.entity)
    if evaluation.rollup is not None:
        _apply_rollup(response, evaluation.rollup)
    return response


def _enqueue_stale_denormalization(
    denormalizer: EvaluationDenormalizer | None,
    *,
    workspace: str,
    reads: list[EvaluationRead],
) -> None:
    """Self-heal the denormalized name facets on read.

    When a read's live rollup names differ from the entity's stored facets, queue the evaluation for
    the refresher. This backfills evaluations that were ingested before the facets existed (and corrects
    any drift) the first time they're read, with no separate migration to run on each instance — the
    live rollup was already fetched to build this response, so the comparison is free and the write is
    deferred to the background worker (which re-checks and skips no-ops).
    """
    if denormalizer is None:
        return
    for read in reads:
        rollup = read.rollup
        if rollup is None:
            continue
        entity = read.entity
        if (
            entity.agent_names != rollup.agent_names
            or entity.agent_versions != rollup.agent_versions
            or entity.model_names != rollup.model_names
        ):
            denormalizer.mark_dirty(workspace=workspace, evaluation_name=entity.name)


async def _evaluation_response_with_rollup(
    read_service: EvaluationReadServiceDep,
    *,
    workspace: str,
    evaluation: Evaluation,
) -> EvaluationResponse:
    batch = await read_service.attach_rollups(workspace=workspace, evaluations=[evaluation])
    return _to_evaluation_response(batch.evaluations[0])


def _evaluation_not_found_http_error(exc: EvaluationNotFoundError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Evaluation '{exc.workspace}/{exc.name}' not found.",
    )


def _apply_rollup(response: EvaluationResponse, rollup: EvaluationRollup) -> None:
    response.evaluator_names = rollup.evaluator_names
    response.model_names = rollup.model_names
    response.agent_names = rollup.agent_names
    response.agent_versions = rollup.agent_versions
    response.aggregate_scores = {name: _aggregate(score) for name, score in rollup.evaluator_scores.items()} or None
    response.run_count = rollup.run_count
    response.test_case_count = rollup.test_case_count
    response.cost_usd = _aggregate(rollup.cost_usd) if rollup.cost_usd is not None else None
    response.latency_ms = _aggregate(rollup.latency_ms) if rollup.latency_ms is not None else None
    response.tokens = _aggregate(rollup.tokens) if rollup.tokens is not None else None


def _aggregate(rollup: ScoreRollup) -> EvaluatorAggregate:
    return EvaluatorAggregate(
        sum=rollup.sum,
        mean=rollup.mean,
        median=rollup.median,
        p90=rollup.p90,
        p95=rollup.p95,
        p99=rollup.p99,
        count=rollup.count,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible URL aliases (deprecated)
#
# The parent resource moved from `/experiment-groups` to `/experiments`. Register the old
# `/experiment-groups...` paths as hidden aliases (``include_in_schema=False``) pointing at the
# same handlers, so existing callers keep working until they migrate to `/experiments`.
# ---------------------------------------------------------------------------
for _legacy_route in list(router.routes):
    if isinstance(_legacy_route, APIRoute) and "/experiments" in _legacy_route.path:
        router.add_api_route(
            _legacy_route.path.replace("/experiments", "/experiment-groups", 1),
            _legacy_route.endpoint,
            methods=sorted(_legacy_route.methods),
            response_model=_legacy_route.response_model,
            status_code=_legacy_route.status_code,
            include_in_schema=False,
            name=f"{_legacy_route.name}_experiment_groups_alias",
        )
