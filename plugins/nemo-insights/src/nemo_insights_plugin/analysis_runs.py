# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level Insights analysis-run request helpers.

This module is a thin facade over the generic ``agents.execute`` job. The
current ``AnalyzeJob`` can continue to run side-by-side while this path
exercises the Analyst-as-Agent implementation.

Insights persists one :class:`~nemo_insights_plugin.entities.AnalysisRun` per
run, holding only what the Jobs layer cannot know: that a job was an analysis
run, and over which agent and scope. Execution state is *not* copied — it is
read back from the job on demand.

The run is named before the job is submitted and the job takes that same name,
so the two are linked by construction rather than by a write-back. That is what
makes the failure modes distinguishable: a run whose job 404s never landed, and
a resubmission that conflicts is proof the original did.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Mapping
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_insights_plugin._perms import AnalysisRunPerms
from nemo_insights_plugin.analyst.agent_config import AGENT_CONFIG_FORMAT, build_analyst_agent_config
from nemo_insights_plugin.authz import scope
from nemo_insights_plugin.entities import AnalysisRun
from nemo_insights_plugin.schema import AnalysisRunPage, AnalysisRunResponse, CreateAnalysisRunRequest
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.agents.client import AsyncAgentsClient
from nemo_platform_plugin.agents.types import CreateExecuteJobRequest, JsonObject
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoHTTPError, NemoTransportError, NotFoundError
from nemo_platform_plugin.client.response import NemoResponse
from nemo_platform_plugin.dependencies import get_sdk_client
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError, get_entity_client
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.models.types import ModelEntity
from nemo_platform_plugin.schema import PaginationData
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)

INSIGHTS_ANALYSIS_EXTENSION_KIND = "insights.analysis"
ANALYSIS_RUN_NAME_PREFIX = "insights-run-"

router = APIRouter(tags=["Insights Analysis Runs"])
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)


class ModelLookupClient(Protocol):
    def get_model(self, *, name: str, workspace: str | None = None) -> Awaitable[NemoResponse[ModelEntity]]: ...


class ExecuteJobClient(Protocol):
    def create_execute_job(
        self, *, body: CreateExecuteJobRequest, workspace: str | None = None
    ) -> Awaitable[NemoResponse[JsonObject]]: ...

    def get_execute_job(self, *, name: str, workspace: str | None = None) -> Awaitable[NemoResponse[JsonObject]]: ...


def _json_object(value: object) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def mint_analysis_run_name() -> str:
    """Mint the name shared by an analysis run and its backing job.

    Minted by Insights rather than assigned by the Jobs service so the run can
    be persisted *before* the job is submitted. Uniqueness comes from the uuid,
    not from the agent — an earlier version derived the name from the agent and
    collided on the second run.
    """
    return f"{ANALYSIS_RUN_NAME_PREFIX}{uuid.uuid4().hex}"


@router.post("/analysis-runs", response_model=AnalysisRunResponse, status_code=201)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AnalysisRunPerms.CREATE])
async def create_analysis_run(
    workspace: str,
    request: CreateAnalysisRunRequest,
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AnalysisRunResponse:
    """Create an Insights analysis run backed by the generic ``agents.execute`` job."""
    agents_client = client_from_platform(sdk, AsyncAgentsClient)
    models_client = client_from_platform(sdk, AsyncModelsClient)
    # Resolve before recording anything: a bogus ref would otherwise persist a
    # run and submit a job that cannot start, and the request carries the only
    # copy of the operator's intent.
    request = await _resolve_model_refs(models_client, request, workspace=workspace)
    run = AnalysisRun(
        name=mint_analysis_run_name(),
        workspace=workspace,
        agent=request.agent,
        since=request.since,
        evaluation_id=request.evaluation_id or "",
        default_model=request.default_model,
        fast_model=request.fast_model,
    )
    try:
        saved = await entity_client.create(run)
    except Exception as exc:
        safe_agent = request.agent.replace("\r", "").replace("\n", "")
        logger.exception("Failed to record analysis run for agent '%s'", safe_agent)
        raise HTTPException(status_code=500, detail="Failed to record the analysis run.") from exc

    spec = build_execute_agent_job_config(request, workspace=workspace, run_name=saved.name)
    try:
        job = (
            await agents_client.create_execute_job(
                workspace=workspace,
                body=CreateExecuteJobRequest(spec=spec, name=saved.name),
            )
        ).data()
    except NemoHTTPError as exc:
        # The run record is deliberately left in place. Deleting it here could
        # remove the only pointer to a job that was in fact created (a create
        # that timed out client-side still lands), which is the untracked-job
        # case this design exists to prevent. A run with no job reads as "never
        # submitted"; recovering one is a read today, since every create mints a
        # new name and no resubmit-under-an-existing-name route exists yet.
        logger.warning("Analysis run %r recorded but its job was not created: %s", saved.name, exc)
        raise HTTPException(
            status_code=exc.status_code,
            # ``error`` keeps the Agents service's own detail — a missing entity
            # or invalid config reads better from the service that rejected it —
            # while ``run`` names the record this call stranded.
            detail={"error": _error_detail(exc), "run": saved.name},
        ) from exc
    except NemoTransportError as exc:
        # Same stranded run, but a transport error rather than an HTTP status
        # error, so it needs its own arm — and it is the case most likely to
        # be worth retrying, since the request may never have reached Jobs.
        # Log the run name: without it the orphan cannot be found afterwards.
        logger.warning("Analysis run %r recorded but the Jobs service was unreachable: %s", saved.name, exc)
        raise HTTPException(
            status_code=503,
            # Same shape as the status arm above: ``run`` names the record this
            # call stranded, which is the caller's only chance to learn what was
            # persisted before the response is all they have.
            detail={
                "error": "Could not reach the Jobs service to submit the analysis run.",
                "run": saved.name,
            },
        ) from exc
    return AnalysisRunResponse(run=saved, job=dict(job))


@router.get("/analysis-runs", response_model=AnalysisRunPage)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AnalysisRunPerms.LIST])
async def list_analysis_runs(
    workspace: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    sort: str = Query(default="-created_at", description="Sort field."),
    agent: str | None = Query(default=None, description="Filter by the agent that was analyzed."),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AnalysisRunPage:
    """List analysis runs. Job state is not joined here — read a run to get it."""
    filter_obj: dict[str, object] = {}
    if agent is not None:
        filter_obj["agent"] = agent
    try:
        result = await entity_client.list(
            AnalysisRun,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_obj or None,
        )
    except Exception as exc:
        logger.exception("Failed to list analysis runs")
        raise HTTPException(status_code=500, detail="Failed to list analysis runs.") from exc

    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return AnalysisRunPage(data=result.data, pagination=pagination, sort=sort, filter=filter_obj or None)


@router.get("/analysis-runs/{name}", response_model=AnalysisRunResponse)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AnalysisRunPerms.READ])
async def get_analysis_run(
    workspace: str,
    name: str,
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AnalysisRunResponse:
    """Get one analysis run, joined with the live state of its backing job."""
    agents_client = client_from_platform(sdk, AsyncAgentsClient)
    try:
        run = await entity_client.get(AnalysisRun, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Analysis run '{name}' not found in workspace '{workspace}'."
        ) from exc
    return AnalysisRunResponse(run=run, job=await _backing_job(agents_client, workspace=workspace, name=name))


async def _backing_job(client: ExecuteJobClient, *, workspace: str, name: str) -> JsonObject | None:
    """Read the job sharing this run's name, or None when submission never landed."""
    try:
        return (await client.get_execute_job(name=name, workspace=workspace)).data()
    except NotFoundError:
        return None
    except NemoHTTPError:
        raise


async def _resolve_model_refs(
    client: ModelLookupClient,
    request: CreateAnalysisRunRequest,
    *,
    workspace: str,
) -> CreateAnalysisRunRequest:
    """Normalize the model pair to ``workspace/name`` and confirm both exist.

    A bare name resolves against the run's own workspace. A qualified ref is
    looked up in the workspace it names, so a run may use a model from another
    workspace the caller can read; the entity store's own authorization decides
    that, and a denial surfaces with the status it returned. The normalized form
    is what gets stored and put in the job spec, because the Analyst only accepts
    qualified refs.
    """
    default_model = await _resolve_model_ref(
        client,
        request.default_model,
        field="default_model",
        workspace=workspace,
    )
    fast_model = await _resolve_model_ref(
        client,
        request.fast_model,
        field="fast_model",
        workspace=workspace,
    )
    return request.model_copy(update={"default_model": default_model, "fast_model": fast_model})


async def _resolve_model_ref(client: ModelLookupClient, ref: str, *, field: str, workspace: str) -> str:
    """Return ``<workspace>/<name>`` for an existing Model Entity, or raise 422."""
    match ref.split("/"):
        case [name]:
            model_workspace = workspace
        case [model_workspace, name] if model_workspace and name:
            pass
        case _:
            raise HTTPException(
                status_code=422,
                detail=f"{field} must be a Model Entity name or '<workspace>/<name>' ref, got {ref!r}.",
            )

    try:
        await client.get_model(name=name, workspace=model_workspace)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field} '{model_workspace}/{name}' is not a known Model Entity.",
        ) from exc
    except NemoHTTPError as exc:
        raise HTTPException(status_code=exc.status_code, detail=_error_detail(exc)) from exc
    except NemoTransportError as exc:
        # Same reasoning as the submit path: an unreachable dependency is not
        # the caller's bad request, and nothing has been recorded yet.
        logger.warning("Could not reach the Models service to validate %s %r: %s", field, ref, exc)
        raise HTTPException(
            status_code=503, detail="Could not reach the Models service to validate the model refs."
        ) from exc
    return f"{model_workspace}/{name}"


def build_execute_agent_job_config(request: CreateAnalysisRunRequest, *, workspace: str, run_name: str) -> JsonObject:
    """Translate a high-level Insights request into a generic execute-agent job config."""
    extension_config: JsonObject = {
        "agent": request.agent,
        "workspace": workspace,
    }
    payload = _json_object(
        {
            "agent": _inline_analyst(request, workspace=workspace),
            "input": _analysis_prompt(request.agent),
            "extension": {
                "kind": INSIGHTS_ANALYSIS_EXTENSION_KIND,
                "config": extension_config,
            },
        }
    )
    if request.timeout_seconds is not None:
        payload["timeout_seconds"] = request.timeout_seconds

    return payload


def _inline_analyst(request: CreateAnalysisRunRequest, *, workspace: str) -> JsonObject:
    """Build the ``agent`` arm of the execute job as an inline definition.

    The Analyst has no Agent entity: its config is composed here from the
    request's model pair and read scope. ``ExecuteAgentJobConfig.agent`` also
    accepts an entity ref, so pointing a run at a stored Analyst is a small
    change if a use case appears.
    """
    return _json_object(
        {
            # AgentInline defaults to the legacy NAT format, matching the Agent
            # entity; the Analyst is a Fabric spec agent, so say so explicitly.
            "config_format": AGENT_CONFIG_FORMAT,
            "config": build_analyst_agent_config(
                agent=request.agent,
                workspace=workspace,
                default_model=request.default_model,
                fast_model=request.fast_model,
                ethos=request.ethos,
                since=request.since,
                evaluation_id=request.evaluation_id,
                # The Analyst's self-observability builds its own OTLP exporter from
                # the operator's local CLI config, which does not exist in a task
                # pod, so on a cluster it exports unauthenticated at best. Telemetry
                # for these runs belongs in the agent config's relay endpoints,
                # where the agents layer already carries auth; until that is wired,
                # a run must not depend on it.
                enable_observability=False,
            ),
        }
    )


def _error_detail(error: NemoHTTPError) -> object:
    """Unwrap the Agents service's ``detail`` from a failed job-create call."""
    body = error.body
    if isinstance(body, Mapping):
        detail = body.get("detail")
        if detail is not None:
            return detail
    return body if body is not None else error.detail


def _analysis_prompt(agent: str) -> str:
    return f"Analyze recent telemetry and maintain durable insights for {agent}."
