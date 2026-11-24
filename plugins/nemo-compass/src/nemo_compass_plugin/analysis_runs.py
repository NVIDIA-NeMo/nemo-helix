# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level Compass analysis-run request helpers.

This module is a thin facade over the generic ``agents.execute`` job.

Compass persists one :class:`~nemo_compass_plugin.entities.AnalysisRun` per
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
from nemo_compass_plugin._perms import AnalysisRunPerms
from nemo_compass_plugin.analysis.models import ModelLookupClient, resolve_model_ref
from nemo_compass_plugin.analyst import SAVE_INSIGHTS_ACTION_KIND, CompassAnalyst, CompassPreparedContext
from nemo_compass_plugin.authz import scope
from nemo_compass_plugin.config import CompassConfig
from nemo_compass_plugin.entities import AnalysisRun
from nemo_compass_plugin.schema import AnalysisRunPage, AnalysisRunResponse, CreateAnalysisRunRequest
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.plugin_managed_agents import PluginManagedAgentInputError
from nemo_helix_plugin.agents.types import NEMO_AGENTS_SPEC_CONFIG_FORMAT, CreateExecuteJobRequest, JsonObject
from nemo_helix_plugin.api.pagination import PageParams, make_page_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError, NotFoundError
from nemo_helix_plugin.client.response import NemoResponse
from nemo_helix_plugin.config import get_nemo_config
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError, get_entity_client
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.schema import PaginationData
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)

ANALYSIS_RUN_NAME_PREFIX = "compass-run-"
# The config name of an inline Compass run, which has no Agent entity to name it.
COMPASS_AGENT_NAME = "compass"

router = APIRouter(tags=["Compass Analysis Runs"])
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)


class ExecuteJobClient(Protocol):
    def create_execute_job(
        self, *, body: CreateExecuteJobRequest, workspace: str | None = None
    ) -> Awaitable[NemoResponse[JsonObject]]: ...

    def get_execute_job(self, *, name: str, workspace: str | None = None) -> Awaitable[NemoResponse[JsonObject]]: ...


def _json_object(value: object) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def mint_analysis_run_name() -> str:
    """Mint the name shared by an analysis run and its backing job.

    Minted by Compass rather than assigned by the Jobs service so the run can
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
    client: AsyncNemoClient = Depends(get_nemo_client),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AnalysisRunResponse:
    """Create a Compass analysis run backed by the generic ``agents.execute`` job."""
    config = get_nemo_config(CompassConfig)
    return await submit_analysis_run(
        workspace=workspace,
        request=request,
        agents_client=AsyncAgentsClient.from_client(client),
        models_client=AsyncModelsClient.from_client(client),
        entity_client=entity_client,
        profile=config.analysis.job_profile,
    )


async def submit_analysis_run(
    *,
    workspace: str,
    request: CreateAnalysisRunRequest,
    agents_client: ExecuteJobClient,
    models_client: ModelLookupClient,
    entity_client: NemoEntitiesClient,
    name: str | None = None,
    profile: str | None = None,
) -> AnalysisRunResponse:
    """Shared submission path for the API and the Compass scheduler.

    The scheduler supplies a name so a persisted intent remains discoverable
    even if submission or the subsequent scheduler-status write fails.
    On-demand runs do not advance the scheduled cursor: their evaluation or
    time scope may omit telemetry that the next scheduled run must still cover.
    """
    # Resolve before recording anything: a bogus ref would otherwise persist a
    # run and submit a job that cannot start, and the request carries the only
    # copy of the operator's intent.
    request = await _resolve_model_refs(models_client, request, workspace=workspace)
    run = AnalysisRun(
        name=name or mint_analysis_run_name(),
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

    spec = build_execute_agent_job_config(request)
    try:
        job = (
            await agents_client.create_execute_job(
                workspace=workspace,
                body=CreateExecuteJobRequest(
                    spec=spec,
                    name=saved.name,
                    profile=profile,
                    # An inline run has no spec.subject, so this tag is how the
                    # scheduler's overlap check sees it. Goes with this module.
                    custom_fields={"insights_analysis_agent": request.agent},
                ),
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
    page_params: PageParams = Depends(make_page_dep(default_size=20, max_size=100)),
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
            page=page_params.page,
            page_size=page_params.page_size,
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
    client: AsyncNemoClient = Depends(get_nemo_client),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AnalysisRunResponse:
    """Get one analysis run, joined with the live state of its backing job."""
    agents_client = AsyncAgentsClient.from_client(client)
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
    is what gets stored and put in the job spec, because Compass only accepts
    qualified refs.
    """
    try:
        default_model = await resolve_model_ref(
            client, request.default_model, field="default_model", workspace=workspace
        )
        fast_model = await resolve_model_ref(client, request.fast_model, field="fast_model", workspace=workspace)
    except PluginManagedAgentInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NemoHTTPError as exc:
        raise HTTPException(status_code=exc.status_code, detail=_error_detail(exc)) from exc
    except NemoTransportError as exc:
        # Same reasoning as the submit path: an unreachable dependency is not
        # the caller's bad request, and nothing has been recorded yet.
        logger.warning("Could not reach the Models service to validate the model refs: %s", exc)
        raise HTTPException(
            status_code=503, detail="Could not reach the Models service to validate the model refs."
        ) from exc
    return request.model_copy(update={"default_model": default_model, "fast_model": fast_model})


def build_execute_agent_job_config(request: CreateAnalysisRunRequest) -> JsonObject:
    """Translate a high-level Compass request into a generic execute-agent job config.

    The run is inline, but in ``compass.analyst``'s shape: its config is the
    type's, and its context is already what the type's ``prepare`` would
    produce (the model refs were resolved by the caller, and the Ethos is the
    request's own), so the adapter has one code path.
    """
    context = CompassPreparedContext(
        agent=request.agent,
        default_model=request.default_model,
        fast_model=request.fast_model,
        ethos=request.ethos,
        since=request.since,
        evaluation_id=request.evaluation_id,
    )
    payload = _json_object(
        {
            "agent": {
                # AgentInline defaults to the legacy NAT format, matching the Agent
                # entity; Compass is a Fabric spec agent, so say so explicitly.
                "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT,
                "config": CompassAnalyst.config(name=COMPASS_AGENT_NAME),
            },
            "input": CompassAnalyst.default_input,
            "context": context.model_dump(mode="json", exclude_none=True),
            "post_run_action": {"kind": SAVE_INSIGHTS_ACTION_KIND},
        }
    )
    if request.timeout_seconds is not None:
        payload["timeout_seconds"] = request.timeout_seconds

    return payload


def _error_detail(error: NemoHTTPError) -> object:
    """Unwrap the Agents service's ``detail`` from a failed job-create call."""
    body = error.body
    if isinstance(body, Mapping):
        detail = body.get("detail")
        if detail is not None:
            return detail
    return body if body is not None else error.detail
