# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic Compass analysis scheduler.

Each run is an ``agents.execute`` job on the workspace's ``compass.analyst``
agent; only cadence, overlap checks, and the incremental success cursor are
owned by this controller.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import ClassVar, TypeVar
from zoneinfo import ZoneInfo

from nemo_compass_plugin.analysis.backend import make_backend
from nemo_compass_plugin.analysis_runs import mint_analysis_run_name
from nemo_compass_plugin.analyst import COMPASS_ANALYST_TYPE_ID, SAVE_INSIGHTS_ACTION_KIND, CompassAnalyst
from nemo_compass_plugin.config import CompassConfig
from nemo_compass_plugin.entities import AnalysisConfig, AnalysisConfigStatus, AnalysisRunStatus
from nemo_compass_plugin.schedule import is_due
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.types import CreateExecuteJobRequest, JsonObject
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.config import get_nemo_config
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.types import HelixJobResponse, ListJobsQueryParams

logger = logging.getLogger(__name__)

_ACTIVE_JOB_STATUSES = [
    "creation_pending",
    "deletion_pending",
    "created",
    "queued",
    "pending",
    "active",
    "cancelling",
    "paused",
    "pausing",
    "resuming",
]

# Minimum new traces (since the last successful run) before a scheduled job is
# worth launching. Only enforced once an agent has an incremental cursor, so the
# first scheduled run still bootstraps.
_MIN_NEW_TRACES_FOR_ANALYSIS = 10

_T = TypeVar("_T")


def _require(value: _T | None, attr: str) -> _T:
    """Return *value* or raise if the controller was used before startup."""
    if value is None:
        raise RuntimeError(f"CompassAnalysisController.{attr} accessed before startup")
    return value


class CompassAnalysisController(NemoController):
    """Submit insights analyzer jobs for enabled agents on a global cadence."""

    name: ClassVar[str] = "compass-analysis"
    dependencies: ClassVar[list[str]] = ["entities", "jobs", "agents", "compass"]

    def __init__(self) -> None:
        self._client: AsyncNemoClient | None = None
        self._entities: NemoEntitiesClient | None = None
        self._jobs: AsyncJobsClient | None = None
        self._config: CompassConfig | None = None

    @property
    def client(self) -> AsyncNemoClient:
        return _require(self._client, "client")

    @property
    def entities(self) -> NemoEntitiesClient:
        return _require(self._entities, "entities")

    @property
    def jobs(self) -> AsyncJobsClient:
        return _require(self._jobs, "jobs")

    @property
    def compass_config(self) -> CompassConfig:
        return _require(self._config, "compass_config")

    @property
    def interval_seconds(self) -> float:
        # Override NemoController's 10s default; per-config due/throttle checks
        # mean a coarser reconcile cadence is enough.
        return 60.0

    async def on_startup(self) -> None:
        """Initialise the service-principal client and typed clients."""
        self._config = get_nemo_config(CompassConfig)
        self._client = get_async_nemo_client(as_service="compass", internal=True)
        self._entities = NemoEntitiesClient(AsyncEntitiesClient.from_client(self._client))
        self._jobs = AsyncJobsClient.from_client(self._client)
        logger.info("CompassAnalysisController started.")

    async def on_shutdown(self) -> None:
        logger.info("CompassAnalysisController shut down.")

    async def list_objects(self) -> list[AnalysisConfig]:
        """Include disabled configs so in-flight runs still get reconciled."""
        if not self.compass_config.analysis.enabled:
            return []
        try:
            configs: list[AnalysisConfig] = []
            page = 1
            while True:
                result = await self.entities.list(AnalysisConfig, workspace="-", page=page, page_size=100)
                configs.extend(result.data)
                if page >= result.pagination.total_pages:
                    return configs
                page += 1
        except Exception:
            logger.exception("Failed to list insights analysis configs")
            return []

    async def reconcile_one(self, obj: object) -> None:
        config = obj if isinstance(obj, AnalysisConfig) else AnalysisConfig.model_validate(obj)
        try:
            await self._reconcile_config(config)
        except NemoEntityConflictError:
            logger.debug(
                "Optimistic lock conflict on analysis config '%s' in workspace '%s'",
                config.name,
                config.workspace,
            )

    async def _reconcile_config(self, config: AnalysisConfig) -> None:
        # Check both legacy and execute jobs, including on-demand analysis.
        if await self._has_active_job(config):
            return
        status = await self._get_run_status(config)
        status = await self._reconcile_run(config, status)
        if not config.enabled or (status is not None and status.status == AnalysisConfigStatus.RUNNING):
            return
        if not config.default_model:
            logger.error(
                "Analysis config for agent '%s' in workspace '%s' has no model selection; "
                "run `nemo compass analysis enable --agent %s --workspace %s` again",
                config.agent,
                config.workspace,
                config.agent,
                config.workspace,
            )
            return
        now = datetime.now(timezone.utc)
        if not self._is_due(status, now):
            return
        if not await self._has_enough_new_traces(config, status):
            return
        await self._submit_analysis_job(config, status, now)

    async def _reconcile_run(
        self, config: AnalysisConfig, status: AnalysisRunStatus | None
    ) -> AnalysisRunStatus | None:
        """Reconcile only the pending attempt recorded before submission."""
        if status is None or not status.last_submitted_job or status.status != AnalysisConfigStatus.RUNNING:
            return status
        updated = status.model_copy()
        try:
            job = (await self.jobs.get_job(workspace=config.workspace, name=status.last_submitted_job)).data()
        except NotFoundError:
            updated.status = AnalysisConfigStatus.ERROR
            updated.last_error = "Analysis job was not submitted or no longer exists"
        else:
            if not _job_targets_agent(job, config.agent):
                updated.status = AnalysisConfigStatus.ERROR
                updated.last_error = "Reconciled job did not belong to this agent"
            elif not job.status.is_terminal():
                return status
            elif job.status == HelixJobStatus.COMPLETED:
                updated.status = AnalysisConfigStatus.IDLE
                # The pre-submission boundary keeps telemetry arriving during
                # execution eligible for the next analysis.
                updated.last_successful_run_at = status.last_attempted_at or status.last_successful_run_at
                updated.last_error = ""
            else:
                updated.status = AnalysisConfigStatus.ERROR
                updated.last_error = f"Analysis job {job.status.value}"
        updated.last_completed_at = datetime.now(timezone.utc)
        return await self.entities.update(updated)

    async def _get_run_status(self, config: AnalysisConfig) -> AnalysisRunStatus | None:
        try:
            return await self.entities.get(AnalysisRunStatus, name=config.agent, workspace=config.workspace)
        except NemoEntityNotFoundError:
            return None
        except Exception:
            logger.exception(
                "Failed to read analysis run status for agent '%s'; deferring",
                config.agent,
            )
            raise

    async def _has_enough_new_traces(self, config: AnalysisConfig, status: AnalysisRunStatus | None) -> bool:
        """Whether enough new traces exist to justify submitting a job.

        Mirrors ``run_compass``'s preflight: the floor is only enforced once an
        incremental cursor exists, so the first scheduled run still bootstraps.
        Checking here avoids launching a job that would immediately skip; the
        job keeps the same floor as a backstop. On count failure we defer to the
        next reconcile rather than launch a job we can't justify.
        """
        since = status.last_successful_run_at if status is not None else None
        if since is None:
            return True
        try:
            backend = make_backend(client=self.client, insights_output=None)
            trace_count = await backend.count_agent_sessions(
                agent=config.agent, workspace=config.workspace, since=since
            )
        except Exception:
            logger.exception(
                "Failed to count new traces for agent '%s'; deferring submission",
                config.agent,
            )
            return False
        if trace_count < _MIN_NEW_TRACES_FOR_ANALYSIS:
            logger.debug(
                "Agent '%s' has %d new trace(s) since %s; below threshold %d, deferring",
                config.agent,
                trace_count,
                since.isoformat(),
                _MIN_NEW_TRACES_FOR_ANALYSIS,
            )
            return False
        return True

    async def _has_active_job(self, config: AnalysisConfig) -> bool:
        try:
            query_params: ListJobsQueryParams = {
                "filter": json.dumps(
                    {
                        "status": {"$in": _ACTIVE_JOB_STATUSES},
                        # Include legacy analysis jobs that may still be running.
                        "source": {"$in": ["nemo-agents-plugin-execute", "insights"]},
                    }
                ),
                "page_size": 100,
                "sort": "-created_at",
            }
            jobs = await self.jobs.list_jobs(
                workspace=config.workspace,
                query_params=query_params,
            )
        except Exception:
            logger.debug(
                "Could not list insights analysis jobs for agent '%s'",
                config.agent,
                exc_info=True,
            )
            return True

        try:
            async for item in jobs.items():
                if _job_targets_agent(item, config.agent):
                    return True
        except Exception:
            logger.debug(
                "Could not inspect insights analysis jobs for agent '%s'",
                config.agent,
                exc_info=True,
            )
            return True
        return False

    def _is_due(self, status: AnalysisRunStatus | None, now: datetime) -> bool:
        compass = self.compass_config.analysis
        anchor = status.last_successful_run_at if status is not None else None
        return is_due(
            now,
            anchor,
            frequency=compass.frequency,
            run_at_hour=compass.run_at_hour,
            run_on_weekday=int(compass.run_on_weekday),
            tz=ZoneInfo(compass.timezone),
        )

    async def _compass_agent_name(self, workspace: str) -> str | None:
        """Return the name of *workspace*'s Compass agent, or None when it has none."""
        agent = await CompassAnalyst.find_agent(workspace=workspace, client=self.client)
        return agent.name if agent is not None else None

    async def _create_execute_job(self, *, workspace: str, body: CreateExecuteJobRequest) -> None:
        await AsyncAgentsClient.from_client(self.client).create_execute_job(workspace=workspace, body=body)

    async def _record_error(self, config: AnalysisConfig, status: AnalysisRunStatus | None, message: str) -> None:
        """Mark the agent's analysis as failed with *message*, unless it already is."""
        if status is not None and status.status == AnalysisConfigStatus.ERROR and status.last_error == message:
            return
        logger.warning(
            "Insights analysis for agent '%s' in workspace '%s': %s", config.agent, config.workspace, message
        )
        updated = (
            status.model_copy()
            if status is not None
            else AnalysisRunStatus(name=config.agent, workspace=config.workspace, agent=config.agent)
        )
        updated.status = AnalysisConfigStatus.ERROR
        updated.last_error = message
        updated.last_completed_at = datetime.now(timezone.utc)
        if status is None:
            await self.entities.create(updated)
        else:
            await self.entities.update(updated)

    async def _submit_analysis_job(
        self,
        config: AnalysisConfig,
        status: AnalysisRunStatus | None,
        submitted_at: datetime,
    ) -> None:
        compass_agent = await self._compass_agent_name(config.workspace)
        if compass_agent is None:
            # Saving the analysis config creates the agent, as the user; this
            # service principal never does. A config saved before that existed
            # needs it created once.
            await self._record_error(
                config,
                status,
                f"Workspace '{config.workspace}' has no Compass agent; run "
                f"`nemo agents create {COMPASS_ANALYST_TYPE_ID} --workspace {config.workspace}`, "
                f"adding `--name <name>` if another agent is already named '{CompassAnalyst.default_name}'.",
            )
            return
        since = status.last_successful_run_at if status is not None else None
        context: JsonObject = {
            "agent": config.agent,
            "default_model": config.default_model,
            "fast_model": config.fast_model or config.default_model,
        }
        if since is not None:
            context["since"] = since.isoformat()
        job_name = mint_analysis_run_name()
        pending = (
            status.model_copy()
            if status
            else AnalysisRunStatus(name=config.agent, workspace=config.workspace, agent=config.agent)
        )
        pending.status = AnalysisConfigStatus.RUNNING
        pending.last_submitted_job = job_name
        pending.last_attempted_at = submitted_at
        pending.last_completed_at = None
        pending.last_error = ""
        # Persist the name before the request: even a timeout or process crash
        # leaves a specific job to check on the next controller pass.
        pending = await (self.entities.create(pending) if status is None else self.entities.update(pending))
        try:
            await self._create_execute_job(
                workspace=config.workspace,
                body=CreateExecuteJobRequest(
                    name=job_name,
                    spec={
                        "agent": compass_agent,
                        "context": context,
                        "post_run_action": {"kind": SAVE_INSIGHTS_ACTION_KIND},
                    },
                ),
            )
        except NemoHTTPError as exc:
            if exc.status_code >= 500 or exc.status_code == 409:
                # The job may exist anyway: a server error can follow its creation, and a
                # conflict means a job already has this name. Leave the attempt pending so
                # the next pass resolves it by name, as after a timeout.
                logger.warning(
                    "Submitting insights analysis job '%s' for agent '%s' failed with HTTP %d; "
                    "checking for it on the next pass",
                    job_name,
                    config.agent,
                    exc.status_code,
                )
                return
            # Rejected before any job was created (an unknown model, say):
            # record why instead of leaving a pending attempt to time out.
            await self._record_error(config, pending, f"Analysis job was rejected: {exc.detail}")
            return
        logger.info(
            "Submitted insights analysis job '%s' for agent '%s' in workspace '%s'",
            job_name,
            config.agent,
            config.workspace,
        )


def _job_targets_agent(job: HelixJobResponse, agent: str) -> bool:
    """Whether *job* is a Compass run analyzing *agent*."""
    spec = job.spec or {}
    if spec.get("plugin_managed_agent_type") == COMPASS_ANALYST_TYPE_ID:
        return spec.get("subject") == agent
    # An inline run from the analysis-runs API has no subject, only this tag.
    # Goes when that API does.
    return (job.custom_fields or {}).get("insights_analysis_agent") == agent
