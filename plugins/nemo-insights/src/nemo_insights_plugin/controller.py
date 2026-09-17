# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-managed periodic controller for insights analysis."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import ClassVar, TypeVar
from zoneinfo import ZoneInfo

from nemo_insights_plugin.analyst.analyst_backend import make_analyst_backend
from nemo_insights_plugin.config import InsightsConfig
from nemo_insights_plugin.entities import AnalysisConfig, AnalysisRunStatus
from nemo_insights_plugin.schedule import is_due
from nemo_insights_plugin.sdk_resources.analysis_jobs import (
    AnalysisJob,
    AsyncAnalysisJobsClient,
    CreateAnalysisJobRequest,
    ListAnalysisJobsQueryParams,
)
from nemo_insights_plugin.types import AnalyzeSpec
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.config import get_nemo_config
from nemo_platform_plugin.controller import NemoController
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)
from nemo_platform_plugin.sdk_provider import get_async_platform_sdk

logger = logging.getLogger(__name__)

_SAFE_JOB_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_ACTIVE_JOB_STATUSES = [
    "created",
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
        raise RuntimeError(f"InsightsAnalysisController.{attr} accessed before startup")
    return value


class InsightsAnalysisController(NemoController):
    """Submit insights analyzer jobs for enabled agents on a global cadence."""

    name: ClassVar[str] = "insights-analysis"
    dependencies: ClassVar[list[str]] = ["entities", "jobs"]

    def __init__(self) -> None:
        self._sdk: AsyncNeMoPlatform | None = None
        self._entities: NemoEntitiesClient | None = None
        self._jobs: AsyncAnalysisJobsClient | None = None
        self._config: InsightsConfig | None = None

    @property
    def sdk(self) -> AsyncNeMoPlatform:
        return _require(self._sdk, "sdk")

    @property
    def entities(self) -> NemoEntitiesClient:
        return _require(self._entities, "entities")

    @property
    def jobs(self) -> AsyncAnalysisJobsClient:
        return _require(self._jobs, "jobs")

    @property
    def insights_config(self) -> InsightsConfig:
        return _require(self._config, "insights_config")

    @property
    def interval_seconds(self) -> float:
        # Override NemoController's 10s default; per-config due/throttle checks
        # mean a coarser reconcile cadence is enough.
        return 60.0

    async def on_startup(self) -> None:
        """Initialise service-principal SDK and typed clients."""
        self._config = get_nemo_config(InsightsConfig)
        self._sdk = get_async_platform_sdk(as_service="insights", internal=True)
        self._entities = NemoEntitiesClient(client_from_platform(self._sdk, AsyncEntitiesClient))
        self._jobs = client_from_platform(self._sdk, AsyncAnalysisJobsClient)
        logger.info("InsightsAnalysisController started.")

    async def on_shutdown(self) -> None:
        logger.info("InsightsAnalysisController shut down.")

    async def list_objects(self) -> list:
        """List enabled analysis configs across all workspaces."""
        if not self.insights_config.analyst.enabled:
            return []
        try:
            result = await self.entities.list(
                AnalysisConfig,
                workspace="-",
                filter_obj={"enabled": True},
            )
            return result.data
        except Exception:
            logger.exception("Failed to list enabled insights analysis configs")
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
        if not config.enabled:
            return
        if not config.default_model:
            logger.error(
                "Analysis config for agent '%s' in workspace '%s' has no model selection; "
                "run `nemo insights analysis enable --agent %s --workspace %s` again",
                config.agent,
                config.workspace,
                config.agent,
                config.workspace,
            )
            return
        if await self._has_active_job(config):
            return
        status = await self._get_run_status(config)
        now = datetime.now(timezone.utc)
        if not self._is_due(status, now):
            return
        if not await self._has_enough_new_traces(config, status):
            return
        await self._submit_analysis_job(config, status, now)

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

        Mirrors ``run_analyst``'s preflight: the floor is only enforced once an
        incremental cursor exists, so the first scheduled run still bootstraps.
        Checking here avoids launching a job that would immediately skip; the
        job keeps the same floor as a backstop. On count failure we defer to the
        next reconcile rather than launch a job we can't justify.
        """
        since = status.last_successful_run_at if status is not None else None
        if since is None:
            return True
        try:
            backend = make_analyst_backend(client=self.sdk, insights_output=None)
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
            query_params: ListAnalysisJobsQueryParams = {
                "filter": json.dumps({"status": {"$in": _ACTIVE_JOB_STATUSES}}),
                "page_size": 100,
                "sort": "-created_at",
            }
            jobs = await self.jobs.list_analysis_jobs(
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
        analyst = self.insights_config.analyst
        anchor = status.last_successful_run_at if status is not None else None
        return is_due(
            now,
            anchor,
            frequency=analyst.frequency,
            run_at_hour=analyst.run_at_hour,
            run_on_weekday=int(analyst.run_on_weekday),
            tz=ZoneInfo(analyst.timezone),
        )

    async def _submit_analysis_job(
        self,
        config: AnalysisConfig,
        status: AnalysisRunStatus | None,
        submitted_at: datetime,
    ) -> None:
        spec = AnalyzeSpec(
            agent=config.agent,
            base_url=self.insights_config.analyst.base_url,
            since=status.last_successful_run_at if status is not None else None,
            default_model=config.default_model,
            fast_model=config.fast_model or config.default_model,
        )
        job_name = _job_name(config, submitted_at)
        (
            await self.jobs.create_analysis_job(
                workspace=config.workspace,
                body=CreateAnalysisJobRequest(
                    name=job_name,
                    spec=spec,
                    custom_fields={"insights_analysis_agent": config.agent},
                ),
            )
        ).data()
        logger.info(
            "Submitted insights analysis job '%s' for agent '%s' in workspace '%s'",
            job_name,
            config.agent,
            config.workspace,
        )


def _job_targets_agent(job: AnalysisJob, agent: str) -> bool:
    return (job.custom_fields or {}).get("insights_analysis_agent") == agent


def _job_name(config: AnalysisConfig, submitted_at: datetime) -> str:
    workspace = _SAFE_JOB_NAME.sub("-", config.workspace).strip("-") or "workspace"
    agent = _SAFE_JOB_NAME.sub("-", config.agent).strip("-") or "agent"
    stamp = submitted_at.strftime("%Y%m%d%H%M%S")
    # The Jobs service also creates a fileset named ``job-fileset-{job_name}``,
    # and entity names cap at 63 characters. Keep our generated name short
    # enough for that derived fileset while preserving agent/workspace context.
    prefix = "opt-analyze"
    max_name = 63 - len("job-fileset-")
    suffix = f"-{stamp}"
    available = max_name - len(prefix) - len(suffix) - 2
    workspace_part = workspace[: max(1, available // 3)]
    agent_part = agent[: max(1, available - len(workspace_part))]
    return f"{prefix}-{workspace_part}-{agent_part}{suffix}"
