# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response schemas for the insights plugin HTTP API.

These live here rather than beside their routes so the SDK resources can share
them: importing :mod:`nemo_insights_plugin.analysis_runs` from an SDK client
would drag FastAPI and the whole Analyst agent stack in with it.
"""

from datetime import datetime
from typing import Annotated, Any

from nemo_insights_plugin.entities import (
    AnalysisConfig,
    AnalysisConfigStatus,
    AnalysisRun,
    AnalysisRunStatus,
    Insight,
    InsightStatus,
)
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.schema import NemoListResponse
from pydantic import BaseModel, Field, StringConstraints


class CreateInsightRequest(BaseModel):
    """Body for ``POST /insights``.

    ``status`` defaults to :attr:`InsightStatus.OPEN`; callers that want to
    mint an insight already in another lifecycle state set it explicitly.
    """

    title: str = Field(
        description=(
            "A short, human-readable sentence naming the core issue. The full "
            "problem statement goes in 'description'. The store's slug name is "
            "auto-generated, so no name is supplied here."
        ),
    )
    agent: str = Field(
        description="Name of the registered agent this insight is about.",
    )
    description: str = Field(
        description=("The problem statement: specific enough to act on. This is editable by the developer."),
    )
    status: InsightStatus = Field(default=InsightStatus.OPEN)
    trace_refs: list[str] = Field(default_factory=list)


class UpdateInsightRequest(BaseModel):
    """Body for ``PATCH /insights/{insight_id}``. Omitted fields are unchanged."""

    title: str | None = None
    agent: str | None = None
    description: str | None = None
    status: InsightStatus | None = None
    trace_refs: list[str] | None = None


class InsightListItem(Insight, entity_type="insights_insight"):
    """Insight representation used only in the paginated list response."""

    experiment_group_count: int | None = Field(
        default=None,
        description="Number of live experiment groups linked to this insight.",
    )
    last_seen_at: datetime | None = Field(
        default=None,
        description="Newest start timestamp among the insight's currently referenced traces.",
    )


InsightPage = NemoListResponse[InsightListItem]


class EnableAnalysisConfigRequest(BaseModel):
    """Model selection captured when periodic analysis is enabled."""

    default_model: str = Field(min_length=1)
    fast_model: str = Field(min_length=1)


class UpdateAnalysisConfigRequest(BaseModel):
    """Body for ``PATCH /analysis-configs/{agent}``."""

    enabled: bool | None = None


NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CreateAnalysisRunRequest(BaseModel):
    """Client-facing request to run the Insights Analyst through ``agents.execute``."""

    agent: NonBlankString = Field(description="Agent whose telemetry should be analyzed.")
    default_model: NonBlankString = Field(
        description=(
            "Model Entity the Analyst uses for analysis work, as a name or a "
            "'<workspace>/<name>' ref; a bare name resolves in the run's workspace and is "
            "stored qualified. Required: the model pair is the operator's choice and is not "
            "persisted anywhere the Platform process can read it."
        ),
    )
    fast_model: NonBlankString = Field(
        description=(
            "Model Entity used for context summarization, in the same form as default_model. "
            "Required for the same reason."
        ),
    )
    ethos: NonBlankString | None = Field(
        default=None,
        description=(
            "Optional Ethos Markdown for the agent under test. Sent inline rather than as a "
            "reference: the execute job's Fabric adapter has no Files access."
        ),
    )
    since: datetime | None = Field(
        default=None,
        description="Optional lower bound enforced on the Analyst's trace/span reads.",
    )
    evaluation_id: NonBlankString | None = Field(
        default=None,
        description="Optional run scope AND-pinned onto every span read.",
    )
    timeout_seconds: float | None = Field(default=None, gt=0, description="Optional execute job timeout.")


class AnalysisRunResponse(BaseModel):
    """An analysis run plus the live state of the job backing it."""

    run: AnalysisRun
    job: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The backing execute-agent job, or null when no job exists under the run's name — "
            "meaning submission never landed and the run can be resubmitted."
        ),
    )

    @property
    def job_status(self) -> str | None:
        """Status of the backing job, or None when no job exists under the run's name."""
        if self.job is None:
            return None
        status = self.job.get("status")
        return str(status) if status else None

    @property
    def job_is_terminal(self) -> bool:
        """Whether the backing job has finished, successfully or not.

        A run with no job is *not* terminal: it never landed and can be
        resubmitted, which is a different outcome from a job that ran and
        failed.
        """
        status = self.job_status
        return status in {terminal.value for terminal in HelixJobStatus.terminals()}


class UpdateAnalysisRunStatusRequest(BaseModel):
    """Body for ``PATCH /analysis-run-statuses/{agent}``."""

    status: AnalysisConfigStatus | None = None
    last_successful_run_at: datetime | None = None
    last_attempted_at: datetime | None = None
    last_completed_at: datetime | None = None
    last_submitted_job: str | None = None
    last_error: str | None = None


AnalysisConfigPage = NemoListResponse[AnalysisConfig]
AnalysisRunPage = NemoListResponse[AnalysisRun]
AnalysisRunStatusPage = NemoListResponse[AnalysisRunStatus]
