# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insights plugin entity definitions — stored in the NeMo Platform entity store."""

from datetime import datetime
from enum import StrEnum

from nemo_platform_plugin.entity import NemoEntity
from pydantic import Field


class InsightStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    DELETED = "deleted"


class AnalysisConfigStatus(StrEnum):
    """Lifecycle state for periodic insights analysis of one agent."""

    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"


class Insight(NemoEntity, entity_type="insights_insight"):
    """A persistent problem, theme, or category of issues in the agent under test."""

    title: str = Field(
        description=(
            "A short, human-readable sentence naming the core issue common to "
            "the linked traces. Editable by the developer."
        ),
    )
    description: str = Field(
        description=(
            "The problem statement: specific enough to act on. Editable by the "
            "developer. A paragraph or two with detail on what exactly is going "
            "wrong, general enough to apply to many traces rather than to a "
            "single problematic instance."
        ),
    )
    agent: str = Field(
        description=(
            "Name of the registered agent this insight is about, or a local "
            "filesystem path (as a string) to the agent directory when running "
            "offline."
        ),
    )
    status: InsightStatus = Field(
        default=InsightStatus.OPEN,
        description=(
            "An insight starts as open. It can be resolved if the developer "
            "thinks the issue has been fixed. It can be deleted if the "
            "developer thinks the issue is not actually a problem or if it is "
            "not a good insight for their domain."
        ),
    )
    trace_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Intake trace ids the analyst identified as evidence for this "
            "insight. This is used as evidence for the insight UI to "
            "communicate to the developer what traces triggered the issue, and "
            "can also be used to identify other similar traces that might "
            "experience the same issue."
        ),
    )


class AnalysisRun(NemoEntity, entity_type="insights_analysis_run"):
    """One Insights analysis run — what was asked for, not how it went.

    The record's ``name`` is also the name of the backing ``agents.execute``
    job. Insights mints that name before submitting, so the link between a run
    and its job is *derivable* rather than written back after the fact. That
    removes the dual write, and with it the failure mode where a job exists
    that no run points at: a run whose job name 404s simply never landed. There
    is no resubmit-under-an-existing-name route yet, so such a run is a read
    today, not something a caller can drive back to completion.

    Progress is deliberately absent. Status, logs, and results belong to the
    Job and are read from it; mirroring them here would go stale immediately.
    Insights owns only what no lower layer knows: that this job was an analysis
    run, over which agent and scope.
    """

    agent: str = Field(description="Agent whose telemetry this run analyzed.")
    since: datetime | None = Field(
        default=None,
        description="Lower bound the run enforced on trace/span reads.",
    )
    evaluation_id: str = Field(
        default="",
        description="Run scope AND-pinned onto every span read, when one was requested.",
    )
    default_model: str = Field(
        default="",
        description="Workspace-qualified Model Entity the run used for analysis work.",
    )
    fast_model: str = Field(
        default="",
        description="Workspace-qualified Model Entity the run used for context summarization.",
    )


class AnalysisConfig(NemoEntity, entity_type="insights_analysis_config"):
    """Per-agent opt-in state for framework-managed periodic analysis.

    The cadence is intentionally global insights configuration. The model pair
    is captured when analysis is enabled because the controller runs in the
    Platform process and cannot read the operator's local CLI configuration.
    Machine-written run state lives on :class:`AnalysisRunStatus` so the
    controller never races the running job.
    """

    agent: str = Field(description="Name of the agent this analysis config targets.")
    enabled: bool = Field(
        default=True,
        description="Whether the periodic insights controller should analyze this agent.",
    )
    default_model: str = Field(
        default="",
        description="Workspace-qualified Model Entity used for quality-critical analysis work.",
    )
    fast_model: str = Field(
        default="",
        description="Workspace-qualified Model Entity used for latency-sensitive analysis work.",
    )


class AnalysisRunStatus(NemoEntity, entity_type="insights_analysis_run_status"):
    """Machine-written run state for periodic analysis of one agent."""

    agent: str = Field(description="Name of the agent this run status targets.")
    status: AnalysisConfigStatus = Field(
        default=AnalysisConfigStatus.IDLE,
        description="Last known periodic analysis state for this agent.",
    )
    last_successful_run_at: datetime | None = Field(
        default=None,
        description=(
            "Cursor for incremental analysis. Scheduled runs only consider telemetry at or after this timestamp."
        ),
    )
    last_attempted_at: datetime | None = Field(
        default=None,
        description="When the controller or job last attempted analysis.",
    )
    last_completed_at: datetime | None = Field(
        default=None,
        description="When the last analysis attempt reached a terminal state.",
    )
    last_submitted_job: str = Field(
        default="",
        description="Most recent platform job name submitted for this agent.",
    )
    last_error: str = Field(
        default="",
        description="Most recent error or skip reason from scheduled analysis.",
    )
