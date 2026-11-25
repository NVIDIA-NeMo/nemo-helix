# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response schemas for Compass plugin HTTP API.

These live here rather than beside their routes so the SDK resources can share
them without importing FastAPI and the routes' dependencies.
"""

from datetime import datetime

from nemo_compass_plugin.entities import (
    AnalysisConfig,
    AnalysisConfigStatus,
    AnalysisRunStatus,
    Insight,
    InsightStatus,
    NullableInsightDate,
    NullableInsightRating,
)
from nemo_compass_plugin.evidence import EvidenceCompatibility, TraceEvidence
from nemo_helix_plugin.schema import NemoListResponse
from pydantic import BaseModel, Field


class CreateInsightRequest(EvidenceCompatibility):
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
    severity: NullableInsightRating = None
    confidence: NullableInsightRating = None
    updated_date: NullableInsightDate = None
    evidence: list[TraceEvidence] = Field(default_factory=list)


class UpdateInsightRequest(EvidenceCompatibility):
    """Body for ``PATCH /insights/{insight_id}``. Omitted fields are unchanged."""

    title: str | None = None
    agent: str | None = None
    description: str | None = None
    status: InsightStatus | None = None
    severity: NullableInsightRating = None
    confidence: NullableInsightRating = None
    updated_date: NullableInsightDate = None
    evidence: list[TraceEvidence] | None = None


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


class UpdateAnalysisRunStatusRequest(BaseModel):
    """Body for ``PATCH /analysis-run-statuses/{agent}``."""

    status: AnalysisConfigStatus | None = None
    last_successful_run_at: datetime | None = None
    last_attempted_at: datetime | None = None
    last_completed_at: datetime | None = None
    last_submitted_job: str | None = None
    last_error: str | None = None


AnalysisConfigPage = NemoListResponse[AnalysisConfig]
AnalysisRunStatusPage = NemoListResponse[AnalysisRunStatus]
