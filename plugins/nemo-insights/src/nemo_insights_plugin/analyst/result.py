# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform persistence contract for Compass's reconciled insights.

The adapter translates package Insights into this change-set after validating
their storage IDs. Both the CLI and the Fabric execute extension persist it
through the same backend.
"""

from datetime import datetime

from nemo_insights_plugin.entities import InsightStatus
from nemo_insights_plugin.evidence import EvidenceCompatibility, TraceEvidence
from pydantic import BaseModel, ConfigDict, Field


class NewInsight(EvidenceCompatibility):
    """A brand-new Insight the analyst wants to file."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        description=(
            "A short, human-readable sentence naming the insight (e.g. "
            "'Retrieval drops relevant context near the token limit'). The "
            "full problem statement goes in 'description', not here."
        ),
    )
    description: str = Field(
        min_length=1,
        description=(
            "Problem statement: the failure mode, the affected tool or model "
            "call, the conditions that trigger it, and a hypothesis for the "
            "cause. Specific enough to act on, general enough to recur."
        ),
    )
    status: InsightStatus = Field(
        default=InsightStatus.OPEN,
        description="Lifecycle status for the new insight (usually 'open').",
    )
    updated_date: datetime | None = None
    evidence: list[TraceEvidence] = Field(
        default_factory=list,
        description="Supporting traces, optional source URLs, and relevant spans.",
    )


class InsightUpdate(EvidenceCompatibility):
    """New evidence to add to an existing Insight."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        description=(
            "Store-assigned id of the existing insight to add evidence to, as "
            "shown in the ``list_insights`` output (e.g. "
            "'insight-5Q2LoF8z8M9JZxZsHwJKNn'). Not the human-readable title."
        ),
    )
    updated_date: datetime | None = None
    evidence: list[TraceEvidence] = Field(
        default_factory=list,
        description=("Evidence to merge with the insight's existing traces and spans, preserving saved links."),
    )


class AnalystResult(BaseModel):
    """Validated creations and evidence updates for one analysis run."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        min_length=1,
        description=(
            "Brief natural-language summary of the analysis and the "
            "highest-impact findings, for the developer reading the run."
        ),
    )
    new_insights: list[NewInsight] = Field(
        default_factory=list,
        description="Insights to create that do not already exist for the agent.",
    )
    updated_insights: list[InsightUpdate] = Field(
        default_factory=list,
        description=(
            "New trace and span evidence for insights that already exist for "
            "the agent. Only evidence can be added to an existing insight — to "
            "record anything else, file a new insight."
        ),
    )
