# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF trajectory ingest for ClickHouse-backed Intake spans."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.intake.config import IntakeConfig
from nhx.intake.spans.api.dependencies import DenormalizerDep, SpansServiceDep, require_workspace_access
from nhx.intake.spans.domain import TraceBatch
from nhx.intake.spans.ingest.atif_domain import (
    AtifAgent,
    AtifFinalMetrics,
    AtifSchemaVersion,
    AtifStep,
    AtifTrajectory,
    validate_atif_step_ids,
    validate_atif_subagent_trajectory_ids,
    validate_atif_tool_call_references,
    validate_atif_v17_subagent_ref_resolution_keys,
)
from nhx.intake.spans.ingest.atif_mapping import (
    AtifTrajectoryDepthError,
    trajectory_to_evaluator_results,
    trajectory_to_spans,
)
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from nhx.intake.spans.ingest.evaluation_context_validation import validate_evaluation_context
from nhx.intake.spans.storage import utc_now
from pydantic import BaseModel, ConfigDict, Field, model_validator

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Ingest"
EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]


def _atif_max_subagent_depth(request: Request) -> int:
    """Read the configured ATIF subagent depth cap from the Intake service."""
    service = getattr(request.app.state, "intake_service", None) or getattr(request.app.state, "service", None)
    cfg: IntakeConfig | None = getattr(service, "service_config", None) if service is not None else None
    if cfg is None:
        cfg = IntakeConfig()
    return cfg.atif_max_subagent_depth


class AtifIngestRequest(BaseModel):
    """Span-based ATIF ingest request.

    ATIF project scoping is intentionally not accepted here; use the workspace
    route and ``evaluation_context`` for evaluation identity.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation_context: EvaluationContext | None = None
    schema_version: AtifSchemaVersion
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: AtifAgent
    final_metrics: AtifFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    notes: str | None = None
    extra: dict[str, Any] | None = None
    steps: list[AtifStep] = Field(default_factory=list)
    subagent_trajectories: list[AtifTrajectory] | None = None

    @model_validator(mode="after")
    def validate_steps(self) -> AtifIngestRequest:
        """Validate step references and embedded subagent trajectory IDs."""
        validate_atif_step_ids(self.steps)
        validate_atif_tool_call_references(self.steps)
        if self.schema_version == "ATIF-v1.7":
            validate_atif_v17_subagent_ref_resolution_keys(self.steps)
        validate_atif_subagent_trajectory_ids(self.subagent_trajectories)
        return self

    def to_trajectory(self) -> AtifTrajectory:
        """Convert this API request into the recursive ATIF domain model."""
        kwargs: dict[str, Any] = {} if self.session_id is None else {"session_id": self.session_id}
        return AtifTrajectory(
            schema_version=self.schema_version,
            trajectory_id=self.trajectory_id,
            agent=self.agent,
            steps=self.steps,
            final_metrics=self.final_metrics,
            continued_trajectory_ref=self.continued_trajectory_ref,
            notes=self.notes,
            extra=self.extra,
            subagent_trajectories=self.subagent_trajectories,
            evaluation_context=self.evaluation_context,
            **kwargs,
        )


@router.post(
    "/v2/workspaces/{workspace}/ingest/atif",
    tags=[API_TAG],
    status_code=status.HTTP_201_CREATED,
    response_class=Response,
)
async def ingest_atif(
    workspace: str,
    body: AtifIngestRequest,
    request: Request,
    service: SpansServiceDep,
    entity_client: EntityClientDep,
    denormalizer: DenormalizerDep,
) -> Response:
    await validate_evaluation_context(
        workspace=workspace,
        context=body.evaluation_context,
        entity_client=entity_client,
    )
    ingested_at = utc_now()
    trajectory = body.to_trajectory()
    max_subagent_depth = _atif_max_subagent_depth(request)
    try:
        spans = trajectory_to_spans(
            workspace=workspace,
            trajectory=trajectory,
            ingested_at=ingested_at,
            max_subagent_depth=max_subagent_depth,
        )
    except AtifTrajectoryDepthError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    evaluator_results = trajectory_to_evaluator_results(
        workspace=workspace,
        trajectory=trajectory,
        spans=spans,
        ingested_at=ingested_at,
        max_subagent_depth=max_subagent_depth,
    )
    await service.ingest_batch(TraceBatch(spans=spans, evaluator_results=evaluator_results))
    context = body.evaluation_context
    if denormalizer is not None and context is not None and context.evaluation_name:
        denormalizer.mark_dirty(workspace=workspace, evaluation_name=context.evaluation_name)
    return Response(status_code=status.HTTP_201_CREATED)
