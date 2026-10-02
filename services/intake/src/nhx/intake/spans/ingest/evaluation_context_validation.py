# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for evaluation-scoped ingest."""

from fastapi import HTTPException, status
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.intake.entities.experiments import Experiment
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext


async def validate_evaluation_context(
    *,
    workspace: str,
    context: EvaluationContext | None,
    entity_client: EntityClient,
) -> None:
    if context is None:
        return
    evaluation_name = context.evaluation_name
    if not evaluation_name:
        return
    try:
        experiment = await entity_client.get(Experiment, name=evaluation_name, workspace=workspace)
    except EntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation '{evaluation_name}' must be created before it can be logged.",
        ) from exc
    if experiment.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation '{evaluation_name}' has been deleted and cannot accept new sessions.",
        )
