# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed dependencies for resources owned by the running Intake service."""

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException
from nhx.common.service.dependencies import service_instance_dependency
from nhx.intake.config import IntakeConfig
from nhx.intake.experiments.denormalizer import EvaluationDenormalizer
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.service import IntakeService
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient, bootstrap_schema

logger = logging.getLogger(__name__)

IntakeServiceDep = Annotated[IntakeService, Depends(service_instance_dependency("intake", IntakeService))]


async def get_clickhouse_client(intake: IntakeServiceDep) -> AsyncIterator[ClickHouseSpanClient]:
    """Require service-owned spans storage, including schema bootstrap."""
    client = intake.clickhouse_client
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="ClickHouse spans storage is disabled or was not initialized during Intake startup",
        )
    try:
        await bootstrap_schema(client)
    except Exception as exc:
        logger.exception("ClickHouse spans storage is unavailable")
        raise HTTPException(status_code=503, detail=CLICKHOUSE_UNAVAILABLE_MESSAGE) from exc
    yield client


def get_optional_clickhouse_client(intake: IntakeServiceDep) -> ClickHouseSpanClient | None:
    return intake.clickhouse_client


def get_denormalizer(intake: IntakeServiceDep) -> EvaluationDenormalizer | None:
    return intake.denormalizer


DenormalizerDep = Annotated[EvaluationDenormalizer | None, Depends(get_denormalizer)]


def get_intake_config(intake: IntakeServiceDep) -> IntakeConfig:
    return intake.service_config or IntakeConfig()


IntakeConfigDep = Annotated[IntakeConfig, Depends(get_intake_config)]
