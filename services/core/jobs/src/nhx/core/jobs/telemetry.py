# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort anonymous usage telemetry for platform jobs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from typing import Any

from nemo_helix_plugin.jobs.telemetry import get_job_telemetry_plugins, get_job_telemetry_session_id
from nemo_helix_plugin.telemetry.events import JobRunEvent, TaskStatusEnum, ToolCategory, job_tool_category
from nemo_helix_plugin.telemetry.handler import TelemetryHandler

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    "completed": TaskStatusEnum.COMPLETED,
    "error": TaskStatusEnum.ERROR,
    "cancelled": TaskStatusEnum.CANCELED,
}

_SAFE_JOB_TYPE_BUCKETS = frozenset(
    {
        "agent",
        "agents",
        "anonymizer",
        "audit",
        "customization",
        "data-designer",
        "evaluation",
        "evaluator",
        "insights",
        "job",
    }
)


@dataclass(frozen=True)
class JobRunObservation:
    """Normalized lifecycle data, independent of anonymous telemetry opt-in/session."""

    job_type: str
    tool_category: ToolCategory
    status: str
    duration_sec: float
    plugins: list[str]
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class JobRunTelemetry:
    event: JobRunEvent
    session_id: str


def _source_client_version() -> str:
    try:
        return metadata.version("nhx-jobs")
    except metadata.PackageNotFoundError:
        return "undefined"


def _job_type_bucket(source: str) -> str:
    label = str(source or "").strip().lower().replace("_", "-").replace(" ", "-")
    return label if label in _SAFE_JOB_TYPE_BUCKETS else "custom"


def _model_data_bucket(raw_model: object) -> str:
    model = str(raw_model).strip() if raw_model is not None else ""
    return "defined" if model else "undefined"


def _token_count(details: dict[str, Any], key: str) -> int:
    raw_value = details.get(key)
    if raw_value is None or isinstance(raw_value, bool):
        return -1
    if isinstance(raw_value, float) and not raw_value.is_integer():
        return -1
    try:
        count = int(raw_value)
    except (TypeError, ValueError, OverflowError):
        return -1
    return count if 0 <= count <= 9223372036854775807 else -1


def _duration_sec(created_at: datetime | None, updated_at: datetime | None) -> float:
    if created_at is None or updated_at is None:
        return -1.0
    try:
        return max(0.0, updated_at.timestamp() - created_at.timestamp())
    except (TypeError, OSError):
        return -1.0


def build_job_run_observation(
    *,
    source: str,
    status: str,
    status_details: dict[str, Any] | None,
    custom_fields: dict[str, Any] | None,
    created_at: datetime | None,
    updated_at: datetime | None,
) -> JobRunObservation:
    details = status_details if isinstance(status_details, dict) else {}
    plugins = get_job_telemetry_plugins(custom_fields)
    return JobRunObservation(
        job_type=_job_type_bucket(source),
        tool_category=job_tool_category(source.strip().lower().replace("_", "-"), plugins),
        status=status if status in {*_STATUS_MAP, "paused"} else "undefined",
        duration_sec=_duration_sec(created_at, updated_at),
        plugins=plugins,
        model=_model_data_bucket(details.get("model")),
        input_tokens=_token_count(details, "input_tokens"),
        output_tokens=_token_count(details, "output_tokens"),
    )


def build_job_run_telemetry_from_observation(
    observation: JobRunObservation, custom_fields: dict[str, Any] | None
) -> JobRunTelemetry | None:
    session_id = get_job_telemetry_session_id(custom_fields)
    if session_id is None:
        return None
    event = JobRunEvent(
        job_type=observation.job_type,
        tool_category=observation.tool_category,
        token_counts_source=(
            "job-summary" if observation.input_tokens >= 0 or observation.output_tokens >= 0 else "unknown"
        ),
        task_status=_STATUS_MAP.get(observation.status, TaskStatusEnum.UNDEFINED),
        duration_sec=observation.duration_sec,
        plugins=observation.plugins,
        model=observation.model,
        input_tokens=observation.input_tokens,
        output_tokens=observation.output_tokens,
    )
    return JobRunTelemetry(event=event, session_id=session_id)


def build_job_run_telemetry(
    *,
    source: str,
    status: str,
    status_details: dict[str, Any] | None,
    custom_fields: dict[str, Any] | None,
    created_at: datetime | None,
    updated_at: datetime | None,
) -> JobRunTelemetry | None:
    if get_job_telemetry_session_id(custom_fields) is None:
        return None
    observation = build_job_run_observation(
        source=source,
        status=status,
        status_details=status_details,
        custom_fields=custom_fields,
        created_at=created_at,
        updated_at=updated_at,
    )
    return build_job_run_telemetry_from_observation(observation, custom_fields)


async def _send_job_run_event(telemetry: JobRunTelemetry) -> None:
    handler = TelemetryHandler(
        source_client_version=_source_client_version(), session_id=telemetry.session_id, max_retries=0
    )
    handler.enqueue(telemetry.event)
    await handler.astop()


def emit_job_run_event(telemetry: JobRunTelemetry | None) -> None:
    if telemetry is None:
        return
    try:
        task = asyncio.create_task(_send_job_run_event(telemetry))
        task.add_done_callback(_consume_task_exception)
    except RuntimeError:
        logger.debug("No running event loop for job_run telemetry", exc_info=True)


def _consume_task_exception(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except Exception:
        logger.debug("job_run telemetry task failed", exc_info=True)
