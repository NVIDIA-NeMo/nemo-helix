# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort anonymous usage telemetry for platform jobs."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
from nemo_platform_plugin.jobs.telemetry import get_job_telemetry_session_id

logger = logging.getLogger(__name__)

CLIENT_ID = "184482118588404"
NEMO_TELEMETRY_VERSION = "nemo-telemetry/1.0"
DEFAULT_ENDPOINT = "https://events.telemetry.data.nvidia.com/v1.1/events/json"
SEND_TIMEOUT_SECONDS = 2.0
SCHEMA_VERSION = "1.10"

_FALSEY = ("", "0", "false", "no", "off")
_CI_ENV_VARS = (
    "CI",
    "GITLAB_CI",
    "GITHUB_ACTIONS",
    "BUILDKITE",
    "CIRCLECI",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
    "TRAVIS",
)


class _TaskStatus(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    CANCELED = "canceled"
    UNDEFINED = "undefined"


_STATUS_MAP = {
    "completed": _TaskStatus.COMPLETED,
    "error": _TaskStatus.ERROR,
    "cancelled": _TaskStatus.CANCELED,
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
class JobRunTelemetry:
    job_type: str
    status: str
    duration_sec: float
    model: str
    input_tokens: int
    output_tokens: int
    session_id: str


def _telemetry_enabled() -> bool:
    value = os.getenv("NEMO_TELEMETRY_ENABLED")
    if value is None:
        return True
    return value.strip().lower() == "true"


def _deployment_type() -> str:
    raw = os.getenv("NEMO_DEPLOYMENT_TYPE", "cli").strip().lower()
    return raw if raw in {"cli", "sdk", "nvidia-internal", "undefined"} else "undefined"


def _is_ci_environment() -> bool:
    return any(os.getenv(v, "").lower() not in _FALSEY for v in _CI_ENV_VARS)


def _cpu_architecture() -> str:
    return platform.machine() or "undefined"


def _telemetry_endpoint() -> str | None:
    endpoint = os.getenv("NEMO_TELEMETRY_ENDPOINT", DEFAULT_ENDPOINT)
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        logger.debug("Skipping job_run telemetry because endpoint is invalid: %s", _redact_endpoint(endpoint))
        return None
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        logger.debug("Skipping job_run telemetry because endpoint is not HTTPS: %s", _redact_endpoint(endpoint))
        return None
    return endpoint


def _redacted_netloc(parsed: SplitResult) -> str:
    host = parsed.hostname
    if host is None:
        return parsed.netloc.rsplit("@", 1)[-1]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port is not None else host


def _redact_endpoint(endpoint: str) -> str:
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return "<invalid-endpoint>"
    query = "<redacted>" if parsed.query else ""
    return urlunsplit((parsed.scheme, _redacted_netloc(parsed), parsed.path, query, parsed.fragment))


def _source_client_version() -> str:
    try:
        return metadata.version("nmp-jobs")
    except metadata.PackageNotFoundError:
        return "undefined"


def _get_iso_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _job_type_bucket(source: str) -> str:
    label = str(source or "").strip().lower().replace("_", "-").replace(" ", "-")
    return label if label in _SAFE_JOB_TYPE_BUCKETS else "custom"


def _model_data_bucket(raw_model: object) -> str:
    model = str(raw_model).strip() if raw_model is not None else ""
    return "defined" if model else "undefined"


def _token_count(details: dict[str, Any], key: str) -> int:
    raw_value = details.get(key)
    if raw_value is None:
        return -1
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def _duration_sec(created_at: datetime | None, updated_at: datetime | None) -> float:
    if created_at is None or updated_at is None:
        return -1.0
    try:
        return max(0.0, updated_at.timestamp() - created_at.timestamp())
    except (TypeError, OSError):
        return -1.0


def build_job_run_telemetry(
    *,
    source: str,
    status: str,
    status_details: dict[str, Any] | None,
    custom_fields: dict[str, Any] | None,
    created_at: datetime | None,
    updated_at: datetime | None,
) -> JobRunTelemetry | None:
    session_id = get_job_telemetry_session_id(custom_fields)
    if session_id is None:
        return None

    details = status_details if isinstance(status_details, dict) else {}
    return JobRunTelemetry(
        job_type=_job_type_bucket(source),
        status=status,
        duration_sec=_duration_sec(created_at, updated_at),
        model=_model_data_bucket(details.get("model")),
        input_tokens=_token_count(details, "input_tokens"),
        output_tokens=_token_count(details, "output_tokens"),
        session_id=session_id,
    )


def _event_parameters(event: JobRunTelemetry) -> dict[str, Any]:
    return {
        "nemoSource": "platform",
        "taskStatus": _STATUS_MAP.get(event.status, _TaskStatus.UNDEFINED).value,
        "deploymentType": _deployment_type(),
        "isCi": _is_ci_environment(),
        "jobType": event.job_type,
        "durationSec": event.duration_sec,
        "plugins": [],
        "model": event.model,
        "inputTokens": event.input_tokens,
        "outputTokens": event.output_tokens,
    }


def build_payload(event: JobRunTelemetry, *, timestamp: datetime | None = None) -> dict[str, Any]:
    event_ts = timestamp or datetime.now(timezone.utc)
    return {
        "browserType": "undefined",
        "clientId": CLIENT_ID,
        "clientType": "Native",
        "clientVariant": "Release",
        "clientVer": _source_client_version(),
        "cpuArchitecture": _cpu_architecture(),
        "deviceGdprBehOptIn": "None",
        "deviceGdprFuncOptIn": "None",
        "deviceGdprTechOptIn": "None",
        "deviceId": "undefined",
        "deviceMake": "undefined",
        "deviceModel": "undefined",
        "deviceOS": "undefined",
        "deviceOSVersion": "undefined",
        "deviceType": "undefined",
        "eventProtocol": "1.6",
        "eventSchemaVer": SCHEMA_VERSION,
        "eventSysVer": NEMO_TELEMETRY_VERSION,
        "externalUserId": "undefined",
        "gdprBehOptIn": "None",
        "gdprFuncOptIn": "None",
        "gdprTechOptIn": "None",
        "idpId": "undefined",
        "integrationId": "undefined",
        "productName": "undefined",
        "productVersion": "undefined",
        "sentTs": _get_iso_timestamp(event_ts),
        "sessionId": event.session_id,
        "userId": "undefined",
        "events": [
            {
                "ts": _get_iso_timestamp(event_ts),
                "parameters": _event_parameters(event),
                "name": "job_run",
            }
        ],
    }


async def _send_job_run_event(event: JobRunTelemetry) -> None:
    if not _telemetry_enabled():
        return
    endpoint = _telemetry_endpoint()
    if endpoint is None:
        return
    try:
        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, json=build_payload(event))
            response.raise_for_status()
    except Exception:
        logger.debug("Failed to emit job_run telemetry to %s", _redact_endpoint(endpoint))


def emit_job_run_event(event: JobRunTelemetry | None) -> None:
    if event is None:
        return
    try:
        task = asyncio.create_task(_send_job_run_event(event))
        task.add_done_callback(_consume_task_exception)
    except RuntimeError:
        logger.debug("No running event loop for job_run telemetry", exc_info=True)


def _consume_task_exception(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except Exception:
        logger.debug("job_run telemetry task failed", exc_info=True)
