# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from nemo_platform_plugin.jobs.telemetry import build_job_telemetry_custom_fields
from nmp.core.jobs.telemetry import _redact_endpoint, _send_job_run_event, build_job_run_telemetry, build_payload


def test_build_job_run_telemetry_uses_stamped_session_id() -> None:
    event = build_job_run_telemetry(
        source="customer project",
        status="completed",
        status_details={"model": "private-model", "input_tokens": "12", "output_tokens": 7},
        custom_fields=build_job_telemetry_custom_fields("session-123"),
        created_at=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 10, 12, 0, 5, tzinfo=timezone.utc),
    )

    assert event is not None
    assert event.session_id == "session-123"
    assert event.job_type == "custom"
    assert event.status == "completed"
    assert event.duration_sec == 5.0
    assert event.model == "defined"
    assert event.input_tokens == 12
    assert event.output_tokens == 7


def test_build_job_run_telemetry_returns_none_without_session_id() -> None:
    event = build_job_run_telemetry(
        source="job",
        status="completed",
        status_details={},
        custom_fields={},
        created_at=None,
        updated_at=None,
    )

    assert event is None


def test_build_payload_matches_job_run_wire_contract() -> None:
    event = build_job_run_telemetry(
        source="evaluation",
        status="cancelled",
        status_details={"input_tokens": None, "output_tokens": "bad"},
        custom_fields=build_job_telemetry_custom_fields("session-123"),
        created_at=None,
        updated_at=None,
    )
    assert event is not None

    payload = build_payload(event, timestamp=datetime(2026, 9, 10, 12, 0, 0, 123000, tzinfo=timezone.utc))

    assert payload["eventSchemaVer"] == "1.10"
    assert payload["sessionId"] == "session-123"
    assert payload["sentTs"] == "2026-09-10T12:00:00.123Z"
    assert payload["events"][0]["name"] == "job_run"
    params = payload["events"][0]["parameters"]
    assert params["nemoSource"] == "platform"
    assert params["taskStatus"] == "canceled"
    assert params["jobType"] == "evaluation"
    assert params["plugins"] == []
    assert params["model"] == "undefined"
    assert params["inputTokens"] == -1
    assert params["outputTokens"] == -1


def test_redact_endpoint_strips_query_and_credentials() -> None:
    endpoint = "https://marker" + "@example.test:8443/events?debug=value"
    assert _redact_endpoint(endpoint) == "https://example.test:8443/events?<redacted>"


@pytest.mark.asyncio
async def test_send_job_run_event_skips_non_https_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    event = build_job_run_telemetry(
        source="job",
        status="completed",
        status_details={},
        custom_fields=build_job_telemetry_custom_fields("session-123"),
        created_at=None,
        updated_at=None,
    )
    assert event is not None
    monkeypatch.setenv("NEMO_TELEMETRY_ENDPOINT", "http://marker" + "@example.test/events?debug=value")

    with patch("nmp.core.jobs.telemetry.httpx.AsyncClient") as async_client:
        await _send_job_run_event(event)

    async_client.assert_not_called()
