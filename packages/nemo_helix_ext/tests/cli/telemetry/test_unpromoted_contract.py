# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expanded usage collection must not leak into the current wire contract."""

from datetime import datetime, timezone

import pytest
from nemo_helix_plugin.telemetry.events import (
    CommandInvokedEvent,
    HelixTelemetryEvent,
    JobRunEvent,
    OnboardingStepEvent,
    TaskStatusEnum,
)
from nemo_helix_plugin.telemetry.handler import QueuedEvent, build_payload
from pydantic import ValidationError


@pytest.mark.parametrize(
    "event",
    [
        OnboardingStepEvent(task_status=TaskStatusEnum.COMPLETED, step="models_discovered"),
        CommandInvokedEvent(task_status=TaskStatusEnum.COMPLETED, command="models list", duration_sec=0.912),
        JobRunEvent(task_status=TaskStatusEnum.COMPLETED, job_type="evaluation"),
    ],
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eventId", "8f4cfabf-8a8b-4f18-9af6-1b557c7aa6bb"),
        ("onboardingAttemptId", "1ac98932-4064-4aae-9c29-7da5468610e7"),
        ("invocationOrigin", "agent"),
        ("toolCategory", "evaluation"),
        ("tokenCountsSource", "job-summary"),
    ],
)
def test_unpromoted_fields_cannot_enter_current_events(event: HelixTelemetryEvent, field: str, value: str) -> None:
    params = event.model_dump(mode="json")
    with pytest.raises(ValidationError) as error:
        type(event).model_validate({**params, field: value})
    assert any(detail["type"] == "extra_forbidden" for detail in error.value.errors())

    payload = build_payload(
        [QueuedEvent(event=event, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))],
        source_client_version="test",
        session_id="test",
    )
    assert payload["eventSchemaVer"] == "1.10"
    assert field not in payload["events"][0]["parameters"]
