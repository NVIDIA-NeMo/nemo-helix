# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session detail API tests."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nhx.intake.spans.api.sessions import get_session
from nhx.intake.spans.domain import IntakeSession, SpanStatus
from nhx.intake.spans.service import IntakeSpansService, SessionNotFoundError


@pytest.mark.asyncio
async def test_get_session_returns_aggregate_schema_without_children() -> None:
    domain = IntakeSession(
        id="session-a",
        workspace="workspace-a",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=SpanStatus.SUCCESS,
        trace_count=2,
        span_count=5,
        total_tokens=123,
    )
    service = SimpleNamespace(get_session=AsyncMock(return_value=domain))

    response = await get_session("workspace-a", "session-a", cast(IntakeSpansService, service))

    assert response.id == "session-a"
    assert response.trace_count == 2
    assert response.span_count == 5
    assert response.total_tokens == 123
    fields = type(response).model_fields
    assert "traces" not in fields
    assert "spans" not in fields
    assert "input" not in fields
    assert "output" not in fields


@pytest.mark.asyncio
async def test_get_session_returns_404_when_missing() -> None:
    service = SimpleNamespace(get_session=AsyncMock(side_effect=SessionNotFoundError("workspace-a", "missing")))

    with pytest.raises(HTTPException) as error:
        await get_session("workspace-a", "missing", cast(IntakeSpansService, service))

    assert error.value.status_code == 404
    assert error.value.detail == "Session workspace-a/missing not found"
