# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from nhx.common.auth import AuthClient
from nhx.common.auth.models import Principal
from nhx.core.jobs.api.v2.jobs.endpoints import update_job_attempt_status_details
from nhx.core.jobs.app.dispatcher import JobDispatcher


@pytest.mark.parametrize(
    "principal",
    [
        Principal(id="alice@example.com"),
        Principal(id="service:other"),
        Principal(id="service:jobs", on_behalf_of="alice@example.com"),
    ],
)
async def test_workflow_admission_requires_controller(principal: Principal) -> None:
    auth = Mock(spec=AuthClient, auth_enabled=True, principal=principal)
    with pytest.raises(HTTPException) as failure:
        await update_job_attempt_status_details(
            job="job",
            attempt="attempt",
            workspace="default",
            request={"workflow_admission": {"step_id": "step"}},
            auth_client=auth,
            dispatcher=Mock(spec=JobDispatcher),
            expected_step_id=None,
            expected_step_status=None,
        )
    assert failure.value.status_code == 403


async def test_controller_can_request_workflow_admission() -> None:
    auth = Mock(spec=AuthClient, auth_enabled=True, principal=Principal(id="service:jobs"))
    dispatcher = Mock(spec=JobDispatcher, update_attempt_status_details=AsyncMock(return_value=Mock()))
    await update_job_attempt_status_details(
        job="job",
        attempt="attempt",
        workspace="default",
        request={"workflow_admission": {"step_id": "step"}},
        auth_client=auth,
        dispatcher=dispatcher,
        expected_step_id=None,
        expected_step_status=None,
    )
    dispatcher.update_attempt_status_details.assert_awaited_once()
