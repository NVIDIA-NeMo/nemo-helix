# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from nhx.common.jobs.schemas import HelixJobStatus
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.tasks.file_io_progress_reporter import JobsServiceProgressReporter


def test_progress_reporter_calls_sdk_create_or_update() -> None:
    sdk = MagicMock()
    mock_jobs = MagicMock()
    ctx = NHXJobContext(
        workspace="ws-a",
        job_id="job-1",
        attempt_id="attempt-0",
        step="training",
        task="train-model",
        jobs_url="http://jobs.example.com",
        files_url=None,
        storage_path=Path("/tmp/job"),
        config_path=Path("/tmp/job/config.json"),
    )
    reporter = JobsServiceProgressReporter(sdk, ctx.workspace, ctx.job_id, ctx.step, ctx.normalized_task)

    with patch(
        "nhx.customization_common.tasks.file_io_progress_reporter.client_from_platform",
        return_value=mock_jobs,
    ):
        reporter.update_progress(HelixJobStatus.ACTIVE, status_details={"phase": "training"})

    mock_jobs.update_job_step_task.assert_called_once()
    call_kwargs = mock_jobs.update_job_step_task.call_args.kwargs
    assert call_kwargs["name"] == ctx.normalized_task
    assert call_kwargs["workspace"] == ctx.workspace
    assert call_kwargs["job"] == ctx.job_id
    assert call_kwargs["step"] == ctx.step
    # status + status_details now travel on the HelixJobTaskUpdate body; unset
    # fields (error_details/error_stack) are omitted, leaving their model defaults.
    body = call_kwargs["body"]
    assert body.status == HelixJobStatus.ACTIVE
    assert body.status_details == {"phase": "training"}
    assert body.error_details is None
    assert body.error_stack is None
