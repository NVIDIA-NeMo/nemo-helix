# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

from nhx.customization_common.service.context import NHXJobContext
from nhx.unsloth.app.constants import SERVICE_NAME
from nhx.unsloth.tasks.training.progress import JobsServiceProgressReporter


def test_progress_reporter_calls_sdk_create_or_update() -> None:
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
    mock_jobs = MagicMock()
    jobs_client = MagicMock()
    jobs_client.with_options.return_value = mock_jobs

    with (
        patch("nhx.customization_common.training.progress.get_task_nemo_client") as get_task_nemo_client,
        patch("nhx.customization_common.training.progress.JobsClient.from_client", return_value=jobs_client),
    ):
        reporter = JobsServiceProgressReporter(ctx)
        reporter.report_running(phase="training", step=1, train_loss=2.5, backend="unsloth")

    get_task_nemo_client.assert_called_once_with(SERVICE_NAME)

    mock_jobs.update_job_step_task.assert_called_once()
    call_kwargs = mock_jobs.update_job_step_task.call_args.kwargs
    assert call_kwargs["name"] == ctx.normalized_task
    assert call_kwargs["workspace"] == ctx.workspace
    assert call_kwargs["job"] == ctx.job_id
    assert call_kwargs["step"] == ctx.step
    # status_details is now carried on the HelixJobTaskUpdate body object.
    body = call_kwargs["body"]
    assert body.status_details["train_loss"] == 2.5
    assert body.status_details["backend"] == "unsloth"
