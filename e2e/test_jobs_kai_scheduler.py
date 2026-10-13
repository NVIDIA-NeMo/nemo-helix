# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for platform jobs running with KAI Scheduler.

These tests verify that platform jobs can be configured to run with KAI Scheduler.
"""

from collections.abc import Callable

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
)
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.testing.e2e import wait_for_platform_job

pytestmark = [pytest.mark.timeout(600), pytest.mark.platform("kubernetes"), pytest.mark.feature("kai-scheduler")]


def test_job_config_is_readable_with_kai_scheduler(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job can read its configuration correctly with KAI Scheduler."""

    jobs = JobsClient.from_client(client)
    job = jobs.create_job(
        workspace=workspace,
        body=CreateHelixJobRequest(
            source="e2e-test-jobs-kai-scheduler",
            spec={"test": "value"},
            platform_spec=HelixJobSpec(
                steps=[
                    HelixJobStep(
                        name="echo-step",
                        executor=CPUExecutionProviderSpec(
                            provider="cpu",
                            container=ContainerSpec(
                                image=image("nhx-tasks"),
                                command=["sh", "-c", "echo 'Step config:'; cat $NEMO_JOB_STEP_CONFIG_FILE_PATH;"],
                            ),
                        ),
                        config={
                            "message": "Hello from job config!",
                        },
                    ),
                ],
            ),
        ),
    ).data()

    # Wait for job to complete
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "completed", f"Job failed with status: {completed_job.status}"

    # Get the job logs to verify the step read its config
    step_logs = list(jobs.list_job_logs(workspace=workspace, name=job.name).items())
    assert len(step_logs) == 2, "Expected two step logs"
    assert "Hello from job config!" in step_logs[1].message, "Step logs do not show config was read"
