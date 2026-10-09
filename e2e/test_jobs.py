# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for platform jobs.

These tests verify the core platform jobs API works correctly,
including job creation, execution, and status tracking.
"""

import hashlib
import uuid
from collections.abc import Callable

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
    HelixJobSpec,
    HelixJobStep,
    StepLifecycle,
)
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import (
    DEFAULT_JOB_STORAGE_PATH,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.testing.e2e import wait_for_job_logs, wait_for_platform_job
from pydantic import SecretStr

JOB_SOURCE = "e2e-test-jobs"

# Cooperative pause. The jobs service does not signal the process. The workload polls its
# step until the status is pausing, writes sha256(job id) into persistent storage, and
# exits 75. Resume starts the same command again; the saved token is printed and the
# process exits 0.
_COOPERATIVE_PAUSE_SCRIPT = """
import hashlib
import json
import os
import time
import urllib.request

job_name = os.environ["NEMO_JOB_ID"]
workspace = os.environ["NEMO_JOB_WORKSPACE"]
step = os.environ["NEMO_JOB_STEP"]
storage = os.environ["NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH"]
jobs_url = os.environ["NHX_JOBS_URL"].rstrip("/")
marker = os.path.join(storage, "pause-token")

job_url = f"{jobs_url}/apis/jobs/v2/workspaces/{workspace}/jobs/{job_name}"
with urllib.request.urlopen(job_url, timeout=5) as response:
    job_id = json.load(response)["id"]
token = hashlib.sha256(job_id.encode()).hexdigest()

if os.path.exists(marker):
    saved = open(marker, encoding="utf-8").read().strip()
    print(saved, flush=True)
    raise SystemExit(0 if saved == token else 1)

step_url = f"{job_url}/steps/{step}"
deadline = time.time() + 90
while time.time() < deadline:
    with urllib.request.urlopen(step_url, timeout=5) as response:
        status = json.load(response).get("status")
    if status == "pausing":
        os.makedirs(storage, exist_ok=True)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(token)
        raise SystemExit(75)
    time.sleep(0.5)
raise SystemExit("pause was not requested")
"""

pytestmark = [pytest.mark.timeout(600)]


def _job_diagnostic_message(client: NemoClient, job, workspace: str, prefix: str) -> str:
    """Build a diagnostic message with job error details and logs for assertion failures."""
    parts = [prefix]
    if job.status_details:
        parts.append(f"Status details: {job.status_details}")
    if job.error_details:
        parts.append(f"Error details: {job.error_details}")
    try:
        logs = list(JobsClient.from_client(client).list_job_logs(workspace=workspace, name=job.name).items())
        if logs:
            parts.append(f"Job logs ({len(logs)} entries):")
            for entry in logs:
                parts.append(f"  - {entry.message}")
    except Exception as log_err:
        parts.append(f"Could not fetch job logs: {log_err}")
    return "\n".join(parts)


def test_basic_platform_job_lifecycle(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test a basic platform job lifecycle: create, run, complete.

    This test verifies the platform jobs system works end-to-end:
    1. Create a job with a simple container step
    2. Wait for the job to complete
    3. Verify job and step reach completed status
    4. Retrieve and check step logs
    """

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="echo-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=["echo", "Hello from e2e test!"],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to complete (use job.name for retrieve, not job.id which is the internal ID)
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "completed", f"Job failed with status: {completed_job.status}"

    # Get the job logs to verify the step ran successfully (wait for OTLP batching)
    step_logs = wait_for_job_logs(client, job.name, workspace, min_log_count=1, timeout=240)
    all_messages = " ".join(log.message for log in step_logs.data)
    assert "Hello from e2e test!" in all_messages, "Step logs do not contain expected output"


def test_job_logs_across_multiple_batches(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that logs spanning multiple OTLP batches are correctly stored and retrieved.

    This test verifies the log storage system handles multiple parquet files:
    1. Create a job that outputs logs over time (with delays to trigger multiple OTLP batches)
    2. Wait for the job to complete
    3. Retrieve all logs and verify they are all present
    4. Verify logs are in the correct order

    The OTLP BatchProcessor batches logs before sending, so logs output with delays
    between them will end up in different batches (and potentially different parquet files).
    """
    num_logs = 5
    delay_seconds = 2  # Delay between logs to trigger separate OTLP batches

    # Build a shell command that outputs numbered logs with delays
    # Each log line includes a sequence number for verification
    log_command = "; ".join(
        [f'echo "Log message {i} of {num_logs}"; sleep {delay_seconds}' for i in range(1, num_logs + 1)]
    )

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "multi-batch-logs"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="multi-log-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=["sh", "-c", log_command],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to complete - this job takes ~20 seconds (10 logs * 2s delay)
    completed_job = wait_for_platform_job(client, job.name, workspace, timeout=120)
    assert completed_job.status == "completed", f"Job failed with status: {completed_job.status}"

    # Wait for all logs to be available (OTLP batching may delay final logs)
    step_logs = wait_for_job_logs(client, job.name, workspace, min_log_count=num_logs, timeout=120)

    # Verify we got all the logs
    assert len(step_logs.data) == num_logs, f"Expected {num_logs} logs, got {len(step_logs.data)}"

    # Verify each log message is present and in order
    for i in range(1, num_logs + 1):
        expected_message = f"Log message {i} of {num_logs}"
        # Logs should be in order (index i-1 for 0-based)
        assert expected_message in step_logs.data[i - 1].message, (
            f"Log {i} not found at expected position. "
            f"Expected '{expected_message}', got '{step_logs.data[i - 1].message}'"
        )


def test_job_config_is_readable(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job can read its configuration correctly."""

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="echo-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=[
                                        "sh",
                                        "-c",
                                        "echo 'Step config:'; cat $NEMO_JOB_STEP_CONFIG_FILE_PATH;",
                                    ],
                                ),
                            ),
                            config={
                                "message": "Hello from job config!",
                            },
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to complete
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "completed", f"Job failed with status: {completed_job.status}"

    # Get the job logs to verify the step read its config (wait for OTLP batching)
    step_logs = wait_for_job_logs(client, job.name, workspace, min_log_count=2, timeout=60)
    all_messages = " ".join(log.message for log in step_logs.data)
    assert "Hello from job config!" in all_messages, "Step logs do not show config was read"


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_job_passing_data_between_steps(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that data can be passed between job steps via persistent storage."""
    persistent_storage_env = [
        EnvironmentVariable(
            name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            value=DEFAULT_JOB_STORAGE_PATH,
        )
    ]

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="generate-data-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=[
                                        "sh",
                                        "-c",
                                        'mkdir -p "${NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH}"; '
                                        "echo 'Data from first step' > "
                                        '"${NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH}/data.txt"',
                                    ],
                                ),
                            ),
                            environment=persistent_storage_env,
                        ),
                        HelixJobStep(
                            name="consume-data-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=[
                                        "sh",
                                        "-c",
                                        "echo 'Consuming data:'; "
                                        'cat "${NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH}/data.txt"',
                                    ],
                                ),
                            ),
                            environment=persistent_storage_env,
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to complete
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "completed", _job_diagnostic_message(
        client,
        completed_job,
        workspace,
        f"Job failed with status: {completed_job.status}",
    )

    # Get the job logs to verify data was passed between steps
    step_logs = list(JobsClient.from_client(client).list_job_logs(workspace=workspace, name=job.name).items())
    all_messages = " ".join(log.message for log in step_logs)
    assert "Data from first step" in all_messages, "Second step did not receive data from first step"


@pytest.mark.skip(
    reason="Default CPU Docker e2e uses subprocess-backed jobs without additional container volume mounts"
)
@pytest.mark.platform("docker")
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_job_using_additional_volume(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job can use an additional volume to store data between steps."""
    # Create a job that uses an additional volume
    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "data-between-steps"},
                platform_spec=HelixJobSpec(
                    steps=[
                        # Write data to the additional volume
                        HelixJobStep(
                            name="write-data",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=[
                                        "sh",
                                        "-c",
                                        "echo 'Hello, World!' > /mnt/additional_storage/shared_data.txt; echo 'Successfully wrote data to persistent storage';",
                                    ],
                                ),
                            ),
                        ),
                        # Read data from the additional volume
                        HelixJobStep(
                            name="read-data",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=[
                                        "sh",
                                        "-c",
                                        "cat /mnt/additional_storage/shared_data.txt; echo 'Successfully read data from persistent storage';",
                                    ],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )
        .data()
    )
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "completed", f"Job failed with status: {completed_job.status}"
    step_logs = list(JobsClient.from_client(client).list_job_logs(workspace=workspace, name=job.name).items())
    assert len(step_logs) == 3, "Expected three step logs"
    assert "Successfully wrote data to persistent storage" in step_logs[0].message, (
        "Step logs do not show data was written to additional volume"
    )
    assert "Hello, World!" in step_logs[1].message, "Step logs do not show data was written to additional volume"
    assert "Successfully read data from persistent storage" in step_logs[2].message, (
        "Step logs do not show data was read from additional volume"
    )


def test_job_using_secret_environment_variable(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job can use secret environment variables."""

    secret_name = f"e2e-secret-{uuid.uuid4().hex[:8]}"
    secret_value = "3"
    # Create a secret to consume in the platform
    secret = (
        SecretsClient.from_client(client)
        .create_secret(
            workspace=workspace,
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
        )
        .data()
    )

    # Create a job that uses the secret as an environment variable
    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="secret-envvar-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=[
                                        "sh",
                                        "-c",
                                        'echo "Secret value is: $SECRET_ENV_VAR"',
                                    ],
                                ),
                            ),
                            environment=[
                                EnvironmentVariable(
                                    name="SECRET_ENV_VAR",
                                    from_secret=EnvironmentVariableFromSecret(name=secret.name),
                                )
                            ],
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to complete
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "completed", f"Job failed with status: {completed_job.status}"

    # Get the job logs to verify the secret was used (wait for OTLP batching)
    step_logs = wait_for_job_logs(client, job.name, workspace, min_log_count=1, timeout=120)
    all_messages = " ".join(log.message for log in step_logs.data)
    assert secret_value in all_messages, "Step logs do not show secret environment variable was used"


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_job_with_expected_failure(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job correctly reports failure when a step fails."""

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="failing-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=["sh", "-c", "echo 'This step will fail'; exit 1;"],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to complete
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "error", f"Job should have failed but has status: {completed_job.status}"

    # Get the job logs to verify the step failure (wait for OTLP batching)
    step_logs = wait_for_job_logs(client, job.name, workspace, min_log_count=1, timeout=30)
    assert len(step_logs.data) == 1, "Expected one step log"
    assert "This step will fail" in step_logs.data[0].message, "Step logs do not contain expected output"


def test_job_cancel_immediately(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job can be created and then cancelled immediately."""

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="long-running-step-cancel-immediate",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    command=["sh", "-c", "sleep 60"],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Cancel the job immediately
    JobsClient.from_client(client).cancel_job(workspace=workspace, name=job.name)

    # Wait for job to reach cancelled status
    cancelled_job = wait_for_platform_job(client, job.name, workspace)
    assert cancelled_job.status == "cancelled", f"Job should have been cancelled but has status: {cancelled_job.status}"


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_job_cancel_once_active(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job can be created and then cancelled once it becomes active."""

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="long-running-step-cancel-once-active",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=image("nhx-tasks"),
                                    entrypoint=["nemo-helix"],
                                    command=[
                                        "run",
                                        "task",
                                        "--task",
                                        "nhx.hello_world.tasks.hello_world",
                                    ],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to become active
    active_job = wait_for_platform_job(client, job.name, workspace, status_to_check="active")
    assert active_job.status == "active", _job_diagnostic_message(
        client,
        active_job,
        workspace,
        f"Job did not become active, status: {active_job.status}",
    )

    # Cancel the job
    JobsClient.from_client(client).cancel_job(workspace=workspace, name=job.name)

    # Wait for job to reach cancelled status
    cancelled_job = wait_for_platform_job(client, job.name, workspace)
    assert cancelled_job.status == "cancelled", _job_diagnostic_message(
        client,
        cancelled_job,
        workspace,
        f"Job should have been cancelled but has status: {cancelled_job.status}",
    )


def _cooperative_pause_step(name: str, image: Callable[[str], str]) -> HelixJobStep:
    return HelixJobStep(
        name=name,
        lifecycle=StepLifecycle(pause_deadline_seconds=180),
        executor=CPUExecutionProviderSpec(
            provider="cpu",
            container=ContainerSpec(
                image=image("nhx-tasks"),
                entrypoint=["python3"],
                command=["-c", _COOPERATIVE_PAUSE_SCRIPT],
            ),
        ),
    )


@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_job_pause_resume(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """A workload pauses by exiting 75, then resume reads the token it saved."""
    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[_cooperative_pause_step("long-running-step-pause-resume", image)],
                ),
            ),
        )
        .data()
    )
    token = hashlib.sha256(job.id.encode()).hexdigest()

    active_job = wait_for_platform_job(client, job.name, workspace, status_to_check="active")
    assert active_job.status == "active", _job_diagnostic_message(
        client, active_job, workspace, f"Job did not become active, status: {active_job.status}"
    )

    JobsClient.from_client(client).pause_job(workspace=workspace, name=job.name)

    paused_job = wait_for_platform_job(client, job.name, workspace, status_to_check="paused")
    assert paused_job.status == "paused", _job_diagnostic_message(
        client, paused_job, workspace, f"Job should have been paused but has status: {paused_job.status}"
    )

    JobsClient.from_client(client).resume_job(workspace=workspace, name=job.name)

    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "completed", _job_diagnostic_message(
        client, completed_job, workspace, f"Job failed with status: {completed_job.status}"
    )

    logs = wait_for_job_logs(client, job.name, workspace)
    messages = "\n".join(entry.message or "" for entry in logs.data)
    assert token in messages, _job_diagnostic_message(
        client, completed_job, workspace, f"Resumed job did not print the saved job-id hash {token}"
    )


@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_job_pause_and_cancel(client: NemoClient, workspace: str, image: Callable[[str], str]):
    """Test that a job can be paused and then cancelled after being paused."""
    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[_cooperative_pause_step("long-running-step-pause-cancel", image)],
                ),
            ),
        )
        .data()
    )

    # Wait for job to become active
    active_job = wait_for_platform_job(client, job.name, workspace, status_to_check="active")
    assert active_job.status == "active", f"Job did not become active, status: {active_job.status}"

    # Pause the job
    JobsClient.from_client(client).pause_job(workspace=workspace, name=job.name)

    # Wait for job to reach paused status
    paused_job = wait_for_platform_job(client, job.name, workspace, status_to_check="paused")
    assert paused_job.status == "paused", f"Job should have been paused but has status: {paused_job.status}"

    # Cancel the job
    JobsClient.from_client(client).cancel_job(workspace=workspace, name=job.name)

    # Wait for job to reach cancelled status
    cancelled_job = wait_for_platform_job(client, job.name, workspace)
    assert cancelled_job.status == "cancelled", f"Job should have been cancelled but has status: {cancelled_job.status}"

    # Ensure that a paused and then cancelled job cannot be resumed
    # TODO (@tmutch): Re-enable this test once resume from cancelled throws an error, currently it returns a 200 response
    # with pytest.raises(NemoHTTPError) as exc_info:
    #    JobsClient.from_client(client).resume_job(workspace=workspace, name=job.name)
    # assert exc_info.value.status_code == 400, "Expected 400 error when resuming a cancelled job"
    # assert "cannot be resumed" in str(exc_info.value), "Expected error when resuming a cancelled job"


BAD_IMAGE_BAD_FORMAT = "__invalid_ubuntu:image"
BAD_IMAGE_VALID_FORMAT = "ubuntu:does-not-exist-1234"


@pytest.mark.skip(
    reason="Image validation is bypassed in subprocess mode because cpu/default container steps are translated"
)
@pytest.mark.parametrize("bad_image", [BAD_IMAGE_BAD_FORMAT, BAD_IMAGE_VALID_FORMAT])
def test_job_invalid_image_format(client: NemoClient, workspace: str, bad_image: str):
    """Test that a job with a bad image fails appropriately."""

    job = (
        JobsClient.from_client(client)
        .create_job(
            workspace=workspace,
            body=CreateHelixJobRequest(
                source=JOB_SOURCE,
                spec={"test": "value"},
                platform_spec=HelixJobSpec(
                    steps=[
                        HelixJobStep(
                            name="bad-image-step",
                            executor=CPUExecutionProviderSpec(
                                provider="cpu",
                                container=ContainerSpec(
                                    image=bad_image,
                                    command=["echo", "This should not run"],
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        )
        .data()
    )

    # Wait for job to complete
    completed_job = wait_for_platform_job(client, job.name, workspace)
    assert completed_job.status == "error", f"Job should have failed but has status: {completed_job.status}"

    # Get the job status to verify the error
    job_status = JobsClient.from_client(client).get_job_status(workspace=workspace, name=job.name).data()

    assert job_status.steps[0].status == "error", "Step should have failed"
