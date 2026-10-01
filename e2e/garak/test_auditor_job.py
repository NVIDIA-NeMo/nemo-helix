# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""E2E test for auditor job that runs Garak using a mock IGW provider.

- When the audit job runs, the worker resolves nhx_uri_spec to the IGW URL and
  Garak calls it; the mock returns fixed responses so the job can complete
  without a real NIM/LLM backend.

Requires: quickstart (Docker) with auditor service and jobs backend so the
audit job container can be scheduled and can reach the IGW at nhx-quickstart:8080.
"""

import time

import pytest
from nemo_helix_plugin.garak.client import GarakClient
from nemo_helix_plugin.garak.types import CreateAuditConfigRequest, CreateAuditTargetRequest, SubmitAuditRequest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.projects.client import ProjectsClient
from nemo_helix_plugin.projects.types import CreateProjectRequest
from nhx.testing import MockProviderResponse, add_mock_provider, short_unique_name

AUDITOR_MOCK_MODEL = "e2e-auditor-mock-model"

# Terminal statuses for audit jobs (API may return lowercase)
_AUDIT_TERMINAL_STATUSES = {"completed", "error", "failed", "cancelled"}

# Garak can take a while (probes, retries); use a generous timeout
_AUDIT_JOB_TIMEOUT = 600.0
_AUDIT_JOB_POLL_INTERVAL = 10.0


def _wait_for_audit_job(
    client: NemoClient,
    job_name: str,
    workspace: str,
    timeout: float = _AUDIT_JOB_TIMEOUT,
    poll_interval: float = _AUDIT_JOB_POLL_INTERVAL,
) -> dict:
    """Wait for an audit job to reach a terminal state.

    Polls the platform job status until it is completed, error, failed, or
    cancelled. Returns the final status response as a dict (model_dump).
    """
    jobs = JobsClient.from_client(client)
    start = time.time()
    status_history: list[str] = []

    while time.time() - start < timeout:
        response = jobs.get_job_status(name=job_name, workspace=workspace).data()
        status = str(response.status.value).lower()
        status_dict = response.model_dump()

        if status and status not in status_history:
            status_history.append(status)

        if status in _AUDIT_TERMINAL_STATUSES:
            return status_dict

        time.sleep(poll_interval)

    elapsed = time.time() - start
    raise TimeoutError(
        f"Audit job {job_name} did not reach terminal state within {timeout}s (elapsed: {elapsed:.1f}s). "
        f"Status history: {status_history}"
    )


def test_audit_job_with_mock_provider_runs_garak(
    client: NemoClient,
    workspace: str,
) -> None:
    """Create an audit job that runs Garak against the mock IGW provider.

    Flow:
    1. Create project, minimal audit config (one probe, few generations), and
       audit target pointing at the mock provider via nhx_uri_spec.
    2. Create audit job with that config and target.
    3. Wait for job to reach terminal state (completed or error).
    4. Assert job completed successfully and results can be listed.

    The auditor worker resolves nhx_uri_spec to the IGW URL (ModelsClient
    get_provider_route_openai_url). The job container runs on the Docker network
    and reaches IGW at nhx-quickstart:8080; the mock provider returns fixed
    chat completions so Garak probes complete without a real LLM.
    """
    pytest.skip("Auditor plugin does not expose remote audit job SDK routes yet.")

    auditor = GarakClient.from_client(client)
    project_name = short_unique_name("e2e-auditor-job-project")
    config_name = short_unique_name("e2e-auditor-job-config")
    target_name = short_unique_name("e2e-auditor-job-target")

    # Mock provider for Garak to talk to
    mock_response = {
        "id": "chatcmpl-auditor-mock",
        "object": "chat.completion",
        "created": 1234567890,
        "model": AUDITOR_MOCK_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, are you there?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    provider = add_mock_provider(
        client,
        workspace=workspace,
        name=short_unique_name("auditor-mock-provider"),
        mock_response_body_by_model={
            # workspace/model for model-entity route (e.g. Garak, guardrails-style)
            f"{workspace}/{AUDITOR_MOCK_MODEL}": [MockProviderResponse(response_body=mock_response)],
            # model only for provider route (request body model field, e.g. gateway.provider.post)
            AUDITOR_MOCK_MODEL: [MockProviderResponse(response_body=mock_response)],
        },
        served_models={AUDITOR_MOCK_MODEL: AUDITOR_MOCK_MODEL},
    )

    # Create project
    ProjectsClient.from_client(client).create_project(workspace=workspace, body=CreateProjectRequest(name=project_name))

    # Minimal config: one probe, few generations to keep runtime short
    auditor.create_audit_config(
        workspace=workspace,
        body=CreateAuditConfigRequest(
            name=config_name,
            system={"parallel_attempts": 4, "lite": True},
            run={"generations": 2},
            plugins={"probe_spec": "goodside.Tag"},
            reporting={},
        ),
    )

    # Target pointing at mock provider (worker will resolve nhx_uri_spec to IGW URL)
    auditor.create_audit_target(
        workspace=workspace,
        body=CreateAuditTargetRequest(
            name=target_name,
            type="nim",
            model=AUDITOR_MOCK_MODEL,
            options={
                "nim": {
                    "skip_seq_start": "<think>",
                    "skip_seq_end": "</think>",
                    "max_tokens": 100,
                    "nhx_uri_spec": {
                        "inference_gateway": {
                            "workspace": workspace,
                            "provider": provider.name,
                        }
                    },
                }
            },
        ),
    )

    # Submit the audit job
    job = auditor.submit_audit(
        workspace=workspace,
        body=SubmitAuditRequest(
            spec={
                "config": f"{workspace}/{config_name}",
                "target": f"{workspace}/{target_name}",
            }
        ),
    ).data()
    job_name = job.name
    assert job_name

    # Wait for terminal state
    status_dict = _wait_for_audit_job(client, job_name, workspace)
    status = (status_dict.get("status") or "").lower()

    if status != "completed":
        # Pull job logs to include in failure message for debugging
        logs_snippet = ""
        try:
            log_messages = [
                entry.message
                for entry in JobsClient.from_client(client).list_job_logs(name=job_name, workspace=workspace).items()
            ]
            if log_messages:
                max_lines = 300
                total = len(log_messages)
                if total > max_lines:
                    log_messages = log_messages[-max_lines:]
                    logs_snippet = f"\n--- Job logs (last {max_lines} of {total}) ---\n"
                else:
                    logs_snippet = "\n--- Job logs ---\n"
                logs_snippet += "\n".join(log_messages)
            else:
                logs_snippet = "\n--- Job logs: (none or empty) ---"
        except Exception as e:
            logs_snippet = f"\n--- Failed to fetch job logs: {e} ---"

        pytest.fail(
            f"Expected audit job to complete successfully, got status={status!r}. Response: {status_dict}{logs_snippet}"
        )

    # Verify results are available
    result_names = [
        r.name for r in JobsClient.from_client(client).list_job_results(name=job_name, workspace=workspace).items()
    ]
    assert "report-html" in result_names or len(result_names) >= 1, (
        f"Expected at least one result (e.g. report-html), got: {result_names}"
    )
