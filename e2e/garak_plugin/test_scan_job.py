# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s-only E2E tests for garak_plugin job submission.

These tests submit real scan jobs and poll for completion. They require:
  - ``NHX_BASE_URL`` pointing at a K8s deployment (set via ``container_only`` marker)
  - garak installed at ``/app/.garak_venv/bin/python`` in the nhx-garak-plugin-tasks image
  - mock inference provider support (``mock_provider_prefix: igw-mock-`` in Helm values)

The probe used is ``test.Test`` — garak's single-message blank probe, which is the
fastest possible smoke run and does not require a real safety-relevant model.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import suppress

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.garak_plugin.client import GarakPluginClient
from nemo_helix_plugin.garak_plugin.types import CreateScanConfigRequest, CreateScanTargetRequest, SubmitScanRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import add_mock_provider, short_unique_name

from e2e.garak_plugin.utils import minimal_scan_config, unique_name

pytestmark = [
    pytest.mark.container_only,
    pytest.mark.timeout(1800),
]

SCAN_JOB_TIMEOUT_SECONDS = 900.0
SCAN_JOB_POLL_INTERVAL_SECONDS = 10.0
TERMINAL_STATUSES = frozenset({"completed", "error", "failed", "cancelled"})


def _chat_completion(content: str = "I'm happy to help!") -> dict:
    return {
        "id": "chatcmpl-scan-e2e",
        "object": "chat.completion",
        "model": "scan-mock",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def _wait_for_scan_job(client: NemoClient, job_name: str, workspace: str) -> str:
    deadline = time.monotonic() + SCAN_JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_resp = JobsClient.from_client(client).get_job_status(name=job_name, workspace=workspace)
        status = str(status_resp.status)
        if status in TERMINAL_STATUSES:
            return status
        time.sleep(SCAN_JOB_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Scan job {job_name!r} did not complete within {SCAN_JOB_TIMEOUT_SECONDS}s")


def _cleanup_scan_job(client: NemoClient, job_name: str, workspace: str) -> None:
    jobs = JobsClient.from_client(client)
    with suppress(Exception):
        jobs.cancel_job(name=job_name, workspace=workspace)
    with suppress(Exception):
        jobs.delete_job(name=job_name, workspace=workspace)


def _submit_scan(client: NemoClient, workspace: str, *, config: dict | str, target: dict | str):
    """Submit a scan job spec (inline entities or ``workspace/name`` references)."""
    return (
        GarakPluginClient.from_client(client)
        .submit_scan(workspace=workspace, body=SubmitScanRequest(spec={"config": config, "target": target}))
        .data()
    )


def _add_mock_provider_or_skip(client: NemoClient, workspace: str, name: str) -> str:
    """Create a mock inference provider, skipping the test if the deployment doesn't support one."""
    try:
        provider = add_mock_provider(
            client,
            workspace=workspace,
            name=name,
            mock_response_body=_chat_completion(),
        )
        return provider.name
    except RuntimeError as exc:
        if "mock_provider_prefix is not configured" in str(exc):
            pytest.skip(
                "The running platform does not have mock-provider mode enabled. "
                "Set mock_provider_prefix: igw-mock- in Helm values (already present in "
                "e2e/k8s/values/minikube.yaml) to run this test."
            )
        raise


# ---- Module-scoped fixtures for the shared K8s workspace and mock provider ----


@pytest.fixture(scope="module")
def scan_workspace(client: NemoClient) -> Iterator[str]:
    workspaces = WorkspacesClient.from_client(client)
    name = short_unique_name("e2e-scan")
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    try:
        yield name
    finally:
        with suppress(Exception):
            workspaces.delete_workspace(name=name).data()


@pytest.fixture(scope="module")
def mock_provider_name(client: NemoClient, scan_workspace: str) -> str:
    """Create a canned-response mock provider for the module; workspace deletion cascades cleanup."""
    provider_name = short_unique_name("scan-mock")
    return _add_mock_provider_or_skip(client, scan_workspace, provider_name)


@pytest.fixture(scope="module")
def scan_config_name(client: NemoClient, scan_workspace: str) -> Iterator[str]:
    garak_plugin = GarakPluginClient.from_client(client)
    name = short_unique_name("e2e-scan-cfg")
    garak_plugin.create_scan_config(
        workspace=scan_workspace,
        body=CreateScanConfigRequest(
            name=name, **minimal_scan_config(plugins={"probe_spec": "test.Test", "detector_spec": "auto"})
        ),
    )
    try:
        yield name
    finally:
        with suppress(Exception):
            garak_plugin.delete_scan_config(workspace=scan_workspace, name=name)


@pytest.fixture(scope="module")
def scan_target_name(client: NemoClient, scan_workspace: str, mock_provider_name: str) -> Iterator[str]:
    garak_plugin = GarakPluginClient.from_client(client)
    name = short_unique_name("e2e-scan-tgt")
    garak_plugin.create_scan_target(
        workspace=scan_workspace,
        body=CreateScanTargetRequest(
            name=name,
            type="openai",
            model=mock_provider_name,
            options={
                "openai": {
                    "OpenAICompatible": {
                        "nhx_uri_spec": {
                            "inference_gateway": {
                                "workspace": scan_workspace,
                                "provider": mock_provider_name,
                            }
                        }
                    }
                }
            },
        ),
    )
    try:
        yield name
    finally:
        with suppress(Exception):
            garak_plugin.delete_scan_target(workspace=scan_workspace, name=name)


# ---- Tests ----


@pytest.mark.skip("re-enable after garak_plugin image rebuilt")
def test_scan_job_submit_blank_probe(
    client: NemoClient,
    scan_workspace: str,
    mock_provider_name: str,
) -> None:
    """Submit an inline scan job with test.Test probe and verify it reaches completed status."""
    config = {
        **minimal_scan_config(plugins={"probe_spec": "test.Test", "detector_spec": "auto"}),
        "name": unique_name("inline-cfg"),
        "workspace": scan_workspace,
    }
    target = {
        "name": unique_name("inline-tgt"),
        "workspace": scan_workspace,
        "type": "openai",
        "model": mock_provider_name,
        "options": {
            "openai": {
                "OpenAICompatible": {
                    "nhx_uri_spec": {
                        "inference_gateway": {
                            "workspace": scan_workspace,
                            "provider": mock_provider_name,
                        }
                    }
                }
            }
        },
    }

    job = _submit_scan(client, scan_workspace, config=config, target=target)
    job_name = job.name
    try:
        final_status = _wait_for_scan_job(client, job_name, scan_workspace)
        assert final_status == "completed", (
            f"Scan job {job_name!r} ended with status {final_status!r} instead of 'completed'. "
            "Check that garak is installed at /app/.garak_venv/bin/python in the nhx-garak-plugin-tasks image."
        )
    finally:
        _cleanup_scan_job(client, job_name, scan_workspace)


@pytest.mark.skip("re-enable after garak_plugin image rebuilt")
def test_scan_job_submit_with_entity_refs(
    client: NemoClient,
    scan_workspace: str,
    scan_config_name: str,
    scan_target_name: str,
) -> None:
    """Submit a scan job using stored entity name references and verify completion."""
    job = _submit_scan(
        client,
        scan_workspace,
        config=f"{scan_workspace}/{scan_config_name}",
        target=f"{scan_workspace}/{scan_target_name}",
    )
    job_name = job.name
    try:
        final_status = _wait_for_scan_job(client, job_name, scan_workspace)
        assert final_status == "completed", (
            f"Scan job {job_name!r} with entity refs ended with status {final_status!r}."
        )
    finally:
        _cleanup_scan_job(client, job_name, scan_workspace)


def test_scan_job_appears_in_list(
    client: NemoClient,
    scan_workspace: str,
    scan_config_name: str,
    scan_target_name: str,
) -> None:
    """Submitted scan job appears in list_jobs() with its name."""
    job = _submit_scan(
        client,
        scan_workspace,
        config=f"{scan_workspace}/{scan_config_name}",
        target=f"{scan_workspace}/{scan_target_name}",
    )
    job_name = job.name
    try:
        jobs = GarakPluginClient.from_client(client).list_scan_jobs(workspace=scan_workspace)
        job_names = [j.name for j in jobs.items()]
        assert job_name in job_names, f"Submitted job {job_name!r} not found in list_jobs(): {job_names}"
    finally:
        _cleanup_scan_job(client, job_name, scan_workspace)
