# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K8s-only E2E tests for auditor job submission.

These tests submit real audit jobs and poll for completion. They require:
  - ``NHX_BASE_URL`` pointing at a K8s deployment (set via ``container_only`` marker)
  - garak installed at ``/app/.garak_venv/bin/python`` in the nhx-auditor-tasks image
  - mock inference provider support (``mock_provider_prefix: igw-mock-`` in Helm values)

The probe used is ``test.Test`` — garak's single-message blank probe, which is the
fastest possible smoke run and does not require a real safety-relevant model.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import suppress

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import add_mock_provider, short_unique_name

from e2e.auditor.utils import minimal_audit_config, unique_name

pytestmark = [
    pytest.mark.container_only,
    pytest.mark.timeout(1800),
]

AUDIT_JOB_TIMEOUT_SECONDS = 900.0
AUDIT_JOB_POLL_INTERVAL_SECONDS = 10.0
TERMINAL_STATUSES = frozenset({"completed", "error", "failed", "cancelled"})


def _chat_completion(content: str = "I'm happy to help!") -> dict:
    return {
        "id": "chatcmpl-audit-e2e",
        "object": "chat.completion",
        "model": "audit-mock",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def _wait_for_audit_job(sdk: NeMoHelix, job_name: str, workspace: str) -> str:
    deadline = time.monotonic() + AUDIT_JOB_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_resp = client_from_platform(sdk, JobsClient).get_job_status(name=job_name, workspace=workspace)
        status = str(status_resp.status)
        if status in TERMINAL_STATUSES:
            return status
        time.sleep(AUDIT_JOB_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Audit job {job_name!r} did not complete within {AUDIT_JOB_TIMEOUT_SECONDS}s")


def _cleanup_audit_job(sdk: NeMoHelix, job_name: str, workspace: str) -> None:
    with suppress(Exception):
        jobs = client_from_platform(sdk, JobsClient)
        jobs.cancel_job(name=job_name, workspace=workspace)
    with suppress(Exception):
        jobs.delete_job(name=job_name, workspace=workspace)


def _add_mock_provider_or_skip(sdk: NeMoHelix, workspace: str, name: str) -> str:
    """Create a mock inference provider, skipping the test if the deployment doesn't support one."""
    try:
        provider = add_mock_provider(
            sdk,
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
def audit_workspace(sdk: NeMoHelix) -> Iterator[str]:
    workspaces = client_from_platform(sdk, WorkspacesClient)
    name = short_unique_name("e2e-audit")
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    try:
        yield name
    finally:
        with suppress(Exception):
            workspaces.delete_workspace(name=name).data()


@pytest.fixture(scope="module")
def mock_provider_name(sdk: NeMoHelix, audit_workspace: str) -> str:
    """Create a canned-response mock provider for the module; workspace deletion cascades cleanup."""
    provider_name = short_unique_name("audit-mock")
    return _add_mock_provider_or_skip(sdk, audit_workspace, provider_name)


@pytest.fixture(scope="module")
def audit_config_name(sdk: NeMoHelix, audit_workspace: str) -> Iterator[str]:
    name = short_unique_name("e2e-audit-cfg")
    sdk.auditor.configs.create(
        workspace=audit_workspace,
        name=name,
        **minimal_audit_config(plugins={"probe_spec": "test.Test", "detector_spec": "auto"}),
    )
    try:
        yield name
    finally:
        with suppress(Exception):
            sdk.auditor.configs.delete(workspace=audit_workspace, name=name)


@pytest.fixture(scope="module")
def audit_target_name(sdk: NeMoHelix, audit_workspace: str, mock_provider_name: str) -> Iterator[str]:
    name = short_unique_name("e2e-audit-tgt")
    sdk.auditor.targets.create(
        workspace=audit_workspace,
        name=name,
        type="openai",
        model=mock_provider_name,
        options={
            "openai": {
                "OpenAICompatible": {
                    "nhx_uri_spec": {
                        "inference_gateway": {
                            "workspace": audit_workspace,
                            "provider": mock_provider_name,
                        }
                    }
                }
            }
        },
    )
    try:
        yield name
    finally:
        with suppress(Exception):
            sdk.auditor.targets.delete(workspace=audit_workspace, name=name)


# ---- Tests ----


@pytest.mark.skip("re-enable after auditor image rebuilt")
def test_audit_job_submit_blank_probe(
    sdk: NeMoHelix,
    audit_workspace: str,
    mock_provider_name: str,
) -> None:
    """Submit an inline audit job with test.Test probe and verify it reaches completed status."""
    config = {
        **minimal_audit_config(plugins={"probe_spec": "test.Test", "detector_spec": "auto"}),
        "name": unique_name("inline-cfg"),
        "workspace": audit_workspace,
    }
    target = {
        "name": unique_name("inline-tgt"),
        "workspace": audit_workspace,
        "type": "openai",
        "model": mock_provider_name,
        "options": {
            "openai": {
                "OpenAICompatible": {
                    "nhx_uri_spec": {
                        "inference_gateway": {
                            "workspace": audit_workspace,
                            "provider": mock_provider_name,
                        }
                    }
                }
            }
        },
    }

    job = sdk.auditor.submit(config=config, target=target, workspace=audit_workspace)
    job_name = job.name
    try:
        final_status = _wait_for_audit_job(sdk, job_name, audit_workspace)
        assert final_status == "completed", (
            f"Audit job {job_name!r} ended with status {final_status!r} instead of 'completed'. "
            "Check that garak is installed at /app/.garak_venv/bin/python in the nhx-auditor-tasks image."
        )
    finally:
        _cleanup_audit_job(sdk, job_name, audit_workspace)


@pytest.mark.skip("re-enable after auditor image rebuilt")
def test_audit_job_submit_with_entity_refs(
    sdk: NeMoHelix,
    audit_workspace: str,
    audit_config_name: str,
    audit_target_name: str,
) -> None:
    """Submit an audit job using stored entity name references and verify completion."""
    job = sdk.auditor.submit(
        config=f"{audit_workspace}/{audit_config_name}",
        target=f"{audit_workspace}/{audit_target_name}",
        workspace=audit_workspace,
    )
    job_name = job.name
    try:
        final_status = _wait_for_audit_job(sdk, job_name, audit_workspace)
        assert final_status == "completed", (
            f"Audit job {job_name!r} with entity refs ended with status {final_status!r}."
        )
    finally:
        _cleanup_audit_job(sdk, job_name, audit_workspace)


def test_audit_job_appears_in_list(
    sdk: NeMoHelix,
    audit_workspace: str,
    audit_config_name: str,
    audit_target_name: str,
) -> None:
    """Submitted audit job appears in list_jobs() with its name."""
    job = sdk.auditor.submit(
        config=f"{audit_workspace}/{audit_config_name}",
        target=f"{audit_workspace}/{audit_target_name}",
        workspace=audit_workspace,
    )
    job_name = job.name
    try:
        jobs = sdk.auditor.list_jobs(workspace=audit_workspace)
        job_names = [j["name"] for j in jobs.get("data", [])]
        assert job_name in job_names, f"Submitted job {job_name!r} not found in list_jobs(): {job_names}"
    finally:
        _cleanup_audit_job(sdk, job_name, audit_workspace)
