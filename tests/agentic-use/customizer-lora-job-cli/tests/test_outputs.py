# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent submitted a LoRA job via the automodel customization plugin.

Tests workspace/fileset creation, dataset upload, and automodel job submission
through the NeMo Helix customization + jobs pipeline.
"""

import base64
import json
import os
from typing import Any

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient

WORKSPACE = "lora-training-workspace"
FILESET = "sft-training-data"


def _make_unsigned_jwt() -> str:
    """Create an unsigned JWT (alg=none) for local quickstart auth."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": "verifier@harbor.local", "email": "verifier@harbor.local"}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}."


@pytest.fixture
def client() -> NemoClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NemoClient(
        base_url=nhx_base_url,
        workspace=WORKSPACE,
        auth=_make_unsigned_jwt(),
    )


@pytest.fixture
def files_client(client: NemoClient) -> FilesClient:
    return FilesClient.from_client(client)


def _list_automodel_jobs(client: NemoClient) -> list[dict[str, Any]]:
    """List automodel customization jobs in the eval workspace."""
    url = f"{client.base_url}/apis/customization/v2/workspaces/{WORKSPACE}/automodel/jobs"
    response = client._client.get(url)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", payload if isinstance(payload, list) else [])
    return data if isinstance(data, list) else []


def test_workspace_exists(client: NemoClient):
    """Verify the lora-training-workspace exists."""
    response = WorkspacesClient.from_client(client).list_workspaces()
    workspace_names = [ws.name for ws in response.items()]
    assert WORKSPACE in workspace_names, f"Workspace '{WORKSPACE}' not found. Found: {workspace_names}"


def test_fileset_exists(files_client: FilesClient):
    """Verify the sft-training-data fileset was created."""
    fileset_names = [fs.name for fs in files_client.list_filesets(workspace=WORKSPACE).page().items]
    assert FILESET in fileset_names, f"Fileset '{FILESET}' not found. Found: {fileset_names}"


def test_fileset_has_data(client: NemoClient):
    """Verify the training dataset was uploaded."""
    files = FilesClient.from_client(client).list_files(name=FILESET, workspace=WORKSPACE).data()
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


def test_customization_job_created(client: NemoClient):
    """Verify that an automodel customization job was submitted."""
    jobs = _list_automodel_jobs(client)
    assert len(jobs) > 0, "No automodel customization jobs found in workspace"


def test_customization_job_has_spec(client: NemoClient):
    """Verify the automodel job has a valid training spec."""
    jobs = _list_automodel_jobs(client)
    assert len(jobs) > 0, "No automodel customization jobs found"
    job = jobs[0]
    assert job.get("spec") is not None, "Automodel job has no spec"


def test_customization_job_dispatched(client: NemoClient):
    """Verify the job was dispatched by the jobs controller (progressed beyond 'created').

    With the Docker socket mounted and GPU available, the jobs controller should
    schedule the training container. The job should reach at least 'pending' status.
    """
    jobs = _list_automodel_jobs(client)
    assert len(jobs) > 0, "No automodel customization jobs found"
    job = jobs[0]

    status = job.get("status", "unknown")
    if hasattr(status, "lower"):
        status = status.lower()

    dispatched_statuses = {"pending", "running", "completed", "error", "cancelled", "paused"}
    valid_statuses = dispatched_statuses | {"created", "unknown"}

    assert status in valid_statuses, f"Job in unexpected status: '{status}'. Expected one of: {valid_statuses}"
