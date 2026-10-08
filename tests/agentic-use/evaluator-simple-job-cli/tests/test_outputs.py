# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up an evaluation job via CLI.

Tests workspace/fileset creation, dataset upload, metric creation, and job creation.
Note: Job execution (completion, results) is not tested because the
quickstart environment does not include the job execution worker.
"""

import os

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.evals.client import EvaluatorClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient

WORKSPACE = "eval-test-workspace"
FILESET = "eval-dataset"


def _get_client() -> NemoClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NemoClient(base_url=nhx_base_url)


def _get_files_client() -> FilesClient:
    return FilesClient.from_client(_get_client())


def test_workspace_exists():
    """Verify the eval-test-workspace was created."""
    client = _get_client()
    response = WorkspacesClient.from_client(client).list_workspaces()
    workspace_names = [ws.name for ws in response.items()]
    assert WORKSPACE in workspace_names, f"Workspace '{WORKSPACE}' not found. Found: {workspace_names}"


def test_fileset_exists():
    """Verify the eval-dataset fileset was created."""
    files_client = _get_files_client()
    fileset_names = [fs.name for fs in files_client.list_filesets(workspace=WORKSPACE).page().items]
    assert FILESET in fileset_names, f"Fileset '{FILESET}' not found. Found: {fileset_names}"


def test_fileset_has_data():
    """Verify the dataset was uploaded to the fileset."""
    client = _get_client()
    files = FilesClient.from_client(client).list_files(name=FILESET, workspace=WORKSPACE).data()
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


def test_metric_created():
    """Verify a string-check metric was created in the workspace."""
    client = _get_client()
    metrics = EvaluatorClient.from_client(client).list_metrics(workspace=WORKSPACE).page().items
    string_check_metrics = [m for m in metrics if getattr(m, "metric_type", None) == "string-check"]
    assert len(string_check_metrics) > 0, (
        f"No string-check metrics found in workspace '{WORKSPACE}'. Found metric types: {[getattr(m, 'metric_type', None) for m in metrics]}"
    )


def test_evaluation_job_created():
    """Verify that at least one evaluation metric job was created."""
    client = _get_client()
    jobs = EvaluatorClient.from_client(client).list_evaluate_jobs(workspace=WORKSPACE).page().items
    assert len(jobs) > 0, "No evaluation jobs found"

    job = jobs[0]
    assert job.spec is not None, "Job has no spec"
