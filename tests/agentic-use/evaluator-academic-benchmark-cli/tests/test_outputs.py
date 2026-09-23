# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up an academic benchmark evaluation job via CLI.

Tests workspace creation, benchmark discovery, and benchmark job creation.
Note: Job execution (completion, results) is not tested because the
quickstart environment does not include the job execution worker.
"""

import os

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.workspaces.client import WorkspacesClient

WORKSPACE = "benchmark-eval-workspace"


def _get_client() -> NeMoHelix:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url)


def test_workspace_exists():
    """Verify the benchmark-eval-workspace was created."""
    client = _get_client()
    response = client_from_platform(client, WorkspacesClient).list_workspaces()
    workspace_names = [ws.name for ws in response.items()]
    assert WORKSPACE in workspace_names, f"Workspace '{WORKSPACE}' not found. Found: {workspace_names}"


def test_benchmark_job_created():
    """Verify that at least one benchmark evaluation job was created."""
    client = _get_client()
    jobs = client.evaluation.benchmark_jobs.list(workspace=WORKSPACE)
    assert len(jobs.data) > 0, f"No benchmark evaluation jobs found in workspace '{WORKSPACE}'"


def test_benchmark_job_has_spec():
    """Verify the benchmark job has a valid spec with benchmark and model."""
    client = _get_client()
    jobs = client.evaluation.benchmark_jobs.list(workspace=WORKSPACE)
    assert len(jobs.data) > 0, "No benchmark jobs found"

    job = jobs.data[0]
    job_detail = client.evaluation.benchmark_jobs.retrieve(job.name, workspace=WORKSPACE)

    assert job_detail.spec is not None, "Job has no spec"

    # The spec should have a benchmark reference
    spec = job_detail.spec
    assert hasattr(spec, "benchmark"), f"Job spec missing 'benchmark' field. Spec type: {type(spec).__name__}"
    assert spec.benchmark is not None, "Job spec benchmark is None"

    # The benchmark reference should point to an MMLU-related system benchmark
    benchmark_ref = str(spec.benchmark)
    assert "mmlu" in benchmark_ref.lower(), f"Expected an MMLU-based benchmark reference, got: {benchmark_ref}"


def test_benchmark_job_has_model():
    """Verify the benchmark job spec references a model."""
    client = _get_client()
    jobs = client.evaluation.benchmark_jobs.list(workspace=WORKSPACE)
    assert len(jobs.data) > 0, "No benchmark jobs found"

    job = jobs.data[0]
    job_detail = client.evaluation.benchmark_jobs.retrieve(job.name, workspace=WORKSPACE)

    spec = job_detail.spec
    assert hasattr(spec, "model"), f"Job spec missing 'model' field. Spec type: {type(spec).__name__}"
    assert spec.model is not None, "Job spec model is None"
