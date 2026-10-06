# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up a BFCL-style tool calling evaluation via CLI.

Covers the operations from the Linear ticket:
1. Prepare BFCL-format evaluation dataset - test_fileset_exists, test_fileset_has_data
2. Configure tool calling evaluation - test_tool_calling_metric_exists
3. Run evaluation against model - test_agent_ran_sync_eval_and_examined_scores (trace)
4. Verify metrics: function_name_accuracy, function_name_and_args_accuracy - test_agent_ran_sync_eval_and_examined_scores
"""

import base64
import json
import os
import sys

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.evaluator.client import EvaluatorClient
from nemo_helix_plugin.files.client import FilesClient

sys.path.insert(0, "/tests/shared")
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from trace_reader import get_session

WORKSPACE = "tool-calling-eval-workspace"
FILESET = "tool-calling-dataset"
METRIC_NAME = "tool-calling-accuracy"


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


def _get_client() -> NemoClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NemoClient(base_url=nhx_base_url, workspace=WORKSPACE, auth=_make_unsigned_jwt())


def _get_files_client() -> FilesClient:
    return FilesClient.from_client(_get_client())


# --- Workspace checks ---


def test_workspace_exists():
    """Verify the tool-calling-eval-workspace was created."""
    client = _get_client()
    response = WorkspacesClient.from_client(client).list_workspaces()
    workspace_names = [ws.name for ws in response.items()]
    assert WORKSPACE in workspace_names, f"Workspace '{WORKSPACE}' not found. Found: {workspace_names}"


# --- Dataset checks ---


def test_fileset_exists():
    """Verify the tool-calling-dataset fileset was created."""
    files_client = _get_files_client()
    fileset_names = [fs.name for fs in files_client.list_filesets().page().items]
    assert FILESET in fileset_names, f"Fileset '{FILESET}' not found. Found: {fileset_names}"


def test_fileset_has_data():
    """Verify the dataset fileset has files uploaded."""
    client = _get_client()
    files = FilesClient.from_client(client).list_files(name=FILESET, workspace=WORKSPACE).data()
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


# --- Tool Calling Metric checks ---


def test_tool_calling_metric_exists():
    """Verify a tool-calling metric was created."""
    client = _get_client()
    metrics = EvaluatorClient.from_client(client).list_metrics(workspace=WORKSPACE).page().items
    tc_metrics = [m for m in metrics if getattr(m, "metric_type", None) == "tool-calling"]
    assert len(tc_metrics) > 0, (
        f"No tool-calling metrics found in workspace '{WORKSPACE}'. Found metric types: {[getattr(m, 'metric_type', None) for m in metrics]}"
    )


def test_tool_calling_metric_named_correctly():
    """Verify the metric is named tool-calling-accuracy."""
    client = _get_client()
    metrics = EvaluatorClient.from_client(client).list_metrics(workspace=WORKSPACE).page().items
    metric_names = [m.name for m in metrics]
    assert METRIC_NAME in metric_names, (
        f"Metric '{METRIC_NAME}' not found in workspace '{WORKSPACE}'. Found: {metric_names}"
    )


# --- Trace checks: agent ran sync eval and examined scores ---


def test_agent_ran_sync_eval_and_examined_scores():
    """Verify the agent ran a sync evaluation and examined the scored results.

    Checks that the agent ran the evaluation command and the output
    contained function_name_accuracy scores.
    """
    session = get_session()
    commands = session.get_bash_commands()

    eval_commands = [cmd for cmd in commands if "evaluation" in cmd and "evaluate" in cmd]
    assert len(eval_commands) > 0, f"Agent never ran an evaluation command. Commands: {commands}"

    bash_results = session.get_tool_results("Bash")
    score_results = [r for r in bash_results if "function_name_accuracy" in r.content and not r.is_error]
    assert len(score_results) > 0, (
        "Agent's evaluation output did not contain 'function_name_accuracy' scores. "
        "The agent should have retrieved and examined the scored results."
    )


# --- Evaluation Job checks ---


def test_evaluation_job_created():
    """Verify that at least one evaluation metric job was created."""
    client = _get_client()
    jobs = EvaluatorClient.from_client(client).list_evaluate_jobs(workspace=WORKSPACE).page().items
    assert len(jobs) > 0, f"No evaluation metric jobs found in workspace '{WORKSPACE}'"

    job = jobs[0]
    assert job.spec is not None, "Job has no spec"
