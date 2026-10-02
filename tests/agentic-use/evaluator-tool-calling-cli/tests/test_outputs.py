# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up a BFCL-style tool calling evaluation via CLI.

Covers the operations from the Linear ticket:
1. Prepare BFCL-format evaluation dataset - test_fileset_exists, test_fileset_has_data
2. Configure tool calling evaluation - test_tool_calling_metric_exists, test_tool_calling_metric_config
3. Run evaluation against model - test_sync_evaluation_produces_scores
4. Verify metrics: function_name_accuracy, function_name_and_args_accuracy - test_sync_evaluation_produces_scores
"""

import base64
import json
import os
import sys

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
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


def _get_client() -> NeMoHelix:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE, access_token=_make_unsigned_jwt())


def _get_files_client() -> FilesClient:
    return client_from_platform(_get_client(), FilesClient)


# --- Workspace checks ---


def test_workspace_exists():
    """Verify the tool-calling-eval-workspace was created."""
    client = _get_client()
    response = client_from_platform(client, WorkspacesClient).list_workspaces()
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
    files = client.files.list(fileset=FILESET)
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


# --- Tool Calling Metric checks ---


def test_tool_calling_metric_exists():
    """Verify a tool-calling metric was created."""
    client = _get_client()
    response = client.evaluation.metrics.list()
    metrics = response.data
    tc_metrics = [m for m in metrics if m.type == "tool-calling"]
    assert len(tc_metrics) > 0, (
        f"No tool-calling metrics found in workspace '{WORKSPACE}'. Found metric types: {[m.type for m in metrics]}"
    )


def test_tool_calling_metric_named_correctly():
    """Verify the metric is named tool-calling-accuracy."""
    client = _get_client()
    response = client.evaluation.metrics.list()
    metric_names = [m.name for m in response.data]
    assert METRIC_NAME in metric_names, (
        f"Metric '{METRIC_NAME}' not found in workspace '{WORKSPACE}'. Found: {metric_names}"
    )


def test_tool_calling_metric_config():
    """Verify the metric has correct type and reference field."""
    client = _get_client()
    metric = client.evaluation.metrics.retrieve(name=METRIC_NAME)

    assert metric.type == "tool-calling", f"Metric type should be 'tool-calling', got: {metric.type}"
    assert metric.reference is not None, "Metric reference (ground truth template) is None"
    assert len(metric.reference) > 0, "Metric reference is empty"


# --- Synchronous Evaluation checks ---


@pytest.mark.timeout(30)
def test_sync_evaluation_produces_scores():
    """Verify the metric produces function_name_accuracy and function_name_and_args_accuracy scores.

    Runs the agent's metric against a small inline dataset with known
    correct and incorrect tool call responses to validate scoring.
    """
    client = _get_client()

    # Row 0: correct tool call (should score 1.0 on both metrics)
    # Row 1: wrong function name (should score 0.0 on both metrics)
    # Row 2: correct function name but wrong arguments (should score 1.0 on name, 0.0 on name+args)
    response = client.evaluation.metrics.evaluate(
        metric=f"{WORKSPACE}/{METRIC_NAME}",
        dataset={
            "rows": [
                {
                    "expected_tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}],
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": "NYC"}',
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                },
                {
                    "expected_tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "LA"}}}],
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "wrong_function",
                                                "arguments": '{"city": "LA"}',
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                },
                {
                    "expected_tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "SF"}}}],
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": "wrong_city"}',
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                },
            ]
        },
    )

    assert len(response.row_scores) == 3, f"Expected 3 row scores, got {len(response.row_scores)}"

    # Check that both expected metric names are present in aggregate scores
    agg_names = {s.name for s in response.aggregate_scores}
    assert "function_name_accuracy" in agg_names, f"'function_name_accuracy' not in aggregate scores. Got: {agg_names}"
    assert "function_name_and_args_accuracy" in agg_names, (
        f"'function_name_and_args_accuracy' not in aggregate scores. Got: {agg_names}"
    )

    # Check per-row scores exist
    for row_score in response.row_scores:
        assert row_score.scores is not None, f"Row {row_score.index} produced no scores (error: {row_score.error})"
        assert "function_name_accuracy" in row_score.scores, (
            f"Row {row_score.index} missing 'function_name_accuracy'. Got: {list(row_score.scores.keys())}"
        )
        assert "function_name_and_args_accuracy" in row_score.scores, (
            f"Row {row_score.index} missing 'function_name_and_args_accuracy'. Got: {list(row_score.scores.keys())}"
        )

    # Row 0 (correct match) should score 1.0
    row0_scores = response.row_scores[0].scores
    assert row0_scores["function_name_accuracy"] == 1.0, (
        f"Correct match should have function_name_accuracy=1.0, got {row0_scores['function_name_accuracy']}"
    )
    assert row0_scores["function_name_and_args_accuracy"] == 1.0, (
        f"Correct match should have function_name_and_args_accuracy=1.0, got {row0_scores['function_name_and_args_accuracy']}"
    )

    # Row 1 (wrong function name) should score 0.0 on both
    row1_scores = response.row_scores[1].scores
    assert row1_scores["function_name_accuracy"] == 0.0, (
        f"Wrong function name should have function_name_accuracy=0.0, got {row1_scores['function_name_accuracy']}"
    )
    assert row1_scores["function_name_and_args_accuracy"] == 0.0, (
        f"Wrong function name should have function_name_and_args_accuracy=0.0, got {row1_scores['function_name_and_args_accuracy']}"
    )

    # Row 2 (correct name, wrong args) should score 1.0 on name but 0.0 on name+args
    row2_scores = response.row_scores[2].scores
    assert row2_scores["function_name_accuracy"] == 1.0, (
        f"Correct function name should have function_name_accuracy=1.0, got {row2_scores['function_name_accuracy']}"
    )
    assert row2_scores["function_name_and_args_accuracy"] == 0.0, (
        f"Wrong arguments should have function_name_and_args_accuracy=0.0, got {row2_scores['function_name_and_args_accuracy']}"
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
    response = client.evaluation.metric_jobs.list()
    assert len(response.data) > 0, f"No evaluation metric jobs found in workspace '{WORKSPACE}'"

    job = response.data[0]
    assert job.spec is not None, "Job has no spec"
