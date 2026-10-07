# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up an LLM-as-a-Judge evaluation via CLI.

Covers the operations from the Linear ticket:
1. Configure judge model and rubric - test_llm_judge_metric_exists, test_llm_judge_metric_named_correctly
2. Prepare dataset with model outputs - test_fileset_exists, test_fileset_has_data
3. Launch LLM-as-a-Judge job - test_evaluation_job_created
4. Retrieve scored results - test_agent_ran_sync_eval_and_examined_scores (trace)

Provider, workspace, and secret infrastructure are pre-configured
in the Dockerfile so the agent can focus on evaluator operations.
"""

import base64
import json
import os
import sys

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.evals.client import EvaluatorClient
from nemo_helix_plugin.files.client import FilesClient

sys.path.insert(0, "/tests/shared")
from trace_reader import get_session

WORKSPACE = "eval-judge-workspace"
FILESET = "judge-eval-dataset"
METRIC_NAME = "quality-judge"
SCORE_NAME = "relevance"


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


def _get_nhx_client() -> NemoClient:
    """Get typed client for the eval workspace."""
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NemoClient(base_url=nhx_base_url, workspace=WORKSPACE, auth=_make_unsigned_jwt())


def _get_files_client() -> FilesClient:
    return FilesClient.from_client(_get_nhx_client())


# --- Dataset checks ---


def test_fileset_exists() -> None:
    """Verify the judge-eval-dataset fileset was created."""
    files_client = _get_files_client()
    fileset_names = [fs.name for fs in files_client.list_filesets().page().items]
    assert FILESET in fileset_names, f"Fileset '{FILESET}' not found. Found: {fileset_names}"


def test_fileset_has_data() -> None:
    """Verify the dataset fileset has files uploaded."""
    client = _get_nhx_client()
    files = FilesClient.from_client(client).list_files(name=FILESET, workspace=WORKSPACE).data()
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


# --- LLM Judge Metric checks ---


def test_llm_judge_metric_exists() -> None:
    """Verify an LLM-as-a-Judge metric was created."""
    client = _get_nhx_client()
    metrics = EvaluatorClient.from_client(client).list_metrics(workspace=WORKSPACE).page().items
    judge_metrics = [m for m in metrics if getattr(m, "metric_type", None) == "llm-judge"]
    assert len(judge_metrics) > 0, (
        f"No llm-judge metrics found in workspace '{WORKSPACE}'. Found metric types: {[getattr(m, 'metric_type', None) for m in metrics]}"
    )


def test_llm_judge_metric_named_correctly() -> None:
    """Verify the metric is named quality-judge."""
    client = _get_nhx_client()
    metrics = EvaluatorClient.from_client(client).list_metrics(workspace=WORKSPACE).page().items
    metric_names = [m.name for m in metrics]
    assert METRIC_NAME in metric_names, (
        f"Metric '{METRIC_NAME}' not found in workspace '{WORKSPACE}'. Found: {metric_names}"
    )


# --- Trace checks: agent retrieved scored results ---


def test_agent_ran_sync_eval_and_examined_scores() -> None:
    """Verify the agent ran a sync evaluation and examined the scored results.

    This checks Linear ticket step 5 (Retrieve scored results) by reading
    the agent's session trace to confirm it ran the evaluation command and
    the output contained actual score data.
    """
    session = get_session()
    commands = session.get_bash_commands()

    # Agent should have run a sync evaluation command
    eval_commands = [cmd for cmd in commands if "evaluation" in cmd and "evaluate" in cmd]
    assert len(eval_commands) > 0, f"Agent never ran an evaluation command. Commands: {commands}"

    # Check the tool results for the eval command to verify scores were returned
    bash_results = session.get_tool_results("Bash")
    score_results = [r for r in bash_results if SCORE_NAME in r.content and not r.is_error]
    assert len(score_results) > 0, (
        f"Agent's evaluation output did not contain '{SCORE_NAME}' scores. "
        "The agent should have retrieved and examined the scored results."
    )


# --- Evaluation Job checks ---


def test_evaluation_job_created() -> None:
    """Verify that at least one evaluation metric job was created."""
    client = _get_nhx_client()
    jobs = EvaluatorClient.from_client(client).list_evaluate_jobs(workspace=WORKSPACE).page().items
    assert len(jobs) > 0, f"No evaluation metric jobs found in workspace '{WORKSPACE}'"

    job = jobs[0]
    assert job.spec is not None, "Job has no spec"
    assert METRIC_NAME in str(job.spec), f"Job spec should reference '{METRIC_NAME}'"
