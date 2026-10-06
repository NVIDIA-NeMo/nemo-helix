# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up a zero-config LLM-as-a-Judge evaluation via CLI.

Zero-config means the agent provided only model + scores (with rubric) and did NOT
supply a custom prompt_template or explicit parsers. The system auto-generates:
  - A default judge prompt template based on score definitions
  - Default JSON parsers for each score
  - Default structured output schema from rubric scores

Covers the 4 operations from the Linear ticket:
1. Provide minimal configuration (dataset, target model) -
     test_fileset_exists, test_fileset_has_data
2. Let system use default judge and criteria -
     test_llm_judge_metric_exists
3. Run evaluation - test_agent_ran_sync_eval_and_examined_scores (trace)
4. Retrieve results - test_agent_ran_sync_eval_and_examined_scores (trace)

Provider, workspace, and secret infrastructure are pre-configured
in the Dockerfile so the agent can focus on evaluator operations.
"""

import base64
import json
import os
import sys

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.evaluator.client import EvaluatorClient
from nemo_helix_plugin.files.client import FilesClient

sys.path.insert(0, "/tests/shared")
from trace_reader import get_session

WORKSPACE = "eval-zeroconfig-workspace"
FILESET = "zeroconfig-dataset"
METRIC_NAME = "zeroconfig-judge"
SCORE_NAME = "quality"


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
    """Verify the zeroconfig-dataset fileset was created."""
    files_client = _get_files_client()
    fileset_names = [fs.name for fs in files_client.list_filesets().page().items]
    assert FILESET in fileset_names, f"Fileset '{FILESET}' not found. Found: {fileset_names}"


def test_fileset_has_data() -> None:
    """Verify the dataset fileset has files uploaded."""
    client = _get_nhx_client()
    files = FilesClient.from_client(client).list_files(name=FILESET, workspace=WORKSPACE).data()
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


# --- Zero-Config LLM Judge Metric checks ---


def test_llm_judge_metric_exists() -> None:
    """Verify an LLM-as-a-Judge metric named zeroconfig-judge was created."""
    client = _get_nhx_client()
    metrics = EvaluatorClient.from_client(client).list_metrics(workspace=WORKSPACE).page().items
    metric_names = [m.name for m in metrics]
    assert METRIC_NAME in metric_names, (
        f"Metric '{METRIC_NAME}' not found in workspace '{WORKSPACE}'. Found: {metric_names}"
    )


# --- Trace checks: agent retrieved scored results ---


def test_agent_ran_sync_eval_and_examined_scores() -> None:
    """Verify the agent ran a sync evaluation and examined the scored results.

    This checks the agent actually ran the evaluation command and
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
