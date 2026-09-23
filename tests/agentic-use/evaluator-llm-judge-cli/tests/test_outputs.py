# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up an LLM-as-a-Judge evaluation via CLI.

Covers all 5 operations from the Linear ticket:
1. Configure judge model (via IGW) - test_llm_judge_metric_config (model URL, format, api_key_secret)
2. Define evaluation rubric/criteria - test_llm_judge_metric_config (scores, prompt_template)
3. Prepare dataset with model outputs - test_fileset_exists, test_fileset_has_data
4. Launch LLM-as-a-Judge job - test_evaluation_job_created
5. Retrieve scored results - test_agent_ran_sync_eval_and_examined_scores (trace),
                             test_sync_evaluation_produces_scores (verifier re-runs)

Provider, workspace, and secret infrastructure are pre-configured
in the Dockerfile so the agent can focus on evaluator operations.
"""

import base64
import json
import os
import sys
from urllib.parse import urlparse

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
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


def _get_nhx_client() -> NeMoHelix:
    """Get NeMoHelix client for the eval workspace."""
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE, access_token=_make_unsigned_jwt())


def _get_files_client() -> FilesClient:
    return client_from_platform(_get_nhx_client(), FilesClient)


# --- Dataset checks ---


def test_fileset_exists() -> None:
    """Verify the judge-eval-dataset fileset was created."""
    files_client = _get_files_client()
    fileset_names = [fs.name for fs in files_client.list_filesets().page().items]
    assert FILESET in fileset_names, f"Fileset '{FILESET}' not found. Found: {fileset_names}"


def test_fileset_has_data() -> None:
    """Verify the dataset fileset has files uploaded."""
    client = _get_nhx_client()
    files = client.files._list_files(name=FILESET)
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


# --- LLM Judge Metric checks ---


def test_llm_judge_metric_exists() -> None:
    """Verify an LLM-as-a-Judge metric was created."""
    client = _get_nhx_client()
    response = client.evaluation.metrics.list()
    metrics = response.data
    judge_metrics = [m for m in metrics if m.type == "llm-judge"]
    assert len(judge_metrics) > 0, (
        f"No llm-judge metrics found in workspace '{WORKSPACE}'. Found metric types: {[m.type for m in metrics]}"
    )


def test_llm_judge_metric_named_correctly() -> None:
    """Verify the metric is named quality-judge."""
    client = _get_nhx_client()
    response = client.evaluation.metrics.list()
    metric_names = [m.name for m in response.data]
    assert METRIC_NAME in metric_names, (
        f"Metric '{METRIC_NAME}' not found in workspace '{WORKSPACE}'. Found: {metric_names}"
    )


def test_llm_judge_metric_config() -> None:
    """Verify the metric has correct model config, scores, and prompt template.

    This validates the agent configured the judge model (step 1) and
    defined the evaluation rubric/criteria (step 2) from the Linear ticket.
    """
    client = _get_nhx_client()
    metric = client.evaluation.metrics.retrieve(name=METRIC_NAME)

    # Model config: judge model pointing to inference endpoint
    assert hasattr(metric, "model"), "Metric has no model config"
    model = metric.model
    assert hasattr(model, "url"), f"Model is a reference string, expected inline config: {model}"
    parsed_model_url = urlparse(model.url)
    trusted_igw_host = urlparse(os.environ.get("NHX_BASE_URL", "http://localhost:8080")).hostname
    valid_url = (
        parsed_model_url.hostname == trusted_igw_host and "inference/providers" in (parsed_model_url.path or "")
    ) or parsed_model_url.hostname == "inference-api.nvidia.com"
    assert valid_url, f"Model URL should use IGW provider proxy or NVIDIA inference API, got: {model.url}"
    assert model.api_key_secret == "nvidia-api-key", (
        f"Model api_key_secret should be 'nvidia-api-key', got: {model.api_key_secret}"
    )
    assert model.format == "openai", f"Model format should be 'openai', got: {model.format}"

    # Scores: at least one score named 'relevance' with 1-5 range
    assert len(metric.scores) > 0, "Metric has no scores defined"
    score_names = [s.name for s in metric.scores]
    assert SCORE_NAME in score_names, f"Expected score '{SCORE_NAME}' not found. Got: {score_names}"
    relevance_score = next(s for s in metric.scores if s.name == SCORE_NAME)
    assert relevance_score.minimum == 1, f"Score minimum should be 1, got: {relevance_score.minimum}"
    assert relevance_score.maximum == 5, f"Score maximum should be 5, got: {relevance_score.maximum}"

    # Prompt template: should have messages array
    assert metric.prompt_template is not None, "Metric has no prompt_template"
    assert isinstance(metric.prompt_template, dict), (
        f"prompt_template should be a dict (messages format), got: {type(metric.prompt_template)}"
    )
    assert "messages" in metric.prompt_template, (
        f"prompt_template should have 'messages' key, got: {list(metric.prompt_template.keys())}"
    )


# --- Synchronous Evaluation checks ---


@pytest.mark.timeout(90)
def test_sync_evaluation_produces_scores() -> None:
    """Verify the metric produces valid relevance scores via synchronous evaluation.

    This is the key integration test: it runs the agent's metric against
    a small inline dataset and checks that the judge LLM returns real scores.
    """
    client = _get_nhx_client()
    response = client.evaluation.metrics.evaluate(
        metric=f"{WORKSPACE}/{METRIC_NAME}",
        dataset={
            "rows": [
                {"input": "What is 2+2?", "output": "2+2 equals 4."},
                {"input": "Explain quantum physics.", "output": "I have no idea whatsoever."},
            ]
        },
    )

    assert len(response.row_scores) == 2, f"Expected 2 row scores, got {len(response.row_scores)}"

    for row_score in response.row_scores:
        assert row_score.scores is not None, f"Row {row_score.index} produced no scores (error: {row_score.error})"
        assert SCORE_NAME in row_score.scores, (
            f"Row {row_score.index} missing '{SCORE_NAME}' score. Got: {list(row_score.scores.keys())}"
        )
        score = row_score.scores[SCORE_NAME]
        assert 1.0 <= score <= 5.0, f"Row {row_score.index} relevance score {score} not in expected range [1, 5]"

    assert len(response.aggregate_scores) > 0, "No aggregate scores returned"
    agg = response.aggregate_scores[0]
    assert agg.name == SCORE_NAME, f"Aggregate score name '{agg.name}' != '{SCORE_NAME}'"
    assert agg.mean is not None, "Aggregate mean is None"


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
    response = client.evaluation.metric_jobs.list()
    assert len(response.data) > 0, f"No evaluation metric jobs found in workspace '{WORKSPACE}'"

    job = response.data[0]
    assert job.spec is not None, "Job has no spec"
    assert METRIC_NAME in str(job.spec), f"Job spec should reference '{METRIC_NAME}'"
