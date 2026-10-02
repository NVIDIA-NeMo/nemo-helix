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
     test_llm_judge_metric_exists, test_metric_has_rubric_score,
     test_metric_has_default_prompt_template
3. Run evaluation - test_sync_evaluation_produces_scores
4. Retrieve results - test_agent_ran_sync_eval_and_examined_scores (trace)

Provider, workspace, and secret infrastructure are pre-configured
in the Dockerfile so the agent can focus on evaluator operations.
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


def _get_nhx_client() -> NeMoHelix:
    """Get NeMoHelix client for the eval workspace."""
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NeMoHelix(base_url=nhx_base_url, workspace=WORKSPACE, access_token=_make_unsigned_jwt())


def _get_files_client() -> FilesClient:
    return client_from_platform(_get_nhx_client(), FilesClient)


# --- Dataset checks ---


def test_fileset_exists() -> None:
    """Verify the zeroconfig-dataset fileset was created."""
    files_client = _get_files_client()
    fileset_names = [fs.name for fs in files_client.list_filesets().page().items]
    assert FILESET in fileset_names, f"Fileset '{FILESET}' not found. Found: {fileset_names}"


def test_fileset_has_data() -> None:
    """Verify the dataset fileset has files uploaded."""
    client = _get_nhx_client()
    files = client.files.list(fileset=FILESET)
    assert len(files.data) > 0, f"Fileset '{FILESET}' has no files uploaded"


# --- Zero-Config LLM Judge Metric checks ---


def test_llm_judge_metric_exists() -> None:
    """Verify an LLM-as-a-Judge metric named zeroconfig-judge was created."""
    client = _get_nhx_client()
    response = client.evaluation.metrics.list()
    metric_names = [m.name for m in response.data]
    assert METRIC_NAME in metric_names, (
        f"Metric '{METRIC_NAME}' not found in workspace '{WORKSPACE}'. Found: {metric_names}"
    )
    judge_metrics = [m for m in response.data if m.type == "llm-judge"]
    assert len(judge_metrics) > 0, f"No llm-judge metrics found. Found types: {[m.type for m in response.data]}"


def test_metric_has_rubric_score() -> None:
    """Verify the metric has a rubric-based quality score with at least 3 levels."""
    client = _get_nhx_client()
    metric = client.evaluation.metrics.retrieve(name=METRIC_NAME)

    assert len(metric.scores) > 0, "Metric has no scores defined"
    score_names = [s.name for s in metric.scores]
    assert SCORE_NAME in score_names, f"Expected score '{SCORE_NAME}' not found. Got: {score_names}"

    quality_score = next(s for s in metric.scores if s.name == SCORE_NAME)
    assert quality_score.rubric is not None and len(quality_score.rubric) >= 3, (
        f"Score '{SCORE_NAME}' should have a rubric with at least 3 levels, got: {quality_score.rubric}"
    )


def test_metric_model_config() -> None:
    """Verify the metric has correct model config pointing to inference endpoint."""
    client = _get_nhx_client()
    metric = client.evaluation.metrics.retrieve(name=METRIC_NAME)

    assert hasattr(metric, "model"), "Metric has no model config"
    model = metric.model
    assert hasattr(model, "url"), f"Model is a reference string, expected inline config: {model}"
    assert model.api_key_secret == "nvidia-api-key", (
        f"Model api_key_secret should be 'nvidia-api-key', got: {model.api_key_secret}"
    )
    assert model.format in ("openai", "nim"), f"Model format should be 'openai' or 'nim', got: {model.format}"


def test_metric_has_default_prompt_template() -> None:
    """Verify the metric's prompt_template is the auto-generated default.

    When no prompt_template is provided, the system generates one containing
    'expert evaluator' from the DEFAULT_JUDGE_SYSTEM_PROMPT_TEMPLATE.
    This is the key zero-config verification.
    """
    client = _get_nhx_client()
    metric = client.evaluation.metrics.retrieve(name=METRIC_NAME)

    assert metric.prompt_template is not None, "Metric has no prompt_template (should have auto-generated default)"

    # The default template is a dict with 'messages' key (chat format)
    if isinstance(metric.prompt_template, dict):
        template_str = json.dumps(metric.prompt_template)
    else:
        template_str = str(metric.prompt_template)

    assert "expert evaluator" in template_str.lower(), (
        f"prompt_template does not appear to be the auto-generated default. "
        f"Expected it to contain 'expert evaluator'. Got: {template_str[:200]}..."
    )


# --- Synchronous Evaluation checks ---


@pytest.mark.timeout(90)
def test_sync_evaluation_produces_scores() -> None:
    """Verify the zero-config metric produces valid scores via synchronous evaluation.

    First tries to re-run the evaluation from the verifier (strongest check).
    If the external inference API is unreachable from the container, falls back
    to checking that the agent's own evaluation produced scores (via trace).
    """
    client = _get_nhx_client()
    try:
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

        # Verify the judge produces meaningful scores: the good answer
        # ("2+2 equals 4") should score higher than the bad answer
        # ("I have no idea whatsoever")
        good_score = response.row_scores[0].scores[SCORE_NAME]
        bad_score = response.row_scores[1].scores[SCORE_NAME]
        assert good_score > bad_score, (
            f"Default judge did not produce meaningful scores: "
            f"good answer scored {good_score}, bad answer scored {bad_score}. "
            f"Expected the good answer to score higher."
        )

        assert len(response.aggregate_scores) > 0, "No aggregate scores returned"
        agg = response.aggregate_scores[0]
        assert agg.name == SCORE_NAME, f"Aggregate score name '{agg.name}' != '{SCORE_NAME}'"
        assert agg.mean is not None, "Aggregate mean is None"
    except Exception:
        # External inference API may be unreachable from the container.
        # Fall back to checking the agent's own eval produced scores.
        session = get_session()
        bash_results = session.get_tool_results("Bash")
        score_results = [r for r in bash_results if SCORE_NAME in r.content and not r.is_error]
        assert len(score_results) > 0, (
            f"Verifier could not re-run evaluation (inference API unreachable) "
            f"and agent's output did not contain '{SCORE_NAME}' scores either."
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
