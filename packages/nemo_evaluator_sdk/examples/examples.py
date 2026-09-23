# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Examples demonstrating local evaluator workflows for the SDK."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from nemo_evaluator_sdk.execution.evaluator import Evaluator
from nemo_evaluator_sdk.execution.values import EvaluationError
from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric
from nemo_evaluator_sdk.metrics.llm_judge import LLMJudgeMetric
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec
from nemo_evaluator_sdk.metrics.string_check import StringCheckMetric
from nemo_evaluator_sdk.values import (
    InferenceParams,
    JSONScoreParser,
    MetricDiagnostic,
    MetricResult,
    Model,
    RangeScore,
    RunConfig,
    RunConfigOnlineModel,
    SecretRef,
)

if TYPE_CHECKING:
    import numpy as np


# --- 1. Defining reusable metric configs and custom metrics ---
# Notice the public API is centered on Evaluator. Metrics stay focused on
# scoring logic and configuration rather than owning execution helpers.
HELPFULNESS_PROMPT_V1 = (
    "You are an evaluator. Rate the response's helpfulness from 0-4. "
    'Return only a JSON object with this shape: {"helpfulness": <integer>}.'
)
# Local evaluator execution resolves this as an environment variable name.
DEFAULT_API_KEY_SECRET = os.getenv("NHX_EVALUATOR_DEFAULT_API_KEY_SECRET", "NVIDIA_API_KEY")


def configure_example_logging() -> None:
    """Enable SDK progress logs when this example file is executed directly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


OFFLINE_EXACT_MATCH_DATASET = [
    {"reference": "Paris", "actual": "Paris"},
    {"reference": "London", "actual": "Berlin"},
]
ONLINE_EXACT_MATCH_DATASET = [
    {
        "prompt": "What is the capital of France? Reply in single word.",
        "reference": "Paris",
    },
    {
        "prompt": "How do I make scrambled eggs? Reply in single word.",
        "reference": "Eggs",
    },
]

OFFLINE_JUDGE_DATASET = [
    {
        "prompt": "What is the capital of France?",
        "response": "Paris",
    },
    {
        "prompt": "How do I make scrambled eggs?",
        "response": "Eggs.",
    },
]

OFFLINE_BENCHMARK_DATASET = [
    {
        "reference": "Paris",
        "actual": "Paris",
        "required_phrase": "Paris",
    },
    {
        "reference": "London",
        "actual": "Berlin",
        "required_phrase": "London",
    },
]

ONLINE_BENCHMARK_DATASET = [
    {
        "prompt": "Return exactly this word with no punctuation: Paris",
        "reference": "Paris",
        "required_phrase": "Paris",
    },
    {
        "prompt": "Return exactly this word with no punctuation: Oslo",
        "reference": "London",
        "required_phrase": "London",
    },
]

ONLINE_JUDGE_DATASET = [
    {
        "prompt": "What is the capital of France?",
    },
    {
        "prompt": "How do I make scrambled eggs?",
    },
]

ONLINE_CHAT_PROMPT_TEMPLATE = {"messages": [{"role": "user", "content": "{{item.prompt}}"}]}


model = Model(
    url="https://integrate.api.nvidia.com/v1/chat/completions",
    name=os.getenv("NEMO_DEFAULT_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
    # looks up NVIDIA_API_KEY by default - override via NHX_EVALUATOR_DEFAULT_API_KEY_SECRET
    api_key_secret=SecretRef(root=DEFAULT_API_KEY_SECRET),
)

model_with_custom_headers = model.with_default_headers({"X-My-Header": "value"})


def create_helpfulness_metric(judge_model: Model) -> LLMJudgeMetric:
    """Build a reusable LLM judge metric for helpfulness scoring."""
    return LLMJudgeMetric(
        model=judge_model,
        scores=[
            RangeScore(
                name="helpfulness",
                minimum=0,
                maximum=4,
                parser=JSONScoreParser(json_path="helpfulness"),
                description="How well does the response help the user?",
            )
        ],
        inference=InferenceParams(
            temperature=0.0,
            max_tokens=32768,
        ),
        prompt_template={
            "messages": [
                {"role": "system", "content": HELPFULNESS_PROMPT_V1},
                {
                    "role": "user",
                    "content": (
                        "User prompt: {{item.prompt}}\n\n"
                        "Assistant response: {{sample.output_text | default(item.response)}}\n\n"
                        "Rate this response."
                    ),
                },
            ],
        },
    )


def _print_example_separator(name: str, **params: Any) -> None:
    """Print a visible section header for an independently runnable example."""
    edge = "====="
    inner = name
    if params:
        inner += "(" + ", ".join(f"{k}={v!r}" for k, v in params.items()) + ")"
    middle_line = f"{edge} {inner} {edge}"
    rule = "=" * len(middle_line)
    print(f"\n{rule}\n{middle_line}\n{rule}\n")


def extract_helpfulness_scores(
    row_scores: Sequence[Any],
    *,
    dimension: str = "helpfulness",
    metric_ref: str | None = None,
    judge_response_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned judge and human score arrays from metric-job rows.

    Args:
        row_scores: Row-level metric-job results returned by ``Evaluator.run``
            or ``client.evaluation.metric_jobs.results.row_scores.download``.
        dimension: Dataset field and judge JSON key to compare.
        metric_ref: Optional metric key for benchmark rows where scores are
            already materialized in ``row.metrics``.
        judge_response_index: Request-log index containing the judge response.

    Returns:
        A pair of NumPy arrays: ``(judge_scores, human_scores)``.
    """
    import numpy as np

    judge_scores = []
    human_scores = []
    failed_requests = 0

    for row in row_scores:
        try:
            human_score = float(row.item[dimension])

            if metric_ref is None:
                if not row.requests:
                    raise ValueError("Missing judge request payload")
                judge_response = row.requests[judge_response_index]["response"]["choices"][0]["message"]["content"]
                judge_data = json.loads(judge_response)
                judge_score = float(judge_data[dimension])
            else:
                judge_score = float(row.metrics[metric_ref][0].value)

            human_scores.append(human_score)
            judge_scores.append(judge_score)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            failed_requests += 1
            continue

    print(f"Total errors: {failed_requests}")
    return np.array(judge_scores), np.array(human_scores)


class CustomExactMatchMetric:
    """A user-defined metric that depends on runtime-injected state."""

    type = "custom-exact-match"

    def __init__(self, db_connection_string: str):
        """Store runtime state needed by the custom metric.

        Args:
            db_connection_string: Example dependency injected into the metric.
        """
        self.db_connection_string = db_connection_string

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return the outputs produced by the metric."""
        return [MetricOutputSpec.continuous_score(self.type)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Convert the custom score into the SDK metric result shape.

        Args:
            input: Original dataset row and evaluated candidate output.

        Returns:
            A metric result containing one exact-match score.
        """
        prediction = input.candidate.output_text
        reference = input.row.data.get("actual")
        if reference is None:
            reference = input.row.data.get("reference")
        if prediction is None or reference is None:
            return MetricResult(outputs=[MetricOutput(name=self.type, value=0.0)])
        score = 1.0 if prediction == reference else 0.0
        return MetricResult(outputs=[MetricOutput(name=self.type, value=score)])


class DiffDiagnosticExactMatchMetric:
    """Exact-match metric that attaches an expected-vs-actual diff on mismatch.

    This demonstrates ``MetricResult.diagnostics``: a metric returns its score
    as usual and, when the score is unexpected, attaches ``MetricDiagnostic``
    findings explaining why. Comparison details (expected/actual/diff) go in
    ``details``. The field defaults to empty, so consumers that ignore
    diagnostics are unaffected.
    """

    type = "diff-diagnostic-exact-match"

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return the outputs produced by the metric."""
        return [MetricOutputSpec.continuous_score(self.type)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Score exact match and attach a diff diagnostic when it fails.

        Args:
            input: Original dataset row and evaluated candidate output.

        Returns:
            A metric result with the exact-match score and, on mismatch,
            ``diagnostics`` containing a unified expected-vs-actual diff.
        """
        prediction = input.candidate.output_text or ""
        reference = str(input.row.data.get("reference", ""))
        matched = prediction == reference
        outputs = [MetricOutput(name=self.type, value=1.0 if matched else 0.0)]

        if matched:
            return MetricResult(outputs=outputs)

        diff = "".join(
            difflib.unified_diff(
                reference.splitlines(keepends=True),
                prediction.splitlines(keepends=True),
                fromfile="expected",
                tofile="actual",
            )
        )
        return MetricResult(
            outputs=outputs,
            diagnostics=[
                MetricDiagnostic(
                    message="exact match failed",
                    details={
                        "expected": reference,
                        "actual": prediction,
                        "diff": diff,
                    },
                )
            ],
        )


class CustomFailingMetric:
    """A user-defined metric that raises to demonstrate metric failure handling."""

    type = "custom-metric-with-failure"

    def __init__(self, message: str):
        """Store the error message emitted by this example metric.

        Args:
            message: Text used for the raised runtime error.
        """
        self.message = message

    def output_spec(self) -> list[MetricOutputSpec]:
        """Return the outputs produced by the metric."""
        return [MetricOutputSpec.continuous_score(self.type)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        """Convert the intentionally raised error into normal metric execution.

        Args:
            input: Original dataset row and evaluated candidate output.

        Raises:
            RuntimeError: Always raised to exercise benchmark failure handling.
        """
        del input
        # This example metric is intentionally failing to demonstrate structured
        # benchmark error handling in local evaluator workflows.
        raise RuntimeError(self.message)


# --- 2. Local evaluator workflows ---
async def run_offline_local_exact_match_example() -> None:
    """Run one offline exact-match evaluation example.

    Returns:
        None.
    """

    _print_example_separator(run_offline_local_exact_match_example.__name__)

    evaluator = Evaluator()
    # TODO: fix error message in another branch
    exact_match = ExactMatchMetric(reference="{{item.reference}}", candidate="{{item.actual}}")

    print("Running offline exact match...")

    exact_match_result = await evaluator.run(
        metrics=[exact_match],
        dataset=OFFLINE_EXACT_MATCH_DATASET,
        config=RunConfig(parallelism=4),
    )
    exact_match_result.print_summary()


async def run_online_local_exact_match_example() -> None:
    """Run one local online exact-match evaluation example.

    Returns:
        None.
    """

    _print_example_separator(run_online_local_exact_match_example.__name__)

    evaluator = Evaluator()
    exact_match = ExactMatchMetric(reference="{{item.reference}}")

    print("Running local online exact match...")

    exact_match_result = await evaluator.run(
        metrics=[exact_match],
        target=model,
        dataset=ONLINE_EXACT_MATCH_DATASET,
        prompt_template=ONLINE_CHAT_PROMPT_TEMPLATE,
        config=RunConfigOnlineModel(parallelism=4),
    )
    exact_match_result.print_summary()


async def run_offline_local_multi_metric_example() -> None:
    """Run one local multi-metric evaluation example.

    Returns:
        None.
    """

    _print_example_separator(run_offline_local_multi_metric_example.__name__)

    evaluator = Evaluator()
    custom_metric = CustomExactMatchMetric(db_connection_string="postgresql://localhost@localhost:5432/mydatabase")
    exact_match = ExactMatchMetric(reference="{{item.reference}}", candidate="{{item.actual}}")

    print("\nRunning local multi-metric evaluation...")

    combined_result = await evaluator.run(
        metrics=[exact_match, custom_metric],
        dataset=OFFLINE_EXACT_MATCH_DATASET,
        config=RunConfig(parallelism=4),
    )
    combined_result.print_summary()
    print(f"Per-metric keys: {list(combined_result.per_metric)}")
    print(f"Exact match aggregate scores: {combined_result.metric_result('exact-match').aggregate_scores.scores}")


async def run_offline_local_benchmark_example() -> None:
    """Run one local benchmark example with multiple metrics.

    Returns:
        None.
    """

    _print_example_separator(run_offline_local_benchmark_example.__name__)

    evaluator = Evaluator()
    exact_match = ExactMatchMetric(reference="{{item.reference}}", candidate="{{item.actual}}")
    contains_required_phrase = StringCheckMetric(
        operation="contains",
        left_template="{{item.actual}}",
        right_template="{{item.required_phrase}}",
    )

    print("\nRunning local benchmark evaluation...")

    benchmark_result = await evaluator.run(
        metrics=[exact_match, contains_required_phrase],
        dataset=OFFLINE_BENCHMARK_DATASET,
        config=RunConfig(parallelism=4),
    )
    benchmark_result.print_summary()
    print(f"Benchmark metric keys: {list(benchmark_result.per_metric)}")
    print(f"Exact match scores: {benchmark_result.metric_result('exact-match').aggregate_scores.scores}")
    print(f"String check scores: {benchmark_result.metric_result('string-check').aggregate_scores.scores}")


async def run_online_local_benchmark_example() -> None:
    """Run one local online benchmark example with multiple metrics.

    Returns:
        None.
    """

    _print_example_separator(run_online_local_benchmark_example.__name__)

    evaluator = Evaluator()
    exact_match = ExactMatchMetric(reference="{{item.reference}}")
    contains_required_phrase = StringCheckMetric(
        operation="contains",
        left_template="{{sample.output_text}}",
        right_template="{{item.required_phrase}}",
    )

    print("\nRunning local online benchmark evaluation...")

    benchmark_result = await evaluator.run(
        metrics=[exact_match, contains_required_phrase],
        target=model,
        dataset=ONLINE_BENCHMARK_DATASET,
        prompt_template=ONLINE_CHAT_PROMPT_TEMPLATE,
        config=RunConfigOnlineModel(parallelism=4),
    )
    benchmark_result.print_summary()
    print(f"Online benchmark metric keys: {list(benchmark_result.per_metric)}")
    print(f"Exact match scores: {benchmark_result.metric_result('exact-match').aggregate_scores.scores}")
    print(f"String check scores: {benchmark_result.metric_result('string-check').aggregate_scores.scores}")


async def run_local_benchmark_with_metric_failure_example() -> None:
    """Run one benchmark where a sibling metric survives another metric failure."""

    _print_example_separator(run_local_benchmark_with_metric_failure_example.__name__)

    evaluator = Evaluator()
    exact_match = ExactMatchMetric(reference="{{item.reference}}", candidate="{{item.actual}}")
    failing_metric = CustomFailingMetric(message="intentional benchmark metric failure")

    print("\nRunning local benchmark evaluation with one failing metric...")

    try:
        await evaluator.run(
            metrics=[exact_match, failing_metric],
            dataset=OFFLINE_BENCHMARK_DATASET,
            config=RunConfig(parallelism=4),
        )
    except EvaluationError as error:
        print("Benchmark evaluation failed with structured context:")
        print(f"  error: {error}")
        print(f"  row index: {error.index}")
        print(f"  phase: {error.phase.value}")
        print(f"  metric key: {error.metric_key}")
        print(f"  message: {error.message}")


async def run_local_metric_with_template_failure_example() -> None:
    """Run metric evaluations that expose Jinja template failures clearly.

    Returns:
        None.
    """

    _print_example_separator(run_local_metric_with_template_failure_example.__name__)

    evaluator = Evaluator()
    invalid_metric = ExactMatchMetric(
        reference="{{item.missing_reference}}",
        candidate="{{item.actual}}",
    )
    dataset = OFFLINE_EXACT_MATCH_DATASET[:1]

    print("\nRunning local metric evaluation with an invalid metric template...")
    try:
        await evaluator.run(
            metrics=[invalid_metric],
            dataset=dataset,
            config=RunConfig(parallelism=1),
        )
    except EvaluationError as error:
        print("Metric evaluation failed with structured context:")
        print(f"  error: {error}")
        print(f"  row index: {error.index}")
        print(f"  phase: {error.phase.value}")
        print(f"  metric key: {error.metric_key}")
        print(f"  message: {error.message}")
        if error.__cause__ is not None:
            print(f"  cause: {type(error.__cause__).__name__}: {error.__cause__}")


async def run_offline_local_llm_judge_example() -> None:
    """Run one local LLM-judge evaluation example with run overrides.

    Returns:
        None.
    """

    _print_example_separator(run_offline_local_llm_judge_example.__name__)

    evaluator = Evaluator()
    llm_judge_metric = create_helpfulness_metric(model_with_custom_headers)

    print("\nRunning local LLM judge evaluation...")

    llm_judge_result = await evaluator.run(
        metrics=[llm_judge_metric],
        dataset=OFFLINE_JUDGE_DATASET,
        config=RunConfig(parallelism=2),
    )
    llm_judge_result.print_summary()


async def run_online_local_llm_judge_example() -> None:
    """Run one local online LLM-judge evaluation example with run overrides.

    Returns:
        None.
    """

    _print_example_separator(run_online_local_llm_judge_example.__name__)

    evaluator = Evaluator()
    llm_judge_metric = create_helpfulness_metric(model_with_custom_headers)

    print("\nRunning local online LLM judge evaluation...")

    llm_judge_result = await evaluator.run(
        metrics=[llm_judge_metric],
        target=model_with_custom_headers,
        dataset=ONLINE_JUDGE_DATASET,
        prompt_template=ONLINE_CHAT_PROMPT_TEMPLATE,
        config=RunConfigOnlineModel(parallelism=2),
    )
    llm_judge_result.print_summary()


def run_sync_example() -> None:
    """Run a minimal synchronous evaluator workflow.

    Returns:
        None.
    """

    evaluator = Evaluator()
    result = evaluator.run_sync(
        metrics=[ExactMatchMetric(reference="{{item.reference}}", candidate="{{item.actual}}")],
        dataset=OFFLINE_EXACT_MATCH_DATASET[:1],  # Only run the first sample
        config=RunConfig(parallelism=1),
    )
    print("\nRunning sync exact match...")
    result.print_summary()


async def run_examples() -> None:
    """Execute the example workflows exposed by this module.

    Returns:
        None.
    """
    #### Local backend examples ####
    await run_offline_local_exact_match_example()
    await run_online_local_exact_match_example()
    await run_offline_local_multi_metric_example()
    await run_offline_local_llm_judge_example()
    await run_online_local_llm_judge_example()
    await run_offline_local_benchmark_example()
    await run_online_local_benchmark_example()
    await run_local_benchmark_with_metric_failure_example()
    await run_local_metric_with_template_failure_example()
    run_sync_example()


if __name__ == "__main__":
    configure_example_logging()
    asyncio.run(run_examples())
