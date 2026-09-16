# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same nine GPA metrics, scored by TruLens's own feedback functions.

TruLens's trace-level feedback functions accept the trace as a plain string and return
``(score between 0 and 1, {"reason": ...})``. That is exactly what the Evaluator ``Metric``
protocol needs, so wrapping one is a few lines: render the trajectory, call the function, return
``score`` and ``reason``.

TruLens is not a dependency of this repository. Install it yourself::

    uv pip install trulens-core trulens-feedback trulens-providers-openai
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from gpa_metrics import GPA_METRICS, NVIDIA_BASE_URL, describe_trial, read_trajectory, render_trajectory
from nemo_evaluator_sdk.metrics.protocol import (
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)

# Which TruLens feedback function scores each dimension. The six trace-level ones take the rendered
# trace; the three Goal ones take text pieces of the trial.
FEEDBACK_FUNCTIONS: dict[str, str] = {
    "answer_correctness": "agent_goal_accuracy_with_cot_reasons",
    "answer_relevance": "relevance_with_cot_reasons",
    "groundedness": "groundedness_measure_with_cot_reasons",
    "plan_quality": "plan_quality_with_cot_reasons",
    "plan_adherence": "plan_adherence_with_cot_reasons",
    "tool_selection": "tool_selection_with_cot_reasons",
    "tool_calling": "tool_calling_with_cot_reasons",
    "execution_efficiency": "execution_efficiency_with_cot_reasons",
    "logical_consistency": "logical_consistency_with_cot_reasons",
}
assert set(FEEDBACK_FUNCTIONS) == set(GPA_METRICS)


def nvidia_provider(model: str, api_key_env: str = "NVIDIA_API_KEY") -> Any:
    """TruLens's OpenAI provider, pointed at build.nvidia.com."""
    from trulens.providers.openai import OpenAI  # ty: ignore[unresolved-import,unused-ignore-comment]

    options: dict[str, Any] = {"model_engine": model, "base_url": NVIDIA_BASE_URL, "api_key": os.environ[api_key_env]}
    return OpenAI(**options)


class TruLensGpaMetric:
    """One GPA dimension scored by a TruLens feedback function."""

    def __init__(self, name: str, provider: Any) -> None:
        self.name = name
        self.provider = provider

    @property
    def type(self) -> str:
        return self.name

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score("score", required=False),
            MetricOutputSpec.label("reason", required=False),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trajectory = await read_trajectory(input)
        if trajectory is None:
            return MetricResult(outputs=[])

        instruction = input.row.data.get("inputs", {}).get("instruction", "")
        answer = next((s.message for s in reversed(trajectory.steps) if s.source == "agent" and s.message.strip()), "")
        observed = "\n".join(str(r.content) for s in trajectory.steps if s.observation for r in s.observation.results)
        goal = input.row.data.get("reference", {}).get("goal")

        feedback = getattr(self.provider, FEEDBACK_FUNCTIONS[self.name])
        if self.name == "answer_correctness":
            if goal is None:
                return MetricResult(outputs=[])
            call = lambda: feedback(records=describe_trial(input, trajectory), reference_goal=goal)  # noqa: E731
        elif self.name == "answer_relevance":
            call = lambda: feedback(prompt=instruction, response=answer)  # noqa: E731
        elif self.name == "groundedness":
            call = lambda: feedback(source=observed, statement=answer)  # noqa: E731
        else:
            call = lambda: feedback(trace=render_trajectory(trajectory))  # noqa: E731

        try:
            score, reasons = await asyncio.to_thread(call)  # TruLens providers are synchronous
        except Exception as error:
            return MetricResult(outputs=[], diagnostics=[MetricDiagnostic(message=str(error))])
        reason = reasons.get("reason") or "\n".join(map(str, reasons.get("reasons", []))) or str(reasons)
        return MetricResult(
            outputs=[MetricOutput(name="score", value=float(score)), MetricOutput(name="reason", value=reason)]
        )


def trulens_gpa_metrics(provider: Any) -> list[TruLensGpaMetric]:
    """All nine GPA metrics, each backed by its TruLens feedback function."""
    return [TruLensGpaMetric(name, provider) for name in GPA_METRICS]
