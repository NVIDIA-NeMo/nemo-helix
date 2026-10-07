# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Hermes Responses SSE agent with a custom translator and metric."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import httpx
from nhx_evals_sdk import (
    AgentStreamTranslation,
    AgentStreamTranslationContext,
    GenericAgent,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
    SecretRef,
    SseFrame,
)
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.results import AgentEvalResult
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nhx_evals_sdk.agent_inference import make_agent_inference_fn
from nhx_evals_sdk.values.atif import Agent as AtifAgent
from nhx_evals_sdk.values.atif import FinalMetrics, Step, Trajectory

# To run Hermes locally:
#   1. Set API_SERVER_ENABLED=true and API_SERVER_KEY=<key> in ~/.hermes/.env.
#   2. Start the API server with `hermes gateway`.
#   3. Export the same key as HERMES_TOKEN for this evaluator client.
# See HERMES_API_SERVER_DOCS_URL below for the complete setup. Update the URL if
# Hermes is running on a different port or host.
DEFAULT_HERMES_URL = "http://127.0.0.1:8642/v1/responses"
EXPECTED_TEXT = "streaming works"
# Export the key configured for the Hermes API server under this name.
HERMES_TOKEN_ENV_NAME = "HERMES_TOKEN"
# Hermes documents how to configure the API server key here.
HERMES_API_SERVER_DOCS_URL = (
    "https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server#1-enable-the-api-server"
)


class KeywordMatchMetric:
    """Score whether the agent response contains the expected text."""

    @property
    def type(self) -> str:
        return "keyword_match"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        expected = str(input.row.data.get("reference", {}).get("expected", "")).casefold()
        answer = (input.candidate.output_text or "").casefold()
        score = 1.0 if expected and expected in answer else 0.0
        return MetricResult(outputs=[MetricOutput(name="score", value=score)])


def _atif_steps(
    response: dict[str, Any],
    context: AgentStreamTranslationContext,
) -> list[Step]:
    input_message = context.request_payload.get("input")
    if not isinstance(input_message, str) or not input_message:
        raise ValueError("Hermes request did not contain input text")

    messages: list[tuple[Literal["user", "agent"], str]] = [("user", input_message)]
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
                and part["text"]
            ]
            if text_parts:
                messages.append(("agent", "\n".join(text_parts)))

    if len(messages) == 1:
        raise ValueError("Hermes response.completed did not contain assistant output text")

    return [
        Step(step_id=step_id, source=source, message=message)
        for step_id, (source, message) in enumerate(messages, start=1)
    ]


class HermesStreamTranslator:
    """Translate a completed Hermes Responses stream into ATIF-v1.7."""

    def __call__(
        self,
        frames: Sequence[SseFrame],
        *,
        context: AgentStreamTranslationContext,
    ) -> AgentStreamTranslation:
        data_payloads = [
            frame.payload for frame in frames if frame.channel == "data" and isinstance(frame.payload, dict)
        ]
        completed = data_payloads[-1] if data_payloads else None
        if (
            not isinstance(completed, dict)
            or completed.get("type") != "response.completed"
            or not isinstance(completed.get("response"), dict)
        ):
            raise ValueError("Hermes stream did not include response.completed")

        response = completed["response"]
        if response.get("status") != "completed":
            raise ValueError("Hermes response.completed did not contain a completed response")
        if not context.output_text:
            raise ValueError("Hermes response.completed did not contain output text")

        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("Hermes response.completed did not contain a response id")

        response_model = response.get("model")
        agent = AtifAgent(
            name=context.agent_name,
            model_name=response_model if isinstance(response_model, str) else None,
        )
        steps = _atif_steps(response, context)

        final_metrics = None
        usage = response.get("usage")
        if isinstance(usage, dict):
            final_metrics = FinalMetrics(
                total_prompt_tokens=usage.get("input_tokens"),
                total_completion_tokens=usage.get("output_tokens"),
                total_steps=len(steps),
            )

        trajectory = Trajectory(
            schema_version="ATIF-v1.7",
            session_id=response_id,
            trajectory_id=context.invocation_id,
            agent=agent,
            steps=steps,
            final_metrics=final_metrics,
        )
        return AgentStreamTranslation(trajectory=trajectory.model_dump(mode="python", exclude_none=True))


def _require_hermes_token() -> None:
    if os.getenv(HERMES_TOKEN_ENV_NAME):
        return
    raise RuntimeError(
        f"{HERMES_TOKEN_ENV_NAME} is not set. Export the Hermes API server key before running this example:\n"
        f"  export {HERMES_TOKEN_ENV_NAME}=<your-hermes-token>\n"
        f"See {HERMES_API_SERVER_DOCS_URL}"
    )


async def evaluate(
    *,
    agent_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AgentEvalResult:
    """Run the Hermes streaming evaluation."""
    _require_hermes_token()
    evaluator = AgentEvaluator(
        agent_inference_fn_factory=partial(
            make_agent_inference_fn,
            stream_translator=HermesStreamTranslator(),
            capture_evidence=True,
        ),
        client=client,
        default_headers={"Accept": "text/event-stream"},
    )

    task = AgentEvalTask(
        id="hermes-streaming",
        intent="Return the requested phrase over SSE.",
        inputs={"instruction": f"Reply with exactly: {EXPECTED_TEXT}."},
        reference={"expected": EXPECTED_TEXT},
        metrics=[KeywordMatchMetric()],
    )

    agent = GenericAgent(
        name="hermes-agent",
        url=agent_url or os.getenv("HERMES_AGENT_URL", DEFAULT_HERMES_URL),
        api_key_secret=SecretRef(HERMES_TOKEN_ENV_NAME),
        body={
            "model": "hermes-agent",
            "input": "{{ instruction }}",
            "stream": True,
            "store": False,
        },
        response_path="$..text",
        stream=True,
    )

    return await evaluator.run(
        tasks=[task],
        target=agent,
        config=AgentEvalRunConfig(parallelism=1),
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    result = await evaluate()

    answer = result.trials[0].output.output_text if result.trials[0].output else None
    aggregate = result.summary.score("keyword_match.score")

    print(f"response: {answer}")
    print(f"{aggregate.name}: {aggregate.mean}")


if __name__ == "__main__":
    asyncio.run(main())
