# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the customer-facing Hermes Responses SSE example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from nhx_evals_sdk import AgentStreamTranslationContext, SseFrame
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus
from nhx_evals_sdk.execution.samples import build_metric_input

_MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "hermes" / "example.py"
_spec = importlib.util.spec_from_file_location("hermes_example", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
hermes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hermes)

_SSE_RESPONSE = """\
event: response.created
data: {"type":"response.created","response":{"id":"resp-test","status":"in_progress","model":"hermes-agent","output":[]},"sequence_number":0}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","status":"in_progress","role":"assistant","content":[]},"sequence_number":1}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"streaming works","sequence_number":2}

event: response.output_text.done
data: {"type":"response.output_text.done","text":"streaming works","sequence_number":3}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"streaming works"}]},"sequence_number":4}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp-test","status":"completed","model":"hermes-agent","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"streaming works"}]}],"usage":{"input_tokens":15671,"output_tokens":7,"total_tokens":15678}},"sequence_number":5}
"""


@pytest.mark.asyncio
async def test_hermes_evaluation_replays_responses_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["authorization"] = request.headers.get("authorization")
        seen["accept"] = request.headers.get("accept")
        return httpx.Response(
            200,
            content=_SSE_RESPONSE,
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setenv(hermes.HERMES_TOKEN_ENV_NAME, "test-token")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await hermes.evaluate(agent_url="http://hermes.test/v1/responses", client=client)

    assert seen == {
        "body": {
            "model": "hermes-agent",
            "input": "Reply with exactly: streaming works.",
            "stream": True,
            "store": False,
        },
        "authorization": "Bearer test-token",
        "accept": "text/event-stream",
    }

    trial = result.trials[0]
    assert trial.status is AgentEvalTrialStatus.COMPLETED
    assert trial.output is not None
    assert trial.output.output_text == "streaming works"

    aggregate = result.summary.score("keyword_match.score")
    assert aggregate.mean == 1.0

    assert trial.evidence is not None
    raw_stream = trial.evidence.require("raw_stream").data
    assert isinstance(raw_stream, str)
    assert "event: response.completed" in raw_stream

    stream_events = trial.evidence.require("stream_events").data
    assert isinstance(stream_events, list)
    assert len(stream_events) == 6
    assert {event["channel"] for event in stream_events} == {"data"}

    trace = trial.evidence.require("trace", kind="trace")
    assert trace.format == "atif"
    assert isinstance(trace.data, dict)
    assert trace.data["session_id"] == "resp-test"
    assert isinstance(trace.data["trajectory_id"], str)
    assert trace.data["trajectory_id"]
    assert trace.data["agent"] == {"name": "hermes-agent", "model_name": "hermes-agent"}
    assert [step["source"] for step in trace.data["steps"]] == ["user", "agent"]
    assert trace.data["steps"][0]["message"] == "Reply with exactly: streaming works."
    assert trace.data["steps"][1]["message"] == "streaming works"
    assert trace.data["final_metrics"] == {
        "total_prompt_tokens": 15671,
        "total_completion_tokens": 7,
        "total_steps": 2,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "expected_score"),
    [("Streaming works.", 1.0), ("streaming failed", 0.0)],
)
async def test_keyword_match_metric_pass_and_fail(answer: str, expected_score: float) -> None:
    result = await hermes.KeywordMatchMetric().compute_scores(
        build_metric_input(
            {"reference": {"expected": "streaming works"}},
            {"output_text": answer},
            index=0,
        )
    )

    assert result.outputs[0].value == expected_score


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [
            SseFrame(
                channel="data",
                payload={
                    "type": "response.completed",
                    "response": {"id": "resp-test", "status": "in_progress"},
                },
                raw="data: {}",
            )
        ],
    ],
)
def test_translator_rejects_missing_or_incomplete_completion(frames: list[SseFrame]) -> None:
    context = AgentStreamTranslationContext(
        agent_name="hermes-agent",
        endpoint="http://hermes.test/v1/responses",
        request_payload={"input": "Reply with exactly: streaming works."},
        output_text="streaming works",
        invocation_id="invocation-1",
    )

    with pytest.raises(ValueError, match="Hermes"):
        hermes.HermesStreamTranslator()(frames, context=context)


def test_translator_builds_steps_from_completed_response_messages() -> None:
    frames = [
        SseFrame(
            channel="data",
            payload={
                "type": "response.completed",
                "response": {
                    "id": "resp-test",
                    "status": "completed",
                    "model": "hermes-agent",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "First message."}],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Final message."}],
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 4},
                },
            },
            raw="data: {}",
        )
    ]
    context = AgentStreamTranslationContext(
        agent_name="hermes-agent",
        endpoint="http://hermes.test/v1/responses",
        request_payload={"input": "Run two steps."},
        output_text="Final message.",
        invocation_id="invocation-1",
    )

    translation = hermes.HermesStreamTranslator()(frames, context=context)

    assert translation.trajectory["steps"] == [
        {"step_id": 1, "source": "user", "message": "Run two steps."},
        {"step_id": 2, "source": "agent", "message": "First message."},
        {"step_id": 3, "source": "agent", "message": "Final message."},
    ]
    assert translation.trajectory["final_metrics"]["total_steps"] == 3


@pytest.mark.asyncio
async def test_incomplete_stream_records_translation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    incomplete = """\
event: response.output_text.done
data: {"type":"response.output_text.done","text":"streaming works"}
"""
    monkeypatch.setenv(hermes.HERMES_TOKEN_ENV_NAME, "test-token")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=incomplete, headers={"content-type": "text/event-stream"})
    )

    async with httpx.AsyncClient(transport=transport) as client:
        result = await hermes.evaluate(agent_url="http://hermes.test/v1/responses", client=client)

    trial = result.trials[0]
    assert trial.status is AgentEvalTrialStatus.FAILED
    assert trial.evidence is not None
    error = trial.evidence.require("translation_error", kind="error")
    assert isinstance(error.data, dict)
    assert error.data["error"] == "Hermes stream did not include response.completed"


@pytest.mark.asyncio
async def test_missing_hermes_token_fails_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    request_count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500, request=request)

    monkeypatch.delenv(hermes.HERMES_TOKEN_ENV_NAME, raising=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RuntimeError) as exc_info:
            await hermes.evaluate(agent_url="http://hermes.test/v1/responses", client=client)

    message = str(exc_info.value)
    assert request_count == 0
    assert "export HERMES_TOKEN=<your-hermes-token>" in message
    assert hermes.HERMES_API_SERVER_DOCS_URL in message
