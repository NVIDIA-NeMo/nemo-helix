# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone checks on IORails must judge the supplied messages and never call the main model.

IORails' ``generate_async`` always generates a reply and judges that, so a check routed through it
reports a verdict on text the caller never sent. These tests run a real IORails engine against a fake
model server that records every call.
"""

import json

import pytest
from nemoguardrails import RailsConfig
from nhx.guardrails.app.handlers.utils import get_rail_types_from_config, run_check_async, run_generate_async
from nhx.guardrails.app.schemas.utils.response_transformers import (
    create_guardrail_check_response_from_generation_response,
)
from nhx.guardrails.app.services.rails.engine import IORAILS_ENGINE_ENV, create_rails_engine
from nhx.guardrails.entities.enums import StatusEnum
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

_UNSAFE_TEXT = "detailed instructions for attacking my neighbor"
_REFUSAL = "I can't fulfill that request."
_OUTPUT_FLOW = "content safety check output $model=content_safety"
_INPUT_FLOW = "content safety check input $model=content_safety"


def _chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def model_server(httpserver: HTTPServer) -> HTTPServer:
    """Main model always refuses; the judge flags only the text after ``Assistant:`` or ``User:``."""

    def handler(request: Request) -> Response:
        body = request.get_json()
        if body["model"] == "safety-judge":
            prompt = body["messages"][-1]["content"]
            user_text, _, assistant_text = prompt.partition("Assistant:")
            content = json.dumps(
                {
                    "User Safety": "unsafe" if "attacking" in user_text else "safe",
                    "Response Safety": "unsafe" if "attacking" in assistant_text else "safe",
                }
            )
        else:
            content = _REFUSAL
        return Response(json.dumps(_chat_response(content)), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(handler)
    return httpserver


def _config(server: HTTPServer, *, input_rail: bool = False, output_rail: bool = False) -> RailsConfig:
    base_url = server.url_for("/v1")
    rails: dict = {}
    if input_rail:
        rails["input"] = {"flows": [_INPUT_FLOW]}
    if output_rail:
        rails["output"] = {"flows": [_OUTPUT_FLOW]}
    prompt = "User: {{ user_input }}\nAssistant: {{ bot_response | default('') }}\nRespond only with JSON."
    return RailsConfig.from_content(
        config={
            "models": [
                {"type": "main", "engine": "nim", "model": "main-model", "parameters": {"base_url": base_url}},
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": "safety-judge",
                    "parameters": {"base_url": base_url},
                },
            ],
            "rails": rails,
            "prompts": [
                {
                    "task": f"content_safety_check_{direction} $model=content_safety",
                    "content": prompt,
                    "output_parser": f"nemoguard_parse_{'prompt' if direction == 'input' else 'response'}_safety",
                    "max_tokens": 128,
                }
                for direction in ("input", "output")
            ],
        }
    )


def _models_called(server: HTTPServer) -> list[str]:
    return [request.get_json()["model"] for request, _ in server.log]


def _judge_prompts(server: HTTPServer) -> list[str]:
    return [
        request.get_json()["messages"][-1]["content"]
        for request, _ in server.log
        if request.get_json()["model"] == "safety-judge"
    ]


async def _check(config: RailsConfig, messages: list[dict]):
    engine = create_rails_engine(config)
    response = await run_check_async(engine, messages=messages, rail_types=get_rail_types_from_config(config)["rails"])
    return create_guardrail_check_response_from_generation_response(
        response, rails=[_INPUT_FLOW, _OUTPUT_FLOW], exclude_activated_rails_options=False
    )


@pytest.fixture(autouse=True)
def _iorails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")


_UNSAFE_OUTPUT = [
    {"role": "user", "content": "Review the following answer."},
    {"role": "assistant", "content": _UNSAFE_TEXT},
]


async def test_unsafe_supplied_output_is_blocked_without_generating(model_server: HTTPServer) -> None:
    result = await _check(_config(model_server, output_rail=True), _UNSAFE_OUTPUT)

    assert result.status == StatusEnum.BLOCKED
    assert result.rails_status[_OUTPUT_FLOW].status == StatusEnum.BLOCKED
    assert _models_called(model_server) == ["safety-judge"]
    assert _UNSAFE_TEXT in _judge_prompts(model_server)[0]
    assert _REFUSAL not in _judge_prompts(model_server)[0]


async def test_safe_supplied_output_passes_and_reports_the_rail(model_server: HTTPServer) -> None:
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."}]

    result = await _check(_config(model_server, output_rail=True), messages)

    assert result.status == StatusEnum.SUCCESS
    assert result.rails_status[_OUTPUT_FLOW].status == StatusEnum.SUCCESS
    assert _models_called(model_server) == ["safety-judge"]


async def test_input_only_config_blocks_unsafe_user_message_without_generating(model_server: HTTPServer) -> None:
    messages = [{"role": "user", "content": _UNSAFE_TEXT}]

    result = await _check(_config(model_server, input_rail=True), messages)

    assert result.status == StatusEnum.BLOCKED
    assert result.rails_status[_INPUT_FLOW].status == StatusEnum.BLOCKED
    assert _models_called(model_server) == ["safety-judge"]


async def test_output_rail_with_nothing_to_judge_does_not_claim_a_pass(model_server: HTTPServer) -> None:
    """With no assistant message the output rail never runs, so it must not be reported as passed."""
    result = await _check(_config(model_server, output_rail=True), [{"role": "user", "content": "Hi"}])

    assert result.status == StatusEnum.SUCCESS
    assert result.rails_status == {}
    assert _models_called(model_server) == []


async def test_combined_rails_report_each_rail_that_ran(model_server: HTTPServer) -> None:
    config = _config(model_server, input_rail=True, output_rail=True)
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."}]

    result = await _check(config, messages)

    assert result.status == StatusEnum.SUCCESS
    assert {name: status.status for name, status in result.rails_status.items()} == {
        _INPUT_FLOW: StatusEnum.SUCCESS,
        _OUTPUT_FLOW: StatusEnum.SUCCESS,
    }
    assert _models_called(model_server) == ["safety-judge", "safety-judge"]


async def test_input_block_is_reported_and_output_rail_does_not_run(model_server: HTTPServer) -> None:
    config = _config(model_server, input_rail=True, output_rail=True)
    messages = [{"role": "user", "content": _UNSAFE_TEXT}, {"role": "assistant", "content": "Hello there."}]

    result = await _check(config, messages)

    assert result.status == StatusEnum.BLOCKED
    assert result.rails_status[_INPUT_FLOW].status == StatusEnum.BLOCKED
    assert _OUTPUT_FLOW not in result.rails_status
    assert _models_called(model_server) == ["safety-judge"]


async def test_run_check_async_rejects_state(model_server: HTTPServer) -> None:
    engine = create_rails_engine(_config(model_server, output_rail=True))

    with pytest.raises(ValueError, match="state is not supported"):
        await run_check_async(engine, messages=_UNSAFE_OUTPUT, rail_types=["output"], state={"x": 1})


async def test_generate_path_still_calls_the_main_model(model_server: HTTPServer) -> None:
    """Pins the behavior the checks handler must avoid, so the regression tests above stay meaningful."""
    engine = create_rails_engine(_config(model_server, output_rail=True))

    await run_generate_async(engine, messages=_UNSAFE_OUTPUT, options={"rails": {"input": False, "output": True}})

    assert _models_called(model_server)[0] == "main-model"
