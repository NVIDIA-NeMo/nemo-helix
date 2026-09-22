# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run streaming inference using a self-check guardrail config.

These tests cover streaming behavior with self-check rails configured. The
streaming SSE wiring is rail-agnostic at the plugin layer, so these tests cover
the streaming workflow without repeating the same cases for every rail type.
"""

from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from nhx.core.inference_gateway.testing.harness import IGWLoopbackHarness
from nhx.testing.mock_chat_completions import (
    ChatCompletion,
    ChatCompletionStream,
    build_chat_completion_stream_from_content,
    build_content_from_chunks,
    chat_completion,
)

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    RailType,
    make_guardrail_config,
    make_guardrails_test_data_names,
    make_middleware_call,
)

pytestmark = [pytest.mark.integration]


class TestStreaming:
    USER_INPUT = "What is the capital of France?"
    BACKEND_RESPONSE = "Paris is the capital of France."
    # Canned refusal text emitted by nemoguardrails when an input or output rail blocks.
    BLOCKED_REFUSAL_TEXT = "I'm sorry, I can't respond to that."

    SELF_CHECK_INPUT_PROMPT_TEMPLATE = "Should the user message be blocked?\n\nUser: {user_input}\nAnswer yes or no:"
    SELF_CHECK_OUTPUT_PROMPT_TEMPLATE = (
        "Your task is to check if the bot message below is harmful or inappropriate.\n\n"
        'Bot message: "{bot_response}"\n\n'
        "Question: Should the message be blocked?\n"
        "Answer yes or no:"
    )

    SELF_CHECK_INPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "self_check_input",
        "content": SELF_CHECK_INPUT_PROMPT_TEMPLATE.replace("{user_input}", "{{ user_input }}"),
    }
    SELF_CHECK_OUTPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "self_check_output",
        "content": SELF_CHECK_OUTPUT_PROMPT_TEMPLATE.replace("{bot_response}", "{{ bot_response }}"),
    }

    @classmethod
    def _expected_input_rail_prompt(cls, user_input: str) -> str:
        return cls.SELF_CHECK_INPUT_PROMPT_TEMPLATE.format(user_input=user_input)

    @classmethod
    def _expected_output_rail_prompt(cls, bot_response: str) -> str:
        return cls.SELF_CHECK_OUTPUT_PROMPT_TEMPLATE.format(bot_response=bot_response)

    @staticmethod
    def _safe_input_self_check_response() -> str:
        return "No"

    @staticmethod
    def _unsafe_input_self_check_response() -> str:
        return "Yes"

    @staticmethod
    def _safe_output_self_check_response() -> str:
        return "No"

    @staticmethod
    def _unsafe_output_self_check_response() -> str:
        return "Yes"

    @classmethod
    def _config_data(
        cls,
        *,
        rail_types: list[RailType],
        stream_first: bool | None = None,
    ) -> dict[str, Any]:
        rails: dict[str, Any] = {}
        prompts: list[dict[str, Any]] = []

        if RailType.INPUT in rail_types:
            rails["input"] = {"flows": ["self check input"]}
            prompts.append(cls.SELF_CHECK_INPUT_PROMPT)

        if RailType.OUTPUT in rail_types:
            output_streaming: dict[str, Any] = {"enabled": True}
            if stream_first is not None:
                output_streaming["stream_first"] = stream_first
            rails["output"] = {
                "flows": ["self check output"],
                "streaming": output_streaming,
            }
            prompts.append(cls.SELF_CHECK_OUTPUT_PROMPT)

        return {"rails": rails, "prompts": prompts}

    @pytest.mark.parametrize(
        "expected_blocked_response",
        [
            pytest.param(False, id="safe"),
            pytest.param(True, id="unsafe"),
        ],
    )
    def test_input_rail_streaming(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        expected_blocked_response: bool,
    ) -> None:
        """Input rails should block unsafe user messages on streaming requests, and otherwise stream the backend's chunks through to the caller.

        When the input rail blocks, the request is never proxied to the backend model,
        and the Guardrails plugin returns the refusal messages as a JSON body rather than
        an SSE stream.
        """
        harness = igw_loopback_harness()
        test_data_names = make_guardrails_test_data_names(
            main_model_prefix="gr-main",
            workspace=harness.workspace,
        )

        self_check_response = (
            self._unsafe_input_self_check_response()
            if expected_blocked_response
            else self._safe_input_self_check_response()
        )
        model_responses: list[ChatCompletion | ChatCompletionStream] = [
            ChatCompletion(body=chat_completion(content=self_check_response))
        ]

        if not expected_blocked_response:
            model_responses.append(build_chat_completion_stream_from_content(self.BACKEND_RESPONSE))

        harness.mock_chat_completions(test_data_names.main_model_served_name, responses=model_responses)
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.main_model_served_name,
            default_model_entity=test_data_names.main_model_entity_ref,
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(rail_types=[RailType.INPUT]),
        )
        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
            )
            response_payload = harness.stream_chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": [{"role": "user", "content": self.USER_INPUT}],
                },
            )

        # First call: the input rail, which sees the fully-rendered
        # self_check_input prompt with the original user message.
        harness.assert_request_messages_contain(
            test_data_names.main_model_served_name,
            self._expected_input_rail_prompt(self.USER_INPUT),
            index=0,
        )

        if expected_blocked_response:
            harness.assert_call_count(test_data_names.main_model_served_name, 1)
            assert isinstance(response_payload, dict)
            assert response_payload["choices"][0]["finish_reason"] == "content_filter"
            assert response_payload["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            harness.assert_call_count(test_data_names.main_model_served_name, 2)
            assert isinstance(response_payload, list)
            assert build_content_from_chunks(response_payload) == self.BACKEND_RESPONSE

    @pytest.mark.parametrize(
        "expected_blocked_response",
        [
            pytest.param(False, id="safe"),
            pytest.param(True, id="unsafe"),
        ],
    )
    @pytest.mark.parametrize(
        "stream_first",
        [
            pytest.param(True, id="stream-first"),
            pytest.param(False, id="rail-first"),
        ],
    )
    def test_output_rail_streaming(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        expected_blocked_response: bool,
        stream_first: bool,
    ) -> None:
        """Output rails should terminate the SSE stream with a content_blocked error token when the bot response is unsafe, and otherwise stream the backend's chunks through unchanged.

        Behaviour on the unsafe branch differs by ``stream_first``: when
        True, backend chunks reach the caller before the rail decides to
        block, so the partial content is still in the stream alongside
        the trailing error token; when False, the rail runs before any
        token is forwarded, so the caller sees only the error token.
        """
        harness = igw_loopback_harness()
        test_data_names = make_guardrails_test_data_names(
            main_model_prefix="gr-main",
            workspace=harness.workspace,
        )

        self_check_response = (
            self._unsafe_output_self_check_response()
            if expected_blocked_response
            else self._safe_output_self_check_response()
        )
        model_responses: list[ChatCompletion | ChatCompletionStream] = [
            build_chat_completion_stream_from_content(self.BACKEND_RESPONSE),
            ChatCompletion(body=chat_completion(content=self_check_response)),
        ]

        harness.mock_chat_completions(test_data_names.main_model_served_name, responses=model_responses)
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.main_model_served_name,
            default_model_entity=test_data_names.main_model_entity_ref,
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(
                rail_types=[RailType.OUTPUT],
                stream_first=stream_first,
            ),
        )
        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                response_middleware=[make_middleware_call(guardrail_config)],
            )
            response_payload = harness.stream_chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": [{"role": "user", "content": self.USER_INPUT}],
                },
            )

        assert isinstance(response_payload, list)
        chunks = response_payload
        harness.assert_call_count(test_data_names.main_model_served_name, 2)
        # First call: the streaming backend hop. The original user message
        # passes through unchanged.
        harness.assert_request_messages_contain(test_data_names.main_model_served_name, self.USER_INPUT, index=0)
        # Second call: the output rail, which sees the fully-rendered
        # self_check_output prompt with the (joined) backend response.
        harness.assert_request_messages_contain(
            test_data_names.main_model_served_name,
            self._expected_output_rail_prompt(self.BACKEND_RESPONSE),
            index=1,
        )

        if expected_blocked_response:
            content_chunks = chunks[:-1]
            final_chunk = chunks[-1]
            returned_content = build_content_from_chunks(content_chunks)

            assert final_chunk["error"]["code"] == "content_blocked"
            assert final_chunk["error"]["param"] == "self check output"

            if stream_first:
                # Backend tokens reached the caller before the rail blocked the message.
                assert "Paris" in returned_content
            else:
                # Output rail ran and blocked the message before any token was forwarded.
                assert returned_content == ""
        else:
            assert build_content_from_chunks(chunks) == self.BACKEND_RESPONSE
