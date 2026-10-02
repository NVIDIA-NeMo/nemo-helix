# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run inference using a self-check guardrail config.

These tests cover various input and output rail outputs to verify the plugin handles
self-check rails correctly.
"""

from collections.abc import Callable
from typing import Any

import pytest
from nhx.core.inference_gateway.testing.harness import IGWLoopbackHarness, IGWPluginHarness
from nhx.testing.mock_chat_completions import ChatCompletion, chat_completion

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    RailType,
    make_guardrail_config,
    make_guardrails_test_data_names,
    make_middleware_call,
)

pytestmark = [pytest.mark.integration]


class TestSelfCheck:
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

    @classmethod
    def _messages_with_tool_history(cls) -> list[dict[str, Any]]:
        return [
            {"role": "user", "content": "Look up the weather in New York."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"New York"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "Sunny, 75 degrees Fahrenheit.",
            },
            {"role": "user", "content": cls.USER_INPUT},
        ]

    @classmethod
    def _build_self_check_input_prompt(cls) -> dict[str, Any]:
        return {
            "task": "self_check_input",
            "content": cls.SELF_CHECK_INPUT_PROMPT_TEMPLATE.replace("{user_input}", "{{ user_input }}"),
        }

    @classmethod
    def _build_self_check_output_prompt(cls) -> dict[str, Any]:
        return {
            "task": "self_check_output",
            "content": cls.SELF_CHECK_OUTPUT_PROMPT_TEMPLATE.replace("{bot_response}", "{{ bot_response }}"),
        }

    @classmethod
    def _expected_input_rail_prompt(cls, user_input: str) -> str:
        """The fully-rendered ``self_check_input`` prompt, as the rail LLM should see it."""
        return cls.SELF_CHECK_INPUT_PROMPT_TEMPLATE.format(user_input=user_input)

    @classmethod
    def _expected_output_rail_prompt(cls, bot_response: str) -> str:
        """The fully-rendered ``self_check_output`` prompt, as the rail LLM should see it."""
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
        main_base_url: str | None,
    ) -> dict[str, Any]:
        """Build the ``data`` block of a self-check guardrail config.

        ``rail_types`` selects which rail flows are wired in (input,
        output, or both). When ``main_base_url`` is given, the rail's
        ``main`` LLM is pointed at it directly; pass ``None`` to omit the
        ``main`` entry entirely and let the plugin's resolver fill the URL
        in at request time (callers must register a passthrough VM for the
        resolver to find).
        """
        models: list[dict[str, Any]] = []
        if main_base_url is not None:
            models.append(
                {
                    "type": "main",
                    "engine": "nim",
                    "parameters": {"base_url": main_base_url},
                }
            )

        rails: dict[str, Any] = {}
        prompts: list[dict[str, Any]] = []

        if RailType.INPUT in rail_types:
            rails["input"] = {"flows": ["self check input"]}
            prompts.append(cls._build_self_check_input_prompt())

        if RailType.OUTPUT in rail_types:
            rails["output"] = {
                "flows": ["self check output"],
                "streaming": {"enabled": True},
            }
            prompts.append(cls._build_self_check_output_prompt())

        return {"models": models, "rails": rails, "prompts": prompts}

    @pytest.mark.parametrize(
        "expected_blocked_response",
        [
            pytest.param(False, id="safe"),
            pytest.param(True, id="unsafe"),
        ],
    )
    def test_input_rail(
        self,
        igw_plugin_harness: IGWPluginHarness,
        expected_blocked_response: bool,
    ) -> None:
        """Input rails should block unsafe user messages before they reach the backend, and let safe user messages through.

        Resolver-path coverage lives in
        :meth:`test_resolver_fills_main_base_url`.
        """
        harness = igw_plugin_harness
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)

        self_check_response = (
            self._unsafe_input_self_check_response()
            if expected_blocked_response
            else self._safe_input_self_check_response()
        )

        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=self_check_response))],
        )
        if not expected_blocked_response:
            harness.mock_chat_completions(
                test_data_names.main_model_served_name,
                responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
            )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(rail_types=[RailType.INPUT], main_base_url=harness.nim_base_url),
        )
        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
            )
            response = harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": [{"role": "user", "content": self.USER_INPUT}],
                },
            )

        # The input rail sees the fully-rendered self_check_input prompt
        # with the original user message.
        harness.assert_called_once(test_data_names.main_model_entity_ref)
        harness.assert_request_messages_contain(
            test_data_names.main_model_entity_ref,
            self._expected_input_rail_prompt(self.USER_INPUT),
        )

        if expected_blocked_response:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            harness.assert_called_once(test_data_names.main_model_served_name)
            harness.assert_call_order([test_data_names.main_model_entity_ref, test_data_names.main_model_served_name])
            # Backend hop: original user message passes through unchanged.
            harness.assert_request_messages_contain(test_data_names.main_model_served_name, self.USER_INPUT)
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    def test_input_rail_checks_current_user_after_tool_history(
        self,
        igw_plugin_harness: IGWPluginHarness,
    ) -> None:
        harness = igw_plugin_harness
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)
        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=self._unsafe_input_self_check_response()))],
        )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )
        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(rail_types=[RailType.INPUT], main_base_url=harness.nim_base_url),
        )

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
            )
            response = harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": self._messages_with_tool_history(),
                },
            )

        harness.assert_called_once(test_data_names.main_model_entity_ref)
        harness.assert_no_calls_to(test_data_names.main_model_served_name)
        harness.assert_request_messages_contain(
            test_data_names.main_model_entity_ref,
            self._expected_input_rail_prompt(self.USER_INPUT),
        )
        assert response["choices"][0]["finish_reason"] == "content_filter"
        assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT

    @pytest.mark.parametrize(
        "expected_blocked_response",
        [
            pytest.param(False, id="safe"),
            pytest.param(True, id="unsafe"),
        ],
    )
    def test_output_rail(
        self,
        igw_plugin_harness: IGWPluginHarness,
        expected_blocked_response: bool,
    ) -> None:
        """Output rails should rewrite unsafe bot responses with a refusal and let safe responses through unchanged.

        Unlike input rails, the backend always runs first — the rail only
        gets to review what the model returned. Resolver-path coverage
        lives in :meth:`test_resolver_fills_main_base_url`.
        """
        harness = igw_plugin_harness
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)
        self_check_response = (
            self._unsafe_output_self_check_response()
            if expected_blocked_response
            else self._safe_output_self_check_response()
        )

        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
        )
        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=self_check_response))],
        )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(rail_types=[RailType.OUTPUT], main_base_url=harness.nim_base_url),
        )
        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                response_middleware=[make_middleware_call(guardrail_config)],
            )
            response = harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": [{"role": "user", "content": self.USER_INPUT}],
                },
            )

        harness.assert_called_once(test_data_names.main_model_served_name)
        harness.assert_called_once(test_data_names.main_model_entity_ref)
        harness.assert_call_order([test_data_names.main_model_served_name, test_data_names.main_model_entity_ref])
        # The output rail sees the fully-rendered self_check_output prompt
        # with the backend response.
        harness.assert_request_messages_contain(
            test_data_names.main_model_entity_ref,
            self._expected_output_rail_prompt(self.BACKEND_RESPONSE),
        )

        if expected_blocked_response:
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    def test_output_rail_checks_response_after_tool_history(
        self,
        igw_plugin_harness: IGWPluginHarness,
    ) -> None:
        harness = igw_plugin_harness
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)
        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
        )
        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=self._unsafe_output_self_check_response()))],
        )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )
        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(rail_types=[RailType.OUTPUT], main_base_url=harness.nim_base_url),
        )

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                response_middleware=[make_middleware_call(guardrail_config)],
            )
            response = harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": self._messages_with_tool_history(),
                },
            )

        harness.assert_call_order([test_data_names.main_model_served_name, test_data_names.main_model_entity_ref])
        harness.assert_request_messages_contain(
            test_data_names.main_model_entity_ref,
            self._expected_output_rail_prompt(self.BACKEND_RESPONSE),
        )
        assert response["choices"][0]["finish_reason"] == "content_filter"
        assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT

    @pytest.mark.parametrize(
        "input_blocked,output_blocked,expected_blocked_response",
        [
            pytest.param(False, False, False, id="all-safe"),
            pytest.param(True, False, True, id="unsafe-input"),
            pytest.param(False, True, True, id="unsafe-output"),
        ],
    )
    def test_input_and_output_rails(
        self,
        igw_plugin_harness: IGWPluginHarness,
        input_blocked: bool,
        output_blocked: bool,
        expected_blocked_response: bool,
    ) -> None:
        """A single guardrail config that wires both input and output rails should compose them correctly.

        Real configs commonly defend against both unsafe prompts and
        unsafe responses, so we cover all three meaningful outcomes:
        everything safe, blocked on the way in, and blocked on the way
        out. There's also a subtle behaviour worth pinning here — when
        the input rail blocks, the output rail still runs against the
        canned refusal that gets injected. The ``unsafe-input`` case
        pins the output rail's response to ``"No"`` so the refusal text
        reaches the caller intact.
        """
        harness = igw_plugin_harness
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)

        input_verdict = (
            self._unsafe_input_self_check_response() if input_blocked else self._safe_input_self_check_response()
        )
        output_verdict = (
            self._unsafe_output_self_check_response() if output_blocked else self._safe_output_self_check_response()
        )

        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[
                ChatCompletion(body=chat_completion(content=input_verdict)),
                ChatCompletion(body=chat_completion(content=output_verdict)),
            ],
        )
        if not input_blocked:
            harness.mock_chat_completions(
                test_data_names.main_model_served_name,
                responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
            )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(
                rail_types=[RailType.INPUT, RailType.OUTPUT],
                main_base_url=harness.nim_base_url,
            ),
        )
        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
                response_middleware=[make_middleware_call(guardrail_config)],
            )
            response = harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": [{"role": "user", "content": self.USER_INPUT}],
                },
            )

        # Both rails always run on this socket; only the backend is skipped
        # when the input rail blocks.
        harness.assert_call_count(test_data_names.main_model_entity_ref, 2)
        # First call: the input rail, which sees the fully-rendered
        # self_check_input prompt with the original user message.
        harness.assert_request_messages_contain(
            test_data_names.main_model_entity_ref,
            self._expected_input_rail_prompt(self.USER_INPUT),
            index=0,
        )

        if input_blocked:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            harness.assert_call_order([test_data_names.main_model_entity_ref, test_data_names.main_model_entity_ref])
            # Second call: the output rail runs against the canned refusal
            # that the input rail's block injected, so the rail sees the
            # refusal as the bot response under review.
            harness.assert_request_messages_contain(
                test_data_names.main_model_entity_ref,
                self._expected_output_rail_prompt(self.BLOCKED_REFUSAL_TEXT),
                index=1,
            )
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            harness.assert_called_once(test_data_names.main_model_served_name)
            harness.assert_call_order(
                [
                    test_data_names.main_model_entity_ref,
                    test_data_names.main_model_served_name,
                    test_data_names.main_model_entity_ref,
                ]
            )
            # Second call: the output rail, which sees the fully-rendered
            # self_check_output prompt with the backend response.
            harness.assert_request_messages_contain(
                test_data_names.main_model_entity_ref,
                self._expected_output_rail_prompt(self.BACKEND_RESPONSE),
                index=1,
            )
            if expected_blocked_response:
                assert response["choices"][0]["finish_reason"] == "content_filter"
                assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
            else:
                assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    def test_resolver_fills_main_base_url(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
    ) -> None:
        """Smoke test: omitting ``parameters.base_url`` on the rail's ``main`` model should resolve via :class:`VirtualModelCache`.

        Pins exactly one thing — that the resolver fills the URL and the
        rail call lands at the upstream — by exercising the input rail
        on a safe message. Block/pass behaviour is rail-agnostic and
        covered by :meth:`test_input_rail` / :meth:`test_output_rail`.
        """
        harness = igw_loopback_harness()
        test_data_names = make_guardrails_test_data_names(
            main_model_prefix="gr-main",
            workspace=harness.workspace,
        )

        # The rail call (resolver-filled URL → IGW loopback → passthrough VM
        # proxy) and the backend completion both hit the same socket since
        # IGW rewrites body["model"] to the served name at each VM hop.
        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[
                ChatCompletion(body=chat_completion(content="No")),
                ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE)),
            ],
        )
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
            data=self._config_data(rail_types=[RailType.INPUT], main_base_url=None),
        )
        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
            )
            response = harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": [{"role": "user", "content": self.USER_INPUT}],
                },
            )

        # The rail call landing at all is the assertion — if the resolver
        # silently fell back, we'd see one call (the backend) instead of two.
        harness.assert_call_count(test_data_names.main_model_served_name, 2)
        harness.assert_request_messages_contain(
            test_data_names.main_model_served_name,
            self._expected_input_rail_prompt(self.USER_INPUT),
            index=0,
        )
        assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE
