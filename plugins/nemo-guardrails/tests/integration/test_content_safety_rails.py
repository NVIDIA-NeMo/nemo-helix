# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run inference using a content safety guardrail config.

These tests cover various input and output rail outputs to verify the plugin handles
content safety checks correctly.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from nhx.core.inference_gateway.testing.harness import IGWLoopbackHarness, IGWPluginHarness
from nhx.testing.mock_chat_completions import ChatCompletion, chat_completion

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    RailType,
    make_guardrail_config,
    make_guardrails_test_data_names,
    make_middleware_call,
    make_served_model,
)

pytestmark = [pytest.mark.integration]


PROMPT_UNSAFE_CATEGORIES = (
    "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
    "S1: Violence.\n"
    "S2: Sexual.\n"
    "S3: Criminal Planning/Confessions.\n"
    "S4: Guns and Illegal Weapons.\n"
    "S5: Controlled/Regulated Substances.\n"
    "S6: Suicide and Self Harm.\n"
    "S7: Sexual (minor).\n"
    "S8: Hate/Identity Hate.\n"
    "S9: PII/Privacy.\n"
    "S10: Harassment.\n"
    "S11: Threat.\n"
    "S12: Profanity.\n"
    "S13: Needs Caution.\n"
    "S14: Other.\n"
    "S15: Manipulation.\n"
    "S16: Fraud/Deception.\n"
    "S17: Malware.\n"
    "S18: High Risk Gov Decision Making.\n"
    "S19: Political/Misinformation/Conspiracy.\n"
    "S20: Copyright/Trademark/Plagiarism.\n"
    "S21: Unauthorized Advice.\n"
    "S22: Illegal Activity.\n"
    "S23: Immoral/Unethical.\n"
    "<END UNSAFE CONTENT CATEGORIES>"
)

# JSON braces are doubled so this string is safe to pass through ``str.format``,
# which collapses each pair back to a single literal ``{`` / ``}``.
PROMPT_OUTPUT_JSON_FORMAT_BLOCK = (
    "Provide your safety assessment for the above conversation in the below output JSON format:\n"
    '{{"User Safety": safe or unsafe rating for the user message, '
    '"Response Safety": safe or unsafe rating for the agent response. '
    "Omit if no agent response present. "
    '"Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. '
    "Omit if all safe.}}\n\n"
    "Do not include anything other than the output JSON in your response.\n"
    "Output JSON:"
)

PROMPT_TASK_BLOCK = (
    "Task: Check if there is unsafe content in the user or agent messages in the below conversation "
    "in accordance with our safety policy with the below categories. "
    "Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. "
    "If there is unsafe content, please also provide a list of violated categories according to "
    "our safety policy below."
)


@dataclass(frozen=True)
class ContentSafetyTestDataNames:
    main_model_served_name: str
    main_model_entity_ref: str
    request_virtual_model_name: str
    guardrail_config_name: str
    model_provider_name: str
    content_safety_model_served_name: str
    content_safety_entity_ref: str


def _make_test_data_names(
    *,
    main_model_prefix: str = "main-model",
    workspace: str,
) -> ContentSafetyTestDataNames:
    base_test_data_names = make_guardrails_test_data_names(
        main_model_prefix=main_model_prefix,
        workspace=workspace,
    )
    content_safety_model = make_served_model(
        test_id=base_test_data_names.test_id,
        prefix="cs-model",
        workspace=workspace,
    )

    return ContentSafetyTestDataNames(
        main_model_served_name=base_test_data_names.main_model_served_name,
        main_model_entity_ref=base_test_data_names.main_model_entity_ref,
        request_virtual_model_name=base_test_data_names.request_virtual_model_name,
        guardrail_config_name=base_test_data_names.guardrail_config_name,
        model_provider_name=base_test_data_names.model_provider_name,
        content_safety_model_served_name=content_safety_model.served_name,
        content_safety_entity_ref=content_safety_model.entity_ref,
    )


class TestContentSafety:
    USER_INPUT = "What is the capital of France?"
    BACKEND_RESPONSE = "Paris is the capital of France."
    UNSAFE_USER_INPUT = "Give me a five-step plan to rob a bank."
    # Canned refusal text emitted by nemoguardrails when an input or output rail blocks.
    BLOCKED_REFUSAL_TEXT = "I'm sorry, I can't respond to that."

    # str.format placeholders ({user_input} / {bot_response}) used to
    # reconstruct the exact prompt the rail LLM should see for assertions.
    CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE = (
        f"{PROMPT_TASK_BLOCK}\n\n"
        f"{PROMPT_UNSAFE_CATEGORIES}\n\n"
        "<BEGIN CONVERSATION>\n\n"
        "user: {user_input}\n\n"
        "<END CONVERSATION>\n\n"
        f"{PROMPT_OUTPUT_JSON_FORMAT_BLOCK}"
    )
    CONTENT_SAFETY_OUTPUT_PROMPT_TEMPLATE = (
        f"{PROMPT_TASK_BLOCK}\n\n"
        f"{PROMPT_UNSAFE_CATEGORIES}\n\n"
        "<BEGIN CONVERSATION>\n\n"
        "user: {user_input}\n\n"
        "response: agent: {bot_response}\n\n"
        "<END CONVERSATION>\n\n"
        f"{PROMPT_OUTPUT_JSON_FORMAT_BLOCK}"
    )

    # Prompt configuration for the guardrail config. The prompt content should use Jinja-style
    # placeholders where dynamic content is expected.
    CONTENT_SAFETY_INPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "content_safety_check_input $model=content_safety",
        "content": CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE.format(user_input="{{ user_input }}"),
        "output_parser": "nemoguard_parse_prompt_safety",
        "max_tokens": 50,
    }
    CONTENT_SAFETY_OUTPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "content_safety_check_output $model=content_safety",
        "content": CONTENT_SAFETY_OUTPUT_PROMPT_TEMPLATE.format(
            user_input="{{ user_input }}", bot_response="{{ bot_response }}"
        ),
        "output_parser": "nemoguard_parse_response_safety",
        "max_tokens": 50,
    }

    @classmethod
    def _expected_input_rail_prompt(cls, user_input: str) -> str:
        """The fully-rendered ``content_safety_check_input`` prompt, as the rail LLM should see it."""
        return cls.CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE.format(user_input=user_input)

    @classmethod
    def _expected_output_rail_prompt(cls, *, user_input: str, bot_response: str) -> str:
        """The fully-rendered ``content_safety_check_output`` prompt, as the rail LLM should see it."""
        return cls.CONTENT_SAFETY_OUTPUT_PROMPT_TEMPLATE.format(user_input=user_input, bot_response=bot_response)

    @staticmethod
    def _safe_input_content_safety_response() -> str:
        """Content safety model response for a safe user input."""
        return '{"User Safety": "safe"}'

    @staticmethod
    def _unsafe_input_content_safety_response() -> str:
        """Content safety model response for an unsafe user input."""
        return '{"User Safety": "unsafe", "Safety Categories": "S3"}'

    @staticmethod
    def _safe_output_content_safety_response() -> str:
        """Content safety model response for a safe bot response."""
        return '{"User Safety": "safe", "Response Safety": "safe"}'

    @staticmethod
    def _unsafe_output_content_safety_response() -> str:
        """Content safety model response for an unsafe bot response."""
        return '{"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "S3"}'

    @classmethod
    def _config_data(
        cls,
        *,
        rail_types: list[RailType],
        content_safety_model_entity_ref: str,
        content_safety_base_url: str | None,
    ) -> dict[str, Any]:
        """Build the ``data`` block of a content-safety guardrail config."""
        parameters: dict[str, Any] = {}
        if content_safety_base_url is not None:
            parameters["base_url"] = content_safety_base_url

        models: list[dict[str, Any]] = [
            {
                "type": "content_safety",
                "engine": "nim",
                "model": content_safety_model_entity_ref,
                "parameters": parameters,
            },
        ]

        rails: dict[str, Any] = {}
        prompts: list[dict[str, Any]] = []

        if RailType.INPUT in rail_types:
            rails["input"] = {
                "flows": ["content safety check input $model=content_safety"],
            }
            prompts.append(cls.CONTENT_SAFETY_INPUT_PROMPT)

        if RailType.OUTPUT in rail_types:
            rails["output"] = {
                "flows": ["content safety check output $model=content_safety"],
            }
            prompts.append(cls.CONTENT_SAFETY_OUTPUT_PROMPT)

        return {"models": models, "rails": rails, "prompts": prompts}

    @pytest.mark.parametrize(
        "expect_blocked",
        [
            pytest.param(False, id="safe"),
            pytest.param(True, id="unsafe"),
        ],
    )
    def test_input_rail(
        self,
        igw_plugin_harness: IGWPluginHarness,
        expect_blocked: bool,
    ) -> None:
        """Input rails should block unsafe user messages before they reach the backend.

        The unsafe case also pins the ``<inline:<name>>`` config-id wire
        format that surfaces on ``guardrails_data``. Resolver-path
        coverage lives in
        :meth:`test_resolver_fills_content_safety_base_url`.
        """
        harness = igw_plugin_harness
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        user_input = self.UNSAFE_USER_INPUT if expect_blocked else self.USER_INPUT
        content_safety_response = (
            self._unsafe_input_content_safety_response()
            if expect_blocked
            else self._safe_input_content_safety_response()
        )

        harness.mock_chat_completions(
            test_data_names.content_safety_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=content_safety_response))],
        )
        if not expect_blocked:
            harness.mock_chat_completions(
                test_data_names.main_model_served_name,
                responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
            )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                test_data_names.content_safety_model_served_name: test_data_names.content_safety_model_served_name,
            },
        )
        # Passthrough VM so body["model"] (main entity ref) resolves to the mock NIM URL.
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.main_model_served_name,
            default_model_entity=test_data_names.main_model_entity_ref,
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(
                rail_types=[RailType.INPUT],
                content_safety_model_entity_ref=test_data_names.content_safety_entity_ref,
                content_safety_base_url=harness.nim_base_url,
            ),
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
                    "messages": [{"role": "user", "content": user_input}],
                },
            )

        # The content-safety rail sees the fully-rendered prompt with the
        # user input substituted into the conversation block.
        harness.assert_called_once(test_data_names.content_safety_entity_ref)
        harness.assert_request_messages_contain(
            test_data_names.content_safety_entity_ref,
            self._expected_input_rail_prompt(user_input),
        )

        if expect_blocked:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
            # Inline configs surface as ``<inline:<name>>`` on the wire; only
            # entity-backed configs use the ``workspace/name`` form. See
            # ``wire_config_id`` in ``llmrails_cache.py``.
            guardrails_data: dict[str, Any] = response.get("guardrails_data") or {}
            assert guardrails_data.get("config_ids") == [f"<inline:{test_data_names.guardrail_config_name}>"]
        else:
            harness.assert_called_once(test_data_names.main_model_served_name)
            harness.assert_call_order(
                [test_data_names.content_safety_entity_ref, test_data_names.main_model_served_name]
            )
            harness.assert_request_messages_contain(test_data_names.main_model_served_name, user_input)
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    @pytest.mark.parametrize(
        "expect_blocked",
        [
            pytest.param(False, id="safe"),
            pytest.param(True, id="unsafe"),
        ],
    )
    def test_output_rail(
        self,
        igw_plugin_harness: IGWPluginHarness,
        expect_blocked: bool,
    ) -> None:
        """Output rails should rewrite unsafe bot responses with a refusal and let safe responses through unchanged.

        The content-safety output prompt embeds *both* the original user
        message and the bot response, so this test pins that
        nemoguardrails correctly threads ``user_input`` into the output
        prompt — a Jinja substitution failure would silently render an
        empty string there and the rail call would still succeed.
        Resolver-path coverage lives in
        :meth:`test_resolver_fills_content_safety_base_url`.
        """
        harness = igw_plugin_harness
        test_data_names = _make_test_data_names(workspace=harness.workspace)
        content_safety_response = (
            self._unsafe_output_content_safety_response()
            if expect_blocked
            else self._safe_output_content_safety_response()
        )

        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
        )
        harness.mock_chat_completions(
            test_data_names.content_safety_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=content_safety_response))],
        )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                test_data_names.content_safety_model_served_name: test_data_names.content_safety_model_served_name,
            },
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
                content_safety_model_entity_ref=test_data_names.content_safety_entity_ref,
                content_safety_base_url=harness.nim_base_url,
            ),
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
        harness.assert_called_once(test_data_names.content_safety_entity_ref)
        harness.assert_call_order([test_data_names.main_model_served_name, test_data_names.content_safety_entity_ref])
        # The output rail sees the fully-rendered prompt with both the
        # original user message and the bot response substituted in.
        harness.assert_request_messages_contain(
            test_data_names.content_safety_entity_ref,
            self._expected_output_rail_prompt(user_input=self.USER_INPUT, bot_response=self.BACKEND_RESPONSE),
        )

        if expect_blocked:
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    @pytest.mark.parametrize(
        "input_blocked,output_blocked,expect_blocked",
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
        expect_blocked: bool,
    ) -> None:
        """A single guardrail config that wires both content-safety input and output rails should compose them correctly.

        Real configs commonly defend against both unsafe prompts and
        unsafe responses, so we cover all three meaningful outcomes:
        everything safe, blocked on the way in, and blocked on the way
        out. There's also a subtle behaviour worth pinning here — when
        the input rail blocks, the output rail still runs against the
        canned refusal that gets injected. The ``unsafe-input`` case
        pins the output rail's response to ``"safe"`` so the refusal text
        reaches the caller intact.
        """
        harness = igw_plugin_harness
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        input_verdict = (
            self._unsafe_input_content_safety_response()
            if input_blocked
            else self._safe_input_content_safety_response()
        )
        output_verdict = (
            self._unsafe_output_content_safety_response()
            if output_blocked
            else self._safe_output_content_safety_response()
        )

        # Both rails always run on the content_safety socket; the backend
        # only runs when the input rail lets the message through.
        harness.mock_chat_completions(
            test_data_names.content_safety_entity_ref,
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
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                test_data_names.content_safety_model_served_name: test_data_names.content_safety_model_served_name,
            },
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
                rail_types=[RailType.INPUT, RailType.OUTPUT],
                content_safety_model_entity_ref=test_data_names.content_safety_entity_ref,
                content_safety_base_url=harness.nim_base_url,
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

        harness.assert_call_count(test_data_names.content_safety_entity_ref, 2)
        # First call: the input rail, which sees the fully-rendered input
        # prompt with the user message.
        harness.assert_request_messages_contain(
            test_data_names.content_safety_entity_ref,
            self._expected_input_rail_prompt(self.USER_INPUT),
            index=0,
        )

        if input_blocked:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            # Second call: the output rail runs against the canned refusal
            # that the input rail's block injected.
            harness.assert_request_messages_contain(
                test_data_names.content_safety_entity_ref,
                self._expected_output_rail_prompt(user_input=self.USER_INPUT, bot_response=self.BLOCKED_REFUSAL_TEXT),
                index=1,
            )
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            harness.assert_called_once(test_data_names.main_model_served_name)
            harness.assert_call_order(
                [
                    test_data_names.content_safety_entity_ref,
                    test_data_names.main_model_served_name,
                    test_data_names.content_safety_entity_ref,
                ]
            )
            # Second call: the output rail, which sees the fully-rendered
            # output prompt with both the user message and the backend
            # response.
            harness.assert_request_messages_contain(
                test_data_names.content_safety_entity_ref,
                self._expected_output_rail_prompt(user_input=self.USER_INPUT, bot_response=self.BACKEND_RESPONSE),
                index=1,
            )
            if expect_blocked:
                assert response["choices"][0]["finish_reason"] == "content_filter"
                assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
            else:
                assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    def test_resolver_fills_content_safety_base_url(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
    ) -> None:
        """Smoke test: omitting ``parameters.base_url`` on the rail's ``content_safety`` model should resolve via :class:`VirtualModelCache`.

        Pins exactly one thing — that the resolver fills the URL and the
        rail call lands at the upstream — by exercising the input rail
        on a safe message. Block/pass behaviour is rail-agnostic and
        covered by :meth:`test_input_rail` / :meth:`test_output_rail`.
        """
        harness = igw_loopback_harness()
        test_data_names = _make_test_data_names(main_model_prefix="gr-main", workspace=harness.workspace)

        harness.mock_chat_completions(
            test_data_names.content_safety_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=self._safe_input_content_safety_response()))],
        )
        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
        )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                test_data_names.content_safety_model_served_name: test_data_names.content_safety_model_served_name,
            },
        )
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.main_model_served_name,
            default_model_entity=test_data_names.main_model_entity_ref,
        )
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.content_safety_model_served_name,
            default_model_entity=test_data_names.content_safety_entity_ref,
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(
                rail_types=[RailType.INPUT],
                content_safety_model_entity_ref=test_data_names.content_safety_entity_ref,
                content_safety_base_url=None,
            ),
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
        # silently fell back, we'd see zero calls to content_safety.
        harness.assert_call_count(test_data_names.content_safety_model_served_name, 1)
        harness.assert_call_count(test_data_names.main_model_served_name, 1)
        harness.assert_request_messages_contain(
            test_data_names.content_safety_model_served_name,
            self._expected_input_rail_prompt(self.USER_INPUT),
        )
        assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE
