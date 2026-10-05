# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run inference using a reasoning content-safety guardrail config.

These tests cover input and output rails wired to the
``Nemotron-Content-Safety-Reasoning-4B`` model, including the
``reasoning.enabled`` toggle that switches the model between its
``/think`` and ``/no_think`` modes.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from nhx.core.inference_gateway.testing.harness import IGWPluginHarness
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


PROMPT_TAXONOMY_BLOCK = (
    "Taxonomy: The following are the list of harmful categories that the prompt and response may fall into.\n\n"
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
    "S14: Manipulation.\n"
    "S15: Fraud/Deception.\n"
    "S16: Malware.\n"
    "S17: High Risk Gov Decision Making.\n"
    "S18: Political/Misinformation/Conspiracy.\n"
    "S19: Copyright/Trademark/Plagiarism.\n"
    "S20: Unauthorized Advice.\n"
    "S21: Illegal Activity.\n"
    "S22: Immoral/Unethical."
)

PROMPT_INTRO = (
    "You are a classifier for analyzing interactions between humans and LLMs. "
    "Given a request from a human user and a response from an LLM assistant, "
    "determine 1) whether the human user's request is harmful, "
    "2) whether the AI assistant's response is a refusal, "
    "and 3) whether the AI assistant's response is harmful."
)

PROMPT_OUTPUT_FORMAT_BLOCK = (
    "Use the taxonomy to determine if the prompt and response fall into any of the harmful categories. "
    "Output your answer in the following format:\n\n"
    '"Prompt harm: <harmful/unharmful>\n'
    'Response harm: <harmful/unharmful>"\n\n'
    "Only output the answer in the format specified above after thinking about the request and\n"
    'response. If there is no response, output "Response harm: <harmful/unharmful>" as None.'
)


@dataclass(frozen=True)
class ContentSafetyReasoningTestDataNames:
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
) -> ContentSafetyReasoningTestDataNames:
    base_test_data_names = make_guardrails_test_data_names(
        main_model_prefix=main_model_prefix,
        workspace=workspace,
    )
    content_safety_model = make_served_model(
        test_id=base_test_data_names.test_id,
        prefix="cs-reason-model",
        workspace=workspace,
    )

    return ContentSafetyReasoningTestDataNames(
        main_model_served_name=base_test_data_names.main_model_served_name,
        main_model_entity_ref=base_test_data_names.main_model_entity_ref,
        request_virtual_model_name=base_test_data_names.request_virtual_model_name,
        guardrail_config_name=base_test_data_names.guardrail_config_name,
        model_provider_name=base_test_data_names.model_provider_name,
        content_safety_model_served_name=content_safety_model.served_name,
        content_safety_entity_ref=content_safety_model.entity_ref,
    )


# Jinja branch the prompt template uses for its trailing directive. At
# request time, nemoguardrails substitutes ``reasoning_enabled`` from
# ``rails.config.content_safety.reasoning.enabled`` and the prompt ends
# with either ``/think`` or ``/no_think`` (the token the model was
# trained to read to switch between reasoning-on and reasoning-off).
THINK_DIRECTIVE_JINJA = "{% if reasoning_enabled %}/think{% else %}/no_think{% endif %}"


def _think_directive(reasoning_enabled: bool) -> str:
    """Return the ``/think`` or ``/no_think`` token ``THINK_DIRECTIVE_JINJA`` renders to."""
    return "/think" if reasoning_enabled else "/no_think"


class TestContentSafetyReasoning:
    USER_INPUT = "What is the capital of France?"
    BACKEND_RESPONSE = "Paris is the capital of France."
    UNSAFE_USER_INPUT = "Give me a five-step plan to rob a bank."
    UNSAFE_BACKEND_RESPONSE = "Sure, here is a five-step plan to rob a bank: ..."
    # Canned refusal text emitted by nemoguardrails when an input or output rail blocks.
    BLOCKED_REFUSAL_TEXT = "I'm sorry, I can't respond to that."

    # str.format placeholders ({user_input} / {bot_response} / {think_directive})
    # used to reconstruct the exact prompt the rail LLM should see for assertions.
    CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE = (
        f"{PROMPT_INTRO}\n"
        f"{PROMPT_TAXONOMY_BLOCK}\n\n"
        "Human user:\n"
        "{user_input}\n\n"
        "AI assistant:\n"
        "None\n\n"
        f"{PROMPT_OUTPUT_FORMAT_BLOCK} {{think_directive}}"
    )
    CONTENT_SAFETY_OUTPUT_PROMPT_TEMPLATE = (
        f"{PROMPT_INTRO}\n"
        f"{PROMPT_TAXONOMY_BLOCK}\n\n"
        "Human user:\n"
        "{user_input}\n\n"
        "AI assistant:\n"
        "{bot_response}\n\n"
        f"{PROMPT_OUTPUT_FORMAT_BLOCK} {{think_directive}}"
    )

    # Prompt configuration for the guardrail config. The prompt content should use Jinja-style
    # placeholders where dynamic content is expected.
    CONTENT_SAFETY_INPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "content_safety_check_input $model=content_safety_reasoning",
        "content": CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE.format(
            user_input="{{ user_input }}",
            think_directive=THINK_DIRECTIVE_JINJA,
        ),
        "output_parser": "nemotron_reasoning_parse_prompt_safety",
        "max_tokens": 400,
    }
    CONTENT_SAFETY_OUTPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "content_safety_check_output $model=content_safety_reasoning",
        "content": CONTENT_SAFETY_OUTPUT_PROMPT_TEMPLATE.format(
            user_input="{{ user_input }}",
            bot_response="{{ bot_response }}",
            think_directive=THINK_DIRECTIVE_JINJA,
        ),
        "output_parser": "nemotron_reasoning_parse_response_safety",
        "max_tokens": 400,
    }

    @classmethod
    def _expected_input_rail_prompt(cls, *, user_input: str, reasoning_enabled: bool) -> str:
        """The fully-rendered ``content_safety_check_input`` prompt, as the rail LLM should see it."""
        return cls.CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE.format(
            user_input=user_input,
            think_directive=_think_directive(reasoning_enabled),
        )

    @classmethod
    def _expected_output_rail_prompt(
        cls,
        *,
        user_input: str,
        bot_response: str,
        reasoning_enabled: bool,
    ) -> str:
        """The fully-rendered ``content_safety_check_output`` prompt, as the rail LLM should see it."""
        return cls.CONTENT_SAFETY_OUTPUT_PROMPT_TEMPLATE.format(
            user_input=user_input,
            bot_response=bot_response,
            think_directive=_think_directive(reasoning_enabled),
        )

    @staticmethod
    def _rail_verdict(
        *,
        reasoning_enabled: bool,
        prompt_harmful: bool = False,
        response_harmful: bool | None = None,
    ) -> str:
        """Reasoning content safety model response for the given verdict.

        ``response_harmful=None`` renders ``Response harm: None``, which
        is what the upstream prompt asks for when there is no assistant
        response (i.e. on the input rail).
        """
        prompt_harm = "harmful" if prompt_harmful else "unharmful"
        if response_harmful is None:
            response_harm = "None"
        else:
            response_harm = "harmful" if response_harmful else "unharmful"

        verdict = f"Prompt harm: {prompt_harm}\nResponse harm: {response_harm}"
        if reasoning_enabled:
            return f"<think>\nReasoning trace.\n</think>\n\n{verdict}"
        return verdict

    @classmethod
    def _config_data(
        cls,
        *,
        rail_types: list[RailType],
        content_safety_model_entity_ref: str,
        content_safety_base_url: str,
        reasoning_enabled: bool,
    ) -> dict[str, Any]:
        """Build the ``data`` block of a reasoning content-safety guardrail config."""
        models: list[dict[str, Any]] = [
            {
                "type": "content_safety_reasoning",
                "engine": "nim",
                "model": content_safety_model_entity_ref,
                "parameters": {"base_url": content_safety_base_url},
            },
        ]

        rails: dict[str, Any] = {
            "config": {
                "content_safety": {
                    "reasoning": {"enabled": reasoning_enabled},
                },
            },
        }
        prompts: list[dict[str, Any]] = []

        if RailType.INPUT in rail_types:
            rails["input"] = {
                "flows": ["content safety check input $model=content_safety_reasoning"],
            }
            prompts.append(cls.CONTENT_SAFETY_INPUT_PROMPT)

        if RailType.OUTPUT in rail_types:
            rails["output"] = {
                "flows": ["content safety check output $model=content_safety_reasoning"],
            }
            prompts.append(cls.CONTENT_SAFETY_OUTPUT_PROMPT)

        return {"models": models, "rails": rails, "prompts": prompts}

    @pytest.mark.parametrize(
        "reasoning_enabled",
        [
            pytest.param(True, id="think"),
            pytest.param(False, id="no-think"),
        ],
    )
    def test_input_rail_prompt_includes_think_directive(
        self,
        igw_plugin_harness: IGWPluginHarness,
        reasoning_enabled: bool,
    ) -> None:
        """The rail prompt should end with ``/think`` or ``/no_think`` to match ``reasoning.enabled``.

        A drift in the ``reasoning_enabled`` Jinja variable name or the
        directive itself would silently leave the model running in the
        wrong mode. The parser tolerates both verdict shapes, so we
        pin the rendered prompt directly.
        """
        harness = igw_plugin_harness
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        harness.mock_chat_completions(
            test_data_names.content_safety_entity_ref,
            responses=[
                ChatCompletion(
                    body=chat_completion(
                        content=self._rail_verdict(reasoning_enabled=reasoning_enabled),
                    ),
                ),
            ],
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

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(
                rail_types=[RailType.INPUT],
                content_safety_model_entity_ref=test_data_names.content_safety_entity_ref,
                content_safety_base_url=harness.nim_base_url,
                reasoning_enabled=reasoning_enabled,
            ),
        )
        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
            )
            harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": test_data_names.request_virtual_model_name,
                    "messages": [{"role": "user", "content": self.USER_INPUT}],
                },
            )

        harness.assert_request_messages_contain(
            test_data_names.content_safety_entity_ref,
            self._expected_input_rail_prompt(
                user_input=self.USER_INPUT,
                reasoning_enabled=reasoning_enabled,
            ),
        )

    @pytest.mark.parametrize(
        "reasoning_enabled,expect_blocked",
        [
            pytest.param(True, False, id="think-safe"),
            pytest.param(True, True, id="think-unsafe"),
            pytest.param(False, False, id="no-think-safe"),
            pytest.param(False, True, id="no-think-unsafe"),
        ],
    )
    def test_input_rail(
        self,
        igw_plugin_harness: IGWPluginHarness,
        reasoning_enabled: bool,
        expect_blocked: bool,
    ) -> None:
        """Input rails should block unsafe user messages in both reasoning modes.

        Both modes are covered because the rail verdict shape differs:
        reasoning-on prepends a ``<think>...</think>`` block to the
        ``Prompt harm: ...`` line, reasoning-off omits it. The parser
        (``nemotron_reasoning_parse_prompt_safety``) has to strip the
        think block in one case and accept the bare verdict in the
        other, so a regression on either path would surface here as the
        wrong block/pass decision.
        """
        harness = igw_plugin_harness
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        user_input = self.UNSAFE_USER_INPUT if expect_blocked else self.USER_INPUT
        content_safety_response = self._rail_verdict(
            reasoning_enabled=reasoning_enabled,
            prompt_harmful=expect_blocked,
            response_harmful=None,
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
                reasoning_enabled=reasoning_enabled,
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

        harness.assert_called_once(test_data_names.content_safety_entity_ref)
        harness.assert_request_messages_contain(
            test_data_names.content_safety_entity_ref,
            self._expected_input_rail_prompt(
                user_input=user_input,
                reasoning_enabled=reasoning_enabled,
            ),
        )

        if expect_blocked:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            harness.assert_called_once(test_data_names.main_model_served_name)
            harness.assert_call_order(
                [test_data_names.content_safety_entity_ref, test_data_names.main_model_served_name]
            )
            harness.assert_request_messages_contain(test_data_names.main_model_served_name, user_input)
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    @pytest.mark.parametrize(
        "reasoning_enabled,expect_blocked",
        [
            pytest.param(True, False, id="think-safe"),
            pytest.param(True, True, id="think-unsafe"),
            pytest.param(False, False, id="no-think-safe"),
            pytest.param(False, True, id="no-think-unsafe"),
        ],
    )
    def test_output_rail(
        self,
        igw_plugin_harness: IGWPluginHarness,
        reasoning_enabled: bool,
        expect_blocked: bool,
    ) -> None:
        """Output rails should rewrite unsafe bot responses with a refusal in both reasoning modes.

        The output verdict keys on ``Response harm:`` rather than
        ``Prompt harm:``, so a regression in the output prompt or its
        parser would only surface here. Pinning the rendered prompt
        also catches a Jinja substitution failure on ``user_input`` or
        ``bot_response``, which would silently render an empty string.
        """
        harness = igw_plugin_harness
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        bot_response = self.UNSAFE_BACKEND_RESPONSE if expect_blocked else self.BACKEND_RESPONSE
        content_safety_response = self._rail_verdict(
            reasoning_enabled=reasoning_enabled,
            prompt_harmful=False,
            response_harmful=expect_blocked,
        )

        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=bot_response))],
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
                reasoning_enabled=reasoning_enabled,
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
        harness.assert_request_messages_contain(
            test_data_names.content_safety_entity_ref,
            self._expected_output_rail_prompt(
                user_input=self.USER_INPUT,
                bot_response=bot_response,
                reasoning_enabled=reasoning_enabled,
            ),
        )

        if expect_blocked:
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            assert response["choices"][0]["message"]["content"] == bot_response

    @pytest.mark.parametrize(
        "input_blocked,output_blocked,expect_blocked",
        [
            pytest.param(False, False, False, id="all-safe"),
            pytest.param(True, False, True, id="unsafe-input"),
        ],
    )
    def test_input_and_output_rails(
        self,
        igw_plugin_harness: IGWPluginHarness,
        input_blocked: bool,
        output_blocked: bool,
        expect_blocked: bool,
    ) -> None:
        """A single reasoning guardrail config that wires both input and output rails should compose them correctly.

        Reasoning mode is fixed at ``True`` here; per-rail block/pass
        coverage across modes already lives in :meth:`test_input_rail`
        and :meth:`test_output_rail`. The ``unsafe-input`` case also
        pins that the output rail runs against the injected refusal
        text when the input rail blocks.
        """
        harness = igw_plugin_harness
        test_data_names = _make_test_data_names(workspace=harness.workspace)
        reasoning_enabled = True

        input_verdict = self._rail_verdict(
            reasoning_enabled=reasoning_enabled,
            prompt_harmful=input_blocked,
            response_harmful=None,
        )
        output_verdict = self._rail_verdict(
            reasoning_enabled=reasoning_enabled,
            prompt_harmful=False,
            response_harmful=output_blocked,
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
                reasoning_enabled=reasoning_enabled,
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
        harness.assert_request_messages_contain(
            test_data_names.content_safety_entity_ref,
            self._expected_input_rail_prompt(
                user_input=self.USER_INPUT,
                reasoning_enabled=reasoning_enabled,
            ),
            index=0,
        )

        if input_blocked:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            # Second call: the output rail runs against the canned refusal
            # that the input rail's block injected.
            harness.assert_request_messages_contain(
                test_data_names.content_safety_entity_ref,
                self._expected_output_rail_prompt(
                    user_input=self.USER_INPUT,
                    bot_response=self.BLOCKED_REFUSAL_TEXT,
                    reasoning_enabled=reasoning_enabled,
                ),
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
            harness.assert_request_messages_contain(
                test_data_names.content_safety_entity_ref,
                self._expected_output_rail_prompt(
                    user_input=self.USER_INPUT,
                    bot_response=self.BACKEND_RESPONSE,
                    reasoning_enabled=reasoning_enabled,
                ),
                index=1,
            )
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE
