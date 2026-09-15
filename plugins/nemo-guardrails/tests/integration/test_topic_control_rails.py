# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run inference using a topic control guardrail config.

These tests cover input topic-control rails, which use a task LLM to decide
whether user messages are on-topic before the backend model is called.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from nemo_guardrails_plugin.constants import GUARDRAILS_PLUGIN_CONFIG_TYPE
from nmp.core.inference_gateway.testing.harness import IGWLoopbackHarness
from nmp.testing.mock_chat_completions import ChatCompletion, chat_completion

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    EntityGuardrailsMiddlewareCall,
    RailType,
    create_guardrail_config,
    delete_guardrail_config_if_present,
    make_guardrails_test_data_names,
    make_served_model,
)

pytestmark = [pytest.mark.integration]


@dataclass(frozen=True)
class TopicControlTestDataNames:
    main_model_served_name: str
    main_model_entity_ref: str
    request_virtual_model_name: str
    guardrail_config_name: str
    model_provider_name: str
    topic_control_model_served_name: str
    topic_control_entity_ref: str


def _make_test_data_names(
    *,
    main_model_prefix: str = "main-model",
    workspace: str,
) -> TopicControlTestDataNames:
    base_test_data_names = make_guardrails_test_data_names(
        main_model_prefix=main_model_prefix,
        workspace=workspace,
    )
    topic_control_model = make_served_model(
        test_id=base_test_data_names.test_id,
        prefix="tc-model",
        workspace=workspace,
    )

    return TopicControlTestDataNames(
        main_model_served_name=base_test_data_names.main_model_served_name,
        main_model_entity_ref=base_test_data_names.main_model_entity_ref,
        request_virtual_model_name=base_test_data_names.request_virtual_model_name,
        guardrail_config_name=base_test_data_names.guardrail_config_name,
        model_provider_name=base_test_data_names.model_provider_name,
        topic_control_model_served_name=topic_control_model.served_name,
        topic_control_entity_ref=topic_control_model.entity_ref,
    )


class TestTopicControl:
    USER_INPUT_ON_TOPIC = "What is your return policy?"
    USER_INPUT_OFF_TOPIC = "Tell me a joke about quantum gravity."
    BACKEND_RESPONSE = "You can return unopened items within 30 days."
    # Canned refusal text emitted by nemoguardrails when an input rail blocks.
    BLOCKED_REFUSAL_TEXT = "I'm sorry, I can't respond to that."

    TOPIC_CONTROL_INPUT_PROMPT_TEMPLATE = (
        "You are to act as a customer service agent, providing users with factual information "
        "in accordance to the knowledge base. Your role is to ensure that you respond only to "
        "relevant queries and adhere to the following guidelines\n\n"
        "Guidelines for the user messages:\n"
        "- Do not answer questions related to personal opinions or advice on user's order, future recommendations\n"
        "- Do not provide any information on non-company products or services.\n"
        "- Do not answer enquiries unrelated to the company policies.\n"
        "- Do not answer questions asking for personal details about the agent or its creators.\n"
        "- Do not answer questions about sensitive topics related to politics, religion, or other sensitive subjects.\n"
        "- If a user asks topics irrelevant to the company's customer service relations, politely redirect the conversation or end the interaction.\n"
        "- Your responses should be professional, accurate, and compliant with customer relations guidelines, focusing solely on providing transparent, up-to-date information about the company that is already publicly available.\n"
        "- allow user comments that are related to small talk and chit-chat.\n\n"
        'User message: "{user_input}"'
    )
    TOPIC_CONTROL_RESPONSE_INSTRUCTIONS = (
        '\n\nIf any of the above conditions are violated, please respond with "off-topic". '
        'Otherwise, respond with "on-topic". You must respond with "on-topic" or "off-topic".'
    )

    TOPIC_CONTROL_INPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "topic_safety_check_input $model=topic_control",
        "content": TOPIC_CONTROL_INPUT_PROMPT_TEMPLATE.format(user_input="{{ user_input }}"),
        "max_tokens": 50,
    }

    @classmethod
    def _expected_input_rail_system_prompt(cls) -> str:
        """The system message ``topic_safety_check_input`` sends to the rail LLM."""
        return cls.TOPIC_CONTROL_INPUT_PROMPT_TEMPLATE.format(user_input="") + cls.TOPIC_CONTROL_RESPONSE_INSTRUCTIONS

    @staticmethod
    def _safe_input_topic_control_response() -> str:
        """Topic control model response for an on-topic user input."""
        return "on-topic"

    @staticmethod
    def _unsafe_input_topic_control_response() -> str:
        """Topic control model response for an off-topic user input."""
        return "off-topic"

    @staticmethod
    def _middleware_call(workspace: str, config_name: str) -> EntityGuardrailsMiddlewareCall:
        return {
            "name": GUARDRAILS_PLUGIN_NAME,
            "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
            "config_id": f"{workspace}/{config_name}",
        }

    @staticmethod
    def _delete_config_if_present(harness: IGWLoopbackHarness, config_name: str) -> None:
        # The service refuses to delete a config a VirtualModel still applies; harness
        # cleanup removes those routes, but it runs after this teardown.
        delete_guardrail_config_if_present(harness, config_name)

    @classmethod
    def _config_data(
        cls,
        *,
        rail_types: list[RailType],
        topic_control_model_entity_ref: str,
        topic_control_base_url: str | None,
    ) -> dict[str, Any]:
        """Build the ``data`` block of a topic-control guardrail config."""
        parameters: dict[str, Any] = {}
        if topic_control_base_url is not None:
            parameters["base_url"] = topic_control_base_url

        models: list[dict[str, Any]] = [
            {
                "type": "topic_control",
                "engine": "nim",
                "model": topic_control_model_entity_ref,
                "parameters": parameters,
            },
        ]

        rails: dict[str, Any] = {}
        prompts: list[dict[str, Any]] = []

        if RailType.INPUT in rail_types:
            rails["input"] = {
                "flows": ["topic safety check input $model=topic_control"],
            }
            prompts.append(cls.TOPIC_CONTROL_INPUT_PROMPT)

        return {"models": models, "rails": rails, "prompts": prompts}

    def _setup_entity_backed_vm(self, harness: IGWLoopbackHarness) -> TopicControlTestDataNames:
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                test_data_names.topic_control_model_served_name: test_data_names.topic_control_model_served_name,
            },
        )
        create_guardrail_config(
            harness,
            name=test_data_names.guardrail_config_name,
            description="Entity-backed topic-control config for integration tests",
            data=self._config_data(
                rail_types=[RailType.INPUT],
                topic_control_model_entity_ref=test_data_names.topic_control_entity_ref,
                topic_control_base_url=harness.nim_base_url,
            ),
        )
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.main_model_served_name,
            default_model_entity=test_data_names.main_model_entity_ref,
        )
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.request_virtual_model_name,
            default_model_entity=test_data_names.main_model_entity_ref,
            request_middleware=[self._middleware_call(harness.workspace, test_data_names.guardrail_config_name)],
        )
        return test_data_names

    def _assert_inference_uses_config(
        self,
        harness: IGWLoopbackHarness,
        test_data_names: TopicControlTestDataNames,
        *,
        user_input: str,
        topic_control_response: str,
        expect_blocked: bool,
    ) -> None:
        """Run one inference request and assert topic-control rail behavior."""
        harness.handler.reset()
        harness.mock_chat_completions(
            test_data_names.topic_control_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=topic_control_response))],
        )
        if not expect_blocked:
            harness.mock_chat_completions(
                test_data_names.main_model_served_name,
                responses=[ChatCompletion(body=chat_completion(content=self.BACKEND_RESPONSE))],
            )

        response = harness.chat_completions(
            workspace=harness.workspace,
            body={
                "model": f"{harness.workspace}/{test_data_names.request_virtual_model_name}",
                "messages": [{"role": "user", "content": user_input}],
            },
        )

        harness.assert_called_once(test_data_names.topic_control_entity_ref)
        harness.assert_request_messages_contain(
            test_data_names.topic_control_entity_ref,
            self._expected_input_rail_system_prompt(),
        )
        harness.assert_request_messages_contain(test_data_names.topic_control_entity_ref, user_input)
        guardrails_data: dict[str, Any] = response.get("guardrails_data") or {}
        assert guardrails_data.get("config_ids") == [f"{harness.workspace}/{test_data_names.guardrail_config_name}"]

        if expect_blocked:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            harness.assert_called_once(test_data_names.main_model_served_name)
            harness.assert_call_order(
                [test_data_names.topic_control_entity_ref, test_data_names.main_model_served_name]
            )
            harness.assert_request_messages_contain(test_data_names.main_model_served_name, user_input)
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    @pytest.mark.parametrize(
        "expect_blocked",
        [
            pytest.param(False, id="on-topic"),
            pytest.param(True, id="off-topic"),
        ],
    )
    def test_input_rail(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        expect_blocked: bool,
    ) -> None:
        """Input topic-control rails should block off-topic user messages before they reach the backend."""
        harness = igw_loopback_harness()

        user_input = self.USER_INPUT_OFF_TOPIC if expect_blocked else self.USER_INPUT_ON_TOPIC
        topic_control_response = (
            self._unsafe_input_topic_control_response() if expect_blocked else self._safe_input_topic_control_response()
        )

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(harness)
            try:
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    user_input=user_input,
                    topic_control_response=topic_control_response,
                    expect_blocked=expect_blocked,
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)
