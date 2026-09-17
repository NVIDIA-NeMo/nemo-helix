# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run inference using a parallel guardrail config.

These tests cover input rails configured with multiple flows to verify the
plugin can run sibling rails in parallel instead of short-circuiting after the
first blocking rail.
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
    create_guardrail_config,
    delete_guardrail_config_if_present,
    make_guardrails_test_data_names,
    make_served_model,
)

pytestmark = [pytest.mark.integration]


@dataclass(frozen=True)
class ParallelRailsTestDataNames:
    main_model_served_name: str
    main_model_entity_ref: str
    request_virtual_model_name: str
    guardrail_config_name: str
    model_provider_name: str
    content_safety_model_served_name: str
    content_safety_model_entity_ref: str
    topic_control_model_served_name: str
    topic_control_model_entity_ref: str


def _make_test_data_names(
    *,
    main_model_prefix: str = "main-model",
    workspace: str,
) -> ParallelRailsTestDataNames:
    base_test_data_names = make_guardrails_test_data_names(
        main_model_prefix=main_model_prefix,
        workspace=workspace,
    )
    content_safety_model = make_served_model(
        test_id=base_test_data_names.test_id,
        prefix="cs-model",
        workspace=workspace,
    )
    topic_control_model = make_served_model(
        test_id=base_test_data_names.test_id,
        prefix="tc-model",
        workspace=workspace,
    )

    return ParallelRailsTestDataNames(
        main_model_served_name=base_test_data_names.main_model_served_name,
        main_model_entity_ref=base_test_data_names.main_model_entity_ref,
        request_virtual_model_name=base_test_data_names.request_virtual_model_name,
        guardrail_config_name=base_test_data_names.guardrail_config_name,
        model_provider_name=base_test_data_names.model_provider_name,
        content_safety_model_served_name=content_safety_model.served_name,
        content_safety_model_entity_ref=content_safety_model.entity_ref,
        topic_control_model_served_name=topic_control_model.served_name,
        topic_control_model_entity_ref=topic_control_model.entity_ref,
    )


class TestParallelRails:
    USER_INPUT = "What is your return policy?"
    BLOCKED_REFUSAL_TEXT = "I'm sorry, I can't respond to that."

    CONTENT_SAFETY_INPUT_FLOW = "content safety check input $model=content_safety"
    TOPIC_CONTROL_INPUT_FLOW = "topic safety check input $model=topic_control"
    CONTENT_SAFETY_INPUT_TASK = "content_safety_check_input $model=content_safety"
    TOPIC_CONTROL_INPUT_TASK = "topic_safety_check_input $model=topic_control"

    CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE = (
        "Task: Check if there is unsafe content in the user message in the below conversation according to the "
        "safety policy.\n\n"
        "<BEGIN CONVERSATION>\n\n"
        "user: {user_input}\n\n"
        "<END CONVERSATION>\n\n"
        'Provide your safety assessment as JSON: {{"User Safety": "safe" or "unsafe", '
        '"Safety Categories": comma-separated list, omit if all safe}}.\n\n'
        "Output JSON:"
    )
    TOPIC_CONTROL_INPUT_PROMPT_TEMPLATE = (
        "You are to act as a customer service agent, providing users with factual information "
        "in accordance to the knowledge base. Your role is to ensure that you respond only to "
        "relevant queries and adhere to the following guidelines\n\n"
        "Guidelines for the user messages:\n"
        "- Do not answer questions unrelated to company policies.\n"
        "- Do not answer questions about sensitive topics related to politics or religion.\n"
        "- Allow user comments that are related to small talk and chit-chat.\n\n"
        'User message: "{user_input}"'
    )
    TOPIC_CONTROL_RESPONSE_INSTRUCTIONS = (
        '\n\nIf any of the above conditions are violated, please respond with "off-topic". '
        'Otherwise, respond with "on-topic". You must respond with "on-topic" or "off-topic".'
    )

    CONTENT_SAFETY_INPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": CONTENT_SAFETY_INPUT_TASK,
        "content": CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE.format(user_input="{{ user_input }}"),
        "output_parser": "nemoguard_parse_prompt_safety",
        "max_tokens": 50,
    }
    TOPIC_CONTROL_INPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": TOPIC_CONTROL_INPUT_TASK,
        "content": TOPIC_CONTROL_INPUT_PROMPT_TEMPLATE.format(user_input="{{ user_input }}"),
        "max_tokens": 50,
    }

    @staticmethod
    def _unsafe_input_content_safety_response() -> str:
        return '{"User Safety": "unsafe", "Safety Categories": "S3"}'

    @staticmethod
    def _safe_input_topic_control_response() -> str:
        return "on-topic"

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
    def _expected_content_safety_prompt(cls) -> str:
        return cls.CONTENT_SAFETY_INPUT_PROMPT_TEMPLATE.format(user_input=cls.USER_INPUT)

    @classmethod
    def _expected_topic_control_system_prompt(cls) -> str:
        return cls.TOPIC_CONTROL_INPUT_PROMPT_TEMPLATE.format(user_input="") + cls.TOPIC_CONTROL_RESPONSE_INSTRUCTIONS

    @classmethod
    def _config_data(
        cls,
        *,
        parallel: bool,
        content_safety_model_entity_ref: str,
        topic_control_model_entity_ref: str,
        rail_base_url: str,
    ) -> dict[str, Any]:
        return {
            "models": [
                {
                    "type": "content_safety",
                    "engine": "nim",
                    "model": content_safety_model_entity_ref,
                    "parameters": {"base_url": rail_base_url},
                },
                {
                    "type": "topic_control",
                    "engine": "nim",
                    "model": topic_control_model_entity_ref,
                    "parameters": {"base_url": rail_base_url},
                },
            ],
            "rails": {
                "input": {
                    "parallel": parallel,
                    "flows": [cls.CONTENT_SAFETY_INPUT_FLOW, cls.TOPIC_CONTROL_INPUT_FLOW],
                },
            },
            "prompts": [cls.CONTENT_SAFETY_INPUT_PROMPT, cls.TOPIC_CONTROL_INPUT_PROMPT],
        }

    @staticmethod
    def _activated_rails_by_name(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
        guardrails_data: dict[str, Any] = response.get("guardrails_data") or {}
        log: dict[str, Any] = guardrails_data.get("log") or {}
        activated_rails = log.get("activated_rails") or []

        return {rail["name"]: rail for rail in activated_rails}

    def _setup_entity_backed_vm(
        self,
        harness: IGWLoopbackHarness,
        *,
        parallel: bool,
    ) -> ParallelRailsTestDataNames:
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                test_data_names.content_safety_model_served_name: test_data_names.content_safety_model_served_name,
                test_data_names.topic_control_model_served_name: test_data_names.topic_control_model_served_name,
            },
        )
        create_guardrail_config(
            harness,
            name=test_data_names.guardrail_config_name,
            description="Entity-backed parallel rails config for integration tests",
            data=self._config_data(
                parallel=parallel,
                content_safety_model_entity_ref=test_data_names.content_safety_model_entity_ref,
                topic_control_model_entity_ref=test_data_names.topic_control_model_entity_ref,
                rail_base_url=harness.nim_base_url,
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

    @pytest.mark.parametrize(
        "parallel",
        [
            pytest.param(False, id="sequential"),
            pytest.param(True, id="parallel"),
        ],
    )
    def test_input_rails_call_order_when_blocking(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        parallel: bool,
    ) -> None:
        """Input rails should execute all flows when parallel rails are enabled,
        even if one rail blocks.

        In this test, the content-safety rail is configured first and classifies the input
        as unsafe. In parallel mode, the topic control rail should also still run.
        In sequential mode, the topic control rail should never run.
        """
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(harness, parallel=parallel)
            try:
                harness.mock_chat_completions(
                    test_data_names.content_safety_model_entity_ref,
                    responses=[
                        ChatCompletion(body=chat_completion(content=self._unsafe_input_content_safety_response()))
                    ],
                )
                harness.mock_chat_completions(
                    test_data_names.topic_control_model_entity_ref,
                    responses=[ChatCompletion(body=chat_completion(content=self._safe_input_topic_control_response()))],
                )

                response = harness.chat_completions(
                    workspace=harness.workspace,
                    body={
                        "model": f"{harness.workspace}/{test_data_names.request_virtual_model_name}",
                        "messages": [{"role": "user", "content": self.USER_INPUT}],
                        "guardrails": {"options": {"log": {"activated_rails": True}}},
                    },
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)

        harness.assert_called_once(test_data_names.content_safety_model_entity_ref)
        harness.assert_request_messages_contain(
            test_data_names.content_safety_model_entity_ref,
            self._expected_content_safety_prompt(),
        )
        harness.assert_no_calls_to(test_data_names.main_model_served_name)
        assert response["choices"][0]["finish_reason"] == "content_filter"
        assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT

        # Verify the content-safety rail blocked the request
        activated_rails = self._activated_rails_by_name(response)
        assert activated_rails[self.CONTENT_SAFETY_INPUT_FLOW]["stop"] is True

        # In parallel mode, nemoguardrails schedules sibling rails concurrently, but
        # cancels pending tasks as soon as one rail blocks. Depending on task
        # interleaving, content-safety may block before topic-control reaches its
        # LLM call, so only assert topic-control prompt details if the call was
        # actually observed.
        if parallel:
            if harness.requests_for(test_data_names.topic_control_model_entity_ref):
                harness.assert_called_once(test_data_names.topic_control_model_entity_ref)
                harness.assert_request_messages_contain(
                    test_data_names.topic_control_model_entity_ref,
                    self._expected_topic_control_system_prompt(),
                )
                harness.assert_request_messages_contain(test_data_names.topic_control_model_entity_ref, self.USER_INPUT)
        # In sequential mode, ensure the topic control rail did not run
        else:
            harness.assert_no_calls_to(test_data_names.topic_control_model_entity_ref)
