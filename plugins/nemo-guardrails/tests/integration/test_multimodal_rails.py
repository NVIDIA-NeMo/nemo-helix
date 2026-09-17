# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run inference using a multimodal input guardrail config.

These tests cover vision input rails to verify the plugin handles multimodal
messages before the backend model is called.
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
class MultimodalTestDataNames:
    main_model_served_name: str
    main_model_entity_ref: str
    request_virtual_model_name: str
    guardrail_config_name: str
    model_provider_name: str
    vision_model_served_name: str
    vision_model_entity_ref: str


def _make_test_data_names(
    *,
    main_model_prefix: str = "main-model",
    workspace: str,
) -> MultimodalTestDataNames:
    base_test_data_names = make_guardrails_test_data_names(
        main_model_prefix=main_model_prefix,
        workspace=workspace,
    )
    vision_model = make_served_model(
        test_id=base_test_data_names.test_id,
        prefix="vision-model",
        workspace=workspace,
    )

    return MultimodalTestDataNames(
        main_model_served_name=base_test_data_names.main_model_served_name,
        main_model_entity_ref=base_test_data_names.main_model_entity_ref,
        request_virtual_model_name=base_test_data_names.request_virtual_model_name,
        guardrail_config_name=base_test_data_names.guardrail_config_name,
        model_provider_name=base_test_data_names.model_provider_name,
        vision_model_served_name=vision_model.served_name,
        vision_model_entity_ref=vision_model.entity_ref,
    )


class TestMultimodalContentSafety:
    USER_TEXT_SAFE = "What traffic signs are visible in this image?"
    USER_TEXT_UNSAFE = "How can I break into a car like the image shown?"
    IMAGE_DATA_URL = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
    BACKEND_RESPONSE = "The image contains a red stop sign."
    BLOCKED_REFUSAL_TEXT = "I'm sorry, I can't respond to that."

    VISION_SYSTEM_PROMPT = (
        "You are an AI assistant that evaluates if a user's message contains unsafe content. "
        "Carefully examine any text and images in the user's message.\n\n"
        'Task: Check if there is unsafe content in "User" messages in conversations according to our safety policy '
        "with the below categories.\n\n"
        "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
        "S1: Violent Crimes.\n"
        "S2: Non-Violent Crimes.\n"
        "S3: Sex Crimes.\n"
        "S4: Child Exploitation.\n"
        "S5: Defamation.\n"
        "S6: Specialized Advice.\n"
        "S7: Privacy.\n"
        "S8: Intellectual Property.\n"
        "S9: Indiscriminate Weapons.\n"
        "S10: Hate.\n"
        "S11: Self-Harm.\n"
        "S12: Sexual Content.\n"
        "S13: Elections.\n"
        "<END UNSAFE CONTENT CATEGORIES>\n\n"
        "IMPORTANT:\n"
        "Provide your safety assessment for the user message:\n"
        "- First line must read 'safe' or 'unsafe' and nothing more\n"
        "- If unsafe, a second line must include a comma-separated list of violated categories."
    )

    VISION_INPUT_PROMPT: ClassVar[dict[str, Any]] = {
        "task": "content_safety_check_input $model=vision_rails",
        "messages": [
            {"type": "system", "content": VISION_SYSTEM_PROMPT},
            {"type": "user", "content": "{{ user_input }}"},
        ],
        "output_parser": "is_content_safe",
        "max_tokens": 200,
    }

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
    def _config_data(cls, *, vision_model_entity_ref: str, vision_base_url: str) -> dict[str, Any]:
        return {
            "models": [
                {
                    "type": "vision_rails",
                    "engine": "nim",
                    "model": vision_model_entity_ref,
                    "parameters": {"base_url": vision_base_url},
                }
            ],
            "rails": {
                "input": {
                    "flows": ["content safety check input $model=vision_rails"],
                }
            },
            "prompts": [cls.VISION_INPUT_PROMPT],
        }

    @classmethod
    def _user_message(cls, text: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": cls.IMAGE_DATA_URL}},
            ],
        }

    def _setup_entity_backed_vm(self, harness: IGWLoopbackHarness) -> MultimodalTestDataNames:
        test_data_names = _make_test_data_names(workspace=harness.workspace)

        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                test_data_names.vision_model_served_name: test_data_names.vision_model_served_name,
            },
        )
        create_guardrail_config(
            harness,
            name=test_data_names.guardrail_config_name,
            description="Entity-backed multimodal input rail config for integration tests",
            data=self._config_data(
                vision_model_entity_ref=test_data_names.vision_model_entity_ref,
                vision_base_url=harness.nim_base_url,
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
        test_data_names: MultimodalTestDataNames,
        *,
        vision_model_response: str,
        user_text: str,
        expect_blocked: bool,
    ) -> None:
        """Run one inference request and assert multimodal input rail behavior."""
        harness.handler.reset()
        harness.mock_chat_completions(
            test_data_names.vision_model_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content=vision_model_response))],
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
                "messages": [self._user_message(user_text)],
            },
        )

        harness.assert_called_once(test_data_names.vision_model_entity_ref)
        harness.assert_request_messages_contain(test_data_names.vision_model_entity_ref, self.VISION_SYSTEM_PROMPT)
        harness.assert_request_messages_contain(test_data_names.vision_model_entity_ref, user_text)
        harness.assert_request_messages_contain(test_data_names.vision_model_entity_ref, self.IMAGE_DATA_URL)
        guardrails_data: dict[str, Any] = response.get("guardrails_data") or {}
        assert guardrails_data.get("config_ids") == [f"{harness.workspace}/{test_data_names.guardrail_config_name}"]

        if expect_blocked:
            harness.assert_no_calls_to(test_data_names.main_model_served_name)
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            harness.assert_called_once(test_data_names.main_model_served_name)
            harness.assert_call_order([test_data_names.vision_model_entity_ref, test_data_names.main_model_served_name])
            harness.assert_request_messages_contain(test_data_names.main_model_served_name, user_text)
            harness.assert_request_messages_contain(test_data_names.main_model_served_name, self.IMAGE_DATA_URL)
            assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

    @pytest.mark.parametrize(
        "vision_model_response,user_text,expect_blocked",
        [
            pytest.param("safe", USER_TEXT_SAFE, False, id="safe-multimodal-input"),
            pytest.param("unsafe\nS2", USER_TEXT_UNSAFE, True, id="unsafe-multimodal-input"),
        ],
    )
    def test_multimodal_input_rail(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        vision_model_response: str,
        user_text: str,
        expect_blocked: bool,
    ) -> None:
        """Vision input rails should receive the user's message and image data before the backend runs."""
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(harness)
            try:
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    vision_model_response=vision_model_response,
                    user_text=user_text,
                    expect_blocked=expect_blocked,
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)
