# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that run inference using an injection detection guardrail config.

These tests cover output injection detection rails to verify the plugin blocks
responses that match built-in or custom injection rules.
"""

from collections.abc import Callable
from typing import Any

import pytest
from nemo_guardrails_plugin.constants import GUARDRAILS_PLUGIN_CONFIG_TYPE
from nmp.core.inference_gateway.testing.harness import IGWLoopbackHarness
from nmp.testing.mock_chat_completions import ChatCompletion, chat_completion

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    EntityGuardrailsMiddlewareCall,
    GuardrailsTestDataNames,
    create_guardrail_config,
    delete_guardrail_config_if_present,
    make_guardrails_test_data_names,
)

pytestmark = [pytest.mark.integration]


class TestInjectionDetection:
    USER_INPUT = "Tell me about Cape Hatteras National Seashore in 50 words or less."
    SAFE_BACKEND_RESPONSE = (
        "Cape Hatteras National Seashore protects barrier islands, beaches, and lighthouses "
        "along North Carolina's Outer Banks."
    )
    CODE_BACKEND_RESPONSE = (
        "```python\n"
        "import requests\n"
        "import urllib.request\n\n"
        "response = requests.get('https://example.com/weather')\n"
        "print(urllib.request.urlopen(response.url).read())\n"
        "```"
    )
    ETHERNET_BACKEND_RESPONSE = "Ethernet headers include source and destination MAC addresses."
    REFUSAL_PREFIX = "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of"

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

    @staticmethod
    def _builtin_injection_detection_config() -> dict[str, Any]:
        return {
            "rails": {
                "config": {
                    "injection_detection": {
                        "injections": ["code", "sqli", "template", "xss"],
                        "action": "reject",
                    }
                },
                "output": {"flows": ["injection detection"]},
            },
        }

    @staticmethod
    def _custom_yara_injection_detection_config() -> dict[str, Any]:
        return {
            "rails": {
                "config": {
                    "injection_detection": {
                        "injections": ["reject_ethernet"],
                        "yara_rules": {
                            "reject_ethernet": (
                                "rule reject_ethernet {\n"
                                " strings:\n"
                                '  $string = "ethernet" nocase\n'
                                " condition:\n"
                                "  $string\n"
                                "}"
                            )
                        },
                        "action": "reject",
                    }
                },
                "output": {"flows": ["injection detection"]},
            },
        }

    def _setup_entity_backed_vm(
        self,
        harness: IGWLoopbackHarness,
        *,
        config_data: dict[str, Any],
    ) -> GuardrailsTestDataNames:
        test_data_names = make_guardrails_test_data_names(
            main_model_prefix="main-model",
            workspace=harness.workspace,
        )

        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )
        create_guardrail_config(
            harness,
            name=test_data_names.guardrail_config_name,
            description="Entity-backed injection detection config for integration tests",
            data=config_data,
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
            response_middleware=[self._middleware_call(harness.workspace, test_data_names.guardrail_config_name)],
        )
        return test_data_names

    def _chat_completions(self, harness: IGWLoopbackHarness, *, model: str) -> dict[str, Any]:
        return harness.chat_completions(
            workspace=harness.workspace,
            body={
                "model": f"{harness.workspace}/{model}",
                "messages": [{"role": "user", "content": self.USER_INPUT}],
            },
        )

    def _assert_inference_uses_config(
        self,
        harness: IGWLoopbackHarness,
        test_data_names: GuardrailsTestDataNames,
        *,
        backend_response: str,
        expected_blocked_content: str | None = None,
    ) -> None:
        """Run one inference request and assert injection detection behavior."""
        harness.handler.reset()
        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=backend_response))],
        )

        response = self._chat_completions(harness, model=test_data_names.request_virtual_model_name)

        harness.assert_called_once(test_data_names.main_model_served_name)
        guardrails_data: dict[str, Any] = response.get("guardrails_data") or {}
        assert guardrails_data.get("config_ids") == [f"{harness.workspace}/{test_data_names.guardrail_config_name}"]

        if expected_blocked_content is not None:
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == expected_blocked_content
        else:
            assert response["choices"][0]["message"]["content"] == self.SAFE_BACKEND_RESPONSE

    @pytest.mark.parametrize(
        "backend_response,expect_blocked",
        [
            pytest.param(SAFE_BACKEND_RESPONSE, False, id="safe-output"),
            pytest.param(CODE_BACKEND_RESPONSE, True, id="unsafe-output"),
        ],
    )
    def test_builtin_injection_detection(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        backend_response: str,
        expect_blocked: bool,
    ) -> None:
        """Built-in injection detection should reject unsafe model output and allow safe output."""
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(
                harness,
                config_data=self._builtin_injection_detection_config(),
            )
            try:
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    backend_response=backend_response,
                    expected_blocked_content=f"{self.REFUSAL_PREFIX} import_networking." if expect_blocked else None,
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)

    @pytest.mark.parametrize(
        "backend_response,expect_blocked",
        [
            pytest.param(SAFE_BACKEND_RESPONSE, False, id="safe-output"),
            pytest.param(ETHERNET_BACKEND_RESPONSE, True, id="custom-yara-reject"),
        ],
    )
    def test_custom_yara_rule(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        backend_response: str,
        expect_blocked: bool,
    ) -> None:
        """Custom inline YARA rules should reject matching model output and allow non-matches."""
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(
                harness,
                config_data=self._custom_yara_injection_detection_config(),
            )
            try:
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    backend_response=backend_response,
                    expected_blocked_content=f"{self.REFUSAL_PREFIX} reject_ethernet." if expect_blocked else None,
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)
