# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for entity-backed Guardrails middleware config caching.

These tests exercise the boundary between IGW's pre-resolved ``config_id``
middleware cache and the Guardrails plugin's in-memory LLMRails cache.
"""

from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import pytest
from nemo_guardrails_plugin.constants import GUARDRAILS_PLUGIN_CONFIG_TYPE
from nmp.core.inference_gateway.testing.harness import IGWLoopbackHarness, IGWPluginHarness
from nmp.testing.mock_chat_completions import ChatCompletion, chat_completion

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    EntityGuardrailsMiddlewareCall,
    GuardrailsTestDataNames,
    create_guardrail_config,
    delete_guardrail_config_if_present,
    expect_harness_http_error,
    make_guardrails_test_data_names,
    update_guardrail_config,
)

pytestmark = [pytest.mark.integration]


class TestMiddlewareConfigCaching:
    USER_INPUT = "What is the capital of France?"
    BLOCKED_REFUSAL_TEXT = "I'm sorry, I can't respond to that."

    SELF_CHECK_INPUT_PROMPT_TEMPLATE = (
        "{version} policy: decide whether this user message should be blocked.\n\nUser: {user_input}\nAnswer yes or no:"
    )
    SELF_CHECK_OUTPUT_PROMPT_TEMPLATE = (
        "{version} policy: decide whether this bot message should be blocked.\n\nBot: {bot_response}\nAnswer yes or no:"
    )

    @classmethod
    def _config_data(cls, *, version: str, main_base_url: str) -> dict[str, Any]:
        return {
            "models": [
                {
                    "type": "main",
                    "engine": "nim",
                    "parameters": {"base_url": main_base_url},
                }
            ],
            "rails": {
                "input": {"flows": ["self check input"]},
                "output": {"flows": ["self check output"]},
            },
            "prompts": [
                {
                    "task": "self_check_input",
                    "content": cls.SELF_CHECK_INPUT_PROMPT_TEMPLATE.format(
                        version=version,
                        user_input="{{ user_input }}",
                    ),
                },
                {
                    "task": "self_check_output",
                    "content": cls.SELF_CHECK_OUTPUT_PROMPT_TEMPLATE.format(
                        version=version,
                        bot_response="{{ bot_response }}",
                    ),
                },
            ],
        }

    @classmethod
    def _expected_input_prompt(cls, *, version: str) -> str:
        return cls.SELF_CHECK_INPUT_PROMPT_TEMPLATE.format(version=version, user_input=cls.USER_INPUT)

    @classmethod
    def _expected_output_prompt(cls, *, version: str) -> str:
        return cls.SELF_CHECK_OUTPUT_PROMPT_TEMPLATE.format(version=version, bot_response=cls._backend_response())

    @staticmethod
    def _safe_input_self_check_response() -> str:
        return "No"

    @staticmethod
    def _safe_output_self_check_response() -> str:
        return "No"

    @staticmethod
    def _unsafe_output_self_check_response() -> str:
        return "Yes"

    @staticmethod
    def _backend_response() -> str:
        return "Paris is the capital of France."

    @staticmethod
    def _middleware_call(workspace: str, config_name: str) -> EntityGuardrailsMiddlewareCall:
        return {
            "name": GUARDRAILS_PLUGIN_NAME,
            "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
            "config_id": f"{workspace}/{config_name}",
        }

    @staticmethod
    def _delete_config_entity_directly(harness: IGWPluginHarness, config_name: str) -> None:
        """Delete the GuardrailConfig entity straight from the entity store.

        The Guardrails API deliberately refuses to delete a config a VirtualModel still applies,
        which is the very state these tests need: a route pointing at a config that is gone. The
        state is still reachable in production — an entity removed out-of-band, or a reference
        that predates that guard — and IGW must keep failing closed when it happens, so the tests
        build it the way it actually arises rather than through the guarded endpoint.
        """
        harness.sdk.entities.delete_entity_by_name(
            name=config_name,
            workspace=harness.workspace,
            entity_type=GUARDRAILS_PLUGIN_CONFIG_TYPE,
        )

    @staticmethod
    def _delete_config_if_present(harness: IGWPluginHarness, config_name: str) -> None:
        # The service refuses to delete a config a VirtualModel still applies; harness
        # cleanup removes those routes, but it runs after this teardown.
        delete_guardrail_config_if_present(harness, config_name)

    @staticmethod
    def _refresh_caches(harness: IGWPluginHarness) -> None:
        """Refresh IGW's Model and VM caches. The integration test harness has no background controller
        loop, so we must manually refresh the in-memory caches to simulate the background refresh task."""
        harness.refresh_caches()

    @classmethod
    def _assert_chat_completions_unavailable(
        cls,
        harness: IGWPluginHarness,
        *,
        model: str,
    ) -> None:
        error = expect_harness_http_error(
            lambda: harness.chat_completions(
                workspace=harness.workspace,
                body={
                    "model": model,
                    "messages": [{"role": "user", "content": cls.USER_INPUT}],
                },
            ),
            HTTPStatus.SERVICE_UNAVAILABLE,
        )

        body = error.body
        assert isinstance(body, dict)
        detail = body.get("detail")
        assert isinstance(detail, str)
        assert "Middleware configuration unavailable" in detail

    @classmethod
    def _chat_completions_body(cls, *, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": cls.USER_INPUT}],
        }

    def _setup_entity_backed_vm(
        self,
        harness: IGWPluginHarness,
        *,
        config_version: str,
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
            description="Entity-backed self-check config for middleware cache tests",
            data=self._config_data(version=config_version, main_base_url=harness.nim_base_url),
        )

        harness.add_virtual_model(
            workspace=harness.workspace,
            name=test_data_names.request_virtual_model_name,
            default_model_entity=test_data_names.main_model_entity_ref,
            request_middleware=[self._middleware_call(harness.workspace, test_data_names.guardrail_config_name)],
            response_middleware=[self._middleware_call(harness.workspace, test_data_names.guardrail_config_name)],
        )
        return test_data_names

    def _assert_inference_uses_config(
        self,
        harness: IGWPluginHarness,
        test_data_names: GuardrailsTestDataNames,
        *,
        config_version: str,
        output_blocked: bool = False,
    ) -> None:
        """Run one inference request and assert the rails use the expected config version."""
        harness.handler.reset()
        output_self_check_response = (
            self._unsafe_output_self_check_response() if output_blocked else self._safe_output_self_check_response()
        )

        # Mock self-check responses for the input and output rails.
        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[
                ChatCompletion(body=chat_completion(content=self._safe_input_self_check_response())),
                ChatCompletion(body=chat_completion(content=output_self_check_response)),
            ],
        )
        # Mock the backend response.
        harness.mock_chat_completions(
            test_data_names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=self._backend_response()))],
        )

        response = harness.chat_completions(
            workspace=harness.workspace,
            body=self._chat_completions_body(model=test_data_names.request_virtual_model_name),
        )

        harness.assert_request_messages_contain(
            test_data_names.main_model_entity_ref,
            self._expected_input_prompt(version=config_version),
            index=0,
        )
        harness.assert_request_messages_contain(
            test_data_names.main_model_entity_ref,
            self._expected_output_prompt(version=config_version),
            index=1,
        )
        harness.assert_call_count(test_data_names.main_model_served_name, 1)

        if output_blocked:
            assert response["choices"][0]["finish_reason"] == "content_filter"
            assert response["choices"][0]["message"]["content"] == self.BLOCKED_REFUSAL_TEXT
        else:
            assert response["choices"][0]["message"]["content"] == self._backend_response()

    def test_entity_config_update_reflected_in_next_request(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
    ) -> None:
        """A refreshed entity-backed config should affect the next real Guardrails execution."""
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(harness, config_version="v1")
            try:
                # Make the first inference request, which should use the v1 config.
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    config_version="v1",
                )

                # Update the config referenced by the VirtualModel
                update_guardrail_config(
                    harness,
                    name=test_data_names.guardrail_config_name,
                    data=self._config_data(version="v2", main_base_url=harness.nim_base_url),
                )

                # Make the second inference request, before the cache is refreshed.
                # This request should still use the v1 config.
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    config_version="v1",
                )

                # Refresh the caches to update the VM cache with the new config.
                self._refresh_caches(harness)

                # Make the third inference request, which should use the v2 config.
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    config_version="v2",
                    output_blocked=True,
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)

    def test_entity_config_delete_blocks_next_request(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
    ) -> None:
        """Deleting a referenced config should fail closed after IGW refreshes middleware config refs."""
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(harness, config_version="v1")
            try:
                # Make the first inference request, which uses the v1 config.
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    config_version="v1",
                )

                # Delete the config referenced by the VirtualModel, out from under it.
                self._delete_config_entity_directly(harness, test_data_names.guardrail_config_name)
                self._refresh_caches(harness)

                # Verify the next inference request fails closed because the referenced
                # GuardrailConfig is no longer available.
                self._assert_chat_completions_unavailable(
                    harness,
                    model=test_data_names.request_virtual_model_name,
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)

    def test_entity_config_recreate_reflected_in_next_request(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
    ) -> None:
        """Recreating the same config_id should let the next refresh recover the failing VM."""
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            test_data_names = self._setup_entity_backed_vm(harness, config_version="v1")
            try:
                # Delete the config referenced by the VirtualModel, out from under it.
                self._delete_config_entity_directly(harness, test_data_names.guardrail_config_name)
                self._refresh_caches(harness)

                # Verify the next inference request fails closed because the referenced
                # GuardrailConfig is no longer available.
                self._assert_chat_completions_unavailable(
                    harness,
                    model=test_data_names.request_virtual_model_name,
                )

                # Recreate the config referenced by the VirtualModel.
                create_guardrail_config(
                    harness,
                    name=test_data_names.guardrail_config_name,
                    description="Recreated self-check config for middleware cache tests",
                    data=self._config_data(version="v2", main_base_url=harness.nim_base_url),
                )
                self._refresh_caches(harness)

                # Make the inference request, which should use the v2 config.
                self._assert_inference_uses_config(
                    harness,
                    test_data_names,
                    config_version="v2",
                )
            finally:
                self._delete_config_if_present(harness, test_data_names.guardrail_config_name)
