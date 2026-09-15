# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``guardrails.options.log`` on VirtualModel chat requests.

Validates that each user-requested log field is populated in
``guardrails_data.log`` and that unrequested fields are omitted from the
serialized response.
"""

import json
from http import HTTPStatus
from typing import Any

import pytest
from nemo_guardrails_plugin.constants import GUARDRAILS_DATA_MESSAGE_ROLE
from nmp.core.inference_gateway.testing.harness import IGWPluginHarness
from nmp.testing.mock_chat_completions import ChatCompletion, chat_completion

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    GuardrailsTestDataNames,
    expect_harness_http_error,
    make_guardrail_config,
    make_guardrails_test_data_names,
    make_middleware_call,
)

pytestmark = [pytest.mark.integration]

REQUEST_LOG_FIELDS = (
    "activated_rails",
    "llm_calls",
    "internal_events",
    "stats",
    "colang_history",
)


class TestRequestLogging:
    USER_INPUT = "What is the capital of France?"
    BACKEND_RESPONSE = "Paris is the capital of France."

    INPUT_RAIL_NAME = "self check input"
    OUTPUT_RAIL_NAME = "self check output"

    @classmethod
    def _config_data(cls, *, main_base_url: str) -> dict[str, Any]:
        return {
            "models": [
                {
                    "type": "main",
                    "engine": "nim",
                    "model": "main-model",
                    "parameters": {"base_url": main_base_url},
                }
            ],
            "rails": {
                "input": {"flows": [cls.INPUT_RAIL_NAME]},
                "output": {
                    "flows": [cls.OUTPUT_RAIL_NAME],
                    "streaming": {"enabled": True},
                },
            },
            "prompts": [
                {
                    "task": "self_check_input",
                    "content": ("Should the user message be blocked?\n\nUser: {{ user_input }}\nAnswer yes or no:"),
                },
                {
                    "task": "self_check_output",
                    "content": (
                        "Your task is to check if the bot message below is harmful or inappropriate.\n\n"
                        'Bot message: "{{ bot_response }}"\n\n'
                        "Question: Should the message be blocked?\n"
                        "Answer yes or no:"
                    ),
                },
            ],
        }

    @staticmethod
    def _guardrails_data(response: dict[str, Any]) -> dict[str, Any]:
        guardrails_data = response.get("guardrails_data")
        assert isinstance(guardrails_data, dict), "expected top-level guardrails_data on the response"
        return guardrails_data

    @staticmethod
    def _log_dict(response: dict[str, Any]) -> dict[str, Any]:
        log = TestRequestLogging._guardrails_data(response).get("log")
        assert isinstance(log, dict), "expected guardrails_data.log to be present"
        return log

    @classmethod
    def _assert_log_includes_fields(cls, log: dict[str, Any], *, fields: set[str]) -> None:
        """Assert ``log`` contains exactly ``fields``, each with a populated value."""
        assert set(log.keys()) == fields

        if "activated_rails" in fields:
            activated_rails = log["activated_rails"]
            assert isinstance(activated_rails, list) and activated_rails
            rail_names = {rail["name"] for rail in activated_rails}
            assert cls.INPUT_RAIL_NAME in rail_names
            assert cls.OUTPUT_RAIL_NAME in rail_names
            for rail in activated_rails:
                assert isinstance(rail.get("name"), str)
                assert isinstance(rail.get("type"), str)

        if "llm_calls" in fields:
            llm_calls = log["llm_calls"]
            assert isinstance(llm_calls, list) and llm_calls
            first_call = llm_calls[0]
            assert isinstance(first_call.get("llm_model_name"), str)
            assert first_call.get("prompt") or first_call.get("completion")

        if "internal_events" in fields:
            internal_events = log["internal_events"]
            assert isinstance(internal_events, list) and internal_events
            assert isinstance(internal_events[0], dict)

        if "stats" in fields:
            stats = log["stats"]
            assert isinstance(stats, dict)
            assert any(
                isinstance(stats.get(field), (int, float))
                for field in (
                    "total_duration",
                    "llm_calls_count",
                    "llm_calls_total_tokens",
                    "input_rails_duration",
                    "output_rails_duration",
                )
            )

        if "colang_history" in fields:
            colang_history = log["colang_history"]
            assert isinstance(colang_history, str) and colang_history.strip()

    def _invoke_all_safe_request(
        self,
        harness: IGWPluginHarness,
        *,
        log_options: dict[str, bool] | None = None,
        return_choice: bool = False,
    ) -> tuple[dict[str, Any], GuardrailsTestDataNames]:
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)

        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[
                ChatCompletion(body=chat_completion(content="No")),
                ChatCompletion(body=chat_completion(content="No")),
            ],
        )
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
            data=self._config_data(main_base_url=harness.nim_base_url),
        )

        guardrails_request: dict[str, Any] = {}
        if log_options is not None:
            guardrails_request["options"] = {"log": log_options}
        if return_choice:
            guardrails_request["return_choice"] = True

        body: dict[str, Any] = {
            "model": test_data_names.request_virtual_model_name,
            "messages": [{"role": "user", "content": self.USER_INPUT}],
        }
        if guardrails_request:
            body["guardrails"] = guardrails_request

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
                response_middleware=[make_middleware_call(guardrail_config)],
            )
            response = harness.chat_completions(workspace=harness.workspace, body=body)

        return response, test_data_names

    @pytest.mark.parametrize("requested_field", REQUEST_LOG_FIELDS)
    def test_requested_log_field_populated_and_others_omitted(
        self,
        igw_plugin_harness: IGWPluginHarness,
        requested_field: str,
    ) -> None:
        """Each log flag should surface only its corresponding response field."""
        response, test_data_names = self._invoke_all_safe_request(
            igw_plugin_harness,
            log_options={requested_field: True},
        )

        assert response["choices"][0]["message"]["content"] == self.BACKEND_RESPONSE

        self._assert_log_includes_fields(self._log_dict(response), fields={requested_field})

        guardrails_data = self._guardrails_data(response)
        assert guardrails_data["config_ids"] == [f"<inline:{test_data_names.guardrail_config_name}>"]

    def test_all_log_fields_requested_together(self, igw_plugin_harness: IGWPluginHarness) -> None:
        """Requesting every log flag should populate every log field."""
        log_options: dict[str, bool] = dict.fromkeys(REQUEST_LOG_FIELDS, True)
        response, _ = self._invoke_all_safe_request(igw_plugin_harness, log_options=log_options)

        self._assert_log_includes_fields(self._log_dict(response), fields=set(REQUEST_LOG_FIELDS))

    def test_log_omitted_when_not_requested(self, igw_plugin_harness: IGWPluginHarness) -> None:
        """Omitting guardrails.options.log should not include guardrails_data.log."""
        response, test_data_names = self._invoke_all_safe_request(igw_plugin_harness)

        guardrails_data = self._guardrails_data(response)
        assert guardrails_data["config_ids"] == [f"<inline:{test_data_names.guardrail_config_name}>"]
        assert "log" not in guardrails_data

    def test_input_and_output_logs_merged_for_list_fields(self, igw_plugin_harness: IGWPluginHarness) -> None:
        """List-shaped log fields should concatenate input-rail and output-rail entries."""
        response, _ = self._invoke_all_safe_request(
            igw_plugin_harness,
            log_options={"activated_rails": True, "llm_calls": True, "internal_events": True},
        )

        log = self._log_dict(response)

        activated_rail_types = {rail["type"] for rail in log["activated_rails"]}
        assert "input" in activated_rail_types
        assert "output" in activated_rail_types
        assert len(log["llm_calls"]) == 2
        assert len(log["internal_events"]) >= 2

    def test_return_choice_includes_requested_log_fields(self, igw_plugin_harness: IGWPluginHarness) -> None:
        """return_choice should embed guardrails_data (including log) in an extra choice."""
        response, test_data_names = self._invoke_all_safe_request(
            igw_plugin_harness,
            log_options={"activated_rails": True, "llm_calls": True},
            return_choice=True,
        )

        assert "guardrails_data" not in response

        guardrails_choices = [
            choice for choice in response["choices"] if choice["message"]["role"] == GUARDRAILS_DATA_MESSAGE_ROLE
        ]
        assert len(guardrails_choices) == 1

        guardrails_data = json.loads(guardrails_choices[0]["message"]["content"])
        assert guardrails_data["config_ids"] == [f"<inline:{test_data_names.guardrail_config_name}>"]
        self._assert_log_includes_fields(guardrails_data["log"], fields={"activated_rails", "llm_calls"})

    def test_unknown_log_field_rejected(self, igw_plugin_harness: IGWPluginHarness) -> None:
        """Unsupported log options should fail request validation."""
        harness = igw_plugin_harness
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)

        harness.mock_chat_completions(
            test_data_names.main_model_entity_ref,
            responses=[ChatCompletion(body=chat_completion(content="No"))],
        )
        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={test_data_names.main_model_served_name: test_data_names.main_model_served_name},
        )

        guardrail_config = make_guardrail_config(
            harness.workspace,
            test_data_names.guardrail_config_name,
            data=self._config_data(main_base_url=harness.nim_base_url),
        )

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            harness.add_virtual_model(
                workspace=harness.workspace,
                name=test_data_names.request_virtual_model_name,
                default_model_entity=test_data_names.main_model_entity_ref,
                request_middleware=[make_middleware_call(guardrail_config)],
            )
            expect_harness_http_error(
                lambda: harness.chat_completions(
                    workspace=harness.workspace,
                    body={
                        "model": test_data_names.request_virtual_model_name,
                        "messages": [{"role": "user", "content": self.USER_INPUT}],
                        "guardrails": {"options": {"log": {"unknown": True}}},
                    },
                ),
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
