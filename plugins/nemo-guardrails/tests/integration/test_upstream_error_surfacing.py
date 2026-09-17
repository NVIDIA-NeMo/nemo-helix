# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests that exercise the real ``langchain_nvidia_ai_endpoints``
error path (nemoguardrails' bundled ``nim`` engine), not a hand-constructed
exception. ``responses.extract_upstream_error`` assumes that library formats
every raised error as a ``"[<status>] <detail>"``-prefixed message — these
tests call the real library against a mocked upstream so a future
``langchain_nvidia_ai_endpoints`` upgrade that changes that convention fails
loudly here, instead of only in production.
"""

from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import pytest
from nemo_guardrails_plugin.constants import GUARDRAILS_PLUGIN_CONFIG_TYPE
from nmp.core.inference_gateway.testing.harness import IGWLoopbackHarness
from nmp.testing.mock_chat_completions import ErrorResponse

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    EntityGuardrailsMiddlewareCall,
    create_guardrail_config,
    delete_guardrail_config_if_present,
    expect_harness_http_error,
    make_guardrails_test_data_names,
    make_served_model,
)

pytestmark = [pytest.mark.integration]

# Sample upstream error body vLLM returns for "too many images" error.
TOO_MANY_IMAGES_BODY = {
    "object": "error",
    "message": (
        "At most 1 image(s) may be provided in one request. "
        "You can set `--limit-mm-per-prompt` to increase this limit if the model supports it."
    ),
    "type": "BadRequestError",
    "param": None,
    "code": 400,
}
TOO_MANY_IMAGES_MESSAGE = TOO_MANY_IMAGES_BODY["message"]


@dataclass(frozen=True)
class _Fixture:
    vm_name: str
    config_name: str
    vision_model_entity_ref: str


class TestUpstreamErrorSurfacing:
    IMAGE_DATA_URL = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"

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
            "prompts": [
                {
                    "task": "content_safety_check_input $model=vision_rails",
                    "messages": [
                        {"type": "system", "content": "Evaluate whether the user message is safe."},
                        {"type": "user", "content": "{{ user_input }}"},
                    ],
                    "output_parser": "is_content_safe",
                    "max_tokens": 200,
                }
            ],
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

    def _setup(self, harness: IGWLoopbackHarness) -> _Fixture:
        """Wire a vision-judge model + guarded VirtualModel."""
        test_data_names = make_guardrails_test_data_names(workspace=harness.workspace)
        vision_model = make_served_model(
            test_id=test_data_names.test_id,
            prefix="vision-model",
            workspace=harness.workspace,
        )

        harness.add_provider(
            workspace=harness.workspace,
            name=test_data_names.model_provider_name,
            served_models={
                test_data_names.main_model_served_name: test_data_names.main_model_served_name,
                vision_model.served_name: vision_model.served_name,
            },
        )
        create_guardrail_config(
            harness,
            name=test_data_names.guardrail_config_name,
            description="Upstream error surfacing test config",
            data=self._config_data(
                vision_model_entity_ref=vision_model.entity_ref,
                vision_base_url=harness.nim_base_url,
            ),
        )
        # Config is created; clean it up if VirtualModel wiring fails before we
        # hand a _Fixture (and its teardown responsibility) back to the caller.
        try:
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
        except Exception:
            self._delete_config_if_present(harness, test_data_names.guardrail_config_name)
            raise
        return _Fixture(
            vm_name=test_data_names.request_virtual_model_name,
            config_name=test_data_names.guardrail_config_name,
            vision_model_entity_ref=vision_model.entity_ref,
        )

    def _send_message(self, harness: IGWLoopbackHarness, vm_name: str) -> dict[str, Any]:
        return harness.chat_completions(
            workspace=harness.workspace,
            body={
                "model": f"{harness.workspace}/{vm_name}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Compare these two images."},
                            {"type": "image_url", "image_url": {"url": self.IMAGE_DATA_URL}},
                            {"type": "image_url", "image_url": {"url": self.IMAGE_DATA_URL}},
                        ],
                    }
                ],
            },
        )

    def test_upstream_400_error_keeps_status_code(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
    ) -> None:
        """A real 400 from the vision-judge model (via nemoguardrails' bundled
        ``langchain_nvidia_ai_endpoints``-backed ``nim`` engine) must reach the
        caller as a 400 with the upstream detail, not the middleware's generic
        503. Exercises the real library end-to-end — no hand-built exception —
        so it fails if a library upgrade changes the ``"[<status>] ..."``
        convention ``extract_upstream_error`` depends on.
        """
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            fixture = self._setup(harness)
            try:
                harness.mock_chat_completions(
                    fixture.vision_model_entity_ref,
                    responses=[ErrorResponse(status_code=400, body=TOO_MANY_IMAGES_BODY)],
                )

                error = expect_harness_http_error(
                    lambda: self._send_message(harness, fixture.vm_name),
                    HTTPStatus.BAD_REQUEST,
                )
                body = error.body
                assert isinstance(body, dict)
                detail = body.get("detail")
                assert isinstance(detail, str)
                assert TOO_MANY_IMAGES_MESSAGE in detail
            finally:
                self._delete_config_if_present(harness, fixture.config_name)

    def test_upstream_500_from_vision_judge_is_also_preserved(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
    ) -> None:
        """A genuine upstream 5xx is propagated verbatim too, same as a 4xx —
        the middleware's generic 503 fallback is reserved for failures with
        no recoverable upstream status at all (e.g. a connection error).
        """
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            fixture = self._setup(harness)
            try:
                harness.mock_chat_completions(
                    fixture.vision_model_entity_ref,
                    responses=[
                        ErrorResponse(
                            status_code=500,
                            body={"object": "error", "message": "Internal server error", "code": 500},
                        )
                    ],
                )

                expect_harness_http_error(
                    lambda: self._send_message(harness, fixture.vm_name),
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            finally:
                self._delete_config_if_present(harness, fixture.config_name)
