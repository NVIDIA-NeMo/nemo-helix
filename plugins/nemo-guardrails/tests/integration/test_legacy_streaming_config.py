# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for configs saved before nemoguardrails 0.24.

The guardrails service used to save ``streaming = {enabled: true, chunk_size: 200, context_size: 50,
stream_first: true}`` on every config, even when the user never configured streaming. nemoguardrails 0.24
rejects those values next to an output rail that rewrites the response (``injection detection`` here),
which would stop every such config from loading. These tests store a row in exactly that old shape,
bypassing the service's model so nothing repairs it on the way in, and check that it still serves
inference through the whole stack: entity store, guardrails service, IGW and the plugin.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from nemo_guardrails_plugin.constants import GUARDRAILS_PLUGIN_CONFIG_TYPE
from nemo_guardrails_plugin.iorails_runtime import IORAILS_ENGINE_ENV
from nemo_helix_plugin.entities import EntityBase
from nhx.core.inference_gateway.testing.harness import IGWLoopbackHarness
from nhx.testing.mock_chat_completions import (
    ChatCompletion,
    build_chat_completion_stream_from_content,
    build_content_from_chunks,
    chat_completion,
)

from .utils import (
    GUARDRAILS_PLUGIN_NAME,
    EntityGuardrailsMiddlewareCall,
    GuardrailsTestDataNames,
    delete_guardrail_config_if_present,
    guardrail_client,
    make_guardrails_test_data_names,
)

pytestmark = [pytest.mark.integration]


LEGACY_STREAMING = {"enabled": True, "chunk_size": 200, "context_size": 50, "stream_first": True}
"""The streaming block the old service saved on every config."""

INJECTION_DETECTION = {"injections": ["code", "sqli", "template", "xss"], "action": "reject"}


class _StoredGuardrailConfig(EntityBase):
    """A guardrail config row exactly as the entity store holds it.

    Shares the entity type of the service's own model but leaves ``data`` as the raw dict that was
    saved, so a row can be written in the shape old versions wrote without the service's model
    repairing it on the way in.
    """

    __entity_type__ = "guardrail_config"

    description: str | None = None
    data: dict[str, Any] | None = None


class TestLegacyStreamingConfig:
    USER_INPUT = "Tell me about Cape Hatteras National Seashore in 50 words or less."
    SAFE_BACKEND_RESPONSE = "Cape Hatteras National Seashore protects barrier islands along the Outer Banks."
    CODE_BACKEND_RESPONSE = (
        "```python\n"
        "import requests\n"
        "import urllib.request\n\n"
        "response = requests.get('https://example.com/weather')\n"
        "print(urllib.request.urlopen(response.url).read())\n"
        "```"
    )

    @staticmethod
    def _config_data(streaming: dict[str, Any] | None) -> dict[str, Any]:
        output: dict[str, Any] = {"flows": ["injection detection"]}
        if streaming is not None:
            output["streaming"] = streaming
        return {"rails": {"config": {"injection_detection": INJECTION_DETECTION}, "output": output}}

    @staticmethod
    def _set_engine(monkeypatch: pytest.MonkeyPatch, *, iorails: bool) -> None:
        """Choose the engine before the plugin loads, because it reads the switch at startup."""
        if iorails:
            monkeypatch.setenv(IORAILS_ENGINE_ENV, "true")
        else:
            monkeypatch.delenv(IORAILS_ENGINE_ENV, raising=False)

    @staticmethod
    def _store_legacy_config(harness: IGWLoopbackHarness, name: str, data: dict[str, Any]) -> None:
        stored = _StoredGuardrailConfig(
            name=name, workspace=harness.workspace, description="Config saved before 0.24", data=data
        )
        asyncio.run(harness.entity_client.create(stored))

    @staticmethod
    def _stored_streaming(harness: IGWLoopbackHarness, name: str) -> dict[str, Any]:
        """The streaming block as the entity store holds it, read back without any model in between."""
        stored = asyncio.run(harness.entity_client.get(_StoredGuardrailConfig, name=name, workspace=harness.workspace))
        assert stored.data is not None
        return stored.data["rails"]["output"]["streaming"]

    @staticmethod
    def _served_streaming(harness: IGWLoopbackHarness, name: str):
        """The streaming block the service returns, which is also what the plugin loads."""
        config = guardrail_client(harness).get_guardrail_config(name=name, workspace=harness.workspace).data()
        assert config.data is not None
        assert config.data.rails is not None
        assert config.data.rails.output is not None
        assert config.data.rails.output.streaming is not None
        return config.data.rails.output.streaming

    @staticmethod
    def _middleware_call(workspace: str, config_name: str) -> EntityGuardrailsMiddlewareCall:
        return {
            "name": GUARDRAILS_PLUGIN_NAME,
            "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
            "config_id": f"{workspace}/{config_name}",
        }

    def _add_virtual_model(self, harness: IGWLoopbackHarness) -> GuardrailsTestDataNames:
        names = make_guardrails_test_data_names(main_model_prefix="main-model", workspace=harness.workspace)
        harness.add_provider(
            workspace=harness.workspace,
            name=names.model_provider_name,
            served_models={names.main_model_served_name: names.main_model_served_name},
        )
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=names.main_model_served_name,
            default_model_entity=names.main_model_entity_ref,
        )
        return names

    def _attach_output_rail(self, harness: IGWLoopbackHarness, names: GuardrailsTestDataNames) -> None:
        harness.add_virtual_model(
            workspace=harness.workspace,
            name=names.request_virtual_model_name,
            default_model_entity=names.main_model_entity_ref,
            response_middleware=[self._middleware_call(harness.workspace, names.guardrail_config_name)],
        )

    def _chat(self, harness: IGWLoopbackHarness, names: GuardrailsTestDataNames, backend_response: str) -> dict:
        harness.handler.reset()
        harness.mock_chat_completions(
            names.main_model_served_name,
            responses=[ChatCompletion(body=chat_completion(content=backend_response))],
        )
        return harness.chat_completions(
            workspace=harness.workspace,
            body={
                "model": f"{harness.workspace}/{names.request_virtual_model_name}",
                "messages": [{"role": "user", "content": self.USER_INPUT}],
            },
        )

    @pytest.mark.parametrize("iorails", [False, True], ids=["llmrails", "iorails"])
    @pytest.mark.parametrize(
        "backend_response,expect_blocked",
        [
            pytest.param(SAFE_BACKEND_RESPONSE, False, id="safe-output"),
            pytest.param(CODE_BACKEND_RESPONSE, True, id="unsafe-output"),
        ],
    )
    def test_saved_config_serves_inference(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        monkeypatch: pytest.MonkeyPatch,
        iorails: bool,
        backend_response: str,
        expect_blocked: bool,
    ) -> None:
        """A row in the old shape loads and still blocks and passes output as its rail dictates."""
        self._set_engine(monkeypatch, iorails=iorails)
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            names = self._add_virtual_model(harness)
            self._store_legacy_config(harness, names.guardrail_config_name, self._config_data(LEGACY_STREAMING))
            self._attach_output_rail(harness, names)
            try:
                response = self._chat(harness, names, backend_response)

                harness.assert_called_once(names.main_model_served_name)
                choice = response["choices"][0]
                if expect_blocked:
                    assert choice["finish_reason"] == "content_filter"
                    assert "import requests" not in choice["message"]["content"]
                else:
                    assert choice["message"]["content"] == self.SAFE_BACKEND_RESPONSE
            finally:
                delete_guardrail_config_if_present(harness, names.guardrail_config_name)

    def test_saved_config_is_repaired_when_read_and_left_unchanged_in_storage(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The service serves ``stream_first=false`` and ``context_size=0``; the stored row keeps the old values."""
        self._set_engine(monkeypatch, iorails=False)
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            names = self._add_virtual_model(harness)
            self._store_legacy_config(harness, names.guardrail_config_name, self._config_data(LEGACY_STREAMING))
            try:
                served = self._served_streaming(harness, names.guardrail_config_name)
                assert (served.enabled, served.stream_first, served.context_size) == (True, False, 0)
                assert served.chunk_size == LEGACY_STREAMING["chunk_size"]

                assert self._stored_streaming(harness, names.guardrail_config_name) == LEGACY_STREAMING
            finally:
                delete_guardrail_config_if_present(harness, names.guardrail_config_name)

    @pytest.mark.parametrize(
        "backend_response,expect_blocked",
        [
            pytest.param(SAFE_BACKEND_RESPONSE, False, id="safe-output"),
            pytest.param(CODE_BACKEND_RESPONSE, True, id="unsafe-output"),
        ],
    )
    def test_saved_config_streams_on_iorails(
        self,
        igw_loopback_harness: Callable[..., IGWLoopbackHarness],
        monkeypatch: pytest.MonkeyPatch,
        backend_response: str,
        expect_blocked: bool,
    ) -> None:
        """Streaming through the rewriting rail holds each chunk back until the rail has judged it.

        Runs on IORails only: nemoguardrails 0.24.1's LLMRails fails to stream through this rail with
        any settings (it passes an argument ``injection_detection`` does not accept).
        """
        self._set_engine(monkeypatch, iorails=True)
        harness = igw_loopback_harness()

        with harness.load_plugin(GUARDRAILS_PLUGIN_NAME):
            names = self._add_virtual_model(harness)
            self._store_legacy_config(harness, names.guardrail_config_name, self._config_data(LEGACY_STREAMING))
            self._attach_output_rail(harness, names)
            try:
                harness.handler.reset()
                harness.mock_chat_completions(
                    names.main_model_served_name,
                    responses=[build_chat_completion_stream_from_content(backend_response)],
                )
                chunks = harness.stream_chat_completions(
                    workspace=harness.workspace,
                    body={
                        "model": f"{harness.workspace}/{names.request_virtual_model_name}",
                        "messages": [{"role": "user", "content": self.USER_INPUT}],
                    },
                )

                assert isinstance(chunks, list)
                if expect_blocked:
                    assert chunks[-1]["error"]["code"] == "content_blocked"
                    assert build_content_from_chunks(chunks[:-1]) == ""
                else:
                    assert build_content_from_chunks(chunks) == self.SAFE_BACKEND_RESPONSE
            finally:
                delete_guardrail_config_if_present(harness, names.guardrail_config_name)
