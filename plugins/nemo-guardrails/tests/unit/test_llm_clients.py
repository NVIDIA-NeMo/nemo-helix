# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for request-scoped LLM client header plumbing."""

from typing import Any, Protocol, cast

from nemo_guardrails_plugin import llm_clients
from nemo_guardrails_plugin.llm_clients import (
    build_header_aware_chat_nvidia,
    get_request_headers,
    platform_headers_context,
    register_header_aware_nim_provider,
)
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client_provider import get_forwarding_headers
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
from nemoguardrails.llm.models.initializer import init_llm_model


class HeaderAwareClientForTest(Protocol):
    """Test-only view of the dynamic client returned as ``BaseChatModel``.

    Production code only promises a ``BaseChatModel``, but these tests
    intentionally inspect the private ChatNVIDIA hook we patch. This class
    is a simple wrapper to inform the type checker of the fields we expect
    to exist.
    """

    model: str
    default_headers: dict[str, str] | None

    def _prepare_inputs_and_payload(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, dict[str, str]]: ...


def assert_and_get_header_aware_client(client: object) -> HeaderAwareClientForTest:
    """
    The production helper returns BaseChatModel, but these tests need to
    inspect the dynamic ChatNVIDIA subclass internals. Assert the runtime
    shape first, then cast so the type checker can follow the same contract.
    """
    assert hasattr(client, "model")
    assert hasattr(client, "default_headers")
    assert hasattr(client, "_prepare_inputs_and_payload")
    return cast(HeaderAwareClientForTest, client)


def _make_client(**custom_headers: str) -> AsyncNemoClient:
    return AsyncNemoClient(base_url="http://test:8000", default_headers=custom_headers)


class TestGetForwardingHeaders:
    def test_returns_custom_headers_from_sdk(self) -> None:
        client = _make_client(
            **{
                "X-NMP-Principal-Id": "service:guardrails-test",
                "traceparent": "00-platform",
            }
        )
        assert get_forwarding_headers(client) == {
            "X-NMP-Principal-Id": "service:guardrails-test",
            "traceparent": "00-platform",
        }

    def test_returns_empty_for_sdk_with_no_custom_headers(self) -> None:
        client = _make_client()
        assert get_forwarding_headers(client) == {}


class TestRequestHeadersContext:
    def test_defaults_to_empty_headers(self) -> None:
        assert get_request_headers() == {}

    def test_sets_platform_headers_and_resets_after_exit(self) -> None:
        client = _make_client(
            **{
                "X-NMP-Principal-Id": "service:guardrails-test",
                "traceparent": "00-platform",
            }
        )

        assert get_request_headers() == {}

        with platform_headers_context(client):
            assert get_request_headers() == {
                "traceparent": "00-platform",
                "X-NMP-Principal-Id": "service:guardrails-test",
            }

        assert get_request_headers() == {}

    def test_nested_contexts_restore_previous_headers(self) -> None:
        outer_client = _make_client(traceparent="outer")
        inner_client = _make_client(traceparent="inner")

        with platform_headers_context(outer_client):
            assert get_request_headers() == {"traceparent": "outer"}

            with platform_headers_context(inner_client):
                assert get_request_headers() == {"traceparent": "inner"}

            assert get_request_headers() == {"traceparent": "outer"}

        assert get_request_headers() == {}


class TestHeaderAwareChatNVIDIA:
    def test_merges_static_headers_with_current_platform_headers(self, monkeypatch) -> None:
        class FakeChatNVIDIA:
            def __init__(self, *, model: str, **kwargs) -> None:
                self.model = model
                self.default_headers = kwargs.get("default_headers")

            def _prepare_inputs_and_payload(self, *args, **kwargs) -> tuple[None, None, dict[str, str] | None]:
                return None, None, self.default_headers

        monkeypatch.setattr("nemo_guardrails_plugin.llm_clients._load_chat_nvidia_class", lambda: FakeChatNVIDIA)

        client = assert_and_get_header_aware_client(
            build_header_aware_chat_nvidia(
                model_name="default/safety",
                kwargs={"default_headers": {"X-Static": "yes"}},
            )
        )

        assert client.model == "default/safety"
        assert client.default_headers == {"X-Static": "yes"}

        client_context = _make_client(
            **{
                "X-NMP-Principal-Id": "service:guardrails-test",
                "traceparent": "00-platform",
            }
        )
        with platform_headers_context(client_context):
            _inputs, _payload, headers = client._prepare_inputs_and_payload([])
            assert headers == {
                "X-Static": "yes",
                "X-NMP-Principal-Id": "service:guardrails-test",
                "traceparent": "00-platform",
            }
            assert client.default_headers == {"X-Static": "yes"}

        assert client.default_headers == {"X-Static": "yes"}

    def test_default_headers_remains_static_after_contextual_request(self, monkeypatch) -> None:
        class FakeChatNVIDIA:
            def __init__(self, *, model: str, **kwargs) -> None:
                self.model = model
                self.default_headers = kwargs.get("default_headers")

            def _prepare_inputs_and_payload(self, *args, **kwargs) -> tuple[str, str, dict[str, str]]:
                return "inputs", "payload", dict(self.default_headers or {})

        monkeypatch.setattr("nemo_guardrails_plugin.llm_clients._load_chat_nvidia_class", lambda: FakeChatNVIDIA)

        client = assert_and_get_header_aware_client(
            build_header_aware_chat_nvidia(
                model_name="default/safety",
                kwargs={"default_headers": {"X-Static": "yes"}},
            )
        )

        client_context = _make_client(**{"X-NMP-Principal-Id": "service:guardrails-test"})
        with platform_headers_context(client_context):
            _inputs, _payload, headers = client._prepare_inputs_and_payload([])

        assert headers == {
            "X-Static": "yes",
            "X-NMP-Principal-Id": "service:guardrails-test",
        }
        assert client.default_headers == {"X-Static": "yes"}


class TestRegisterHeaderAwareNimProvider:
    def test_registers_provider_used_by_nemoguardrails_initializer(self, monkeypatch) -> None:
        class FakeChatNVIDIA:
            def __init__(self, *, model: str, **kwargs) -> None:
                self.model = model
                self.default_headers = kwargs.get("default_headers")

            def _prepare_inputs_and_payload(self, *args, **kwargs) -> tuple[None, None, dict[str, str] | None]:
                return None, None, self.default_headers

        monkeypatch.setattr("nemo_guardrails_plugin.llm_clients._load_chat_nvidia_class", lambda: FakeChatNVIDIA)
        monkeypatch.setattr(llm_clients, "_HEADER_AWARE_INITIALIZER_INSTALLED", False)

        register_header_aware_nim_provider()

        # Exercise NeMo Guardrails' real provider lookup for `engine: nim`.
        model = init_llm_model(
            model_name="default/safety",
            provider_name="nim",
            kwargs={"default_headers": {"X-Static": "yes"}},
        )
        assert isinstance(model, LangChainLLMAdapter)

        # NeMo Guardrails 0.23 expects providers to return its LLMModel
        # interface, so our LangChain client is wrapped in LangChainLLMAdapter.
        # Inspect the wrapped client to confirm it is the header-aware subclass.
        client = assert_and_get_header_aware_client(model._llm)
        assert client.model == "default/safety"
        assert client.default_headers == {"X-Static": "yes"}

        client_context = _make_client(traceparent="00-platform")
        with platform_headers_context(client_context):
            # The registered client must keep static config headers and merge
            # request-scoped platform headers at call time.
            _inputs, _payload, headers = client._prepare_inputs_and_payload([])

        assert headers == {"X-Static": "yes", "traceparent": "00-platform"}

    def test_is_idempotent(self, monkeypatch) -> None:
        calls = []

        def register_provider(name: str, provider) -> None:
            calls.append((name, provider))

        monkeypatch.setattr(llm_clients, "register_provider", register_provider)
        monkeypatch.setattr(llm_clients, "_HEADER_AWARE_INITIALIZER_INSTALLED", False)

        register_header_aware_nim_provider()
        register_header_aware_nim_provider()

        assert calls == [("nim", llm_clients._init_header_aware_nim_model)]
