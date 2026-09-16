# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_helix_plugin.sdk_provider`."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.sdk_provider import (
    DefaultSDKProvider,
    SDKProvider,
    _on_behalf_of_headers,
    _read_principal_from_env,
    get_task_sdk,
    set_sdk_provider,
)


def _xnhx(sdk) -> dict[str, str]:
    return {k: v for k, v in sdk.default_headers.items() if k.startswith("X-NHX-")}


# ---------------------------------------------------------------------------
# _read_principal_from_env
# ---------------------------------------------------------------------------


class TestReadPrincipalFromEnv:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
        assert _read_principal_from_env() is None

    def test_returns_none_when_empty(self, monkeypatch):
        monkeypatch.setenv("NHX_PRINCIPAL", "")
        assert _read_principal_from_env() is None

    def test_returns_none_when_id_missing(self, monkeypatch):
        monkeypatch.setenv("NHX_PRINCIPAL", json.dumps({"email": "a@b.com"}))
        assert _read_principal_from_env() is None

    def test_returns_none_when_id_empty(self, monkeypatch):
        monkeypatch.setenv("NHX_PRINCIPAL", json.dumps({"id": ""}))
        assert _read_principal_from_env() is None

    def test_parses_valid_principal(self, monkeypatch):
        principal = {"id": "user@example.com", "email": "user@example.com", "groups": ["team-a"]}
        monkeypatch.setenv("NHX_PRINCIPAL", json.dumps(principal))
        result = _read_principal_from_env()
        assert result == principal

    def test_raises_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("NHX_PRINCIPAL", "not-json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _read_principal_from_env()


# ---------------------------------------------------------------------------
# _on_behalf_of_headers
# ---------------------------------------------------------------------------


class TestOnBehalfOfHeaders:
    def test_simple_principal(self):
        headers = _on_behalf_of_headers({"id": "user@ex.com", "email": "user@ex.com", "groups": ["g1", "g2"]})
        assert headers["X-NHX-Principal-On-Behalf-Of"] == "user@ex.com"
        assert headers["X-NHX-Principal-On-Behalf-Of-Email"] == "user@ex.com"
        assert headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "g1,g2"

    def test_delegated_principal_uses_effective(self):
        principal = {
            "id": "service:evaluator",
            "on_behalf_of": "real-user@ex.com",
            "on_behalf_of_email": "real-user@ex.com",
            "on_behalf_of_groups": ["admin"],
        }
        headers = _on_behalf_of_headers(principal)
        assert headers["X-NHX-Principal-On-Behalf-Of"] == "real-user@ex.com"
        assert headers["X-NHX-Principal-On-Behalf-Of-Email"] == "real-user@ex.com"
        assert headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "admin"

    def test_no_email_or_groups(self):
        headers = _on_behalf_of_headers({"id": "user@ex.com"})
        assert headers == {"X-NHX-Principal-On-Behalf-Of": "user@ex.com"}


# ---------------------------------------------------------------------------
# DefaultSDKProvider
# ---------------------------------------------------------------------------


class TestDefaultSDKProvider:
    def test_get_task_sdk_with_principal(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            json.dumps({"id": "creator@ex.com", "email": "creator@ex.com", "groups": ["team"]}),
        )

        provider = DefaultSDKProvider()
        sdk = provider.get_task_sdk("evaluator")

        assert isinstance(sdk, NeMoHelix)
        assert sdk.base_url == "http://test:9090"
        assert sdk.default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert sdk.default_headers["X-NHX-Internal"] == "true"
        assert sdk.default_headers["X-NHX-Principal-On-Behalf-Of"] == "creator@ex.com"

    def test_get_task_sdk_without_principal(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        provider = DefaultSDKProvider()
        sdk = provider.get_task_sdk("evaluator")

        assert sdk.default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert "X-NHX-Principal-On-Behalf-Of" not in sdk.default_headers

    def test_get_task_sdk_default_base_url(self, monkeypatch):
        monkeypatch.delenv("NHX_BASE_URL", raising=False)
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        provider = DefaultSDKProvider()
        sdk = provider.get_task_sdk("test")
        assert sdk.base_url == "http://localhost:8080"

    def test_get_task_sdk_uses_workload_identity_when_token_file_configured(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-from-file\n", encoding="utf-8")
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            json.dumps({"id": "creator@ex.com", "email": "creator@ex.com", "groups": ["team"]}),
        )

        provider = DefaultSDKProvider()
        sdk = provider.get_task_sdk("evaluator")
        try:
            assert sdk.default_headers["X-NHX-Internal"] == "true"
            assert "X-NHX-Principal-Id" not in sdk.default_headers
            assert "X-NHX-Principal-On-Behalf-Of" not in sdk.default_headers
        finally:
            sdk.close()

    def test_get_platform_sdk_as_service(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        provider = DefaultSDKProvider()
        sdk = provider.get_platform_sdk(as_service="my-svc", internal=True)

        assert sdk.default_headers["X-NHX-Principal-Id"] == "service:my-svc"
        assert sdk.default_headers["X-NHX-Internal"] == "true"

    def test_get_platform_sdk_on_behalf_of(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        provider = DefaultSDKProvider()
        sdk = provider.get_platform_sdk(as_service="svc", on_behalf_of="user@ex.com")

        assert sdk.default_headers["X-NHX-Principal-On-Behalf-Of"] == "user@ex.com"

    def test_get_platform_sdk_propagates_env_principal_on_behalf_of(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            json.dumps(
                {
                    "id": "service:evaluator",
                    "email": "evaluator@service.test",
                    "groups": ["system:serviceaccounts"],
                    "on_behalf_of": "creator@ex.com",
                    "on_behalf_of_email": "creator@ex.com",
                    "on_behalf_of_groups": ["workspace-editors", "ml-team"],
                }
            ),
        )

        sdk = DefaultSDKProvider().get_platform_sdk()

        assert sdk.default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert sdk.default_headers["X-NHX-Principal-Email"] == "evaluator@service.test"
        assert sdk.default_headers["X-NHX-Principal-Groups"] == "system:serviceaccounts"
        assert sdk.default_headers["X-NHX-Principal-On-Behalf-Of"] == "creator@ex.com"
        assert sdk.default_headers["X-NHX-Principal-On-Behalf-Of-Email"] == "creator@ex.com"
        assert sdk.default_headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "workspace-editors,ml-team"


# ---------------------------------------------------------------------------
# get_async_task_sdk — the async sibling of get_task_sdk
# ---------------------------------------------------------------------------


class TestAsyncTaskSdk:
    def test_async_task_sdk_with_principal(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            json.dumps({"id": "creator@ex.com", "email": "creator@ex.com", "groups": ["team"]}),
        )

        sdk = DefaultSDKProvider().get_async_task_sdk("evaluator")

        assert isinstance(sdk, AsyncNeMoHelix)
        assert sdk.base_url == "http://test:9090"
        assert sdk.default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert sdk.default_headers["X-NHX-Internal"] == "true"
        assert sdk.default_headers["X-NHX-Principal-On-Behalf-Of"] == "creator@ex.com"

    def test_async_task_sdk_without_principal(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        sdk = DefaultSDKProvider().get_async_task_sdk("evaluator")

        assert sdk.default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert "X-NHX-Principal-On-Behalf-Of" not in sdk.default_headers

    def test_parity_with_sync_task_sdk(self, monkeypatch):
        # Regression guard: the async task SDK must carry the *full* delegated identity (on-behalf-of
        # id, email, and groups) — wire-identical to get_task_sdk. A prior implementation built on
        # get_async_platform_sdk dropped the -Email/-Groups headers.
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            json.dumps(
                {
                    "id": "service:evaluator",
                    "on_behalf_of": "real-user@ex.com",
                    "on_behalf_of_email": "real-user@ex.com",
                    "on_behalf_of_groups": ["admin", "team"],
                }
            ),
        )

        provider = DefaultSDKProvider()
        assert _xnhx(provider.get_async_task_sdk("evaluator")) == _xnhx(provider.get_task_sdk("evaluator"))

    @pytest.mark.asyncio
    async def test_async_task_sdk_uses_workload_identity_when_token_file_configured(self, monkeypatch, tmp_path):
        subject_token_file = tmp_path / "workload-token"
        subject_token_file.write_text("subject-token-from-file\n", encoding="utf-8")
        monkeypatch.setenv("NHX_BASE_URL", "http://test:9090")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            json.dumps({"id": "creator@ex.com", "email": "creator@ex.com", "groups": ["team"]}),
        )

        sdk = DefaultSDKProvider().get_async_task_sdk("evaluator")
        try:
            assert sdk.default_headers["X-NHX-Internal"] == "true"
            assert "X-NHX-Principal-Id" not in sdk.default_headers
            assert "X-NHX-Principal-On-Behalf-Of" not in sdk.default_headers
        finally:
            await sdk.close()


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


class _CustomProvider:
    def get_task_sdk(self, service_name: str) -> NeMoHelix:
        return NeMoHelix(base_url="http://custom:1234")

    def get_async_task_sdk(self, service_name: str) -> AsyncNeMoHelix:
        return AsyncNeMoHelix(base_url="http://custom:1234")

    def get_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> NeMoHelix:
        return NeMoHelix(base_url="http://custom:1234")

    def get_async_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | None = None,
    ) -> AsyncNeMoHelix:
        return AsyncNeMoHelix(base_url="http://custom:1234")


class _FakeEntryPoint:
    def __init__(
        self,
        name: str,
        obj: object,
        *,
        value: str = "tests:DefaultSDKProvider",
        load_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self._obj = obj
        self._load_error = load_error

    def load(self) -> object:
        if self._load_error is not None:
            raise self._load_error
        return self._obj


class TestProviderResolution:
    def setup_method(self):
        # Reset global state before each test.
        set_sdk_provider(None)

    def teardown_method(self):
        set_sdk_provider(None)

    def test_explicit_provider_takes_precedence(self, monkeypatch):
        monkeypatch.delenv("NHX_BASE_URL", raising=False)
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        set_sdk_provider(_CustomProvider())
        sdk = get_task_sdk("test")
        assert sdk.base_url == "http://custom:1234"

    def test_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://fallback:8080")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        # No explicit provider, no entry-points → default
        with patch("nemo_helix_plugin.sdk_provider.entry_points", return_value=[]):
            sdk = get_task_sdk("test")
        assert sdk.base_url == "http://fallback:8080"

    def test_set_none_clears_and_re_resolves(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://re-resolved:8080")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)

        set_sdk_provider(_CustomProvider())
        assert get_task_sdk("x").base_url == "http://custom:1234"

        # Clear the override
        set_sdk_provider(None)
        with patch("nemo_helix_plugin.sdk_provider.entry_points", return_value=[]):
            sdk = get_task_sdk("x")
        assert sdk.base_url == "http://re-resolved:8080"

    def test_entry_point_not_satisfying_protocol_raises(self):
        class _NotAProvider:
            pass

        eps = [_FakeEntryPoint("platform", _NotAProvider())]
        with patch("nemo_helix_plugin.sdk_provider.entry_points", return_value=eps):
            with pytest.raises(RuntimeError, match="does not satisfy SDKProvider"):
                get_task_sdk("test")

    def test_entry_point_load_exception_raises_and_retries(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://retried:8080")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
        ep = _FakeEntryPoint("platform", DefaultSDKProvider, load_error=ImportError("temporarily unavailable"))
        with patch("nemo_helix_plugin.sdk_provider.entry_points", return_value=[ep]):
            with pytest.raises(RuntimeError, match="Failed to load or construct SDK provider") as exc_info:
                get_task_sdk("test")
            assert isinstance(exc_info.value.__cause__, ImportError)
            ep._load_error = None
            assert get_task_sdk("test").base_url == "http://retried:8080"

    def test_duplicate_name_same_target_is_deduplicated(self, monkeypatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://deduplicated:8080")
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
        eps = [
            _FakeEntryPoint("platform", DefaultSDKProvider, value="tests:DefaultSDKProvider"),
            _FakeEntryPoint("platform", DefaultSDKProvider, value="tests:DefaultSDKProvider"),
        ]
        with patch("nemo_helix_plugin.sdk_provider.entry_points", return_value=eps):
            assert get_task_sdk("test").base_url == "http://deduplicated:8080"

    @pytest.mark.parametrize("reverse", [False, True])
    def test_duplicate_name_different_targets_raises_deterministically(self, reverse):
        eps = [
            _FakeEntryPoint("platform", DefaultSDKProvider, value="z_package:Provider"),
            _FakeEntryPoint("platform", DefaultSDKProvider, value="a_package:Provider"),
        ]
        if reverse:
            eps.reverse()
        with patch("nemo_helix_plugin.sdk_provider.entry_points", return_value=eps):
            with pytest.raises(RuntimeError, match="Conflicting SDK providers") as exc_info:
                get_task_sdk("test")
        assert "a_package:Provider, z_package:Provider" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_default_provider_is_protocol_instance(self):
        assert isinstance(DefaultSDKProvider(), SDKProvider)
