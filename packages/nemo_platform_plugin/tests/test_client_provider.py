# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nemo_platform_plugin.client_provider`.

Covers the env-var default provider and the provider/entry-point resolution
seam.  The rich platform provider (``nmp.common.client_factory``) is tested in
``packages/nmp_common/tests/client_factory``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client_provider import (
    DefaultNemoClientProvider,
    NemoClientProvider,
    _build_headers,
    _read_principal_from_env,
    get_async_nemo_client,
    get_forwarding_headers,
    get_nemo_client,
    set_nemo_client_provider,
)

# ---------------------------------------------------------------------------
# _read_principal_from_env
# ---------------------------------------------------------------------------


class TestReadPrincipalFromEnv:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        assert _read_principal_from_env() is None

    def test_returns_none_when_empty(self, monkeypatch):
        monkeypatch.setenv("NMP_PRINCIPAL", "")
        assert _read_principal_from_env() is None

    def test_returns_none_when_id_missing(self, monkeypatch):
        monkeypatch.setenv("NMP_PRINCIPAL", json.dumps({"email": "a@b.com"}))
        assert _read_principal_from_env() is None

    def test_returns_none_when_id_empty(self, monkeypatch):
        monkeypatch.setenv("NMP_PRINCIPAL", json.dumps({"id": ""}))
        assert _read_principal_from_env() is None

    def test_parses_valid_principal(self, monkeypatch):
        principal = {"id": "user@example.com", "email": "user@example.com", "groups": ["team-a"]}
        monkeypatch.setenv("NMP_PRINCIPAL", json.dumps(principal))
        assert _read_principal_from_env() == principal

    def test_raises_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("NMP_PRINCIPAL", "not-json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _read_principal_from_env()


# ---------------------------------------------------------------------------
# _build_headers
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def test_internal_flag(self):
        assert _build_headers(internal=True)["X-NMP-Internal"] == "true"

    def test_service_principal(self):
        assert _build_headers(as_service="svc")["X-NMP-Principal-Id"] == "service:svc"

    def test_explicit_on_behalf_of(self):
        headers = _build_headers(as_service="svc", on_behalf_of="user@ex.com")
        assert headers["X-NMP-Principal-On-Behalf-Of"] == "user@ex.com"

    def test_principal_from_env_when_no_service(self, monkeypatch):
        monkeypatch.setenv(
            "NMP_PRINCIPAL",
            json.dumps(
                {
                    "id": "user@ex.com",
                    "email": "user@ex.com",
                    "groups": ["g1", "g2"],
                    "on_behalf_of": "boss@ex.com",
                    "on_behalf_of_email": "boss@ex.com",
                    "on_behalf_of_groups": ["admin"],
                }
            ),
        )
        headers = _build_headers()
        assert headers["X-NMP-Principal-Id"] == "user@ex.com"
        assert headers["X-NMP-Principal-Email"] == "user@ex.com"
        assert headers["X-NMP-Principal-Groups"] == "g1,g2"
        assert headers["X-NMP-Principal-On-Behalf-Of"] == "boss@ex.com"
        assert headers["X-NMP-Principal-On-Behalf-Of-Email"] == "boss@ex.com"
        assert headers["X-NMP-Principal-On-Behalf-Of-Groups"] == "admin"

    def test_service_principal_ignores_env_principal(self, monkeypatch):
        # The as_service branch does not read NMP_PRINCIPAL (matches legacy behavior).
        monkeypatch.setenv("NMP_PRINCIPAL", json.dumps({"id": "user@ex.com", "on_behalf_of": "boss@ex.com"}))
        headers = _build_headers(as_service="svc")
        assert headers["X-NMP-Principal-Id"] == "service:svc"
        assert "X-NMP-Principal-On-Behalf-Of" not in headers

    def test_explicit_on_behalf_of_overrides_env_delegation(self, monkeypatch):
        # An explicit on_behalf_of must not leave behind the env principal's
        # delegated email/groups sub-headers: those describe a different
        # identity.  Only the overridden -On-Behalf-Of id should survive, matching
        # nmp.common.sdk_factory._get_default_headers.
        monkeypatch.setenv(
            "NMP_PRINCIPAL",
            json.dumps(
                {
                    "id": "owner@ex.com",
                    "on_behalf_of": "boss@ex.com",
                    "on_behalf_of_email": "boss@ex.com",
                    "on_behalf_of_groups": ["admin"],
                }
            ),
        )
        headers = _build_headers(on_behalf_of="override@ex.com")
        assert headers["X-NMP-Principal-On-Behalf-Of"] == "override@ex.com"
        assert "X-NMP-Principal-On-Behalf-Of-Email" not in headers
        assert "X-NMP-Principal-On-Behalf-Of-Groups" not in headers


# ---------------------------------------------------------------------------
# DefaultNemoClientProvider
# ---------------------------------------------------------------------------


class TestDefaultNemoClientProvider:
    def test_sync_default_base_url(self, monkeypatch):
        monkeypatch.delenv("NMP_BASE_URL", raising=False)
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        client = DefaultNemoClientProvider().get_nemo_client()
        assert isinstance(client, NemoClient)
        assert client.base_url == "http://localhost:8080"

    def test_sync_env_base_url_and_service_internal(self, monkeypatch):
        monkeypatch.setenv("NMP_BASE_URL", "http://test:9090")
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        client = DefaultNemoClientProvider().get_nemo_client(as_service="evaluator", internal=True)
        assert client.base_url == "http://test:9090"
        assert client._default_headers["X-NMP-Principal-Id"] == "service:evaluator"
        assert client._default_headers["X-NMP-Internal"] == "true"

    def test_sync_workspace_passthrough(self, monkeypatch):
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        client = DefaultNemoClientProvider().get_nemo_client(workspace="team-a")
        assert client.workspace == "team-a"

    def test_sync_propagates_env_principal_on_behalf_of(self, monkeypatch):
        monkeypatch.setenv(
            "NMP_PRINCIPAL",
            json.dumps({"id": "creator@ex.com", "on_behalf_of": "real@ex.com"}),
        )
        client = DefaultNemoClientProvider().get_nemo_client()
        assert client._default_headers["X-NMP-Principal-Id"] == "creator@ex.com"
        assert client._default_headers["X-NMP-Principal-On-Behalf-Of"] == "real@ex.com"

    def test_async_service_internal(self, monkeypatch):
        monkeypatch.setenv("NMP_BASE_URL", "http://test:9090")
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        client = DefaultNemoClientProvider().get_async_nemo_client(as_service="evaluator", internal=True)
        assert isinstance(client, AsyncNemoClient)
        assert client.base_url == "http://test:9090"
        assert client._default_headers["X-NMP-Principal-Id"] == "service:evaluator"
        assert client._default_headers["X-NMP-Internal"] == "true"

    def test_async_workspace_passthrough(self, monkeypatch):
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        client = DefaultNemoClientProvider().get_async_nemo_client(workspace="team-a")
        assert client.workspace == "team-a"


class TestForwardingHeaders:
    def test_returns_copy_of_sync_client_default_headers(self) -> None:
        client = NemoClient(base_url="http://test:8000", default_headers={"traceparent": "00-platform"})

        headers = get_forwarding_headers(client)
        headers["traceparent"] = "mutated"

        assert get_forwarding_headers(client) == {"traceparent": "00-platform"}

    def test_returns_copy_of_async_client_default_headers(self) -> None:
        client = AsyncNemoClient(base_url="http://test:8000", default_headers={"X-NMP-Internal": "true"})

        headers = get_forwarding_headers(client)
        headers["X-NMP-Internal"] = "false"

        assert get_forwarding_headers(client) == {"X-NMP-Internal": "true"}


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    def __init__(
        self,
        name: str,
        obj: object,
        *,
        value: str = "tests:_CustomProvider",
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


class _CustomProvider:
    def get_nemo_client(self, **kwargs) -> NemoClient:
        return NemoClient(base_url="http://custom:1234")

    def get_async_nemo_client(self, **kwargs) -> AsyncNemoClient:
        return AsyncNemoClient(base_url="http://custom:1234")

    def get_task_nemo_client(self, service_name, **kwargs) -> NemoClient:
        return NemoClient(base_url="http://custom:1234")

    def get_async_task_nemo_client(self, service_name, **kwargs) -> AsyncNemoClient:
        return AsyncNemoClient(base_url="http://custom:1234")


class TestProviderResolution:
    def setup_method(self):
        set_nemo_client_provider(None)

    def teardown_method(self):
        set_nemo_client_provider(None)

    def test_explicit_provider_takes_precedence(self, monkeypatch):
        monkeypatch.delenv("NMP_BASE_URL", raising=False)
        set_nemo_client_provider(_CustomProvider())
        assert get_nemo_client().base_url == "http://custom:1234"
        assert get_async_nemo_client().base_url == "http://custom:1234"

    def test_falls_back_to_default_when_no_entry_points(self, monkeypatch):
        monkeypatch.setenv("NMP_BASE_URL", "http://fallback:8080")
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=[]):
            client = get_nemo_client()
        assert client.base_url == "http://fallback:8080"

    def test_entry_point_provider_is_discovered(self, monkeypatch):
        monkeypatch.delenv("NMP_BASE_URL", raising=False)
        eps = [_FakeEntryPoint("platform", _CustomProvider)]
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            client = get_nemo_client()
        assert client.base_url == "http://custom:1234"

    def test_entry_point_instance_is_discovered(self, monkeypatch):
        # An entry-point that loads an instance (not a class) is used as-is.
        monkeypatch.delenv("NMP_BASE_URL", raising=False)
        eps = [_FakeEntryPoint("platform", _CustomProvider())]
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            assert get_nemo_client().base_url == "http://custom:1234"

    def test_entry_point_not_satisfying_protocol_raises(self):
        class _NotAProvider:
            pass

        eps = [_FakeEntryPoint("platform", _NotAProvider())]
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            with pytest.raises(RuntimeError, match="does not satisfy NemoClientProvider"):
                get_nemo_client()

    def test_entry_point_load_exception_raises(self):
        eps = [_FakeEntryPoint("platform", _CustomProvider, load_error=ImportError("missing provider"))]
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            with pytest.raises(RuntimeError, match="Failed to load or construct NemoClient provider") as exc_info:
                get_nemo_client()
        assert isinstance(exc_info.value.__cause__, ImportError)

    def test_entry_point_constructor_exception_raises(self):
        class _BrokenProvider:
            def __init__(self) -> None:
                raise ValueError("invalid configuration")

        eps = [_FakeEntryPoint("platform", _BrokenProvider, value="tests:_BrokenProvider")]
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            with pytest.raises(RuntimeError, match="Failed to load or construct NemoClient provider") as exc_info:
                get_nemo_client()
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_resolution_retries_after_entry_point_failure(self):
        ep = _FakeEntryPoint("platform", _CustomProvider, load_error=ImportError("temporarily unavailable"))
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=[ep]):
            with pytest.raises(RuntimeError, match="Failed to load or construct NemoClient provider"):
                get_nemo_client()
            ep._load_error = None
            assert get_nemo_client().base_url == "http://custom:1234"

    def test_multiple_named_providers_raise(self):
        eps = [
            _FakeEntryPoint("platform", _CustomProvider, value="tests:_CustomProvider"),
            _FakeEntryPoint("other", _CustomProvider, value="tests:_OtherProvider"),
        ]
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            with pytest.raises(RuntimeError, match="Multiple NemoClient providers"):
                get_nemo_client()

    def test_duplicate_name_same_target_is_deduplicated(self, monkeypatch):
        monkeypatch.delenv("NMP_BASE_URL", raising=False)
        eps = [
            _FakeEntryPoint("platform", _CustomProvider, value="tests:_CustomProvider"),
            _FakeEntryPoint("platform", _CustomProvider, value="tests:_CustomProvider"),
        ]
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            assert get_nemo_client().base_url == "http://custom:1234"

    @pytest.mark.parametrize("reverse", [False, True])
    def test_duplicate_name_different_targets_raises_deterministically(self, reverse):
        eps = [
            _FakeEntryPoint("platform", _CustomProvider, value="z_package:Provider"),
            _FakeEntryPoint("platform", _CustomProvider, value="a_package:Provider"),
        ]
        if reverse:
            eps.reverse()
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=eps):
            with pytest.raises(RuntimeError, match="Conflicting NemoClient providers") as exc_info:
                get_nemo_client()
        assert "a_package:Provider, z_package:Provider" in str(exc_info.value)

    def test_set_none_clears_and_re_resolves(self, monkeypatch):
        monkeypatch.setenv("NMP_BASE_URL", "http://re-resolved:8080")
        monkeypatch.delenv("NMP_PRINCIPAL", raising=False)
        set_nemo_client_provider(_CustomProvider())
        assert get_nemo_client().base_url == "http://custom:1234"
        set_nemo_client_provider(None)
        with patch("nemo_platform_plugin.client_provider.entry_points", return_value=[]):
            assert get_nemo_client().base_url == "http://re-resolved:8080"

    def test_public_functions_pass_workspace_through(self):
        captured: dict[str, object] = {}

        class _CapturingProvider:
            def get_nemo_client(
                self,
                *,
                as_service: str | None = None,
                internal: bool = False,
                on_behalf_of: str | None = None,
                workspace: str | None = None,
            ) -> NemoClient:
                kwargs = {
                    "as_service": as_service,
                    "internal": internal,
                    "on_behalf_of": on_behalf_of,
                    "workspace": workspace,
                }
                captured.update(kwargs)
                return NemoClient(base_url="http://x")

            def get_async_nemo_client(
                self,
                *,
                as_service: str | None = None,
                internal: bool = False,
                on_behalf_of: str | None = None,
                workspace: str | None = None,
            ) -> AsyncNemoClient:
                kwargs = {
                    "as_service": as_service,
                    "internal": internal,
                    "on_behalf_of": on_behalf_of,
                    "workspace": workspace,
                }
                captured.update(kwargs)
                return AsyncNemoClient(base_url="http://x")

            def get_task_nemo_client(self, service_name: str, *, workspace: str | None = None) -> NemoClient:
                return NemoClient(base_url="http://x", workspace=workspace)

            def get_async_task_nemo_client(self, service_name: str, *, workspace: str | None = None) -> AsyncNemoClient:
                return AsyncNemoClient(base_url="http://x", workspace=workspace)

        set_nemo_client_provider(_CapturingProvider())
        get_nemo_client(as_service="svc", internal=True, on_behalf_of="u@x", workspace="ws1")
        assert captured == {"as_service": "svc", "internal": True, "on_behalf_of": "u@x", "workspace": "ws1"}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_default_provider_is_protocol_instance(self):
        assert isinstance(DefaultNemoClientProvider(), NemoClientProvider)
