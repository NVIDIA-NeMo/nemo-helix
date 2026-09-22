# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`nhx.common.client_factory` — the rich NemoClient provider.

Covers what the platform provider adds over the plugin's env-var default:
per-service URL routing, endpoint-aware HTTP clients, principal/auth +
internal + OTEL headers, workspace defaults, and explicit client injection.
"""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.types import PreparedRequest
from nemo_helix_plugin.client_provider import NemoClientProvider
from nemo_helix_plugin.jobs import endpoints as jobs_endpoints
from nemo_helix_plugin.jobs.client import JobsClient
from nhx.common import client_factory as cf
from nhx.common.auth import Principal, auth_client_context
from nhx.common.auth.client import AuthClient
from nhx.common.config import AuthConfig, Configuration
from nhx.common.config.base import OIDCConfig
from nhx.common.observability.otel import scoped_otel_headers
from nhx.common.platform_endpoint import _AsyncHelixEndpointRoutingTransport, _SyncHelixEndpointRoutingTransport


def _auth_config_with_token_exchange() -> AuthConfig:
    return AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            workload_token_exchange_enabled=True,
            workload_token_private_key_file="/tmp/test-workload-token-private-key.pem",
        ),
    )


@pytest.fixture(autouse=True)
def _reset_client_factory_state():
    """Keep tests order-independent by clearing the config cache."""
    Configuration.clear_cache()
    try:
        yield
    finally:
        Configuration.clear_cache()


def _get(path_template: str, **path_params: str) -> PreparedRequest:
    return PreparedRequest(
        method="GET",
        path_template=path_template,
        path_params=path_params,
        content=None,
        content_type=None,
        response_type=None,
    )


def _mock_client(sink: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        sink.append(request)
        return httpx.Response(200, json={"ok": True})

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Sync construction
# ---------------------------------------------------------------------------


class TestSyncConstruction:
    def test_base_url_from_config(self):
        client = cf.get_nemo_client()
        assert isinstance(client, NemoClient)
        assert client.base_url == str(Configuration.get_platform_config().base_url).rstrip("/")

    def test_service_principal_and_internal_headers(self):
        client = cf.get_nemo_client(as_service="evaluator", internal=True)
        assert client._default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert client._default_headers["X-NHX-Internal"] == "true"
        assert client._default_headers["X-NHX-Actor-Aliases"] == "service:evaluator"

    def test_service_principal_uses_bearer_auth_in_token_exchange_mode(self):
        try:
            Configuration.set_override(_auth_config_with_token_exchange())

            with patch(
                "nhx.common.auth.workload_tokens.ServiceWorkloadAccessTokenProvider.get_access_token",
                return_value="typed-service-token",
            ):
                client = cf.get_nemo_client(as_service="evaluator", internal=True)
                assert client._auth is not None
                assert client._auth.get_access_token() == "typed-service-token"
        finally:
            Configuration.clear_override(AuthConfig)

        assert client._default_headers["X-NHX-Internal"] == "true"
        assert "X-NHX-Principal-Id" not in client._default_headers

    def test_service_principal_rejects_bearer_auth_to_remote_cleartext_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://platform.example.test")
        Configuration.clear_cache()
        try:
            Configuration.set_override(_auth_config_with_token_exchange())

            client = cf.get_nemo_client(as_service="evaluator")
            with pytest.raises(ValueError, match="NemoClient cannot send Authorization.*cleartext remote endpoint"):
                client.send(_get("/apis/entities/v2/foo"))
        finally:
            Configuration.clear_override(AuthConfig)

    def test_on_behalf_of(self):
        client = cf.get_nemo_client(as_service="svc", on_behalf_of="user@example.com")
        assert client._default_headers["X-NHX-Principal-On-Behalf-Of"] == "user@example.com"

    def test_workspace_passthrough(self):
        client = cf.get_nemo_client(workspace="team-a")
        assert client.workspace == "team-a"

    def test_uses_endpoint_sync_http_client(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        monkeypatch.setenv("NHX_ENTITIES_URL", "http://entities-svc:9999")
        Configuration.clear_cache()

        client = cf.get_nemo_client()

        assert isinstance(client._http._transport, _SyncHelixEndpointRoutingTransport)

    def test_explicit_http_client_wins(self):
        with httpx.Client() as explicit:
            client = cf.get_nemo_client(http_client=explicit)
            assert client._http is explicit


# ---------------------------------------------------------------------------
# Async construction
# ---------------------------------------------------------------------------


class TestAsyncConstruction:
    def test_base_url_from_config(self):
        client = cf.get_async_nemo_client()
        assert isinstance(client, AsyncNemoClient)
        assert client.base_url == str(Configuration.get_platform_config().base_url).rstrip("/")

    def test_service_principal_and_internal_headers(self):
        client = cf.get_async_nemo_client(as_service="evaluator", internal=True)
        assert client._default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert client._default_headers["X-NHX-Internal"] == "true"
        assert client._default_headers["X-NHX-Actor-Aliases"] == "service:evaluator"

    async def test_uses_endpoint_async_http_client(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        socket_path = tmp_path / "entities.sock"
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        monkeypatch.setenv("NHX_ENTITIES_URL", f"unix://{socket_path}")
        Configuration.clear_cache()

        client = cf.get_async_nemo_client()

        assert isinstance(client._http._transport, _AsyncHelixEndpointRoutingTransport)


# ---------------------------------------------------------------------------
# URL routing
# ---------------------------------------------------------------------------


class TestUrlRouting:
    def test_routes_service_path_to_discovered_origin(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        monkeypatch.setenv("NHX_ENTITIES_URL", "http://entities-svc:9999")
        Configuration.clear_cache()

        captured: list[httpx.Request] = []
        client = cf.get_nemo_client(as_service="entities", internal=True, http_client=_mock_client(captured))
        client.send(_get("/apis/entities/v2/foo"))

        assert str(captured[0].url) == "http://entities-svc:9999/apis/entities/v2/foo"
        assert captured[0].headers["X-NHX-Principal-Id"] == "service:entities"
        assert captured[0].headers["X-NHX-Internal"] == "true"
        assert captured[0].headers["X-NHX-Actor-Aliases"] == "service:entities"

    def test_allows_credentialed_uds_service_route(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        socket_path = tmp_path / "entities.sock"
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        monkeypatch.setenv("NHX_ENTITIES_URL", f"unix://{socket_path}")
        Configuration.clear_cache()

        captured: list[httpx.Request] = []
        client = cf.get_nemo_client(as_service="entities", internal=True, http_client=_mock_client(captured))
        client.send(_get("/apis/entities/v2/foo"))

        assert str(captured[0].url) == "http://nemo-helix.local/apis/entities/v2/foo"
        assert captured[0].headers["X-NHX-Principal-Id"] == "service:entities"

    def test_preserves_query_string_when_routing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        monkeypatch.setenv("NHX_ENTITIES_URL", "http://entities-svc:9999")
        Configuration.clear_cache()

        captured: list[httpx.Request] = []
        client = cf.get_nemo_client(http_client=_mock_client(captured))
        client.send(_get("/apis/entities/v2/models?limit=5"))

        assert str(captured[0].url) == "http://entities-svc:9999/apis/entities/v2/models?limit=5"

    def test_non_discovered_path_stays_on_platform_origin(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        monkeypatch.delenv("NHX_MODELS_URL", raising=False)
        Configuration.clear_cache()

        captured: list[httpx.Request] = []
        client = cf.get_nemo_client(http_client=_mock_client(captured))
        client.send(_get("/apis/models/v1/bar"))

        assert str(captured[0].url) == "https://nemo-gateway:8080/apis/models/v1/bar"

    def test_workspace_default_fills_path_param(self):
        captured: list[httpx.Request] = []
        client = cf.get_nemo_client(workspace="team-a", http_client=_mock_client(captured))
        client.send(_get("/apis/entities/v2/workspaces/{workspace}/models"))

        assert "/workspaces/team-a/models" in str(captured[0].url)

    def test_typed_client_jobs_property_preserves_factory_runtime_routing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        monkeypatch.setenv("NHX_JOBS_URL", "http://jobs-svc:8080")
        Configuration.clear_cache()

        client = cf.get_nemo_client(workspace="default")
        jobs = client.jobs

        assert isinstance(jobs, JobsClient)
        assert jobs.nemo_client_runtime is client.nemo_client_runtime
        request = jobs_endpoints.list_steps(workspace="default", name="job-1")
        assert jobs._resolve_path(request) == "http://jobs-svc:8080/apis/jobs/v2/workspaces/default/jobs/job-1/steps"


# ---------------------------------------------------------------------------
# Headers / auth
# ---------------------------------------------------------------------------


class TestHeadersAuth:
    def test_propagates_request_principal_when_no_service(self):
        auth_headers = {"X-NHX-Principal-Id": "user@example.com", "X-NHX-Principal-Groups": "g1,g2"}
        # platform_auth_headers reads the request principal via the platform context helper.
        with patch("nhx.common.platform_client_context.current_principal_auth_headers", return_value=auth_headers):
            client = cf.get_nemo_client()
        assert client._default_headers["X-NHX-Principal-Id"] == "user@example.com"
        assert client._default_headers["X-NHX-Principal-Groups"] == "g1,g2"

    def test_merges_otel_propagation_headers_without_adding_auth_context(self):
        with scoped_otel_headers({"traceparent": "00-trace-span-01", "X-NHX-Actor-Aliases": "attacker"}):
            client = cf.get_nemo_client(as_service="svc")
        assert client._default_headers["traceparent"] == "00-trace-span-01"
        assert client._default_headers["X-NHX-Principal-Id"] == "service:svc"
        assert client._default_headers["X-NHX-Actor-Aliases"] == "service:svc"

    def test_explicit_auth_headers_win_over_conflicting_otel_context(self):
        with scoped_otel_headers(
            {
                "traceparent": "00-trace-span-01",
                "x-nhx-principal-id": "attacker@example.com",
                "X-NHX-Principal-Groups": "admins",
                "x-NHX-Subject-Aliases": "attacker",
            }
        ):
            client = cf.get_async_nemo_client(as_service="evaluator", internal=True)
        assert client._default_headers["traceparent"] == "00-trace-span-01"
        assert client._default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert client._default_headers["X-NHX-Internal"] == "true"
        assert client._default_headers["X-NHX-Actor-Aliases"] == "service:evaluator"
        assert all(name.lower() != "x-nhx-principal-groups" for name in client._default_headers)
        assert all(name.lower() != "x-nhx-subject-aliases" for name in client._default_headers)

    def test_no_headers_leaves_default_headers_none(self):
        # No service, no principal context, no OTEL, no internal → no default headers.
        with patch("nhx.common.platform_client_context.current_principal_auth_headers", return_value={}):
            with patch("nhx.common.platform_client_context.principal_from_env", return_value=None):
                client = cf.get_nemo_client()
        assert client._default_headers == {}

    def test_token_exchange_request_context_forwards_bearer_without_trusted_headers(self):
        config = _auth_config_with_token_exchange()
        Configuration.set_override(config)
        context_token = auth_client_context.set(
            AuthClient(
                principal=Principal(id="service:models", authz_aliases=["service:models"]),
                config=config,
                bearer_token="incoming-service-token",
            )
        )
        try:
            client = cf.get_async_nemo_client()
        finally:
            auth_client_context.reset(context_token)
            Configuration.clear_override(AuthConfig)

        assert client._default_headers["Authorization"] == "Bearer incoming-service-token"
        assert all(name.lower() != "x-nhx-principal-id" for name in client._default_headers)
        assert all(name.lower() != "x-nhx-actor-aliases" for name in client._default_headers)

    async def test_token_exchange_request_context_rejects_bearer_to_remote_cleartext_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("NHX_BASE_URL", "http://platform.example.test")
        Configuration.clear_cache()
        config = _auth_config_with_token_exchange()
        Configuration.set_override(config)
        context_token = auth_client_context.set(
            AuthClient(
                principal=Principal(id="service:models", authz_aliases=["service:models"]),
                config=config,
                bearer_token="incoming-service-token",
            )
        )
        try:
            client = cf.get_async_nemo_client()
            with pytest.raises(
                ValueError, match="AsyncNemoClient cannot send Authorization.*cleartext remote endpoint"
            ):
                await client.send(_get("/apis/entities/v2/foo"))
        finally:
            auth_client_context.reset(context_token)
            Configuration.clear_override(AuthConfig)


# ---------------------------------------------------------------------------
# Test-client injection
# ---------------------------------------------------------------------------


class TestTestClientInjection:
    async def test_async_uses_explicit_http_client(self):
        test_client = httpx.AsyncClient(base_url="http://testserver")
        try:
            client = cf.get_async_nemo_client(as_service="evaluator", http_client=test_client)
            assert client._http is test_client
        finally:
            await test_client.aclose()


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class TestHelixNemoClientProvider:
    def test_satisfies_protocol(self):
        assert isinstance(cf.HelixNemoClientProvider(), NemoClientProvider)

    def test_get_nemo_client_returns_routed_sync_client(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "https://nemo-gateway:8080")
        Configuration.clear_cache()
        provider = cf.HelixNemoClientProvider()
        client = provider.get_nemo_client(as_service="svc", internal=True, workspace="ws1")
        assert isinstance(client, NemoClient)
        assert client.base_url == "https://nemo-gateway:8080"
        assert client.workspace == "ws1"
        assert client._default_headers["X-NHX-Principal-Id"] == "service:svc"

    def test_get_async_nemo_client_returns_async_client(self):
        provider = cf.HelixNemoClientProvider()
        client = provider.get_async_nemo_client(as_service="svc")
        assert isinstance(client, AsyncNemoClient)
        assert client._default_headers["X-NHX-Principal-Id"] == "service:svc"


# ---------------------------------------------------------------------------
# Task client: creator delegation (PR-800 claim 1)
# ---------------------------------------------------------------------------


class TestTaskClientDelegation:
    def test_task_client_delegates_to_job_creator(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            '{"id": "user:alice@acme.com", "email": "alice@acme.com", "groups": ["team-a"]}',
        )
        client = cf.get_task_nemo_client("evaluator")
        headers = client._default_headers
        assert headers["X-NHX-Internal"] == "true"
        assert headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert headers["X-NHX-Actor-Aliases"] == "service:evaluator"
        assert headers["X-NHX-Principal-On-Behalf-Of"] == "user:alice@acme.com"
        assert headers["X-NHX-Principal-On-Behalf-Of-Email"] == "alice@acme.com"
        assert headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "team-a"

    def test_task_client_without_principal_warns(self, monkeypatch: pytest.MonkeyPatch, caplog):
        monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
        with caplog.at_level("WARNING"):
            client = cf.get_task_nemo_client("evaluator")
        assert client._default_headers["X-NHX-Principal-Id"] == "service:evaluator"
        assert "X-NHX-Principal-On-Behalf-Of" not in client._default_headers
        assert "without on-behalf-of delegation" in caplog.text

    async def test_async_task_client_delegates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(
            "NHX_PRINCIPAL",
            '{"id": "user:alice@acme.com", "email": "alice@acme.com", "groups": ["team-a"]}',
        )
        client = cf.get_async_task_nemo_client("evaluator")
        assert client._default_headers["X-NHX-Principal-On-Behalf-Of"] == "user:alice@acme.com"

    def test_provider_exposes_task_methods(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_PRINCIPAL", '{"id": "user:alice@acme.com"}')
        provider = cf.HelixNemoClientProvider()
        headers = provider.get_task_nemo_client("evaluator")._default_headers
        assert headers["X-NHX-Principal-On-Behalf-Of"] == "user:alice@acme.com"


# ---------------------------------------------------------------------------
# Task client: workload identity (PR-800 claim 2)
# ---------------------------------------------------------------------------


class _FakeExchangeProvider:
    def get_access_token(self) -> str:
        return "exchanged-token"

    async def get_access_token_async(self) -> str:
        return "exchanged-token"


class TestTaskClientWorkloadIdentity:
    @pytest.fixture
    def _stub_exchange(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, str] = {}

        def _fake(*, base_url, subject_token_file):
            captured["base_url"] = base_url
            captured["subject_token_file"] = str(subject_token_file)
            return _FakeExchangeProvider()

        monkeypatch.setattr(
            "nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider",
            _fake,
        )
        return captured

    def test_task_client_bootstraps_workload_identity(self, monkeypatch, tmp_path, _stub_exchange):
        token_file = tmp_path / "token"
        token_file.write_text("subject-token")
        monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
        monkeypatch.setenv("NHX_PRINCIPAL", '{"id": "user:alice@acme.com"}')  # ignored in WI mode
        Configuration.clear_cache()

        client = cf.get_task_nemo_client("evaluator")
        assert isinstance(client._auth, _FakeExchangeProvider)
        assert _stub_exchange["base_url"] == "http://platform:8080"
        # No trusted principal headers in workload-identity mode.
        assert "X-NHX-Principal-Id" not in client._default_headers
        assert client._default_headers.get("X-NHX-Internal") == "true"

    def test_uds_does_not_bootstrap_workload_identity(self, monkeypatch, tmp_path, _stub_exchange):
        # Matches get_task_sdk exactly: with the WI token file set the task path
        # delegates to get_nemo_client(internal=True); on UDS transport that skips
        # bearer exchange and propagates the env principal as its own identity
        # (no service principal, no bearer auth).
        token_file = tmp_path / "token"
        token_file.write_text("subject-token")
        monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", str(token_file))
        monkeypatch.setenv("NHX_BASE_URL", "unix:///tmp/nemo-helix.sock")
        monkeypatch.setenv("NHX_PRINCIPAL", '{"id": "user:alice@acme.com"}')
        Configuration.clear_cache()

        client = cf.get_task_nemo_client("evaluator")
        assert client._auth is None
        assert client._default_headers["X-NHX-Principal-Id"] == "user:alice@acme.com"
        assert "X-NHX-Principal-On-Behalf-Of" not in client._default_headers


# ---------------------------------------------------------------------------
# UDS endpoint routing + transport (PR-800 claim 3)
# ---------------------------------------------------------------------------


class TestUdsTransport:
    def test_uds_base_url_is_normalized_not_pathed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "unix:///tmp/nemo-helix.sock")
        Configuration.clear_cache()
        client = cf.get_nemo_client()
        # base_url is the routable host, not the raw unix:// socket path.
        assert client.base_url == "http://nemo-helix.local"
        # concatenating an API path yields a valid URL, not a broken one.
        assert client.base_url + "/apis/entities/v2/foo" == "http://nemo-helix.local/apis/entities/v2/foo"

    def test_uds_sync_client_uses_http_transport(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "unix:///tmp/nemo-helix.sock")
        Configuration.clear_cache()
        client = cf.get_nemo_client()
        transport = client._http._transport
        assert isinstance(transport, httpx.HTTPTransport)

    async def test_uds_async_client_uses_http_transport(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "unix:///tmp/nemo-helix.sock")
        Configuration.clear_cache()
        client = cf.get_async_nemo_client()
        transport = client._http._transport
        assert isinstance(transport, httpx.AsyncHTTPTransport)

    def test_tcp_client_uses_platform_base_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
        Configuration.clear_cache()
        client = cf.get_nemo_client()
        assert client.base_url == "http://platform:8080"
