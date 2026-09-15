# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four typed-client builders in ``client.bootstrap``: transport shape, auth wiring, retry, TLS."""

from __future__ import annotations

import json
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml
from nemo_platform_ext.auth.helpers import NMPOIDCConfig
from nemo_platform_ext.client.bootstrap import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_RETRY_POLICY,
    build_async_nemo_client,
    build_direct_async_nemo_client,
    build_direct_nemo_client,
    build_nemo_client,
    resolve_timeout,
)
from nemo_platform_ext.client.tls import NMP_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_platform_plugin.client.auth import TokenProviderAuth
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.endpoint import get
from nemo_platform_plugin.client.types import RetryPolicy
from pydantic import BaseModel


class Probe(BaseModel):
    ok: bool


@get("/apis/test/v2/probe")
def probe() -> Probe:
    raise NotImplementedError


def _wire(client: NemoClient) -> list[httpx.Request]:
    """Swap the transport for a recorder that answers every request, keeping the builder's auth hook."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    client._http = httpx.Client(
        transport=httpx.MockTransport(handler), auth=client._http.auth, headers=client._http.headers
    )
    return seen


_OIDC = NMPOIDCConfig(auth_enabled=True, client_id="nmp-client-id", token_endpoint="https://idp/token")


def _jwt(exp: float) -> str:
    def b64(data: bytes) -> str:
        return urlsafe_b64encode(data).rstrip(b"=").decode()

    return ".".join([b64(b'{"alg":"RS256"}'), b64(json.dumps({"exp": exp, "sub": "u"}).encode()), b64(b"sig")])


def _write_config(tmp_path: Path, *, user: dict, certificate_authority: str | None = None) -> Path:
    cluster: dict = {"name": "default", "base_url": "http://localhost:8080"}
    if certificate_authority:
        cluster["certificate_authority"] = certificate_authority
    config = {
        "current_context": "default",
        "clusters": [cluster],
        "users": [{"name": "default", **user}],
        "contexts": [{"name": "default", "cluster": "default", "user": "default", "workspace": "ws"}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _oauth_config(tmp_path: Path, **kwargs) -> Path:
    token = _jwt(time.time() + 3600)
    return _write_config(tmp_path, user={"type": "oauth", "token": token, "refresh_token": "r"}, **kwargs)


def _api_key_config(tmp_path: Path) -> Path:
    return _write_config(tmp_path, user={"type": "api-key", "api_key": "nvapi-secret"})


# ---------------------------------------------------------------------------
# timeout shape
# ---------------------------------------------------------------------------


def test_resolve_timeout_keeps_a_short_connect_phase_for_bare_numbers() -> None:
    resolved = resolve_timeout(60.0)

    assert resolved.connect == DEFAULT_CONNECT_TIMEOUT
    assert resolved.read == 60.0
    assert resolved.write == 60.0
    assert resolved.pool == 60.0


def test_resolve_timeout_default_matches_the_generated_sdk_shape() -> None:
    resolved = resolve_timeout(None)

    assert resolved == httpx.Timeout(60.0, connect=5.0)


def test_resolve_timeout_respects_an_explicit_httpx_timeout() -> None:
    explicit = httpx.Timeout(10.0, connect=30.0)

    assert resolve_timeout(explicit) is explicit


@pytest.mark.parametrize("timeout", [None, 60.0, 15])
def test_direct_builders_apply_the_connect_cap_to_the_transport_and_per_request(timeout: float | None) -> None:
    """CLIContext passes a bare float; both the httpx client and the per-request timeout must get connect=5."""
    client = build_direct_nemo_client(base_url="http://localhost:8080", timeout=timeout)
    async_client = build_direct_async_nemo_client(base_url="http://localhost:8080", timeout=timeout)

    for built in (client, async_client):
        expected = httpx.Timeout(60.0 if timeout is None else timeout, connect=DEFAULT_CONNECT_TIMEOUT)
        assert built._http.timeout == expected
        assert built._timeout == expected


@patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_OIDC)
def test_config_builders_apply_the_connect_cap(_discover, tmp_path: Path) -> None:
    config = _oauth_config(tmp_path)

    client = build_nemo_client(config_path=config, timeout=45.0)
    async_client = build_async_nemo_client(config_path=config, timeout=45.0)

    for built in (client, async_client):
        assert built._http.timeout == httpx.Timeout(45.0, connect=DEFAULT_CONNECT_TIMEOUT)
        assert built._timeout == httpx.Timeout(45.0, connect=DEFAULT_CONNECT_TIMEOUT)


# ---------------------------------------------------------------------------
# retry, TLS, headers
# ---------------------------------------------------------------------------


def test_direct_builder_defaults_to_the_generated_sdk_retry_policy() -> None:
    client = build_direct_nemo_client(base_url="http://localhost:8080")

    assert client.retry == DEFAULT_RETRY_POLICY
    assert DEFAULT_RETRY_POLICY.max_retries == 2
    assert DEFAULT_RETRY_POLICY.retryable_status_codes == (408, 409, 429)


def test_direct_builder_honours_a_retry_override() -> None:
    policy = RetryPolicy(max_retries=0)

    assert build_direct_nemo_client(base_url="http://localhost:8080", retry=policy).retry is policy
    assert build_direct_nemo_client(base_url="http://localhost:8080", retry=None).retry is None


def test_direct_builder_sends_default_headers_on_the_wire() -> None:
    client = build_direct_nemo_client(
        base_url="http://localhost:8080",
        workspace="ws",
        default_headers={"Authorization": "Bearer api-key", "X-Extra": "1"},
    )
    seen = _wire(client)

    client.send(probe())

    assert seen[0].headers["Authorization"] == "Bearer api-key"
    assert seen[0].headers["X-Extra"] == "1"
    assert client.workspace == "ws"
    assert client._auth is None


def test_direct_builder_uses_the_certificate_authority_for_verification(tmp_path: Path) -> None:
    with patch("nemo_platform_ext.client.bootstrap.httpx.Client") as client_cls:
        build_direct_nemo_client(base_url="https://nmp.example", certificate_authority="/etc/nmp/ca.pem")

    assert client_cls.call_args.kwargs["verify"] == "/etc/nmp/ca.pem"


def test_direct_builder_verifies_by_default() -> None:
    with patch("nemo_platform_ext.client.bootstrap.httpx.Client") as client_cls:
        build_direct_nemo_client(base_url="https://nmp.example")

    assert client_cls.call_args.kwargs["verify"] is True


def test_direct_builder_prefers_the_env_ca_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_ca = tmp_path / "env.pem"
    env_ca.write_text("cert")
    monkeypatch.setenv(NMP_CLIENT_SSL_CERT_FILE_ENVVAR, str(env_ca))

    with patch("nemo_platform_ext.client.bootstrap.httpx.Client") as client_cls:
        build_direct_nemo_client(base_url="https://nmp.example", certificate_authority="/other/ca.pem")

    assert client_cls.call_args.kwargs["verify"] == str(env_ca)


# ---------------------------------------------------------------------------
# auth wiring
# ---------------------------------------------------------------------------


@patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_OIDC)
def test_oauth_builder_installs_the_provider_on_both_layers(_discover, tmp_path: Path) -> None:
    client = build_nemo_client(config_path=_oauth_config(tmp_path))

    assert isinstance(client, NemoClient)
    assert client._auth is not None
    assert isinstance(client._http.auth, TokenProviderAuth)
    assert client.workspace == "ws"
    assert client.base_url.rstrip("/") == "http://localhost:8080"


@patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_OIDC)
def test_oauth_builder_sends_the_stored_token_on_the_wire(_discover, tmp_path: Path) -> None:
    token = _jwt(time.time() + 3600)
    config = _write_config(tmp_path, user={"type": "oauth", "token": token, "refresh_token": "r"})
    client = build_nemo_client(config_path=config)
    seen = _wire(client)

    client.send(probe())

    assert seen[0].headers["Authorization"] == f"Bearer {token}"


@patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_OIDC)
def test_async_oauth_builder_installs_the_provider_on_both_layers(_discover, tmp_path: Path) -> None:
    client = build_async_nemo_client(config_path=_oauth_config(tmp_path))

    assert isinstance(client, AsyncNemoClient)
    assert client._auth is not None
    assert isinstance(client._http.auth, TokenProviderAuth)


@patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_OIDC)
def test_api_key_builder_sends_the_key_as_a_bearer_header(_discover, tmp_path: Path) -> None:
    client = build_nemo_client(config_path=_api_key_config(tmp_path))
    seen = _wire(client)

    client.send(probe())

    assert seen[0].headers["Authorization"] == "Bearer nvapi-secret"


@patch("nemo_platform_ext.client.bootstrap.discover_nmp_config", return_value=_OIDC)
def test_config_builder_workspace_override_wins(_discover, tmp_path: Path) -> None:
    client = build_nemo_client(config_path=_api_key_config(tmp_path), workspace="other")

    assert client.workspace == "other"
