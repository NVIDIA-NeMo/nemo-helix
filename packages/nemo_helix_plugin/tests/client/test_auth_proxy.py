# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the upstream client built by the auth proxy."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from nemo_helix_plugin.client import auth_proxy
from nemo_helix_plugin.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR


def _capture_client_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    real_client = httpx.AsyncClient

    def fake_client(**kwargs: object) -> httpx.AsyncClient:
        captured.update(kwargs)
        return real_client()

    monkeypatch.setattr(auth_proxy, "AsyncClient", fake_client)
    return captured


def test_upstream_client_honors_client_ssl_cert_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/etc/nhx/ca.pem")
    captured = _capture_client_kwargs(monkeypatch)

    auth_proxy.build_auth_proxy_app(base_url="https://nemo.example", headers={"x-nhx-principal-id": "p"})

    assert captured["verify"] == "/etc/nhx/ca.pem"


def test_upstream_client_verifies_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)
    captured = _capture_client_kwargs(monkeypatch)

    auth_proxy.build_auth_proxy_app(base_url="https://nemo.example", headers={"x-nhx-principal-id": "p"})

    assert captured["verify"] is True


@pytest.mark.parametrize("transfer_encoded", [False, True])
@pytest.mark.parametrize("method", ["HEAD", "GET"])
@respx.mock
def test_file_size_metadata_and_response_framing(method: str, transfer_encoded: bool) -> None:
    upstream_headers = {
        "content-length": "12345",
        "x-repo-commit": "a" * 40,
        "etag": '"model-etag"',
        "connection": "x-upstream-hop",
        "x-upstream-hop": "remove-me",
    }
    if transfer_encoded:
        upstream_headers["transfer-encoding"] = "chunked"
    upstream = httpx.Response(200, headers=upstream_headers, stream=httpx.ByteStream(b""))
    respx.request(method, "https://files.test/model/resolve/main/config.json").mock(return_value=upstream)
    app = auth_proxy.build_auth_proxy_app(
        base_url="https://files.test", headers={"x-nhx-principal-id": "service:models"}
    )

    with TestClient(app) as client:
        response = client.request(method, "/model/resolve/main/config.json")

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers.get("content-length") == ("12345" if method == "HEAD" and not transfer_encoded else None)
    assert response.headers["x-repo-commit"] == "a" * 40
    assert response.headers["etag"] == '"model-etag"'
    assert "transfer-encoding" not in response.headers
    assert "connection" not in response.headers
    assert "x-upstream-hop" not in response.headers


@respx.mock
def test_head_does_not_invent_missing_file_size() -> None:
    respx.head("https://files.test/model/resolve/main/config.json").mock(
        return_value=httpx.Response(200, stream=httpx.ByteStream(b""))
    )
    app = auth_proxy.build_auth_proxy_app(
        base_url="https://files.test", headers={"x-nhx-principal-id": "service:models"}
    )

    with TestClient(app) as client:
        response = client.head("/model/resolve/main/config.json")

    assert "content-length" not in response.headers
