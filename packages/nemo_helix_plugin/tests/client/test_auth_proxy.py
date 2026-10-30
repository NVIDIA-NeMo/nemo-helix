# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the upstream client built by the auth proxy."""

from __future__ import annotations

import httpx
import pytest
from nemo_helix_plugin.client import auth_proxy
from nemo_helix_plugin.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR


def _capture_client_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    real_client = httpx.AsyncClient

    def fake_client(**kwargs: object) -> httpx.AsyncClient:
        captured.update(kwargs)
        return real_client()

    monkeypatch.setattr(auth_proxy.httpx, "AsyncClient", fake_client)
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
