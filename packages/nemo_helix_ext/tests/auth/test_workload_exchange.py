# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RFC 8693 workload identity token exchange."""

import ast
import json
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import nemo_helix_ext.auth.workload_exchange as workload_exchange_module
import pytest
from nemo_helix_ext.auth.workload_exchange import (
    ACCESS_TOKEN_TYPE,
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    JWT_TOKEN_TYPE,
    TOKEN_EXCHANGE_GRANT_TYPE,
    WorkloadTokenExchangeError,
    WorkloadTokenExchangeProvider,
    read_subject_token_file,
    token_exchange_grant,
)
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR


def test_workload_exchange_module_has_no_nhx_common_dependency():
    assert workload_exchange_module.__file__ is not None
    tree = ast.parse(Path(workload_exchange_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        assert not any(name == "nhx.common" or name.startswith("nhx.common.") for name in imported)


def _make_jwt(claims: dict) -> str:
    header = {"alg": "RS256", "typ": "JWT"}
    h = urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    s = urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


def test_read_subject_token_file_strips_whitespace(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("subject-token\n", encoding="utf-8")

    assert read_subject_token_file(token_file) == "subject-token"


def test_read_subject_token_file_rejects_empty_file(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match=WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR):
        read_subject_token_file(token_file)


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
def test_token_exchange_grant_sends_rfc8693_request(mock_post):
    access_token = _make_jwt({"exp": int(time.time()) + 3600})
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"access_token": access_token, "expires_in": 300}
    mock_post.return_value = response

    result = token_exchange_grant(
        token_endpoint="https://idp.example.com/token",
        client_id="nemo-helix-workload",
        subject_token="subject-token",
        audience="nemo-helix",
        scope="openid email groups",
    )

    assert result["access_token"] == access_token
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["data"] == {
        "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
        "client_id": "nemo-helix-workload",
        "subject_token": "subject-token",
        "subject_token_type": JWT_TOKEN_TYPE,
        "requested_token_type": ACCESS_TOKEN_TYPE,
        "audience": "nemo-helix",
        "scope": "openid email groups",
    }


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
def test_token_exchange_grant_uses_docker_opaque_subject_token_type(mock_post):
    mock_post.return_value = httpx.Response(200, json={"access_token": "exchanged-token", "expires_in": 300})

    token_exchange_grant(
        token_endpoint="https://idp.example.com/token",
        client_id="nemo-helix-workload",
        subject_token="nhx_obo_v1.delegation.secret",
    )

    assert mock_post.call_args.kwargs["data"]["subject_token_type"] == DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
def test_token_exchange_grant_rejects_http_non_loopback_endpoint_before_sending_subject_token(mock_post):
    with pytest.raises(ValueError, match="must use HTTPS"):
        token_exchange_grant(
            token_endpoint="http://idp.example.com/token",
            client_id="nemo-helix-workload",
            subject_token="subject-token",
        )

    mock_post.assert_not_called()


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
@pytest.mark.parametrize(
    "token_endpoint",
    [
        "http://localhost:18080/token",
        "http://127.0.0.1:18080/token",
        "http://[::1]:18080/token",
    ],
)
def test_token_exchange_grant_allows_http_loopback_endpoints(mock_post, token_endpoint):
    access_token = _make_jwt({"exp": int(time.time()) + 3600})
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"access_token": access_token}
    mock_post.return_value = response

    result = token_exchange_grant(
        token_endpoint=token_endpoint,
        client_id="nemo-helix-workload",
        subject_token="subject-token",
    )

    assert result["access_token"] == access_token
    mock_post.assert_called_once()


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
def test_token_exchange_grant_uses_nemo_scoped_ca_bundle(mock_post, monkeypatch):
    access_token = _make_jwt({"exp": int(time.time()) + 3600})
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"access_token": access_token}
    mock_post.return_value = response
    monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

    result = token_exchange_grant(
        token_endpoint="https://idp.example.com/token",
        client_id="nemo-helix-workload",
        subject_token="subject-token",
    )

    assert result["access_token"] == access_token
    assert mock_post.call_args.kwargs["verify"] == "/tmp/nemo-ca.pem"


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
def test_token_exchange_grant_uses_context_certificate_authority(mock_post, tmp_path, monkeypatch):
    access_token = _make_jwt({"exp": int(time.time()) + 3600})
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"access_token": access_token}
    mock_post.return_value = response
    context_ca = str(tmp_path / "context-ca.pem")
    monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

    result = token_exchange_grant(
        token_endpoint="https://idp.example.com/token",
        client_id="nemo-helix-workload",
        subject_token="subject-token",
        certificate_authority=context_ca,
    )

    assert result["access_token"] == access_token
    assert mock_post.call_args.kwargs["verify"] == context_ca


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
def test_token_exchange_grant_surfaces_idp_error(mock_post):
    response = MagicMock()
    response.status_code = 400
    response.text = "invalid subject token"
    response.headers = {"content-type": "application/json"}
    response.json.return_value = {
        "error": "invalid_request",
        "error_description": "invalid subject token",
    }
    mock_post.return_value = response

    with pytest.raises(WorkloadTokenExchangeError, match="invalid_request - invalid subject token"):
        token_exchange_grant(
            token_endpoint="https://idp.example.com/token",
            client_id="nemo-helix-workload",
            subject_token="bad-subject-token",
        )


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
def test_token_exchange_grant_rejects_non_object_error_payload(mock_post):
    response = MagicMock()
    response.status_code = 400
    response.text = "[]"
    response.headers = {"content-type": "application/json"}
    response.json.return_value = []
    mock_post.return_value = response

    with pytest.raises(WorkloadTokenExchangeError, match="invalid_response - Token endpoint error response"):
        token_exchange_grant(
            token_endpoint="https://idp.example.com/token",
            client_id="nemo-helix-workload",
            subject_token="bad-subject-token",
        )


@patch("nemo_helix_ext.auth.workload_exchange.httpx.post")
@pytest.mark.parametrize("payload", [[], {}, {"access_token": ""}, {"access_token": None}])
def test_token_exchange_grant_rejects_success_response_without_non_empty_access_token(mock_post, payload):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    mock_post.return_value = response

    with pytest.raises(WorkloadTokenExchangeError):
        token_exchange_grant(
            token_endpoint="https://idp.example.com/token",
            client_id="nemo-helix-workload",
            subject_token="subject-token",
        )


@patch("nemo_helix_ext.auth.workload_exchange.token_exchange_grant")
def test_provider_rejects_exchange_response_without_access_token(mock_exchange, tmp_path):
    subject_token_file = tmp_path / "token"
    subject_token_file.write_text("subject-token", encoding="utf-8")
    mock_exchange.return_value = {}

    provider = WorkloadTokenExchangeProvider(
        token_endpoint="https://idp.example.com/token",
        client_id="nemo-helix-workload",
        subject_token_file=subject_token_file,
    )

    with pytest.raises(WorkloadTokenExchangeError, match="non-empty access_token"):
        provider.get_access_token()


@patch("nemo_helix_ext.auth.workload_exchange.token_exchange_grant")
@pytest.mark.parametrize(
    "expires_in",
    [None, "300", True, float("nan"), float("inf"), 10**400],
)
def test_provider_rejects_exchange_response_without_usable_lifetime(mock_exchange, tmp_path, expires_in):
    subject_token_file = tmp_path / "token"
    subject_token_file.write_text("subject-token", encoding="utf-8")
    token_data = {"access_token": "opaque-access-token"}
    if expires_in is not None:
        token_data["expires_in"] = expires_in
    mock_exchange.return_value = token_data

    provider = WorkloadTokenExchangeProvider(
        token_endpoint="https://idp.example.com/token",
        client_id="nemo-helix-workload",
        subject_token_file=subject_token_file,
    )

    with pytest.raises(WorkloadTokenExchangeError, match="usable access_token lifetime"):
        provider.get_access_token()

    assert provider.tokens.access_token == ""


@patch("nemo_helix_ext.auth.workload_exchange.token_exchange_grant")
def test_provider_rejects_expired_exchange_response_and_retries_with_current_subject_token(mock_exchange, tmp_path):
    subject_token_file = tmp_path / "token"
    subject_token_file.write_text("subject-token-one", encoding="utf-8")
    expired_access_token = _make_jwt({"exp": int(time.time()) - 10})
    fresh_access_token = _make_jwt({"exp": int(time.time()) + 3600})
    mock_exchange.side_effect = [
        {"access_token": expired_access_token},
        {"access_token": fresh_access_token},
    ]

    provider = WorkloadTokenExchangeProvider(
        token_endpoint="https://idp.example.com/token",
        client_id="nemo-helix-workload",
        subject_token_file=subject_token_file,
        audience="nemo-helix",
        scope="openid email groups",
        certificate_authority="/tmp/context-ca.pem",
        refresh_margin_seconds=0,
    )

    with pytest.raises(WorkloadTokenExchangeError, match="expired access_token"):
        provider.get_access_token()

    subject_token_file.write_text("subject-token-two", encoding="utf-8")

    assert provider.get_access_token() == fresh_access_token
    assert mock_exchange.call_count == 2
    assert mock_exchange.call_args_list[0].kwargs["subject_token"] == "subject-token-one"
    assert mock_exchange.call_args_list[1].kwargs["subject_token"] == "subject-token-two"
    assert mock_exchange.call_args_list[1].kwargs["audience"] == "nemo-helix"
    assert mock_exchange.call_args_list[1].kwargs["scope"] == "openid email groups"
    assert mock_exchange.call_args_list[1].kwargs["certificate_authority"] == "/tmp/context-ca.pem"
