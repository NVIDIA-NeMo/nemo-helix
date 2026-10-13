# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR, httpx_tls_config_from_env


def test_httpx_tls_config_from_env_defaults_to_certificate_validation(monkeypatch):
    monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

    assert httpx_tls_config_from_env() == {}


def test_httpx_tls_config_from_env_ignores_blank_ca_bundle(monkeypatch):
    monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "   ")

    assert httpx_tls_config_from_env() == {}


def test_httpx_tls_config_from_env_uses_custom_ca_bundle(monkeypatch):
    monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

    assert httpx_tls_config_from_env() == {"verify": "/tmp/nemo-ca.pem"}


def test_httpx_tls_config_from_env_uses_context_certificate_authority(monkeypatch):
    monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

    assert httpx_tls_config_from_env("/tmp/context-ca.pem") == {"verify": "/tmp/context-ca.pem"}


def test_httpx_tls_config_from_env_env_ca_bundle_overrides_context_certificate_authority(monkeypatch):
    monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/env-ca.pem")

    assert httpx_tls_config_from_env("/tmp/context-ca.pem") == {"verify": "/tmp/env-ca.pem"}


def test_httpx_tls_config_from_env_uses_env_overlay(monkeypatch):
    monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/default-ca.pem")

    assert httpx_tls_config_from_env(env={NHX_CLIENT_SSL_CERT_FILE_ENVVAR: "/tmp/override-ca.pem"}) == {
        "verify": "/tmp/override-ca.pem"
    }


def test_httpx_tls_config_from_env_uses_first_nonblank_configured_envvar(monkeypatch):
    monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

    assert httpx_tls_config_from_env(
        env={
            NHX_CLIENT_SSL_CERT_FILE_ENVVAR: "   ",
            "REQUESTS_CA_BUNDLE": "/tmp/requests-ca.pem",
            "SSL_CERT_FILE": "/tmp/ssl-ca.pem",
        },
        cert_file_envvars=(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"),
    ) == {"verify": "/tmp/requests-ca.pem"}
