# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint transport resolution: which address the platform dials for an Endpoint URL."""

from __future__ import annotations

import datetime
import ipaddress
import json
import ssl
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from nemo_deployments_plugin.config import DeploymentsConfig, ExecutorConfigEntry
from nemo_deployments_plugin.endpoint_transport import DIRECT, EndpointTransport, endpoint_transport

MINTED = "https://default--nmp-abc--http.openshell.localhost:8080/v1/chat/completions?stream=true"


def _config(monkeypatch: pytest.MonkeyPatch, *entries: dict[str, Any], default: str | None = None) -> None:
    cfg = DeploymentsConfig(executors=[ExecutorConfigEntry(**e) for e in entries], default_executor=default)
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: cfg))


def test_direct_transport_leaves_the_url_alone() -> None:
    url, headers = DIRECT.target(MINTED)
    assert url == MINTED
    assert headers == {}
    assert DIRECT.client_kwargs() == {}


def test_connect_base_swaps_netloc_and_carries_the_minted_host() -> None:
    transport = EndpointTransport(connect_base="https://openshell.openshell.svc.cluster.local:8080")
    url, headers = transport.target(MINTED)
    assert url == "https://openshell.openshell.svc.cluster.local:8080/v1/chat/completions?stream=true"
    assert headers == {"Host": "default--nmp-abc--http.openshell.localhost:8080"}


def test_plaintext_transport_has_no_client_kwargs() -> None:
    assert EndpointTransport(connect_base="http://gw:17670").client_kwargs() == {}


def _issue(
    subject: str,
    key: ec.EllipticCurvePrivateKey,
    *,
    issuer: x509.Certificate | None,
    issuer_key: ec.EllipticCurvePrivateKey | None,
    san: x509.SubjectAlternativeName | None = None,
    ca: bool = False,
) -> x509.Certificate:
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
    now = datetime.datetime.now(datetime.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer.subject if issuer is not None else name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if ca:
        builder = builder.add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    else:
        assert issuer is not None
        issuer_ski = issuer.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value
        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(issuer_ski), critical=False
        )
    if san is not None:
        builder = builder.add_extension(san, critical=False)
    return builder.sign(issuer_key or key, hashes.SHA256())


def _write_pem(path: Path, *objects: x509.Certificate | ec.EllipticCurvePrivateKey) -> str:
    chunks = []
    for obj in objects:
        if isinstance(obj, x509.Certificate):
            chunks.append(obj.public_bytes(serialization.Encoding.PEM))
        else:
            chunks.append(
                obj.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
    path.write_bytes(b"".join(chunks))
    return str(path)


def _mkdir(path: Path) -> Path:
    path.mkdir()
    return path


class _Pki:
    """A CA with one loopback server leaf and one client leaf, written as PEM files."""

    def __init__(self, root: Path) -> None:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca = _issue("test-ca", ca_key, issuer=None, issuer_key=None, ca=True)
        server_key = ec.generate_private_key(ec.SECP256R1())
        server = _issue(
            "gateway",
            server_key,
            issuer=ca,
            issuer_key=ca_key,
            san=x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
        )
        client_key = ec.generate_private_key(ec.SECP256R1())
        client = _issue("platform", client_key, issuer=ca, issuer_key=ca_key)
        self.ca_cert = _write_pem(root / "ca.crt", ca)
        self.server_cert = _write_pem(root / "server.crt", server)
        self.server_key = _write_pem(root / "server.key", server_key)
        self.client_cert = _write_pem(root / "client.crt", client)
        self.client_key = _write_pem(root / "client.key", client_key)


class _EchoHost(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"host": self.headers.get("Host"), "path": self.path}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def pki(tmp_path: Path) -> _Pki:
    return _Pki(tmp_path)


@pytest.fixture
def mtls_server(pki: _Pki) -> Iterator[str]:
    """A loopback HTTPS server that requires a client certificate from *pki*, like an mTLS gateway."""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=pki.ca_cert)
    context.load_cert_chain(pki.server_cert, pki.server_key)
    context.verify_mode = ssl.CERT_REQUIRED
    server = HTTPServer(("127.0.0.1", 0), _EchoHost)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_transport_completes_mtls_and_delivers_the_minted_host(pki: _Pki, mtls_server: str) -> None:
    transport = EndpointTransport(
        connect_base=mtls_server, ca_cert=pki.ca_cert, client_cert=(pki.client_cert, pki.client_key)
    )
    url, headers = transport.target(MINTED.split("?")[0].replace("/v1/chat/completions", "/health"))
    with httpx.Client(timeout=5, **transport.client_kwargs()) as client:
        response = client.get(url, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"host": "default--nmp-abc--http.openshell.localhost:8080", "path": "/health"}


def test_transport_without_a_client_cert_is_refused_by_an_mtls_server(pki: _Pki, mtls_server: str) -> None:
    transport = EndpointTransport(connect_base=mtls_server, ca_cert=pki.ca_cert)
    url, headers = transport.target(MINTED)
    with pytest.raises(httpx.HTTPError, match="CERTIFICATE_REQUIRED"):
        with httpx.Client(timeout=5, **transport.client_kwargs()) as client:
            client.get(url, headers=headers)


def test_transport_pins_the_configured_ca(pki: _Pki, mtls_server: str, tmp_path: Path) -> None:
    other = _Pki(_mkdir(tmp_path / "other"))
    transport = EndpointTransport(
        connect_base=mtls_server, ca_cert=other.ca_cert, client_cert=(pki.client_cert, pki.client_key)
    )
    url, headers = transport.target(MINTED)
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        with httpx.Client(timeout=5, **transport.client_kwargs()) as client:
            client.get(url, headers=headers)


def test_openshell_executor_dials_the_gateway_with_its_tls_material(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(
        monkeypatch,
        {"name": "local-k8s", "backend": "k8s"},
        {
            "name": "openshell",
            "backend": "openshell",
            "config": {
                "gateway_endpoint": "https://openshell.openshell.svc.cluster.local:8080",
                "tls": {
                    "ca_cert_path": "/etc/openshell-tls/client/ca.crt",
                    "client_cert_path": "/etc/openshell-tls/client/tls.crt",
                    "client_key_path": "/etc/openshell-tls/client/tls.key",
                },
            },
        },
    )
    transport = endpoint_transport("openshell")
    assert transport == EndpointTransport(
        connect_base="https://openshell.openshell.svc.cluster.local:8080",
        ca_cert="/etc/openshell-tls/client/ca.crt",
        client_cert=("/etc/openshell-tls/client/tls.crt", "/etc/openshell-tls/client/tls.key"),
    )


def test_plaintext_openshell_executor_dials_http_with_no_tls_material(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway installed with server.disableTls=true is reached over plain http, minted host preserved."""
    _config(
        monkeypatch,
        {
            "name": "openshell",
            "backend": "openshell",
            "config": {"gateway_endpoint": "http://openshell.openshell.svc.cluster.local:8080"},
        },
    )
    transport = endpoint_transport("openshell")
    url, headers = transport.target("http://default--nmp-abc--http.openshell.localhost:8080/health")
    assert url == "http://openshell.openshell.svc.cluster.local:8080/health"
    assert headers == {"Host": "default--nmp-abc--http.openshell.localhost:8080"}
    assert transport.client_kwargs() == {}


def test_unset_executor_resolves_the_default_like_the_reconciler(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(
        monkeypatch,
        {"name": "openshell", "backend": "openshell", "config": {"gateway_endpoint": "http://127.0.0.1:17670"}},
        default="openshell",
    )
    assert endpoint_transport(None).connect_base == "http://127.0.0.1:17670"


@pytest.mark.parametrize("executor", ["local-k8s", "local-docker", "missing", None])
def test_non_openshell_and_unknown_executors_dial_directly(
    monkeypatch: pytest.MonkeyPatch, executor: str | None
) -> None:
    _config(
        monkeypatch,
        {"name": "local-k8s", "backend": "k8s"},
        {"name": "local-docker", "backend": "docker"},
        default="local-docker",
    )
    assert endpoint_transport(executor) is DIRECT
