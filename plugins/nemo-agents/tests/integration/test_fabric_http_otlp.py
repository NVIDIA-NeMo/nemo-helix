# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real serving/runtime exports; set NEMO_TEST_OTLP_HOST to a local non-loopback IP."""

import ipaddress
import json
import os
import ssl
import threading
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi.testclient import TestClient
from nemo_agents_plugin.fabric.server import create_fabric_serving_app
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

pytestmark = pytest.mark.integration


def _collector_tls(tmp_path: Path, host: str) -> tuple[ssl.SSLContext, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-collector")])
    issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-collector-ca")])
    now = datetime.now(UTC)
    ca = (
        x509.CertificateBuilder()
        .subject_name(issuer)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(host))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "collector.crt", tmp_path / "collector.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ca_path = tmp_path / "collector-ca.crt"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context, ca_path


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_served_deepagent_exports_traces(
    tmp_path: Path, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch, scheme: str
) -> None:
    host = os.environ.get("NEMO_TEST_OTLP_HOST")
    if not host:
        pytest.skip("Set NEMO_TEST_OTLP_HOST to a local non-loopback IP reachable by the agent process")
    assert not ipaddress.ip_address(host).is_loopback
    assert version("nemo-fabric") == "0.6.0"
    assert version("nemo-fabric-runtime") == "0.6.0"
    assert version("nemo-fabric-adapters-deepagents") == "0.6.0"
    assert version("nemo-relay") == "0.11.0"
    monkeypatch.setenv("OTLP_TEST_MODEL_KEY", "test-key")
    monkeypatch.setenv("NO_PROXY", f"{host},localhost,127.0.0.1")
    received = threading.Event()
    exports: list[ExportTraceServiceRequest] = []

    def collect(request: Request) -> Response:
        export = ExportTraceServiceRequest.FromString(request.get_data())
        exports.append(export)
        received.set()
        return Response(b"", content_type="application/x-protobuf")

    def model(request: Request) -> Response:
        payload = request.get_json()
        completion = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
        if payload.get("stream"):
            chunk = {**completion, "object": "chat.completion.chunk"}
            chunk["choices"] = [
                {"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
            ]
            return Response(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n", content_type="text/event-stream")
        return Response(json.dumps(completion), content_type="application/json")

    httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_handler(model)
    tls_context, cert_path = _collector_tls(tmp_path, host) if scheme == "https" else (None, None)
    with HTTPServer(host="0.0.0.0", ssl_context=tls_context) as collector:
        collector.expect_request("/v1/traces", method="POST").respond_with_handler(collect)
        endpoint = {
            "type": "gen_ai",
            "endpoint": f"{scheme}://{host}:{collector.port}/v1/traces",
            "service_name": "http-otlp-test",
            "scheduled_delay_millis": 10,
        }
        if cert_path:
            endpoint["tls"] = {"certificate": str(cert_path)}
        config = {
            "config_format": "nemo-agents-spec-v1",
            "name": "http-otlp-test",
            "default_harness": "deepagents",
            "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "test-model",
                    "base_url": httpserver.url_for("/v1"),
                    "api_key_env": "OTLP_TEST_MODEL_KEY",
                }
            },
            "environment": {"workspace": "./workspace", "artifacts": "./artifacts"},
            "telemetry": {
                "enabled": True,
                "insecure_http_hosts": [host] if scheme == "http" else [],
                "opentelemetry": {"enabled": True, "endpoints": [endpoint]},
            },
        }
        config_path = tmp_path / "agent.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with TestClient(create_fabric_serving_app(config_path)) as client:
            assert client.get("/health").status_code == 200
            result = client.post(
                "/v1/chat/completions",
                json={"model": "http-otlp-test", "messages": [{"role": "user", "content": "Say hello"}]},
            )
            assert result.status_code == 200, result.text
            assert result.json()["choices"][0]["message"]["content"] == "hello"
        assert received.wait(10), "The collector received no OTLP trace export"
        spans = [
            span
            for export in exports
            for resource in export.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        ]
        assert spans, exports
        assert any(span.trace_id and span.span_id for span in spans)
        httpserver.check_assertions()
        collector.check_assertions()
