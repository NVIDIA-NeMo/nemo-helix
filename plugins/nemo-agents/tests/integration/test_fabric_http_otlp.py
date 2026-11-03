# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helix agent.yaml allowlisting reaches the served agent; set NEMO_TEST_OTLP_HOST to a local non-loopback IP."""

import ipaddress
import json
import os
import threading
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from nemo_agents_plugin.fabric.server import create_fabric_serving_app
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

pytestmark = pytest.mark.integration


def test_served_deepagent_exports_to_allowlisted_http_collector(
    tmp_path: Path, httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = os.environ.get("NEMO_TEST_OTLP_HOST")
    if not host:
        pytest.skip("Set NEMO_TEST_OTLP_HOST to a local non-loopback IP reachable by the agent process")
    assert not ipaddress.ip_address(host).is_loopback
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
    with HTTPServer(host="0.0.0.0") as collector:
        collector.expect_request("/v1/traces", method="POST").respond_with_handler(collect)
        endpoint = {
            "type": "gen_ai",
            "endpoint": f"http://{host}:{collector.port}/v1/traces",
            "service_name": "http-otlp-test",
            "scheduled_delay_millis": 10,
        }
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
                "insecure_http_hosts": [host],
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
