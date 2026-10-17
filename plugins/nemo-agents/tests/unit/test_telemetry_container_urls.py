# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container URL rewriting preserves exporter protocols and independent collectors."""

import copy
from urllib.parse import urlsplit

import pytest
from nemo_agents_plugin.runner.deployments_backend import rewrite_fabric_config_base_urls
from nemo_agents_plugin.telemetry.container_urls import telemetry_url_fields


@pytest.fixture(params=["authoring", "translated"])
def envelope(request):
    """Exercise the public authoring and translated Fabric representations."""
    return ("telemetry",) if request.param == "authoring" else ("relay", "observability")


@pytest.fixture(params=["atif", "atof-stream", "atof-endpoint", "otlp"])
def exporter(request):
    """Construct each supported network destination alongside file/S3 destinations."""

    def make(url):
        match request.param:
            case "atif":
                return {
                    "atif": {
                        "storage": [
                            {"type": "http", "endpoint": url},
                            {"type": "s3", "endpoint_url": "http://localhost:9000"},
                        ]
                    }
                }
            case "atof-stream":
                return {
                    "atof": {
                        "sinks": [{"type": "stream", "url": url}, {"type": "file", "output_directory": "./traces"}]
                    }
                }
            case "atof-endpoint":
                return {"atof": {"endpoints": [{"endpoint": url}]}}
            case "otlp":
                return {
                    "opentelemetry": {"endpoints": [{"type": "gen_ai", "endpoint": url, "transport": "http_binary"}]}
                }

    return make


def _config(envelope, exporters):
    config = {}
    target = config
    for key in envelope[:-1]:
        target = target.setdefault(key, {})
    target[envelope[-1]] = exporters
    return config


def test_all_platform_telemetry_formats_are_rebased(envelope, exporter):
    """Platform-bound exporters reach the chosen gateway and preserve request details."""
    config = _config(
        envelope, exporter("http://localhost:8080/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces?q=1")
    )
    original = copy.deepcopy(config)
    rewritten = rewrite_fabric_config_base_urls(config, "http://reachable-gateway:18080")
    destinations = list(telemetry_url_fields(rewritten))
    assert len(destinations) == 1
    holder, key = destinations[0]
    assert holder[key] == "http://reachable-gateway:18080/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces?q=1"
    assert config == original


def test_custom_collectors_keep_their_origin(envelope, exporter):
    """A host collector on another port must never be silently sent to Helix instead."""
    config = _config(envelope, exporter("http://127.0.0.1:4318/v1/traces"))
    assert rewrite_fabric_config_base_urls(config, "http://gateway:8080") == config


def test_remote_collectors_are_not_rewritten(envelope, exporter):
    """Remote exporters remain independent of local platform routing."""
    config = _config(envelope, exporter("https://collector.example:443/v1/traces"))
    assert rewrite_fabric_config_base_urls(config, "http://gateway:8080") == config


@pytest.mark.parametrize(
    "original,gateway,expected",
    [
        ("ws", "http", "ws"),
        ("ws", "https", "wss"),
        ("wss", "http", "wss"),
    ],
)
def test_atof_websocket_transport_is_preserved(original, gateway, expected):
    """Rebasing a stream endpoint must retain WebSocket transport and TLS requirements."""
    config = {
        "relay": {
            "observability": {
                "atof": {"sinks": [{"type": "stream", "url": f"{original}://localhost:8080/apis/intake/stream"}]}
            }
        }
    }
    rewritten = rewrite_fabric_config_base_urls(config, f"{gateway}://gateway:18080")
    holder, key = next(telemetry_url_fields(rewritten))
    assert urlsplit(holder[key]).scheme == expected
    assert urlsplit(holder[key]).netloc == "gateway:18080"
