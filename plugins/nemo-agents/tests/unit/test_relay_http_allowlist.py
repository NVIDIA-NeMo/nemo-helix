# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Fabric/Relay boundary for opt-in HTTP trace exports."""

from pathlib import Path
from typing import Any

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_agents_plugin.runner.deployments_backend import rewrite_fabric_config_base_urls
from nemo_fabric import Fabric
from nemo_relay import plugin


def _plugin_config(tmp_path: Path, endpoint: str, hosts: list[str]) -> dict[str, Any]:
    payload = {
        "config_format": "nemo-agents-spec-v1",
        "name": "http-trace-test",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents"}},
        "models": {"default": {"provider": "openai", "model": "test-model"}},
        "telemetry": {
            "enabled": True,
            "insecure_http_hosts": hosts,
            "opentelemetry": {"enabled": True, "endpoints": [{"type": "gen_ai", "endpoint": endpoint}]},
        },
    }
    rewritten = rewrite_fabric_config_base_urls(payload, "http://127.0.0.1:8090")
    config = AgentConfig.model_validate(rewritten)
    translated = translate_agent_config(config)
    plan = Fabric().plan(translated, base_dir=tmp_path).to_mapping()
    observability = plan["config"]["relay"]["observability"]
    assert observability["insecure_http_hosts"] == hosts
    assert observability["opentelemetry"]["endpoints"][0]["endpoint"] == endpoint
    return {"version": 1, "components": [{"kind": "observability", "enabled": True, "config": observability}]}


@pytest.mark.parametrize(
    ("endpoint", "hosts"),
    [
        (
            "http://collector.observability.svc.cluster.local:4318/v1/traces",
            ["collector.observability.svc.cluster.local"],
        ),
        ("https://collector.example.com/v1/traces", []),
        ("http://127.0.0.1:4318/v1/traces", []),
    ],
)
def test_relay_accepts_supported_trace_destinations(tmp_path: Path, endpoint: str, hosts: list[str]) -> None:
    report = plugin.validate_exact(_plugin_config(tmp_path, endpoint, hosts))
    errors = [item for item in report["config"]["diagnostics"] if item["level"] == "error"]
    assert not errors, errors


@pytest.mark.parametrize("hosts", [[], ["example.com"], ["192.0.2.1"]])
def test_relay_rejects_http_trace_host_without_exact_allowance(tmp_path: Path, hosts: list[str]) -> None:
    report = plugin.validate_exact(_plugin_config(tmp_path, "http://collector.example.com:4318/v1/traces", hosts))
    assert any(item["level"] == "error" for item in report["config"]["diagnostics"]), report


@pytest.mark.parametrize(
    "host",
    ["http://collector", "collector:4318", "collector/path", "*.example.com", ".example.com", "user@collector"],
)
def test_relay_rejects_malformed_allowlist_entries(tmp_path: Path, host: str) -> None:
    report = plugin.validate_exact(_plugin_config(tmp_path, "https://collector.example.com/v1/traces", [host]))
    assert any(item["level"] == "error" for item in report["config"]["diagnostics"]), report
