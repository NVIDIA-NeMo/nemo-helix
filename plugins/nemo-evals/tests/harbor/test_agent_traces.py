# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in agents send their own traces to the gateway, unless their target already sends them elsewhere."""

import json
from pathlib import Path

import pytest
from nemo_evals.harbor.agent_traces import export_agent_traces, trace_endpoint
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig
from nhx_evals_sdk.values import SecretRef

ENDPOINT = trace_endpoint("http://nemo-helix-api:8080/", "evals")


def _config(tmp_path: Path, **fields: object) -> HarborRuntimeConfig:
    return HarborRuntimeConfig.model_validate({"jobs_dir": tmp_path, **fields})


def test_the_endpoint_is_the_gateways_traces_route_for_the_workspace() -> None:
    assert ENDPOINT == "http://nemo-helix-api:8080/apis/inference-gateway/v2/workspaces/evals/otlp/v1/traces"


def test_claude_code_exports_only_traces_over_http_protobuf(tmp_path: Path) -> None:
    configured = export_agent_traces(_config(tmp_path, env_vars={"KEEP": "me"}), "claude-code", ENDPOINT)

    assert configured is not None
    env = configured.env_vars
    assert env["KEEP"] == "me"
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "1"
    assert env["OTEL_TRACES_EXPORTER"] == "otlp"
    assert env["OTEL_METRICS_EXPORTER"] == env["OTEL_LOGS_EXPORTER"] == "none"
    assert env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == ENDPOINT


@pytest.mark.parametrize(
    "fields",
    [
        {"env_vars": {"OTEL_TRACES_EXPORTER": "console"}},
        {"env_vars": {"CLAUDE_CODE_ENABLE_TELEMETRY": "0"}},
        {"env_secrets": {"OTEL_EXPORTER_OTLP_TRACES_HEADERS": SecretRef("collector-token")}},
    ],
    ids=["own-exporter", "telemetry-off", "own-collector"],
)
def test_claude_code_keeps_telemetry_its_target_configured(tmp_path: Path, fields: dict[str, object]) -> None:
    assert export_agent_traces(_config(tmp_path, **fields), "claude-code", ENDPOINT) is None


@pytest.mark.parametrize(
    "native",
    [None, {"model_reasoning_effort": "high"}, json.dumps({"model_reasoning_effort": "high"})],
    ids=["no-config", "inline-object", "inline-json"],
)
def test_codex_gets_a_native_otlp_trace_exporter(tmp_path: Path, native: object) -> None:
    kwargs = {} if native is None else {"config": native}

    configured = export_agent_traces(_config(tmp_path, agent_kwargs=kwargs), "codex", ENDPOINT)

    assert configured is not None
    config = configured.agent_kwargs["config"]
    assert isinstance(config, dict)
    assert config["otel"] == {"trace_exporter": {"otlp-http": {"endpoint": ENDPOINT, "protocol": "binary"}}}
    if native is not None:
        assert config["model_reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "native",
    ["/configs/codex.toml", {"otel": {"exporter": "none"}}],
    ids=["config-file", "own-otel"],
)
def test_codex_keeps_a_config_the_job_cannot_or_should_not_edit(tmp_path: Path, native: object) -> None:
    assert export_agent_traces(_config(tmp_path, agent_kwargs={"config": native}), "codex", ENDPOINT) is None


@pytest.mark.parametrize("agent", ["oracle", "aider", "openhands"])
def test_other_agents_export_nothing(tmp_path: Path, agent: str) -> None:
    assert export_agent_traces(_config(tmp_path), agent, ENDPOINT) is None
