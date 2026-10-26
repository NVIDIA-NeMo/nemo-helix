# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point a built-in agent's own OpenTelemetry traces at the Inference Gateway, live, from its sandbox.

The gateway accepts them only from a leased sandbox and files them under the sandbox's trial, beside the
model calls it recorded. Agents that bring their own telemetry settings are left as they are.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig

logger = logging.getLogger(__name__)

CLAUDE_CODE = "claude-code"
CODEX = "codex"
CODEX_CONFIG_KWARG = "config"

#: Variables that say a Claude Code target already chose where its telemetry goes.
_CLAUDE_CODE_TELEMETRY_VARIABLES = (
    "CLAUDE_CODE_ENABLE_TELEMETRY",
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


def trace_endpoint(origin: str, workspace: str) -> str:
    """The gateway's OTLP/HTTP traces route for ``workspace``, as reached from a sandbox at ``origin``."""
    return f"{origin.rstrip('/')}/apis/inference-gateway/v2/workspaces/{workspace}/otlp/v1/traces"


def _claude_code(config: HarborRuntimeConfig, endpoint: str) -> HarborRuntimeConfig | None:
    configured = {*config.env_vars, *config.env_secrets}
    if any(name in configured for name in _CLAUDE_CODE_TELEMETRY_VARIABLES) or any(
        name.startswith("OTEL_EXPORTER_OTLP_TRACES") for name in configured
    ):
        return None
    env = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        # Traces are behind Claude Code's beta flag; metrics and log events stay off.
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_LOGS_EXPORTER": "none",
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint,
        # A trial can end seconds after its last call, and spans still batched then are lost.
        "OTEL_TRACES_EXPORT_INTERVAL": "1000",
    }
    return config.model_copy(update={"env_vars": {**config.env_vars, **env}})


def _codex(config: HarborRuntimeConfig, endpoint: str) -> HarborRuntimeConfig | None:
    native: Any = config.agent_kwargs.get(CODEX_CONFIG_KWARG)
    if isinstance(native, str):
        try:
            native = json.loads(native)
        except json.JSONDecodeError:
            logger.info("Codex traces stay in the trial: its config is a file path the job doesn't edit")
            return None
    if native is None:
        native = {}
    if not isinstance(native, dict) or "otel" in native:
        return None
    otel = {"trace_exporter": {"otlp-http": {"endpoint": endpoint, "protocol": "binary"}}}
    kwargs = {**config.agent_kwargs, CODEX_CONFIG_KWARG: {**native, "otel": otel}}
    return config.model_copy(update={"agent_kwargs": kwargs})


def export_agent_traces(config: HarborRuntimeConfig, agent: str, endpoint: str) -> HarborRuntimeConfig | None:
    """``config`` with built-in ``agent`` exporting its traces to ``endpoint``, or ``None`` when it doesn't."""
    if agent == CLAUDE_CODE:
        return _claude_code(config, endpoint)
    if agent == CODEX:
        return _codex(config, endpoint)
    return None
