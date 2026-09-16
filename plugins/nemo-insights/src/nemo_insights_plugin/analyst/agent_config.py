# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Insights Analyst's agent definition, built per request.

The Analyst's config is derived, not authored: its models are chosen by the
caller and its harness settings are scoped to a single run, so there is nothing
durable to store. Building it here keeps the one part that *is* fixed — the
Fabric adapter and the shape of its settings — next to the adapter it pairs
with, and lets ``agents.execute`` take it as an inline definition rather than an
Agent entity somebody has to provision and keep current.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

ANALYST_AGENT_NAME = "insights-analyst"
# The Analyst runs on the platform's generic NOOA adapter rather than an
# adapter of its own; `entrypoint` below is what makes it the Analyst.
ANALYST_ADAPTER_ID = "nvidia.nemo-platform.nooa"
ANALYST_ENTRYPOINT = "nemo_insights_plugin.fabric_adapter:run"
ANALYST_HARNESS_NAME = "insights"
AGENT_CONFIG_FORMAT = "nemo-agents-spec-v1"

# `provider` is required by the agent config schema but unused by this adapter:
# it resolves models through Platform Model Entities, not a provider SDK.
_MODEL_PROVIDER = "platform"


def build_analyst_agent_config(
    *,
    agent: str,
    workspace: str,
    default_model: str,
    fast_model: str,
    ethos: str | None = None,
    since: datetime | None = None,
    evaluation_id: str | None = None,
    base_url: str | None = None,
    enable_observability: bool | None = None,
) -> dict[str, Any]:
    """Build the Analyst's ``nemo-agents-spec-v1`` config for one run.

    Args:
        agent: Agent under test. The Analyst matches this against each span's
            normalized ``agent_name``.
        workspace: Workspace the Analyst reads telemetry from and writes
            Insights to.
        default_model: Workspace-qualified Model Entity ref for analysis work.
        fast_model: Workspace-qualified Model Entity ref for context
            summarization.
        ethos: Optional Ethos Markdown for the agent under test. Inlined into
            the harness settings because the adapter has no Files access.
        since: Optional lower bound enforced on trace/span reads.
        evaluation_id: Optional run scope AND-pinned onto every span read.
        base_url: Optional Platform base URL. Unset lets the job's own
            environment supply it.
        enable_observability: Whether the Analyst may export its own OTLP trace.

    Returns:
        A config dict suitable for ``AgentInline.config``.
    """
    settings: dict[str, Any] = {
        "entrypoint": ANALYST_ENTRYPOINT,
        "agent": agent,
        "workspace": workspace,
    }
    if ethos is not None:
        settings["ethos"] = ethos
    if since is not None:
        settings["since"] = since.isoformat()
    if evaluation_id is not None:
        settings["evaluation_id"] = evaluation_id
    if base_url is not None:
        settings["base_url"] = base_url
    if enable_observability is not None:
        settings["enable_observability"] = enable_observability

    return {
        "config_format": AGENT_CONFIG_FORMAT,
        "name": ANALYST_AGENT_NAME,
        "description": f"Insights Analyst run for agent '{agent}'.",
        "default_harness": ANALYST_HARNESS_NAME,
        "harnesses": {ANALYST_HARNESS_NAME: {"kind": ANALYST_ADAPTER_ID, "settings": settings}},
        # `default` drives the analysis; `fast` is used for context
        # summarization and falls back to `default` when the adapter cannot
        # find it.
        "models": {
            "default": {"provider": _MODEL_PROVIDER, "model": default_model},
            "fast": {"provider": _MODEL_PROVIDER, "model": fast_model},
        },
        # agents.execute only supports local Fabric environments.
        "environment": {"provider": "local"},
    }
