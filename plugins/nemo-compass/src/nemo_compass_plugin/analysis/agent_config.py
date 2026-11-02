# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass's agent definition, built per request.

Compass's config is derived, not authored: its models are chosen by the
caller and its harness settings are scoped to a single run, so there is nothing
durable to store. A run's read scope (``since``, ``evaluation_id``) is not part
of the config at all; it travels in the execute job's ``context``. Building it here keeps the one part that *is* fixed — the
Fabric adapter and the shape of its settings — next to the adapter it pairs
with, and lets ``agents.execute`` take it as an inline definition rather than an
Agent entity somebody has to provision and keep current.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

COMPASS_AGENT_NAME = "compass"
COMPASS_ADAPTER_ID = "nvidia.helix.compass.analyst"
COMPASS_HARNESS_NAME = "compass"
AGENT_CONFIG_FORMAT = "nemo-agents-spec-v1"

# `provider` is required by the agent config schema but unused by this adapter:
# it resolves models through Platform Model Entities, not a provider SDK.
_MODEL_PROVIDER = "platform"

# Keys the adapter reads a run's scope from in the Fabric request context.
SINCE_CONTEXT_KEY = "since"
EVALUATION_ID_CONTEXT_KEY = "evaluation_id"


def build_compass_agent_config(
    *,
    agent: str,
    default_model: str,
    fast_model: str,
    ethos: str | None = None,
    base_url: str | None = None,
    enable_observability: bool | None = None,
) -> dict[str, Any]:
    """Build Compass's ``nemo-agents-spec-v1`` config for one run.

    Args:
        agent: Agent under test. Compass matches this against each span's
            normalized ``agent_name``.
        default_model: Workspace-qualified Model Entity ref for analysis work.
        fast_model: Workspace-qualified Model Entity ref for context
            summarization.
        ethos: Optional Ethos Markdown for the agent under test. Inlined into
            the harness settings because the adapter has no Files access.
        base_url: Optional Platform base URL. Unset lets the job's own
            environment supply it.
        enable_observability: Whether Compass may export its own OTLP trace.

    Returns:
        A config dict suitable for ``AgentInline.config``.
    """
    settings: dict[str, Any] = {"agent": agent}
    if ethos is not None:
        settings["ethos"] = ethos
    if base_url is not None:
        settings["base_url"] = base_url
    if enable_observability is not None:
        settings["enable_observability"] = enable_observability

    return {
        "config_format": AGENT_CONFIG_FORMAT,
        "name": COMPASS_AGENT_NAME,
        "description": f"Compass run for agent '{agent}'.",
        "default_harness": COMPASS_HARNESS_NAME,
        "harnesses": {COMPASS_HARNESS_NAME: {"kind": COMPASS_ADAPTER_ID, "settings": settings}},
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


def build_compass_run_context(*, since: datetime | None = None, evaluation_id: str | None = None) -> dict[str, Any]:
    """Build the execute job ``context`` that scopes one Compass run.

    Args:
        since: Optional lower bound enforced on trace/span reads.
        evaluation_id: Optional run scope AND-pinned onto every span read.

    Returns:
        A dict suitable for ``ExecuteAgentJobConfig.context``; empty when the
        run is unscoped.
    """
    context: dict[str, Any] = {}
    if since is not None:
        context[SINCE_CONTEXT_KEY] = since.isoformat()
    if evaluation_id is not None:
        context[EVALUATION_ID_CONTEXT_KEY] = evaluation_id
    return context
