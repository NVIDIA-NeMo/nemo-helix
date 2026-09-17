# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Insights analyst as a platform NOOA agent.

This used to be a Fabric adapter of its own — a runtime class, a descriptor, and
a closed settings schema — whose only analyst-specific behaviour was one call to
``run_analyst_change_set``. It is now an entrypoint for the generic
``nvidia.nemo-platform.nooa`` adapter, which owns the lifecycle, the Relay
activation, and the error mapping.

The Analyst is the generic adapter's first consumer. A calling convention with
no first-party consumer is one nobody exercises.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_insights_plugin.analyst.run import run_analyst_change_set
from nemo_platform_plugin.agents.nooa_contract import NooaInvocation
from nemo_platform_plugin.nooa_model_client import ConfiguredModelRefs
from nemo_platform_plugin.sdk_provider import get_async_task_sdk

ANALYST_RELAY_SCOPE = "insights-analyst"


class AnalystAdapterConfigError(ValueError):
    """The Fabric-projected analyst adapter configuration is invalid."""


async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
    """Run one Insights analysis and return its change set.

    The generic adapter maps a raised exception onto a failed ``AgentRunResult``
    and logs it, so nothing here catches broadly.
    """
    result = await _run_analysis(invocation)
    return contract.AgentRunResult(
        status=contract.AgentRunStatus.SUCCEEDED,
        output={
            "response": result.summary,
            "analyst_result": result.model_dump(mode="json"),
        },
    )


async def _run_analysis(invocation: NooaInvocation):
    settings = dict(invocation.settings)
    target_agent = _string_setting(settings, "agent")
    if target_agent is None:
        raise AnalystAdapterConfigError("harness.settings.agent is required for the Insights analyst")

    workspace = (
        _string_setting(settings, "workspace")
        or _string_context(invocation.request.context, "job_workspace")
        or os.environ.get("NMP_WORKSPACE")
        or "default"
    )
    base_url = (
        _string_setting(settings, "base_url") or os.environ.get("NMP_BASE_URL") or os.environ.get("NEMO_BASE_URL")
    )
    # Every settings read can raise, so resolve them before opening the
    # client: nothing is worth a live SDK handle that no one closes.
    ethos = _string_setting(settings, "ethos")
    since = _datetime_setting(settings, "since")
    evaluation_id = _string_setting(settings, "evaluation_id")
    enable_observability = _bool_setting(settings, "enable_observability", True)
    model_refs = ConfiguredModelRefs(
        default=_default_model_ref(settings, invocation.models),
        fast=_fast_model_ref(settings, invocation.models),
    )
    # The generic adapter activates Relay when Fabric asked for it, but opening
    # the *scope* needs the agent object, which only exists inside the run --
    # so the scope name is passed down, and must be None when Relay is off.
    # Fabric says which through the context the adapter hands us verbatim. The
    # name stays `insights-analyst` so exported telemetry keeps its identity
    # across the move off a bespoke adapter.
    telemetry = invocation.context.telemetry
    relay_scope = None
    if telemetry is not None and telemetry.relay_enabled:
        relay_scope = _string_setting(settings, "relay_scope") or ANALYST_RELAY_SCOPE
    async with get_async_task_sdk("insights") as client:
        result, _backend = await run_analyst_change_set(
            agent=target_agent,
            ethos=ethos,
            workspace=workspace,
            base_url=base_url,
            client=client,
            since=since,
            evaluation_id=evaluation_id,
            enable_observability=enable_observability,
            relay_scope_name=relay_scope,
            model_refs=model_refs,
        )
    return result


def _string_setting(settings: dict[str, Any], key: str) -> str | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AnalystAdapterConfigError(f"harness.settings.{key} must be a non-empty string")
    return value


def _string_context(context: dict[str, Any], key: str) -> str | None:
    value = context.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _datetime_setting(settings: dict[str, Any], key: str) -> datetime | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AnalystAdapterConfigError(f"harness.settings.{key} must be an ISO-8601 datetime string")
    return datetime.fromisoformat(value)


def _bool_setting(settings: dict[str, Any], key: str, default: bool) -> bool:
    value = settings.get(key, default)
    if not isinstance(value, bool):
        raise AnalystAdapterConfigError(f"harness.settings.{key} must be a boolean, got {type(value).__name__}")
    return value


def _default_model_ref(
    settings: dict[str, Any],
    models: Mapping[str, contract.AgentModelConfig],
) -> str:
    configured = _string_setting(settings, "default_model")
    if configured is not None:
        return configured
    model = models.get("default")
    if model is None:
        raise AnalystAdapterConfigError("models.default or harness.settings.default_model is required")
    return model.model


def _fast_model_ref(
    settings: dict[str, Any],
    models: Mapping[str, contract.AgentModelConfig],
) -> str:
    configured = _string_setting(settings, "fast_model")
    if configured is not None:
        return configured
    model = models.get("fast")
    if model is not None:
        return model.model
    return _default_model_ref(settings, models)
