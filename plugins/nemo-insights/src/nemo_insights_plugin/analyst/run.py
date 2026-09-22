# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable insights analyst run orchestration."""

import os
import sys
from datetime import datetime
from pathlib import Path

from nemo_insights_plugin.analyst.analyst_backend import AnalystBackend, make_analyst_backend
from nemo_insights_plugin.analyst.observability import (
    ANALYST_OBSERVABILITY_ENV,
    AnalystEvaluationContext,
    setup_analyst_observability,
)
from nemo_insights_plugin.analyst.result import AnalystResult
from nemo_insights_plugin.analyst.trace_intel import analyze_snapshot, load_existing_insights
from nemo_insights_plugin.analyst.trace_snapshot import load_trace_snapshot
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.nooa_model_client import (
    ConfiguredModelClients,
    ConfiguredModelRefs,
    activate_model_clients,
    resolve_model_clients,
)
from nooa.context_blocks import EventBase
from nooa.events import LLMComplete, PythonOutput

# Truncate long tool inputs/outputs when echoing the verbose trace so a single
# span dump doesn't flood the terminal.
_VERBOSE_TRUNCATE = 2000


async def run_analyst(
    *,
    agent: str,
    ethos: str | None,
    workspace: str,
    base_url: str | None,
    client: AsyncNeMoPlatform,
    insights_output: str | Path | None = None,
    local_only: bool = False,
    verbose: bool = False,
    since: datetime | None = None,
    evaluation_id: str | None = None,
    analyst_evaluation: AnalystEvaluationContext | None = None,
    enable_observability: bool = True,
    model_refs: ConfiguredModelRefs | None = None,
) -> str:
    """Build and run the analyst agent against an agent's telemetry.

    The trace-volume floor for scheduled runs lives in the periodic controller,
    which decides whether a run is worth launching; this entry point just runs.

    Args:
        agent: Agent under test.
        ethos: Optional Ethos Markdown for the agent under test.
        workspace: Platform workspace.
        base_url: Platform base URL. ``None`` uses the active platform context.
        client: Platform client to use. This function closes it before returning.
        insights_output: Optional local YAML path. Receives a mirror of the
            insights the platform stored, or the only copy under *local_only*.
        local_only: Skip the platform and persist insights to *insights_output*
            alone. Reserved for the insights evaluation — no CLI flag sets it.
            Requires *insights_output*.
        verbose: Whether to stream model/tool events to stderr.
        since: Optional incremental lower bound enforced on trace/span reads.
        evaluation_id: Optional run scope; AND-pinned onto every span read.
        analyst_evaluation: Optional Evaluation and test-case
            identity attached to the Analyst's own OTLP trace.
        enable_observability: Whether this run may export the Analyst's own
            OTLP trace. The environment variable can still disable export.
        model_refs: Optional explicit default/fast Model Entity IDs. Unset uses
            the active Platform CLI context.
    """
    try:
        result, backend = await run_analyst_change_set(
            agent=agent,
            ethos=ethos,
            workspace=workspace,
            base_url=base_url,
            client=client,
            insights_output=insights_output,
            local_only=local_only,
            verbose=verbose,
            since=since,
            evaluation_id=evaluation_id,
            analyst_evaluation=analyst_evaluation,
            enable_observability=enable_observability,
            model_refs=model_refs,
        )
        return await backend.persist_result(workspace=workspace, agent=agent, result=result)
    finally:
        await client.close()


async def run_analyst_change_set(
    *,
    agent: str,
    ethos: str | None = None,
    workspace: str,
    base_url: str | None,
    client: AsyncNeMoPlatform,
    insights_output: str | Path | None = None,
    local_only: bool = False,
    verbose: bool = False,
    since: datetime | None = None,
    evaluation_id: str | None = None,
    analyst_evaluation: AnalystEvaluationContext | None = None,
    enable_observability: bool = True,
    relay_scope_name: str | None = None,
    model_refs: ConfiguredModelRefs | None = None,
) -> tuple[AnalystResult, AnalystBackend]:
    """Build and run the analyst agent without persisting its change-set.

    The caller owns *client* and is responsible for closing it; this function
    never does.

    Compared to run_analyst, this function adds the following args:
        relay_scope_name: Scope to run the agent under when NeMo Relay is
            instrumenting it. ``None`` runs uninstrumented. Independent of
            *enable_observability*, which is the older direct-to-Intake path.
    """
    observability = None
    model_clients: ConfiguredModelClients | None = None
    insights_output_path = str(insights_output) if insights_output else None
    try:
        models_client = client_from_platform(client, AsyncModelsClient)
        model_clients = await resolve_model_clients(models_client, model_refs)
        backend = make_analyst_backend(
            client=client,
            insights_output=insights_output_path,
            local_only=local_only,
        )
        if base_url and enable_observability and _analyst_observability_enabled():
            observability = setup_analyst_observability(
                base_url=base_url,
                workspace=workspace,
                target_agent=agent,
                evaluation_context=analyst_evaluation,
            )
        with activate_model_clients(model_clients):
            existing = await load_existing_insights(backend, workspace=workspace, agent=agent)
            snapshot = await load_trace_snapshot(
                backend.intake,
                workspace=workspace,
                agent=agent,
                base_url=base_url or str(client.base_url).rstrip("/"),
                since=since,
                evaluation_id=evaluation_id,
            )
            result = await analyze_snapshot(
                snapshot,
                existing=existing,
                model_clients=model_clients,
                ethos=ethos,
                relay_scope_name=relay_scope_name,
                event_handler=_echo_event if verbose else None,
            )
        return result, backend
    finally:
        # *client* is deliberately absent here: it belongs to the caller, who
        # closes it once this function's work — and its own — is done.
        try:
            if observability is not None:
                observability.shutdown()
        finally:
            if model_clients is not None:
                await model_clients.aclose()


def _analyst_observability_enabled() -> bool:
    """Return false only when self-observability is explicitly disabled."""
    value = os.environ.get(ANALYST_OBSERVABILITY_ENV)
    return value is None or value.strip().lower() not in {"0", "false", "no", "off"}


def _echo_event(event: EventBase) -> None:
    """Print one useful Nooa event in the legacy verbose CLI format."""
    if isinstance(event, LLMComplete):
        if event.reasoning_content.strip():
            print(f"[thought] {_truncate(event.reasoning_content.strip())}", file=sys.stderr)
        for tool_call in event.tool_calls:
            name = str(tool_call.get("function_name", "tool"))
            arguments = tool_call.get("arguments", "")
            print(f"[tool] {name}({_truncate(str(arguments))})", file=sys.stderr)
        return

    if isinstance(event, PythonOutput):
        parts = [part.rstrip() for part in (event.stdout, event.stderr, event.error) if part.rstrip()]
        if event.value is not None:
            parts.append(repr(event.value))
        detail = "\n".join(parts) or event.execution_status.value
        print(f"[result] execute_python -> {_truncate(detail)}", file=sys.stderr)


def _truncate(text: str, limit: int = _VERBOSE_TRUNCATE) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"
