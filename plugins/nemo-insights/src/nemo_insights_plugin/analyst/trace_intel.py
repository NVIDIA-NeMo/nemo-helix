# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform boundaries around Compass evidence generation and reconciliation."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from insight_agent.config import EvidenceStreamsConfig
from insight_agent.evidence_streams.builtins import registered_builtin_streams
from insight_agent.insight import Insight, resolve_trace_links
from insight_agent.insights_generation.insight_compilation import InsightCompilation
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
from nemo_insights_plugin.analyst.analyst_backend import AnalystBackend
from nemo_insights_plugin.analyst.result import AnalystResult, InsightUpdate, NewInsight
from nemo_insights_plugin.entities import InsightStatus
from nemo_insights_plugin.evidence import TraceEvidence, merge_evidence
from nooa.context_blocks import EventBase
from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel, ValidationError
from trace_ingest.models import TraceSnapshot

logger = logging.getLogger(__name__)


class BorrowedModelClient(UnifiedLLM):
    """Let package streams enter/exit a model client owned by the Platform run."""

    def __init__(self, client: UnifiedLLM) -> None:
        super().__init__(client.model, **client.config)
        self.client = client

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        return self.client.call(messages, tools=tools, output_model=output_model, **kwargs)

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> LLMResponse:
        return await self.client.acall(messages, tools=tools, output_model=output_model, **kwargs)


async def load_existing_insights(backend: AnalystBackend, *, workspace: str, agent: str) -> list[Insight]:
    existing: list[Insight] = []
    page = 1
    while True:
        result = await backend.list_insights(workspace=workspace, agent=agent, status=None, page=page, page_size=100)
        for item in result.data:
            if item.status == InsightStatus.RESOLVED:
                continue
            try:
                if not item.id:
                    raise ValueError("missing storage ID")
                existing.append(
                    Insight.model_validate(
                        {
                            "id": item.id,
                            "name": item.title,
                            "description": item.description,
                            "evidence": [entry.model_dump(mode="json", exclude_none=True) for entry in item.evidence],
                            "updated_date": item.updated_date,
                        }
                    )
                )
            except (ValidationError, ValueError):
                logger.warning("Skipping invalid existing insight %s", item.id, exc_info=True)
        if result.pagination is None:
            raise RuntimeError("Insights response is missing pagination metadata")
        if page >= result.pagination.total_pages:
            return existing
        page += 1


def to_change_set(insights: list[Insight], existing: list[Insight], *, trace_count: int) -> AnalystResult:
    """Validate storage identities before allowing any persistence."""
    known = {item.id: item for item in existing}
    seen: set[str] = set()
    new: list[NewInsight] = []
    updates: list[InsightUpdate] = []
    for item in insights:
        evidence = [TraceEvidence.model_validate(entry.model_dump()) for entry in item.evidence]
        if item.id is None:
            new.append(
                NewInsight(
                    title=item.name, description=item.description, evidence=evidence, updated_date=item.updated_date
                )
            )
            continue
        if item.id not in known or item.id in seen:
            raise ValueError(f"Compass returned an unknown or duplicate insight ID: {item.id!r}")
        seen.add(item.id)
        # Storage identity, title, description and lifecycle remain Platform-owned.
        previous = [TraceEvidence.model_validate(entry.model_dump()) for entry in known[item.id].evidence]
        merged = merge_evidence(previous, evidence)
        previous_by_id = {entry.trace_id: entry for entry in previous}
        added = [entry for entry in merged if entry != previous_by_id.get(entry.trace_id)]
        if added:
            updates.append(InsightUpdate(id=item.id, evidence=added, updated_date=item.updated_date))
    return AnalystResult(
        summary=f"Analyzed {trace_count} traces: {len(new)} new insights, {len(updates)} existing insights with new evidence.",
        new_insights=new,
        updated_insights=updates,
    )


async def analyze_snapshot(
    snapshot: TraceSnapshot,
    *,
    existing: list[Insight],
    model_clients: ConfiguredModelClients,
    ethos: str | None,
    relay_scope_name: str | None = None,
    event_handler: Callable[[EventBase], None] | None = None,
) -> AnalystResult:
    """Run package-owned streams and compilation with Platform model clients."""
    with TemporaryDirectory(prefix="nemo-insights-") as directory:
        config = EvidenceStreamsConfig()
        if ethos and config.ethos_divergence is not None:
            path = Path(directory) / "ETHOS.md"
            path.write_text(ethos, encoding="utf-8")
            config.ethos_divergence = config.ethos_divergence.model_copy(update={"ethos_path": path})
        registry = registered_builtin_streams(
            anomaly_and_patterns=config.anomaly_and_patterns,
            tool_issues=config.tool_issues,
            ethos_divergence=config.ethos_divergence,
            eval_failure_patterns=config.eval_failure_patterns,
            user_sentiment=config.user_sentiment,
            llm_factory=lambda: BorrowedModelClient(model_clients.fast),
        )
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(registry.analyze(name, snapshot)) for name in registry.names]
        evidence = [task.result() for task in tasks]
        if not any(item.problems for item in evidence):
            return to_change_set(existing, existing, trace_count=len(snapshot))
        compiler = InsightCompilation(llm=model_clients.default)
        async with AsyncExitStack() as stack:
            if relay_scope_name is not None:
                from nemo_insights_plugin.analyst.relay_compat import relay_scope

                await stack.enter_async_context(relay_scope(compiler.event_manager, relay_scope_name))
            if event_handler is not None:
                for event in ("LLMComplete", "PythonOutput"):
                    stack.callback(compiler.event_manager.on(event, event_handler))
            insights = await compiler.compile_insights(
                evidence, snapshot, existing, run_timestamp=datetime.now(timezone.utc)
            )
        return to_change_set(resolve_trace_links(insights, snapshot, existing), existing, trace_count=len(snapshot))
