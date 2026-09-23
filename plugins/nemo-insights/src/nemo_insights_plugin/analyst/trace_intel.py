# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform boundaries around trace-intel evidence generation and reconciliation."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from insight_agent.config import EvidenceStreamsConfig
from insight_agent.evidence_streams.builtins import registered_builtin_streams
from insight_agent.insight import Insight
from insight_agent.insights_generation.insight_compilation import InsightCompilation
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
from nemo_insights_plugin.analyst.analyst_backend import AnalystBackend
from nemo_insights_plugin.analyst.result import AnalystResult, InsightUpdate, NewInsight
from nemo_insights_plugin.entities import InsightStatus
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
                            "trace_refs": item.trace_refs,
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
        if item.id is None:
            new.append(NewInsight(title=item.name, description=item.description, trace_refs=item.trace_refs))
            continue
        if item.id not in known or item.id in seen:
            raise ValueError(f"trace-intel returned an unknown or duplicate insight ID: {item.id!r}")
        seen.add(item.id)
        # Storage identity, title, description and lifecycle remain Platform-owned.
        existing_refs = set(known[item.id].trace_refs)
        added_refs = [ref for ref in dict.fromkeys(item.trace_refs) if ref not in existing_refs]
        if added_refs:
            updates.append(InsightUpdate(id=item.id, trace_refs=added_refs))
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
                from nooa.nemo_relay_middleware import nemo_relay_scope

                await stack.enter_async_context(nemo_relay_scope(compiler, relay_scope_name))
            if event_handler is not None:
                for event in ("LLMComplete", "PythonOutput"):
                    stack.callback(compiler.event_manager.on(event, event_handler))
            insights = await compiler.compile_insights(evidence, snapshot, existing)
        return to_change_set(insights, existing, trace_count=len(snapshot))
