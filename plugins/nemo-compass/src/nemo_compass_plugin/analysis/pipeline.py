# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform boundaries around Compass evidence generation and reconciliation."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from compass.compass_signals.anomaly_and_patterns.signal import AnomalyAndPatternsArtifacts
from compass.compass_signals.builtins import registered_builtin_signals
from compass.compass_signals.compass_signals import CompassSignalResult
from compass.compass_signals.tool_issues.signal import ToolIssueEvidenceArtifacts
from compass.config import CompassSignalsConfig
from compass.insight import Insight, resolve_trace_links
from compass.insights_generation.insight_compilation import InsightCompilation
from nemo_compass_plugin.analysis.backend import CompassBackend
from nemo_compass_plugin.analysis.result import CompassResult, InsightUpdate, NewInsight
from nemo_compass_plugin.entities import InsightStatus
from nemo_compass_plugin.evidence import TraceEvidence, merge_evidence
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
from nooa.context_blocks import EventBase
from nooa.unifiedllm import LLMResponse, Tool, UnifiedLLM
from pydantic import BaseModel, ValidationError
from trace_ingest.models import TraceSnapshot

logger = logging.getLogger(__name__)

# Below-threshold detail lines kept in the report; the rest are counted.
MAX_BELOW_THRESHOLD_NOTES = 5
BELOW_THRESHOLD_SUMMARY = (
    "Some observations did not meet minimum recurrence thresholds and were not promoted to insight candidates; "
    "see the analysis report for details."
)


class BorrowedModelClient(UnifiedLLM):
    """Let package signals enter/exit a model client owned by the Platform run."""

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


async def load_existing_insights(backend: CompassBackend, *, workspace: str, agent: str) -> list[Insight]:
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
                            "severity": item.severity,
                            "confidence": item.confidence,
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


def to_change_set(insights: list[Insight], existing: list[Insight], *, trace_count: int) -> CompassResult:
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
                    title=item.name,
                    description=item.description,
                    evidence=evidence,
                    updated_date=item.updated_date,
                    severity=item.severity,
                    confidence=item.confidence,
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
    return CompassResult(
        summary=f"Analyzed {trace_count} traces: {len(new)} new insights, {len(updates)} existing insights with new evidence.",
        new_insights=new,
        updated_insights=updates,
    )


def _one_line(text: str) -> str:
    """Escape line breaks in trace-derived text so it cannot add lines to the report."""
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _tool_issue_shortfalls(artifacts: ToolIssueEvidenceArtifacts) -> list[tuple[int, str]]:
    required = artifacts.config.minimum_independent_cases
    shortfalls = []
    for card in artifacts.cards:
        if card.eligible_for_insights:
            continue
        tools = ", ".join(dict.fromkeys(_one_line(item.tool_name) for item in card.representative_evidence))
        shortfalls.append(
            (
                card.independent_case_count,
                f"- below threshold: tool-issues {tools} {card.issue_type} "
                f"({card.independent_case_count} of {required} independent cases)",
            )
        )
    return shortfalls


def _failure_pattern_shortfalls(artifacts: AnomalyAndPatternsArtifacts) -> list[tuple[int, str]]:
    analysis = artifacts.result
    grouped = {group.signature for group in analysis.failure_groups}
    cross_tool = {group.message_signature for group in analysis.cross_tool_failure_groups}
    traces: dict[str, set[str]] = defaultdict(set)
    for event in analysis.failure_events:
        if event["signature"] in grouped or event.get("message_signature") in cross_tool:
            continue
        traces[str(event["signature"])].add(str(event["trace_id"]))
    required = artifacts.config.minimum_independent_traces
    return [
        (
            len(trace_ids),
            f"- below threshold: anomaly-and-patterns {_one_line(signature)} ({len(trace_ids)} of {required} traces)",
        )
        for signature, trace_ids in traces.items()
    ]


def with_evidence_notes(result: CompassResult, evidence: Sequence[CompassSignalResult]) -> CompassResult:
    """Report evidence that was held below a recurrence threshold or never analyzed."""
    shortfalls: list[tuple[int, str]] = []
    skipped: list[str] = []
    for item in evidence:
        if isinstance(item.artifacts, ToolIssueEvidenceArtifacts):
            shortfalls.extend(_tool_issue_shortfalls(item.artifacts))
        elif isinstance(item.artifacts, AnomalyAndPatternsArtifacts):
            shortfalls.extend(_failure_pattern_shortfalls(item.artifacts))
        if item.skip_reason is not None:
            skipped.append(f"- skipped: {item.signal_name} ({item.skip_reason})")
    # Closest to the threshold first, so a capped list keeps the likeliest issues.
    shortfalls.sort(key=lambda shortfall: -shortfall[0])
    notes = [line for _, line in shortfalls[:MAX_BELOW_THRESHOLD_NOTES]]
    if len(shortfalls) > MAX_BELOW_THRESHOLD_NOTES:
        notes.append(f"- ... and {len(shortfalls) - MAX_BELOW_THRESHOLD_NOTES} more below threshold")
    summary = f"{result.summary} {BELOW_THRESHOLD_SUMMARY}" if shortfalls else result.summary
    return result.model_copy(update={"summary": summary, "notes": [*result.notes, *notes, *skipped]})


async def analyze_snapshot(
    snapshot: TraceSnapshot,
    *,
    existing: list[Insight],
    model_clients: ConfiguredModelClients,
    ethos: str | None,
    relay_scope_name: str | None = None,
    event_handler: Callable[[EventBase], None] | None = None,
) -> CompassResult:
    """Run package-owned signals and compilation with Platform model clients."""
    with TemporaryDirectory(prefix="nemo-compass-") as directory:
        config = CompassSignalsConfig()
        ethos_path = None
        if ethos:
            ethos_path = Path(directory) / "ETHOS.md"
            ethos_path.write_text(ethos, encoding="utf-8")
        registry = registered_builtin_signals(
            anomaly_and_patterns=config.anomaly_and_patterns,
            tool_issues=config.tool_issues,
            ethos_divergence=config.ethos_divergence,
            ethos_path=ethos_path,
            grader_failure_patterns=config.grader_failure_patterns,
            user_sentiment=config.user_sentiment,
            llm_factory=lambda: BorrowedModelClient(model_clients.fast),
        )
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(registry.analyze(name, snapshot)) for name in registry.names]
        evidence = [task.result() for task in tasks]
        if not any(item.problems for item in evidence):
            return with_evidence_notes(to_change_set(existing, existing, trace_count=len(snapshot)), evidence)
        compiler = InsightCompilation(llm=model_clients.default)
        async with AsyncExitStack() as stack:
            if relay_scope_name is not None:
                from nemo_compass_plugin.analysis.relay_compat import relay_scope

                await stack.enter_async_context(relay_scope(compiler.event_manager, relay_scope_name))
            if event_handler is not None:
                for event in ("LLMComplete", "PythonOutput"):
                    stack.callback(compiler.event_manager.on(event, event_handler))
            insights = await compiler.compile_insights(
                evidence, snapshot, existing, run_timestamp=datetime.now(timezone.utc)
            )
        return with_evidence_notes(
            to_change_set(resolve_trace_links(insights, snapshot, existing), existing, trace_count=len(snapshot)),
            evidence,
        )
