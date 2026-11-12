# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform boundaries around Compass evidence generation and reconciliation."""

import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from compass.analysis import AnalysisOptions, BorrowedModelClient, analyze
from compass.insight import Insight
from compass.traces import TraceSnapshot
from nemo_compass_plugin.analysis.backend import CompassBackend
from nemo_compass_plugin.analysis.relay_compat import relay_scope
from nemo_compass_plugin.analysis.result import CompassResult, InsightUpdate, NewInsight
from nemo_compass_plugin.entities import InsightStatus
from nemo_compass_plugin.evidence import TraceEvidence, merge_evidence
from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
from nooa.context_blocks import EventBase
from pydantic import ValidationError

logger = logging.getLogger(__name__)

BELOW_THRESHOLD_SUMMARY = (
    "Some observations did not meet minimum recurrence thresholds and were not promoted to insight candidates; "
    "see the analysis report for details."
)


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


async def analyze_snapshot(
    snapshot: TraceSnapshot,
    *,
    existing: list[Insight],
    model_clients: ConfiguredModelClients,
    ethos: str | None,
    analysis_options: dict[str, Any] | None = None,
    relay_scope_name: str | None = None,
    event_handler: Callable[[EventBase], None] | None = None,
) -> CompassResult:
    """Call Compass's complete pipeline, then project its result into entity changes."""
    options = AnalysisOptions.model_validate(analysis_options or {})
    with TemporaryDirectory(prefix="nemo-compass-") as directory:
        ethos_path = None
        if ethos:
            ethos_path = Path(directory) / "ETHOS.md"
            ethos_path.write_text(ethos, encoding="utf-8")
        analysis = await analyze(
            snapshot,
            existing_insights=existing,
            options=options,
            llm_factory=lambda: BorrowedModelClient(model_clients.default),
            fast_llm_factory=lambda: BorrowedModelClient(model_clients.fast),
            ethos_path=ethos_path,
            event_handler=event_handler,
            instrument_agent=(lambda events: relay_scope(events, relay_scope_name)) if relay_scope_name else None,
        )
    result = to_change_set(analysis.insights, existing, trace_count=len(snapshot))
    return result.model_copy(
        update={
            "summary": f"{result.summary} {BELOW_THRESHOLD_SUMMARY}"
            if analysis.below_threshold_count
            else result.summary,
            "notes": analysis.notes,
            "analysis_result": analysis.model_dump(mode="json"),
        }
    )
