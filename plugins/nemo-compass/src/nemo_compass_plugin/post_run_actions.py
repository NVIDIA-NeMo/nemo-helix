# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass post-run actions for generic ``agents.execute`` jobs.

Both actions read the change-set Compass returned in the Fabric output and save
an ``analysis-report`` result. ``compass.persist`` writes the change-set as
Insight entities; ``compass.preview`` writes nothing and renders what persisting
would have done. The change-set itself is already in the job's
``fabric_run_result`` either way.

Neither action takes config. The agent under test comes from the Compass
harness settings in the agent config the job ran, and the workspace is the
job's own, exactly as the adapter resolves them, so the action can never
disagree with the analysis about what it analyzed.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from nemo_compass_plugin.analysis.agent_config import COMPASS_ADAPTER_ID
from nemo_compass_plugin.analysis.backend import InsightNotFoundError, RemoteCompassBackend
from nemo_compass_plugin.analysis.result import CompassResult, InsightUpdate, NewInsight
from nemo_compass_plugin.evidence import TraceEvidence
from nemo_helix_plugin.agents.post_run_actions import PostRunActionContext
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client_provider import get_async_task_nemo_client
from pydantic import BaseModel, ConfigDict

REPORT_RESULT_NAME = "analysis-report"
REPORT_FILE_NAME = "analysis-report.txt"

COMPASS_PERSIST_ACTION_KIND = "compass.persist"
COMPASS_PREVIEW_ACTION_KIND = "compass.preview"

# The distributions whose versions identify the Analyst implementation. The
# plugin is not its own distribution inside the bundled ``nemo-helix`` wheel, so
# that wheel's version stands in for it there.
_PLUGIN_DISTRIBUTIONS = ("nemo-compass-plugin", "nemo-helix")
_ANALYST_DISTRIBUTION = "nemo-compass"


class CompassActionConfig(BaseModel):
    """Compass actions take no config; the agent config names the target."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class AnalysisTarget:
    agent: str
    workspace: str


@dataclass(frozen=True, slots=True)
class ExistingInsight:
    title: str
    # False when the insight belongs to another workspace or agent: persisting
    # evidence to it fails the job.
    owned: bool


class _CompassAction(ABC):
    # ``after_invoke`` parses through this same attribute, so a config the
    # create request accepted is one this action can actually run.
    config_model: ClassVar[type[CompassActionConfig]] = CompassActionConfig

    def after_invoke(self, context: PostRunActionContext) -> None:
        self.config_model.model_validate(context.config)
        # Resolve the target before parsing the output: a job pointed at the
        # wrong agent is the more fundamental mistake to report.
        target = AnalysisTarget(agent=target_agent(context.agent_config), workspace=context.ctx.workspace)
        result = _compass_result_from_fabric_output(context.fabric_result.output)
        report = asyncio.run(self._report(target=target, result=result))

        report_path = context.ctx.storage.ephemeral / REPORT_FILE_NAME
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"{_version_line()}\n\n{report}", encoding="utf-8")
        context.ctx.results.save(REPORT_RESULT_NAME, report_path)

    @abstractmethod
    async def _report(self, *, target: AnalysisTarget, result: CompassResult) -> str:
        """Act on the change-set and return the report body."""


class CompassPersistAction(_CompassAction):
    """Write the Compass change-set as Insight entities."""

    title: ClassVar[str] = "Compass: save insights"
    description: ClassVar[str] = "Save the insights a Compass analysis found and an analysis report."

    async def _report(self, *, target: AnalysisTarget, result: CompassResult) -> str:
        async with get_async_task_nemo_client("compass") as client:
            return await RemoteCompassBackend(client).persist_result(
                workspace=target.workspace, agent=target.agent, result=result
            )


class CompassPreviewAction(_CompassAction):
    """Render the Compass change-set without writing any Insight."""

    title: ClassVar[str] = "Compass: preview insights"
    description: ClassVar[str] = (
        "Save an analysis report of the insights a Compass analysis found, without saving them."
    )

    async def _report(self, *, target: AnalysisTarget, result: CompassResult) -> str:
        async with get_async_task_nemo_client("compass") as client:
            existing = await _existing_insights(client, target=target, result=result)
        return render_preview(result, existing=existing)


def target_agent(agent_config: Mapping[str, Any]) -> str:
    """Read the agent under test from Compass's harness settings.

    Mirrors the adapter's own resolution, so the action acts on exactly the
    agent the analysis read.
    """
    harness_name = agent_config.get("default_harness")
    harness = (agent_config.get("harnesses") or {}).get(harness_name)
    if not isinstance(harness, Mapping) or harness.get("kind") != COMPASS_ADAPTER_ID:
        raise ValueError(
            f"Compass actions require an agent whose default harness is the Compass adapter "
            f"({COMPASS_ADAPTER_ID!r}); got harness {harness_name!r}."
        )
    settings = harness.get("settings") or {}
    agent = settings.get("agent") or settings.get("target_agent")
    if not isinstance(agent, str) or not agent:
        raise ValueError(f"Compass harness {harness_name!r} does not name the agent to analyze in settings.agent.")
    return agent


def render_preview(result: CompassResult, *, existing: Mapping[str, ExistingInsight | None]) -> str:
    """Render a change-set as a readable report of what persisting would do.

    *existing* maps each updated insight's id to what the lookup found, or to
    ``None`` when no such insight exists; persisting would skip those.
    """
    found: list[tuple[InsightUpdate, ExistingInsight]] = []
    missing: list[InsightUpdate] = []
    foreign: list[tuple[InsightUpdate, ExistingInsight]] = []
    for upd in result.updated_insights:
        insight = existing.get(upd.id)
        if insight is None:
            missing.append(upd)
        # Persisting checks ownership only when it adds evidence.
        elif upd.evidence and not insight.owned:
            foreign.append((upd, insight))
        else:
            found.append((upd, insight))

    lines = ["Preview: no insights were saved."]
    if foreign:
        lines.append("Persisting this change-set would fail: it adds evidence to another agent's insights.")
    lines += ["", result.summary]

    lines += ["", f"Proposed new insights ({len(result.new_insights)})"]
    if not result.new_insights:
        lines.append("- none")
    for position, new in enumerate(result.new_insights, start=1):
        lines += ["", *_render_new_insight(position, new)]

    lines += ["", f"Existing insights that would gain evidence ({len(found)})"]
    if not found:
        lines.append("- none")
    for upd, insight in found:
        lines.append(f"- {insight.title} [{upd.id}]: {_evidence_summary(upd.evidence)}")
        lines += _render_evidence(upd.evidence)
    for upd in missing:
        lines.append(f"- would skip (insight not found): {upd.id}")
    for upd, insight in foreign:
        lines.append(f"- would fail (insight belongs to another agent): {insight.title} [{upd.id}]")

    if result.notes:
        lines += ["", "Notes"]
        lines += [note if note.startswith("- ") else f"- {note}" for note in result.notes]
    return "\n".join(lines)


def _render_new_insight(position: int, new: NewInsight) -> list[str]:
    ratings = [f"status {new.status.value}"]
    if new.severity is not None:
        ratings.append(f"severity {new.severity}")
    if new.confidence is not None:
        ratings.append(f"confidence {new.confidence}")
    lines = [f"{position}. {new.title}", f"   {', '.join(ratings)}"]
    lines += [f"   {line}" if line else "" for line in new.description.splitlines()]
    lines.append(f"   Evidence: {_evidence_summary(new.evidence)}")
    lines += _render_evidence(new.evidence)
    return lines


def _evidence_summary(evidence: list[TraceEvidence]) -> str:
    spans = sum(len(trace.spans) for trace in evidence)
    return f"{_plural(len(evidence), 'trace')}, {_plural(spans, 'span')}"


def _render_evidence(evidence: list[TraceEvidence]) -> list[str]:
    lines: list[str] = []
    for trace in evidence:
        lines.append(f"     - trace {trace.trace_id}" + (f" ({trace.url})" if trace.url else ""))
        lines += [f"       - span {span.span_id}" + (f" ({span.url})" if span.url else "") for span in trace.spans]
    return lines


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


async def _existing_insights(
    client: AsyncNemoClient, *, target: AnalysisTarget, result: CompassResult
) -> dict[str, ExistingInsight | None]:
    """Look up each insight the change-set would add evidence to.

    An id is not readable on its own, so this fetches the title. Uses the same
    lookup and ownership check ``compass.persist`` does, so a preview shows
    ``None`` (would skip) and unowned insights (would fail) exactly where
    persisting would.
    """
    backend = RemoteCompassBackend(client)
    existing: dict[str, ExistingInsight | None] = {}
    for upd in result.updated_insights:
        if upd.id in existing:
            continue
        try:
            insight = await backend.get_insight(workspace=target.workspace, insight_id=upd.id)
        except InsightNotFoundError:
            existing[upd.id] = None
        else:
            owned = insight.workspace == target.workspace and insight.agent == target.agent
            existing[upd.id] = ExistingInsight(title=insight.title, owned=owned)
    return existing


def _compass_result_from_fabric_output(output: Any) -> CompassResult:
    if not isinstance(output, dict):
        raise ValueError("Compass actions expected Fabric output to be a mapping.")
    if "compass_result" not in output:
        # Validating the whole output instead would report missing CompassResult
        # fields, describing the shape we wanted rather than the key we lacked.
        raise ValueError(
            f"Compass actions expected Fabric output to contain 'compass_result'; got keys: {list(output)}."
        )
    return CompassResult.model_validate(output["compass_result"])


def _version_line() -> str:
    """Name the Analyst implementation that produced a report."""
    return (
        f"Produced by {_distribution_version(*_PLUGIN_DISTRIBUTIONS)}, {_distribution_version(_ANALYST_DISTRIBUTION)}"
    )


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return f"{name} {importlib.metadata.version(name)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return f"{names[0]} (version unknown)"
