# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable metrics for scoring agent-evaluation trials.

The metrics are keyed off ``AgentEvalTrial``:

* Metrics (scorers) — ``AgentPhaseSuccessMetric`` reads the agent-phase outcome
  stamped on trial metadata; ``EvidencePresenceMetric`` is a genuine
  *metric-over-evidence* that scores by inspecting ``candidate.evidence`` (a
  filesystem evidence handle) rather than trusting a verifier's stamped reward.

``TrialMeasurements`` is re-exported from this module for import compatibility;
its durable model is owned by :mod:`nemo_evaluator_sdk.agent_eval.trials`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from nemo_evaluator_sdk.agent_eval.trials import TrialMeasurements  # noqa: F401 - compatibility re-export
from nemo_evaluator_sdk.enums import MetricType
from nemo_evaluator_sdk.metrics.protocol import (
    CandidateOutput,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
from nemo_evaluator_sdk.values.atif import Trajectory
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FINAL_STATE,
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_FORMAT_OTLP,
    EVIDENCE_TRACE,
    CandidateEvidence,
)
from nemo_evaluator_sdk.values.metrics import MetricBase
from nemo_evaluator_sdk.values.otlp import span_text_strings
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans
from pydantic import Field, ValidationError

logger = logging.getLogger(__name__)


class AgentPhaseSuccessMetric(MetricBase):
    """Emit ``True`` when the agent phase exited successfully, else ``False``.

    The output name stays ``agent_phase_success`` (which gating reads as a reward
    signal — ``True``/``False`` coerces to ``1.0``/``0.0``).

    A built-in metric type, so it bundles inline and needs no cloudpickle opt-in to be
    stored on a task. ``type`` is therefore a fixed discriminator and no longer
    overridable per caller.
    """

    type: Literal[MetricType.AGENT_PHASE_SUCCESS] = MetricType.AGENT_PHASE_SUCCESS

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.boolean("agent_phase_success")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        # Only an explicit boolean counts as success; a stray truthy string
        # (e.g. "false") must not mark a failed trial as passed.
        raw_agent_ok = input.candidate.metadata.get("agent_ok")
        agent_ok = raw_agent_ok if isinstance(raw_agent_ok, bool) else False
        return MetricResult(outputs=[MetricOutput(name="agent_phase_success", value=agent_ok)])


class EvidencePresenceMetric(MetricBase):
    """Emit ``True`` when a named filesystem evidence directory exists (and is non-empty).

    Reads ``candidate.evidence`` directly — the canonical metric-over-evidence
    pattern — so the result reflects what the agent actually produced on disk,
    not a reward stamped into metadata by a verifier.
    """

    type: Literal[MetricType.EVIDENCE_PRESENCE] = MetricType.EVIDENCE_PRESENCE
    evidence_name: str = Field(default=EVIDENCE_FINAL_STATE, description="Evidence directory to look for.")
    output_name: str = Field(default="evidence_present", description="Name of the emitted boolean score.")
    require_non_empty: bool = Field(
        default=True, description="Require the evidence directory to be non-empty, not merely present."
    )

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.boolean(self.output_name)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        present = False
        evidence = input.candidate.evidence
        if evidence is not None and evidence.get(self.evidence_name) is not None:
            try:
                handle = await evidence.filesystem(self.evidence_name)
                if await handle.exists():
                    present = bool(await handle.iter_paths(recursive=True)) if self.require_non_empty else True
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "EvidencePresenceMetric scored False: could not resolve evidence %r for output %r: %s",
                    self.evidence_name,
                    self.output_name,
                    exc,
                )
        return MetricResult(outputs=[MetricOutput(name=self.output_name, value=present)])


class SkillUsedMetric(MetricBase):
    """Emit ``skill_present`` and ``skill_used`` so an eval can flag a failure to use an injected skill.

    * ``skill_present`` — ``True`` when one or more skills were injected into the trial. Reads
      the ``"skills"`` metadata key a skill-aware runtime stamps — a list of provenance dicts
      (``{"name", "hash", "mode", "adapter_id", "location", ...}``, see ``fabric.skills.SkillProvenance``).
      Baseline trials carry an empty list.
    * ``skill_used`` — best-effort ``True`` when the agent referenced *any* injected skill in its ATIF
      trajectory. It matches each skill's staged ``location`` (a specific, low-false-positive path
      signal — e.g. a read of ``.agents/skills/<name>/SKILL.md``) against tool-call names/arguments,
      step messages, reasoning, and observations. A bare skill-*name* match is intentionally NOT
      counted (the name commonly appears in the task prompt), so ``skill_present=True, skill_used=False``
      flags a *likely* failure to use the skill.

    Limitation: an absent trajectory reference cannot fully distinguish "not used" from "used without
    leaving a filesystem trace" — strongest for codex-style filesystem discovery, weaker for in-context
    skill loading. Authoritative usage detection via harness skill-activation events is a follow-up.
    With no skill present, both outputs are ``False``.
    """

    type: Literal[MetricType.SKILL_USED] = MetricType.SKILL_USED
    trace_evidence: str = Field(default=EVIDENCE_TRACE, description="Trace evidence to scan for skill usage.")

    OUTPUT_PRESENT: ClassVar[str] = "skill_present"
    OUTPUT_USED: ClassVar[str] = "skill_used"
    # Metadata key skill-aware runtimes stamp the provenance list under (matches the fabric runtime).
    _SKILLS_KEY: ClassVar[str] = "skills"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.boolean(self.OUTPUT_PRESENT),
            MetricOutputSpec.boolean(self.OUTPUT_USED),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        provenances = self._extract_provenances(input.candidate.metadata)
        present = bool(provenances)
        used = await self._any_skill_used(input.candidate, provenances) if present else False
        return MetricResult(
            outputs=[
                MetricOutput(name=self.OUTPUT_PRESENT, value=present),
                MetricOutput(name=self.OUTPUT_USED, value=used),
            ]
        )

    def _extract_provenances(self, metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        skills = metadata.get(self._SKILLS_KEY)
        if isinstance(skills, list):
            return [p for p in skills if isinstance(p, Mapping) and p]
        return []

    async def _any_skill_used(self, candidate: CandidateOutput, provenances: list[Mapping[str, Any]]) -> bool:
        locations = [loc for p in provenances if isinstance(loc := p.get("location"), str) and loc]
        if not locations:
            return False
        evidence = candidate.evidence
        if evidence is None or evidence.get(self.trace_evidence) is None:
            return False
        # Each view is resolved and read before the next is considered, so a trace that is
        # discovered but will not parse falls through to the other rather than answering for it.
        unreadable: list[str] = []
        for evidence_format, read in ((EVIDENCE_FORMAT_OTLP, _otlp_used), (EVIDENCE_FORMAT_ATIF, _atif_used)):
            try:
                return await read(evidence, self.trace_evidence, locations)
            except KeyError:
                continue
            except (ValueError, ValidationError, OSError) as exc:
                # ValidationError covers Trajectory.model_validate; OSError the underlying read.
                unreadable.append(f"{evidence_format}: {exc}")
        if unreadable:
            logger.warning(
                "SkillUsedMetric scored skill_used=False: could not read trace %r (%s)",
                self.trace_evidence,
                "; ".join(unreadable),
            )
        return False


class ToolCallCountMetric(MetricBase):
    """Count how often the agent called one tool, and whether that matches an expected count.

    * ``tool_call_count`` — the number of trajectory tool calls whose ``function_name`` is
      ``tool_name`` or ends with ``-<tool_name>`` / ``_<tool_name>``, since harnesses prefix MCP tool
      names with their server (Hermes records ``mcp__<server>__<tool>``).
    * ``tool_call_count_matches`` — ``True`` when that count equals ``expected_calls``.

    Reads the ATIF view of the trace. Without a readable trajectory the count is ``0``, so an agent
    that never produced a trace scores exactly like one that never called the tool: the match is
    ``False`` unless ``expected_calls`` is itself ``0``.
    """

    type: Literal[MetricType.TOOL_CALL_COUNT] = MetricType.TOOL_CALL_COUNT
    tool_name: str = Field(description="Tool to count, without any harness or MCP server prefix.")
    expected_calls: int = Field(default=1, ge=0, description="Call count that scores ``tool_call_count_matches`` True.")
    trace_evidence: str = Field(default=EVIDENCE_TRACE, description="Trace evidence to scan for tool calls.")

    OUTPUT_COUNT: ClassVar[str] = "tool_call_count"
    OUTPUT_MATCHES: ClassVar[str] = "tool_call_count_matches"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.discrete_score(self.OUTPUT_COUNT),
            MetricOutputSpec.boolean(self.OUTPUT_MATCHES),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        count = await self._count_calls(input.candidate)
        return MetricResult(
            outputs=[
                MetricOutput(name=self.OUTPUT_COUNT, value=count),
                MetricOutput(name=self.OUTPUT_MATCHES, value=count == self.expected_calls),
            ]
        )

    async def _count_calls(self, candidate: CandidateOutput) -> int:
        trajectory = await _read_atif(candidate, self.trace_evidence, metric=type(self).__name__)
        if trajectory is None:
            return 0
        return sum(
            1
            for step in trajectory.steps
            for call in step.tool_calls or []
            if _tool_name_matches(call.function_name, self.tool_name)
        )


class ToolArgumentMatchesInputMetric(MetricBase):
    """Emit ``tool_argument_matches_input``: the agent passed a task input to a tool unchanged.

    Scores the agent's tool call, not the tool: for every trajectory call to ``tool_name`` (matched
    like :class:`ToolCallCountMetric`), the ``argument`` value must equal the task input under
    ``input_key`` after ``normalize``. ``True`` only when the tool was called at least once and every
    call matched; ``False`` when it was never called, the argument is missing or not a string, the
    task has no such input, or no readable ATIF trajectory exists.

    Reads the task input from the metric row's ``inputs`` (agent-eval passes ``task.inputs`` through
    verbatim there), so it works against any server, real or mock, in either execution mode.
    """

    type: Literal[MetricType.TOOL_ARGUMENT_MATCHES_INPUT] = MetricType.TOOL_ARGUMENT_MATCHES_INPUT
    tool_name: str = Field(description="Tool whose calls are checked, without any harness or MCP server prefix.")
    argument: str = Field(default="text", description="Tool-call argument that must carry the task input.")
    input_key: str = Field(default="instruction", description="Task input the argument must equal.")
    normalize: Literal["exact", "whitespace"] = Field(
        default="whitespace",
        description="``whitespace`` collapses runs of whitespace and trims before comparing; ``exact`` does not.",
    )
    trace_evidence: str = Field(default=EVIDENCE_TRACE, description="Trace evidence to scan for tool calls.")

    OUTPUT_MATCHES: ClassVar[str] = "tool_argument_matches_input"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.boolean(self.OUTPUT_MATCHES)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        inputs = input.row.data.get("inputs")
        expected = inputs.get(self.input_key) if isinstance(inputs, Mapping) else None
        matches = isinstance(expected, str) and await self._every_call_matches(input.candidate, expected)
        return MetricResult(outputs=[MetricOutput(name=self.OUTPUT_MATCHES, value=matches)])

    async def _every_call_matches(self, candidate: CandidateOutput, expected: str) -> bool:
        trajectory = await _read_atif(candidate, self.trace_evidence, metric=type(self).__name__)
        if trajectory is None:
            return False
        calls = [
            call
            for step in trajectory.steps
            for call in step.tool_calls or []
            if _tool_name_matches(call.function_name, self.tool_name)
        ]
        if not calls:
            return False
        wanted = self._normalize(expected)
        for call in calls:
            value = call.arguments.get(self.argument) if isinstance(call.arguments, Mapping) else None
            if not isinstance(value, str) or self._normalize(value) != wanted:
                return False
        return True

    def _normalize(self, text: str) -> str:
        return " ".join(text.split()) if self.normalize == "whitespace" else text


def _tool_name_matches(function_name: str, tool_name: str) -> bool:
    """Whether a trajectory tool name is ``tool_name``, allowing a harness/MCP-server prefix."""
    return function_name == tool_name or function_name.endswith(("-" + tool_name, "_" + tool_name))


async def _read_atif(candidate: CandidateOutput, trace_evidence: str, *, metric: str) -> Trajectory | None:
    """The ATIF trajectory behind ``trace_evidence``, or None when absent or unreadable (logged)."""
    evidence = candidate.evidence
    if evidence is None or evidence.get(trace_evidence) is None:
        return None
    try:
        return await (await evidence.trace(trace_evidence, format=EVIDENCE_FORMAT_ATIF)).trace()
    except KeyError:
        return None
    except (ValueError, ValidationError, OSError) as exc:
        logger.warning("%s could not read ATIF trace %r (%s)", metric, trace_evidence, exc)
        return None


async def _otlp_used(evidence: CandidateEvidence, name: str, locations: list[str]) -> bool:
    """Whether the OTLP view of a trace references any staged skill location."""
    resource_spans = await (await evidence.trace(name, format=EVIDENCE_FORMAT_OTLP)).resource_spans()
    return any(_otlp_references(resource_spans, location) for location in locations)


async def _atif_used(evidence: CandidateEvidence, name: str, locations: list[str]) -> bool:
    """Whether the ATIF view of a trace references any staged skill location."""
    trajectory = await (await evidence.trace(name, format=EVIDENCE_FORMAT_ATIF)).trace()
    return any(_trajectory_references(trajectory, location) for location in locations)


def _otlp_references(resource_spans: list[ResourceSpans], needle: str) -> bool:
    """Whether any string attribute on the trace's spans or events contains ``needle``."""
    return any(needle in blob for blob in span_text_strings(resource_spans))


def _trajectory_references(trajectory: Trajectory, needle: str) -> bool:
    """Whether ``needle`` appears anywhere an agent action could reference the skill.

    Scans each step's message, reasoning, tool calls (name + arguments), and observation results.
    """
    for step in trajectory.steps:
        if needle in step.message or (step.reasoning_content is not None and needle in step.reasoning_content):
            return True
        for call in step.tool_calls or []:
            if needle in call.function_name:
                return True
            if call.arguments is not None and needle in json.dumps(call.arguments, default=str):
                return True
        if step.observation is not None:
            for result in step.observation.results:
                if result.content is not None and needle in json.dumps(result.content, default=str):
                    return True
    return False
