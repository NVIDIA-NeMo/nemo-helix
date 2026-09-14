# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ToolCallCountMetric (tool_call_count / tool_call_count_matches)."""

from __future__ import annotations

from typing import Any

from nemo_evaluator_sdk.agent_eval.metrics import ToolCallCountMetric
from nemo_evaluator_sdk.execution.samples import build_metric_input
from nemo_evaluator_sdk.values.evidence import EVIDENCE_TRACE, CandidateEvidence, EvidenceDescriptor


def _atif(*calls: str) -> dict[str, Any]:
    steps = [
        {"source": "agent", "message": "call", "tool_calls": [{"function_name": name, "arguments": {}}]}
        for name in calls
    ]
    steps.append({"source": "agent", "message": "done"})
    return {"schema_version": "ATIF-v1.7", "steps": steps}


def _evidence(atif: dict[str, Any]) -> CandidateEvidence:
    return CandidateEvidence(descriptors={EVIDENCE_TRACE: EvidenceDescriptor(kind="trace", format="atif", data=atif)})


async def _score(metric: ToolCallCountMetric, sample: dict[str, Any]) -> dict[str, Any]:
    result = await metric.compute_scores(build_metric_input({}, sample, 0))
    return {output.name: output.value for output in result.outputs}


def test_output_spec_declares_a_count_and_a_match() -> None:
    names = [spec.name for spec in ToolCallCountMetric(tool_name="analyze").output_spec()]
    assert names == ["tool_call_count", "tool_call_count_matches"]


async def test_counts_exact_and_harness_prefixed_names() -> None:
    """Hermes records MCP tools as ``mcp__<server>__<tool>`` (seen live), so a prefixed name is the same tool."""
    sample = {
        "evidence": _evidence(_atif("email_phishing_analyzer", "mcp__email_phishing_analyzer__email_phishing_analyzer"))
    }
    scores = await _score(ToolCallCountMetric(tool_name="email_phishing_analyzer", expected_calls=2), sample)
    assert scores == {"tool_call_count": 2, "tool_call_count_matches": True}


async def test_a_different_tool_is_not_counted() -> None:
    sample = {"evidence": _evidence(_atif("read_file", "email_phishing_analyzer_v2"))}
    scores = await _score(ToolCallCountMetric(tool_name="email_phishing_analyzer"), sample)
    assert scores == {"tool_call_count": 0, "tool_call_count_matches": False}


async def test_repeat_calls_fail_an_exactly_once_expectation() -> None:
    sample = {"evidence": _evidence(_atif("email_phishing_analyzer", "email_phishing_analyzer"))}
    scores = await _score(ToolCallCountMetric(tool_name="email_phishing_analyzer", expected_calls=1), sample)
    assert scores == {"tool_call_count": 2, "tool_call_count_matches": False}


async def test_no_trace_scores_zero_calls() -> None:
    scores = await _score(ToolCallCountMetric(tool_name="email_phishing_analyzer"), {})
    assert scores == {"tool_call_count": 0, "tool_call_count_matches": False}


async def test_an_unreadable_trace_scores_zero_rather_than_raising() -> None:
    broken = _evidence({"schema_version": "ATIF-v1.7", "steps": "not-a-list"})
    scores = await _score(
        ToolCallCountMetric(tool_name="email_phishing_analyzer", expected_calls=0), {"evidence": broken}
    )
    assert scores["tool_call_count"] == 0
    # Zero calls do satisfy an expectation of zero; the metric reports what it can read.
    assert scores["tool_call_count_matches"] is True
