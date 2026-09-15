# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ToolArgumentMatchesInputMetric (tool_argument_matches_input)."""

from __future__ import annotations

from typing import Any

from nemo_evaluator_sdk.agent_eval.metrics import ToolArgumentMatchesInputMetric
from nemo_evaluator_sdk.execution.samples import build_metric_input
from nemo_evaluator_sdk.values.evidence import EVIDENCE_TRACE, CandidateEvidence, EvidenceDescriptor

_EMAIL = "Claim Your Free iPhone Now!\n\nDear valued customer,\nCongratulations!"
# The shape agent-eval hands metrics (``_metric_row``): task inputs nested under ``inputs``.
_ROW = {"task": {"id": "t"}, "inputs": {"instruction": _EMAIL}}


def _atif(*calls: dict[str, Any]) -> dict[str, Any]:
    steps = [{"source": "agent", "message": "call", "tool_calls": [call]} for call in calls]
    steps.append({"source": "agent", "message": "done"})
    return {"schema_version": "ATIF-v1.7", "steps": steps}


def _call(text: object, name: str = "mcp__email_phishing_analyzer__email_phishing_analyzer") -> dict[str, Any]:
    return {"function_name": name, "arguments": {"text": text}}


def _evidence(atif: dict[str, Any]) -> CandidateEvidence:
    return CandidateEvidence(descriptors={EVIDENCE_TRACE: EvidenceDescriptor(kind="trace", format="atif", data=atif)})


async def _score(
    metric: ToolArgumentMatchesInputMetric, atif: dict[str, Any] | None, row: dict[str, Any] = _ROW
) -> bool:
    sample = {"evidence": _evidence(atif)} if atif is not None else {}
    result = await metric.compute_scores(build_metric_input(row, sample, 0))
    (output,) = result.outputs
    assert output.name == "tool_argument_matches_input"
    return bool(output.value)


_METRIC = ToolArgumentMatchesInputMetric(tool_name="email_phishing_analyzer")


async def test_verbatim_argument_matches() -> None:
    assert await _score(_METRIC, _atif(_call(_EMAIL))) is True


async def test_whitespace_differences_are_tolerated_by_default_but_not_under_exact() -> None:
    reflowed = " ".join(_EMAIL.split())
    assert await _score(_METRIC, _atif(_call(reflowed))) is True
    strict = ToolArgumentMatchesInputMetric(tool_name="email_phishing_analyzer", normalize="exact")
    assert await _score(strict, _atif(_call(reflowed))) is False


async def test_an_edited_or_truncated_argument_fails() -> None:
    assert await _score(_METRIC, _atif(_call(_EMAIL.replace("iPhone", "laptop")))) is False
    assert await _score(_METRIC, _atif(_call(_EMAIL.split("\n\n")[1]))) is False  # body only, subject dropped


async def test_every_call_must_match_not_just_one() -> None:
    assert await _score(_METRIC, _atif(_call(_EMAIL), _call("something else"))) is False


async def test_other_tools_are_ignored_and_no_call_fails() -> None:
    assert await _score(_METRIC, _atif(_call(_EMAIL, name="read_file"))) is False
    assert await _score(_METRIC, _atif()) is False
    assert await _score(_METRIC, None) is False


async def test_missing_or_non_string_argument_fails() -> None:
    assert await _score(_METRIC, _atif({"function_name": "email_phishing_analyzer", "arguments": {}})) is False
    assert await _score(_METRIC, _atif(_call({"nested": _EMAIL}))) is False


async def test_a_task_without_the_input_key_fails_rather_than_matching_nothing() -> None:
    assert await _score(_METRIC, _atif(_call(_EMAIL)), row={"task": {"id": "t"}, "inputs": {}}) is False
    # A top-level key is not a task input; only the nested ``inputs`` mapping is.
    assert await _score(_METRIC, _atif(_call(_EMAIL)), row={"instruction": _EMAIL}) is False
    other = ToolArgumentMatchesInputMetric(tool_name="email_phishing_analyzer", input_key="email")
    assert await _score(other, _atif(_call(_EMAIL)), row={"inputs": {"email": _EMAIL}}) is True
