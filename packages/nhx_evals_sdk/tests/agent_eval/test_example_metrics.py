# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the example's reference metrics-over-evidence."""

import importlib.util
import json
from pathlib import Path

import pytest
from nhx_evals_sdk.execution.samples import build_metric_input
from nhx_evals_sdk.values.evidence import (
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_FORMAT_OTLP,
    CandidateEvidence,
    EvidenceDescriptor,
)

_MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "run_agent_eval" / "example_metrics.py"
_spec = importlib.util.spec_from_file_location("example_metrics", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
example_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example_metrics)


def _input_with_evidence(evidence: CandidateEvidence):
    return build_metric_input({"prompt": "q"}, {"evidence": evidence}, index=0)


def _otlp_evidence(spans: list[object], *extra_resource_spans: object) -> CandidateEvidence:
    return CandidateEvidence(
        descriptors={
            "trace": EvidenceDescriptor(
                kind="trace",
                format=EVIDENCE_FORMAT_OTLP,
                data={
                    "resourceSpans": [
                        {"scopeSpans": [{"spans": spans}]},
                        *extra_resource_spans,
                    ]
                },
            )
        }
    )


def _otlp_tool_span(
    name: str,
    arguments: dict[str, object] | None,
    start_ns: int,
    *,
    kind: str = "TOOL",
) -> dict[str, object]:
    attributes: list[dict[str, object]] = [
        {"key": "openinference.span.kind", "value": {"stringValue": kind}},
        {"key": "tool.name", "value": {"stringValue": name}},
    ]
    if arguments is not None:
        attributes.append({"key": "input.value", "value": {"stringValue": json.dumps(arguments)}})
    return {"name": name, "startTimeUnixNano": str(start_ns), "attributes": attributes}


@pytest.mark.asyncio
async def test_tests_pass_and_no_test_cheating(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    final = tmp_path / "final"
    for root in (initial, final):
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "test_x.py").write_text("def test(): assert True", encoding="utf-8")
    (final / "solution.py").write_text("print('done')", encoding="utf-8")

    evidence = CandidateEvidence(
        descriptors={
            "initial_state": EvidenceDescriptor(kind="filesystem", ref=str(initial)),
            "final_state": EvidenceDescriptor(kind="filesystem", ref=str(final)),
        }
    )

    tests_pass = await example_metrics.TestsPassMetric(["test", "-f", "solution.py"]).compute_scores(
        _input_with_evidence(evidence)
    )
    assert tests_pass.outputs[0].value is True

    no_cheat = await example_metrics.NoTestCheatingMetric().compute_scores(_input_with_evidence(evidence))
    assert no_cheat.outputs[0].value is True

    # Mutating a protected test file flips no_test_cheating to False.
    (final / "tests" / "test_x.py").write_text("def test(): assert False", encoding="utf-8")
    evidence_cheated = CandidateEvidence(
        descriptors={
            "initial_state": EvidenceDescriptor(kind="filesystem", ref=str(initial)),
            "final_state": EvidenceDescriptor(kind="filesystem", ref=str(final)),
        }
    )
    cheated = await example_metrics.NoTestCheatingMetric().compute_scores(_input_with_evidence(evidence_cheated))
    assert cheated.outputs[0].value is False


@pytest.mark.asyncio
async def test_inefficient_retry_loop(tmp_path: Path) -> None:
    def trajectory(repeats: int) -> dict:
        calls = [
            {"tool_call_id": f"c{i}", "function_name": "search", "arguments": {"q": "same"}} for i in range(repeats)
        ]
        return {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "demo", "version": "1.0"},
            "steps": [{"step_id": 1, "source": "agent", "message": "", "tool_calls": calls}],
        }

    looping = tmp_path / "loop.json"
    looping.write_text(json.dumps(trajectory(5)), encoding="utf-8")
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps(trajectory(1)), encoding="utf-8")

    metric = example_metrics.InefficientRetryLoopMetric(threshold=2)

    loop_result = await metric.compute_scores(
        _input_with_evidence(
            CandidateEvidence(
                descriptors={"trace": EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_ATIF, ref=str(looping))}
            )
        )
    )
    assert loop_result.outputs[0].value is False
    assert loop_result.outputs[1].value == 5

    clean_result = await metric.compute_scores(
        _input_with_evidence(
            CandidateEvidence(
                descriptors={"trace": EvidenceDescriptor(kind="trace", format=EVIDENCE_FORMAT_ATIF, ref=str(clean))}
            )
        )
    )
    assert clean_result.outputs[0].value is True


@pytest.mark.asyncio
async def test_inefficient_retry_loop_reads_time_ordered_otlp_tool_spans() -> None:
    evidence = _otlp_evidence(
        [
            _otlp_tool_span("search", {"q": "same"}, 30),
            _otlp_tool_span("search", {"q": "same"}, 10),
            _otlp_tool_span("search", {"q": "same"}, 20),
        ]
    )

    result = await example_metrics.InefficientRetryLoopMetric(threshold=2).compute_scores(
        _input_with_evidence(evidence)
    )

    assert result.outputs[0].value is False
    assert result.outputs[1].value == 3


@pytest.mark.asyncio
async def test_inefficient_retry_loop_ignores_non_tool_otlp_spans() -> None:
    evidence = _otlp_evidence(
        [
            _otlp_tool_span("search", {"q": "same"}, 10),
            _otlp_tool_span("search", {"q": "same"}, 20, kind="CHAIN"),
            _otlp_tool_span("search", {"q": "different"}, 30),
        ]
    )

    result = await example_metrics.InefficientRetryLoopMetric(threshold=1).compute_scores(
        _input_with_evidence(evidence)
    )

    assert result.outputs[0].value is True
    assert result.outputs[1].value == 1


@pytest.mark.asyncio
async def test_inefficient_retry_loop_refuses_a_trace_it_cannot_parse() -> None:
    # Reading the spans as OTLP rather than as loose JSON means a structurally invalid trace stops
    # the metric instead of being scored around. Retry identity comes from the spans it could not
    # read, so an answer here would be a guess -- the same stance as the no-arguments case below.
    evidence = _otlp_evidence([_otlp_tool_span("search", {"q": "same"}, 10)], {"scopeSpans": "not-a-list"})

    with pytest.raises(ValueError, match="invalid OTLP payload"):
        await example_metrics.InefficientRetryLoopMetric(threshold=1).compute_scores(_input_with_evidence(evidence))


@pytest.mark.asyncio
async def test_inefficient_retry_loop_rejects_otlp_tool_without_arguments() -> None:
    evidence = _otlp_evidence(
        [
            {
                "name": "search",
                "attributes": [
                    {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                    {"key": "gen_ai.tool.name", "value": {"stringValue": "search"}},
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="retry identity is unavailable"):
        await example_metrics.InefficientRetryLoopMetric().compute_scores(_input_with_evidence(evidence))
