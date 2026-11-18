# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
from unittest.mock import AsyncMock, patch

import nhx_evals_sdk as sdk
import pytest
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nhx_evals_sdk.metrics.protocol import (
    CandidateOutput,
    DatasetRow,
    Metric,
    MetricInput,
    MetricResult,
    validate_metric_result,
)
from nhx_evals_sdk.metrics.types import MetricsUnion
from nhx_evals_sdk.structured_output import StructuredOutputMode
from nhx_evals_sdk.values.metrics import MetricBase
from pydantic import TypeAdapter, ValidationError

_DOC = (
    "---\nname: incident-summary\ndescription: Use when summarizing incidents for shift handoffs.\n---\n"
    "# Incident summary\n## Prerequisites\nRead the incident.\n## Inputs\nIncident text.\n"
    "## Output format\nSymptoms and mitigation.\n## Example\nReport the restart.\n"
    "## Error handling\nAsk for missing incident text.\n## Limitations\nDo not infer a cause.\n"
)


def _artifact(files: dict[str, str] | None = None) -> dict:
    files = files if files is not None else {"SKILL.md": _DOC}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "reference": {
            "skill_snapshot": {
                "directory_name": "incident-summary",
                "digest": digest,
                "document": files.get("SKILL.md"),
                "files": sorted(files),
            },
            "skill_contents": {"artifact_digest": digest, "files": files},
        }
    }


def _trace(calls: list[dict] | None = None, *, complete: bool = True, initial_context: list | None = None) -> dict:
    return {
        "inputs": {"instruction": "Summarize the incident"},
        "trial": {
            "id": "v2",
            "metadata": {
                "skill_tool_trace": {
                    "trial_id": "v2",
                    "complete": complete,
                    "calls": calls or [],
                    "initial_context": initial_context or [],
                }
            },
        },
    }


def _call(id: str, step: int, component: str | None = "MCP:docs/read", **kwargs) -> dict:
    return {"id": id, "step": step, "tool": "invoke", "component": component, **kwargs}


async def _result(metric: Metric, row: dict) -> MetricResult:
    result = await metric.compute_scores(MetricInput(row=DatasetRow(data=row), candidate=CandidateOutput()))
    validate_metric_result(result, metric.output_spec())
    return result


async def _score(metric: Metric, row: dict) -> dict:
    return {output.name: output.value for output in (await _result(metric, row)).outputs}


def _metrics() -> list[MetricBase]:
    model = sdk.Model(name="fixture", url="http://judge.invalid/v1")
    return [
        sdk.SkillStaticCorrectnessMetric(),
        sdk.SkillDescriptionQualityMetric(),
        sdk.SkillReliabilityReadinessMetric(requires_inputs=True, requires_output_contract=True),
        sdk.SkillScriptQualityMetric(max_nesting=3),
        sdk.SkillUnicodeSafetyMetric(scope="package"),
        sdk.SkillSecurityScanMetric(),
        sdk.SkillSensitiveDataMetric(),
        sdk.SkillCodeRiskMetric(),
        sdk.SkillLicenseMetric(),
        sdk.SkillDependencyComplianceMetric(),
        sdk.SkillPackageConfigurationRiskMetric(),
        sdk.SkillBundledTestMetric(minimum_coverage=0.9),
        sdk.AgentSafetyMetric(),
        sdk.SkillOveractivationMetric(skill_name="incident-summary"),
        sdk.SkillRoutingMetric(expected_skill="Skill:incident-summary", acceptable_skills=["Skill:summary"]),
        sdk.ExpectedScriptInvocationMetric(expected_script="scripts/summarize.py"),
        sdk.WorkflowComplianceMetric(rules=[sdk.SkillOrderRule(before="Skill:*", after="MCP:*")]),
        sdk.ErrorRecoveryMetric(),
        sdk.ToolProductivityMetric(),
        sdk.ToolSelectionMetric(
            expected_tools=["MCP:docs/read"], acceptable_tools=["MCP:docs/search"], decoy_tools=["MCP:delete"]
        ),
        sdk.ToolArgumentComplianceMetric(
            rules=[sdk.SkillArgumentRule(tool="MCP:*", schema={"type": "object", "required": ["query"]})]
        ),
        sdk.ToolExecutionOutcomeMetric(tool_pattern="*"),
        sdk.ComponentCoordinationMetric(
            order_rules=[sdk.SkillOrderRule(before="Skill:*", after="MCP:*")],
            handoffs=[sdk.SkillHandoffRule(producer="MCP:docs/read", consumer="Skill:summary", source_path="text")],
            conflicts=[sdk.SkillConflictRule(first="Skill:summary", second="Skill:delete")],
        ),
        sdk.ComponentUsageMetric(
            declared_components=["Skill:summary", "MCP:docs/read"], unavailable_components=["Skill:summary"]
        ),
        sdk.skill_documentation_rubric_metric(model)[0],
        sdk.answer_accuracy_metric(model)[0],
        sdk.task_success_metric(model)[0],
        sdk.skill_outcome_metric(model, ["Reports mitigation."])[0],
    ]


@pytest.mark.parametrize("metric", _metrics(), ids=lambda metric: type(metric).__name__)
def test_remaining_metrics_round_trip_with_public_exports(metric: MetricBase) -> None:
    adapter = TypeAdapter(MetricsUnion)
    restored = adapter.validate_json(adapter.dump_json(metric))
    assert type(restored) is type(metric)
    assert restored.model_dump() == metric.model_dump()
    assert getattr(sdk, type(metric).__name__) is type(metric)


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", [metric for metric in _metrics() if not isinstance(metric, sdk.LLMJudgeMetric)])
async def test_missing_evidence_is_unmeasured(metric: Metric) -> None:
    assert await _score(metric, {}) == {}


@pytest.mark.asyncio
async def test_readiness_checks_separate_complete_guidance_from_defects() -> None:
    row = _artifact()
    for metric in (
        sdk.SkillStaticCorrectnessMetric(),
        sdk.SkillDescriptionQualityMetric(),
        sdk.SkillReliabilityReadinessMetric(requires_inputs=True, requires_output_contract=True),
    ):
        assert await _score(metric, row) == {"score": 1, "defect_count": 0}
    broken = _artifact({"SKILL.md": "---\nname: wrong\ndescription: something\n---\n[missing](missing.md)"})
    assert await _score(sdk.SkillStaticCorrectnessMetric(), broken) == {"score": 0, "defect_count": 3}
    assert await _score(sdk.SkillDescriptionQualityMetric(), broken) == {"score": 0, "defect_count": 3}
    assert await _score(sdk.SkillReliabilityReadinessMetric(), broken) == {"score": 0, "defect_count": 3}


@pytest.mark.asyncio
async def test_script_checks_are_advisory_and_require_all_original_sources() -> None:
    row = _artifact(
        {"SKILL.md": _DOC, "scripts/a.py": "print('error: bad')\nexit(-3)\n", "scripts/b.py": "def broken(:"}
    )
    result = await _result(sdk.SkillScriptQualityMetric(), row)
    rules = {(finding.details or {})["rule"] for finding in result.diagnostics}
    assert {"syntax_error", "missing_shebang", "flat_script", "nonstandard_exit_code", "diagnostic_to_stdout"} <= rules
    del row["reference"]["skill_contents"]["files"]["scripts/b.py"]
    assert await _score(sdk.SkillScriptQualityMetric(), row) == {}


@pytest.mark.asyncio
async def test_unicode_scope_detects_controls_without_logging_contents() -> None:
    row = _artifact({"SKILL.md": "\ufeff" + _DOC, "refs/text.txt": "secret-excerpt\u202e\U000e0061"})
    assert (await _score(sdk.SkillUnicodeSafetyMetric(), row))["clean"] == 1
    metric = sdk.SkillUnicodeSafetyMetric(scope="package")
    result = await _result(metric, row)
    assert {output.name: output.value for output in result.outputs} == {
        "finding_count": 2,
        "files_checked": 2,
        "clean": 0,
    }
    assert "secret-excerpt" not in result.model_dump_json()
    del row["reference"]["skill_contents"]["files"]["refs/text.txt"]
    assert await _score(metric, row) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["digest", "path", "document"])
async def test_source_bundle_cannot_substitute_another_revision(mutation: str) -> None:
    row = _artifact()
    contents = row["reference"]["skill_contents"]
    if mutation == "digest":
        contents["artifact_digest"] = "f" * 64
    elif mutation == "path":
        contents["files"]["../../outside.py"] = ""
    else:
        contents["files"]["SKILL.md"] = "substituted"
    with pytest.raises(ValueError):
        await _score(sdk.SkillUnicodeSafetyMetric(scope="package"), row)


def _scan(row: dict, key: str = "security", *, status: str = "completed", trial: bool = False) -> dict:
    report = {
        "subject_id": "v2" if trial else row["reference"]["skill_snapshot"]["digest"],
        "scanner": "fixture-scanner",
        "scanner_version": "1.0",
        "status": status,
        "files_scanned": 1,
        "files_expected": 1,
        "findings": [
            {
                "rule": "unsafe",
                "category": "canary_exfiltration" if trial else "security",
                "severity": "high",
                "file": "SKILL.md",
                "line": 1,
            }
        ]
        * 2,
    }
    if trial:
        row["trial"]["metadata"]["skill_scans"] = {key: report}
    else:
        row["reference"]["skill_scans"] = {key: report}
    return report


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "partial", "failed", "unavailable"])
async def test_scan_completeness_and_deduplication(status: str) -> None:
    row = _artifact()
    report = _scan(row, status=status)
    scores = await _score(sdk.SkillSecurityScanMetric(), row)
    assert scores["finding_count"] == scores["high"] == 1
    assert scores["complete"] == float(status == "completed")
    assert (scores.get("clean") == 0) if status == "completed" else "clean" not in scores
    report["findings"] = []
    scores = await _score(sdk.SkillSecurityScanMetric(), row)
    assert (scores.get("clean") == 1) if status == "completed" else "clean" not in scores


@pytest.mark.asyncio
async def test_report_identity_coverage_and_canary_leak_accounting() -> None:
    row = _artifact()
    report = _scan(row)
    report["subject_id"] = "f" * 64
    with pytest.raises(ValueError, match="another artifact"):
        await _score(sdk.SkillSecurityScanMetric(), row)
    report["subject_id"] = row["reference"]["skill_snapshot"]["digest"]
    report["files_expected"] = 2
    with pytest.raises(ValidationError, match="every expected"):
        await _score(sdk.SkillSecurityScanMetric(), row)
    row = _trace()
    report = _scan(row, "safety", trial=True)
    assert (await _score(sdk.AgentSafetyMetric(), row))["canary_leaks"] == 1
    report["subject_id"] = "different-trial"
    with pytest.raises(ValueError, match="another trial"):
        await _score(sdk.AgentSafetyMetric(), row)


@pytest.mark.asyncio
async def test_bundled_test_skips_and_incomplete_coverage_are_not_passes() -> None:
    row = _artifact()
    report = {
        "artifact_digest": row["reference"]["skill_snapshot"]["digest"],
        "status": "completed",
        "passed": 3,
        "failed": 1,
        "errors": 0,
        "skipped": 10,
        "coverage": 0.75,
    }
    row["reference"]["skill_test_report"] = report
    metric = sdk.SkillBundledTestMetric()
    scores = await _score(metric, row)
    assert scores["pass_rate"] == 0.75 and scores["meets_coverage"] == 0
    report["status"] = "partial"
    assert "pass_rate" not in await _score(metric, row)
    assert "coverage" not in await _score(metric, row)
    report.update(status="completed", passed=0, failed=0)
    assert "pass_rate" not in await _score(metric, row)
    report["coverage"] = float("nan")
    with pytest.raises(ValidationError):
        await _score(metric, row)


@pytest.mark.parametrize(
    "calls",
    [
        [_call("a", 1), _call("a", 2)],
        [_call("a", 2), _call("b", 1)],
        [_call("a", 1, succeeded=True), _call("b", 2, retry_of="a")],
        [_call("a", 1, succeeded=False), _call("b", 1, retry_of="a")],
        [_call("a", 1, retry_of="missing")],
    ],
)
def test_trace_rejects_ambiguous_order_and_invalid_retry_links(calls: list[dict]) -> None:
    with pytest.raises(ValidationError):
        sdk.SkillToolTrace.model_validate(_trace(calls)["trial"]["metadata"]["skill_tool_trace"])


@pytest.mark.asyncio
async def test_trace_from_another_trial_is_rejected() -> None:
    row = _trace()
    row["trial"]["metadata"]["skill_tool_trace"]["trial_id"] = "v1"
    with pytest.raises(ValueError, match="another trial"):
        await _score(sdk.ErrorRecoveryMetric(), row)


@pytest.mark.asyncio
async def test_routing_and_overactivation_distinguish_unknown_and_wrong_skills() -> None:
    negative = sdk.SkillOveractivationMetric(skill_name="summary")
    row = {"trial": {"metadata": {"skill_activation": {"summary": True}}}}
    assert await _score(negative, row) == {"overactivated": 1, "as_expected": 0}
    metric = sdk.SkillRoutingMetric(expected_skill="Skill:summary", acceptable_skills=["Skill:alternate"])
    assert (await _score(metric, _trace([_call("a", 0, "Skill:alternate")])))["as_expected"] == 1
    assert (await _score(metric, _trace([_call("a", 0, "Skill:wrong")])))["as_expected"] == 0
    assert (await _score(metric, _trace()))["as_expected"] == 0
    assert "as_expected" not in await _score(metric, _trace(complete=False))


@pytest.mark.asyncio
async def test_script_invocation_failure_is_distinct_from_absence() -> None:
    metric = sdk.ExpectedScriptInvocationMetric(expected_script="scripts/run.py")
    scores = await _score(metric, _trace([_call("a", 0, script="scripts/run.py", succeeded=False)], complete=False))
    assert scores["invoked"] == scores["failed"] == 1 and scores["successful"] == 0
    assert (await _score(metric, _trace()))["invoked"] == 0
    assert "invoked" not in await _score(metric, _trace(complete=False))


@pytest.mark.asyncio
@pytest.mark.parametrize(("steps", "score"), [([0, 1], 1), ([1, 1], 0), ([1, 0], 0)])
async def test_workflow_requires_strict_observed_precedence(steps: list[int], score: int) -> None:
    metric = sdk.WorkflowComplianceMetric(rules=[sdk.SkillOrderRule(before="Skill:read", after="Skill:write")])
    calls = sorted(
        [_call("a", steps[0], "Skill:read"), _call("b", steps[1], "Skill:write")], key=lambda call: call["step"]
    )
    assert (await _score(metric, _trace(calls)))["score"] == score
    assert (await _score(metric, _trace()))["score"] == 0
    assert "score" not in await _score(metric, _trace(calls, complete=False))


@pytest.mark.asyncio
async def test_recovery_follows_retry_chains_and_never_invents_failure_attribution() -> None:
    metric = sdk.ErrorRecoveryMetric()
    calls = [
        _call("a", 0, succeeded=False, fault_owner="skill"),
        _call("b", 1, succeeded=False, retry_of="a"),
        _call("c", 2, succeeded=True, retry_of="b"),
    ]
    scores = await _score(metric, _trace(calls))
    assert scores["recovered"] == scores["failures"] == 2 and scores["recovery_rate"] == 1
    assert scores["skill_faults"] == scores["unknown_faults"] == 1
    calls[-1]["succeeded"] = None
    scores = await _score(metric, _trace(calls))
    assert scores["unknown_outcome"] == 1 and "recovery_rate" not in scores and "unresolved" not in scores
    assert "recovery_rate" not in await _score(metric, _trace())


@pytest.mark.asyncio
async def test_productivity_uses_explicit_classifications_and_excludes_neutral_calls() -> None:
    metric = sdk.ToolProductivityMetric()
    calls = [
        _call("a", 0, productivity="productive"),
        _call("b", 1, productivity="wasted"),
        _call("c", 2, productivity="neutral"),
    ]
    assert (await _score(metric, _trace(calls)))["score"] == 0.5
    del calls[-1]["productivity"]
    assert "score" not in await _score(metric, _trace(calls))
    assert "score" not in await _score(metric, _trace())


@pytest.mark.asyncio
async def test_tool_selection_penalizes_decoys_repetition_does_not_inflate_recall() -> None:
    metric = sdk.ToolSelectionMetric(
        expected_tools=["MCP:read", "MCP:search"], acceptable_tools=["MCP:list"], decoy_tools=["MCP:delete"]
    )
    calls = [
        _call("a", 0, "MCP:read"),
        _call("b", 1, "MCP:read"),
        _call("c", 2, "MCP:delete"),
        _call("d", 3, "MCP:list"),
    ]
    scores = await _score(metric, _trace(calls))
    assert scores["precision"] == pytest.approx(2 / 3) and scores["recall"] == 0.5
    assert scores["f1"] == pytest.approx(4 / 7) and scores["decoys"] == 1
    assert (await _score(metric, _trace()))["f1"] == 0
    assert "f1" not in await _score(metric, _trace(calls, complete=False))


@pytest.mark.asyncio
async def test_arguments_check_required_fields_values_and_uncalled_rules() -> None:
    metric = sdk.ToolArgumentComplianceMetric(
        rules=[
            sdk.SkillArgumentRule(
                tool="MCP:read",
                schema={"type": "object", "required": ["query"], "properties": {"query": {"const": "incident"}}},
            )
        ]
    )
    calls = [_call("a", 0, "MCP:read", arguments={"query": "incident"}), _call("b", 1, "MCP:read", arguments={})]
    scores = await _score(metric, _trace(calls))
    assert scores["score"] == 0.5 and scores["checked"] == 2
    assert "score" not in await _score(metric, _trace())
    assert (await _score(metric, _trace()))["unmatched_rules"] == 1
    for schema in (
        {"$ref": "https://example.org/schema"},
        {"$recursiveRef": "https://example.org/schema"},
        {"$id": "https://example.org/schema"},
        {"type": "invalid"},
    ):
        with pytest.raises(ValidationError):
            sdk.SkillArgumentRule(tool="MCP:*", schema=schema)


@pytest.mark.asyncio
async def test_execution_outcomes_keep_unknown_outside_known_denominator() -> None:
    calls = [_call("a", 0, succeeded=True), _call("b", 1, succeeded=False), _call("c", 2)]
    scores = await _score(sdk.ToolExecutionOutcomeMetric(), _trace(calls))
    assert scores == {"complete": 1, "successful": 1, "failed": 1, "unknown": 1, "success_rate": 0.5}
    assert "success_rate" not in await _score(sdk.ToolExecutionOutcomeMetric(), _trace())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "observed"),
    [("normal", 1), ("same_step", 0), ("backwards", 0), ("initial", 0), ("prompt", 0), ("failed", 0)],
)
async def test_value_handoffs_need_new_successful_producer_evidence(mode: str, observed: int) -> None:
    value = "incident-record-123"
    producer = _call("a", 1, "MCP:read", observation={"record": value}, succeeded=mode != "failed")
    consumer = _call(
        "b", 1 if mode == "same_step" else 0 if mode == "backwards" else 2, "Skill:summary", arguments={"id": value}
    )
    row = _trace(
        sorted([producer, consumer], key=lambda call: call["step"]),
        initial_context=[value] if mode == "initial" else [],
    )
    if mode == "prompt":
        row["inputs"]["instruction"] += " " + value
    metric = sdk.ComponentCoordinationMetric(
        handoffs=[
            sdk.SkillHandoffRule(producer="MCP:read", consumer="Skill:summary", source_path="record", target_path="id")
        ]
    )
    scores = await _score(metric, row)
    assert scores["handoffs_observed"] == scores["score"] == observed


@pytest.mark.asyncio
async def test_artifact_handoffs_and_conflicts_are_separate_probes() -> None:
    metric = sdk.ComponentCoordinationMetric(
        handoffs=[sdk.SkillHandoffRule(producer="Skill:write", consumer="Skill:read", kind="artifact")],
        conflicts=[sdk.SkillConflictRule(first="Skill:read", second="Skill:delete")],
    )
    calls = [
        _call("a", 0, "Skill:write", writes=["out/report.json"], succeeded=True),
        _call("b", 1, "Skill:read", reads=["out/report.json"]),
        _call("c", 2, "Skill:delete"),
    ]
    scores = await _score(metric, _trace(calls))
    assert scores["handoffs_observed"] == scores["conflicts_observed"] == 1 and scores["score"] == 0.5
    assert "score" not in await _score(metric, _trace(calls, complete=False))


@pytest.mark.asyncio
async def test_component_coverage_separates_unavailable_unknown_and_never_activated() -> None:
    metric = sdk.ComponentUsageMetric(
        declared_components=["Skill:a", "MCP:b", "MCP:c"], unavailable_components=["MCP:c"]
    )
    scores = await _score(metric, _trace([_call("a", 0, "Skill:a")]))
    assert scores["coverage"] == 0.5 and scores["unavailable"] == scores["never_activated"] == 1
    assert "coverage" not in await _score(metric, _trace([_call("a", 0, component=None)]))
    assert "coverage" not in await _score(metric, _trace(complete=False))
    with pytest.raises(ValueError, match="unavailable"):
        await _score(metric, _trace([_call("a", 0, "MCP:c")]))


@pytest.mark.asyncio
async def test_value_handoff_without_initial_context_coverage_is_unmeasured() -> None:
    row = _trace(
        [
            _call("a", 0, "MCP:read", observation="record-123456", succeeded=True),
            _call("b", 1, "Skill:summary", arguments={"id": "record-123456"}),
        ]
    )
    del row["trial"]["metadata"]["skill_tool_trace"]["initial_context"]
    metric = sdk.ComponentCoordinationMetric(
        handoffs=[sdk.SkillHandoffRule(producer="MCP:read", consumer="Skill:summary")]
    )
    scores = await _score(metric, row)
    assert scores["handoffs_unmeasured"] == 1 and "score" not in scores
    evidence = row["trial"]["metadata"]["skill_tool_trace"]
    evidence["initial_context"] = []
    evidence["calls"][0]["succeeded"] = None
    scores = await _score(metric, row)
    assert scores["handoffs_unmeasured"] == 1 and "score" not in scores
    evidence["calls"][0]["succeeded"] = True
    evidence["calls"][0]["observation"] = None
    scores = await _score(metric, row)
    assert scores["handoffs_unmeasured"] == 1 and "score" not in scores


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [
        sdk.skill_documentation_rubric_metric,
        sdk.answer_accuracy_metric,
        sdk.task_success_metric,
        sdk.skill_outcome_metric,
    ],
)
@pytest.mark.parametrize("complete_reply", [True, False])
async def test_judge_factories_render_native_prompts_and_views_require_all_verdicts(
    factory, complete_reply: bool
) -> None:
    judge, views = factory(sdk.Model(name="fixture", url="http://judge.invalid/v1"))
    numeric = factory is sdk.skill_documentation_rubric_metric
    reply = {score.name: ("8" if numeric else "met") for score in judge.scores}
    if not complete_reply:
        del reply[judge.scores[-1].name]
    inference = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(reply)}}]})
    judge.set_inference_fn(inference)
    task = AgentEvalTask(
        id="handoff",
        intent="Summarize it",
        inputs={"instruction": "Report the incident"},
        reference={**_artifact()["reference"], "ground_truth": "The restart fixed it."},
        metrics=[judge],
        views=views,
    )
    trial = AgentEvalTrial(
        id="v2",
        task_id=task.id,
        status=AgentEvalTrialStatus.COMPLETED,
        output=AgentOutput(output_text="Restarting restored service."),
    )
    with patch(
        "nhx_evals_sdk.metrics.llm_judge.detect_structured_output_mode", return_value=StructuredOutputMode.UNSUPPORTED
    ):
        run = await AgentEvaluator().run(tasks=[task], trials=[trial])
    summaries = {score.name: score.mean for score in run.summary.scores.scores}
    for name, view in views.items():
        all_present = all(signal.output in reply for signal in view.signals)
        assert summaries[f"view.{name}"] == ((8.0 if numeric else 1.0) if all_present else None)
    assert inference.await_args is not None
    prompt = inference.await_args.args[1]["messages"][-1]["content"]
    assert ("Use when summarizing" if numeric else "Restarting restored service.") in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_level", ["11", "-1", "invalid", 8])
async def test_documentation_rubric_cannot_accept_out_of_scale_or_nonlabel_verdicts(invalid_level) -> None:
    judge, _ = sdk.skill_documentation_rubric_metric(sdk.Model(name="fixture", url="http://judge.invalid/v1"))
    reply: dict[str, object] = {score.name: "8" for score in judge.scores}
    reply[judge.scores[0].name] = invalid_level
    judge.set_inference_fn(AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(reply)}}]}))
    with patch(
        "nhx_evals_sdk.metrics.llm_judge.detect_structured_output_mode", return_value=StructuredOutputMode.UNSUPPORTED
    ):
        scores = await _score(judge, _artifact())
    assert math.isnan(scores[judge.scores[0].name])
    assert scores[judge.scores[1].name] == 8
