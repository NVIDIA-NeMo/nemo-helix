# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline fixtures for the remaining skill metrics; all scoring is implemented in the SDK."""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, patch

import nhx_evals_sdk as sdk
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput
from nhx_evals_sdk.structured_output import StructuredOutputMode

_DOC = (
    "---\nname: incident-summary\ndescription: Use when summarizing incidents for shift handoffs.\n---\n"
    "# Incident summary\n## Prerequisites\nRead the incident with the docs MCP tool.\n"
    "## Inputs\nIncident text.\n## Output format\nSymptoms and mitigation.\n"
    "## Example\nReport the restart.\n## Error handling\nRetry a transient read failure once.\n"
    "## Limitations\nDo not infer the root cause.\n"
)


def _fixture(version: str) -> tuple[dict, dict]:
    improved = version == "v2"
    document = _DOC if improved else "---\nname: incident-summary\ndescription: general things\n---\nSummarize.\n"
    files = {"SKILL.md": document, "scripts/publish.py": "print('error: missing incident')\n"}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    snapshot = sdk.SkillSnapshot(
        directory_name="incident-summary", digest=digest, document=document, files=sorted(files)
    )
    finding = sdk.SkillFinding(rule="fixture-policy", category="security", severity="high", file="SKILL.md")
    reference = {
        "skill_snapshot": snapshot.model_dump(mode="json"),
        "skill_contents": sdk.SkillContentBundle(artifact_digest=digest, files=files).model_dump(mode="json"),
        "skill_scans": {
            key: sdk.SkillScanReport(
                subject_id=digest,
                scanner="fixture-only",
                scanner_version="1",
                status="completed",
                findings=[] if improved else [finding],
                files_scanned=2,
                files_expected=2,
            ).model_dump(mode="json")
            for key in ("security", "sensitive_data", "code_risk", "license", "dependencies", "package_configuration")
        },
        "skill_test_report": sdk.SkillTestReport(
            artifact_digest=digest,
            status="completed",
            passed=4 if improved else 2,
            failed=0 if improved else 2,
            errors=0,
            skipped=0,
            coverage=0.9 if improved else 0.5,
        ).model_dump(mode="json"),
        "ground_truth": "Uploads timed out until the service restarted.",
    }
    calls = [
        sdk.SkillToolCall(
            id="activate",
            step=0,
            tool="skill",
            component="Skill:incident-summary",
            succeeded=True,
            productivity="neutral",
        ),
        sdk.SkillToolCall(
            id="read",
            step=1,
            tool="mcp",
            component="MCP:docs/read",
            arguments={"query": "incident"} if improved else {},
            succeeded=False,
            fault_owner="skill",
            productivity="wasted",
        ),
    ]
    if improved:
        calls.append(
            sdk.SkillToolCall(
                id="retry",
                step=2,
                tool="mcp",
                component="MCP:docs/read",
                arguments={"query": "incident"},
                observation={"record": "incident-record-123"},
                succeeded=True,
                retry_of="read",
                productivity="productive",
                writes=["out/incident.json"],
            )
        )
    calls.append(
        sdk.SkillToolCall(
            id="publish",
            step=3,
            tool="python",
            component="Skill:publish",
            arguments={"record": "incident-record-123"} if improved else {},
            script="scripts/publish.py",
            succeeded=improved,
            reads=["out/incident.json"] if improved else [],
            productivity="productive" if improved else "wasted",
        )
    )
    metadata = {
        "skill_activation": {"unrelated-skill": False},
        "skill_tool_trace": sdk.SkillToolTrace(
            trial_id=version, complete=True, calls=calls, initial_context=[]
        ).model_dump(mode="json"),
        "skill_scans": {
            "safety": sdk.SkillScanReport(
                subject_id=version,
                scanner="fixture-safety",
                scanner_version="1",
                status="completed",
                files_scanned=1,
                files_expected=1,
            ).model_dump(mode="json")
        },
    }
    return reference, metadata


def _artifact_metrics() -> list[sdk.Metric]:
    return [
        sdk.SkillStaticCorrectnessMetric(),
        sdk.SkillDescriptionQualityMetric(),
        sdk.SkillReliabilityReadinessMetric(requires_inputs=True, requires_output_contract=True),
        sdk.SkillScriptQualityMetric(),
        sdk.SkillUnicodeSafetyMetric(scope="package"),
        sdk.SkillSecurityScanMetric(),
        sdk.SkillSensitiveDataMetric(),
        sdk.SkillCodeRiskMetric(),
        sdk.SkillLicenseMetric(),
        sdk.SkillDependencyComplianceMetric(),
        sdk.SkillPackageConfigurationRiskMetric(),
        sdk.SkillBundledTestMetric(),
    ]


def _trial_metrics() -> list[sdk.Metric]:
    return [
        sdk.SkillOveractivationMetric(skill_name="unrelated-skill"),
        sdk.SkillRoutingMetric(expected_skill="Skill:incident-summary", acceptable_skills=["Skill:publish"]),
        sdk.ExpectedScriptInvocationMetric(expected_script="scripts/publish.py"),
        sdk.WorkflowComplianceMetric(
            rules=[sdk.SkillOrderRule(before="Skill:incident-summary", after="MCP:docs/read")]
        ),
        sdk.ErrorRecoveryMetric(),
        sdk.ToolProductivityMetric(),
        sdk.ToolSelectionMetric(
            expected_tools=["MCP:docs/read"], acceptable_tools=["Skill:*"], decoy_tools=["MCP:delete"]
        ),
        sdk.ToolArgumentComplianceMetric(
            rules=[
                sdk.SkillArgumentRule(
                    tool="MCP:docs/read",
                    schema={"type": "object", "required": ["query"], "properties": {"query": {"const": "incident"}}},
                )
            ]
        ),
        sdk.ToolExecutionOutcomeMetric(),
        sdk.ComponentCoordinationMetric(
            order_rules=[sdk.SkillOrderRule(before="MCP:docs/read", after="Skill:publish")],
            handoffs=[
                sdk.SkillHandoffRule(
                    producer="MCP:docs/read", consumer="Skill:publish", source_path="record", target_path="record"
                )
            ],
            conflicts=[sdk.SkillConflictRule(first="Skill:incident-summary", second="Skill:delete")],
        ),
        sdk.ComponentUsageMetric(
            declared_components=["Skill:incident-summary", "Skill:publish", "MCP:docs/read", "MCP:archive"],
            unavailable_components=["MCP:archive"],
        ),
        sdk.AgentSafetyMetric(),
    ]


async def _preview(version: str) -> None:
    reference, metadata = _fixture(version)
    model = sdk.Model(name="fixture-only", url="http://judge.invalid/v1")
    doc_judge, doc_views = sdk.skill_documentation_rubric_metric(model)
    outcome_judge, outcome_views = sdk.skill_outcome_metric(model, ["Identifies the restart as the recovery action."])
    for judge, numeric in ((doc_judge, True), (outcome_judge, False)):
        reply = {
            score.name: ("8" if version == "v2" else "4") if numeric else ("met" if version == "v2" else "unmet")
            for score in judge.scores
        }
        judge.set_inference_fn(AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(reply)}}]}))
    tasks = [
        AgentEvalTask(
            id="artifact",
            intent="Review skill readiness",
            inputs={"instruction": "Review the skill document."},
            reference=reference,
            metrics=[*_artifact_metrics(), doc_judge],
            views=doc_views,
        ),
        AgentEvalTask(
            id="runtime",
            intent="Summarize a resolved incident",
            inputs={"instruction": "Summarize the upload incident."},
            reference=reference,
            metrics=[*_trial_metrics(), outcome_judge],
            views=outcome_views,
        ),
    ]
    trials = [
        AgentEvalTrial(
            id=f"{version}-artifact",
            task_id="artifact",
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(output_text=""),
        ),
        AgentEvalTrial(
            id=version,
            task_id="runtime",
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(output_text="Restarting restored uploads." if version == "v2" else "Uploads failed."),
            metadata=metadata,
        ),
    ]
    with patch(
        "nhx_evals_sdk.metrics.llm_judge.detect_structured_output_mode", return_value=StructuredOutputMode.UNSUPPORTED
    ):
        run = await AgentEvaluator().run(tasks=tasks, trials=trials, config=AgentEvalRunConfig(fail_fast=True))
    print(f"\n{version}: synthetic evidence and fixed judge replies")
    for score in run.summary.scores.scores:
        if score.mean is not None and (
            score.name.startswith("view.")
            or score.name.endswith(
                (
                    ".score",
                    ".clean",
                    ".pass_rate",
                    ".recovery_rate",
                    ".f1",
                    ".success_rate",
                    ".coverage",
                    ".as_expected",
                    ".invoked",
                    ".defect_count",
                )
            )
        ):
            print(f"  {score.name}: {score.mean:.4g}")


async def main() -> None:
    for version in ("v1", "v2"):
        await _preview(version)


if __name__ == "__main__":
    asyncio.run(main())
