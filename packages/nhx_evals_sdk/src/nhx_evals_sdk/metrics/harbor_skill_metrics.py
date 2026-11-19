# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score pinned skill packages and Codex evidence in a Harbor task."""

import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, Literal

from nhx_evals_sdk.agent_eval.metrics import AgentPhaseSuccessMetric, SkillUsedMetric
from nhx_evals_sdk.agent_eval.runtimes.harbor.codex_evidence import (
    AGENT_HOME,
    PUBLISH,
    READBACK,
    observe_codex,
    rollout_events,
    tool_trace,
)
from nhx_evals_sdk.enums import MetricType
from nhx_evals_sdk.metrics.protocol import (
    DatasetRow,
    Metric,
    MetricDiagnostic,
    MetricInput,
    MetricOutputSpec,
    MetricResult,
)
from nhx_evals_sdk.metrics.skill_readiness import (
    SkillDescriptionQualityMetric,
    SkillReliabilityReadinessMetric,
    SkillScriptQualityMetric,
    SkillStaticCorrectnessMetric,
    SkillUnicodeSafetyMetric,
)
from nhx_evals_sdk.metrics.skill_scans import (
    AgentSafetyMetric,
    SkillBundledTestMetric,
    SkillCodeRiskMetric,
    SkillDependencyComplianceMetric,
    SkillLicenseMetric,
    SkillPackageConfigurationRiskMetric,
    SkillSecurityScanMetric,
    SkillSensitiveDataMetric,
)
from nhx_evals_sdk.metrics.skill_traces import (
    ComponentCoordinationMetric,
    ComponentUsageMetric,
    ErrorRecoveryMetric,
    SkillArgumentRule,
    SkillOrderRule,
    SkillRoutingMetric,
    ToolArgumentComplianceMetric,
    ToolExecutionOutcomeMetric,
    ToolProductivityMetric,
    ToolSelectionMetric,
    WorkflowComplianceMetric,
)
from nhx_evals_sdk.metrics.skills import (
    SkillActivationMetric,
    SkillContextEfficiencyMetric,
    SkillReferenceIntegrityMetric,
    SkillStructureMetric,
)
from nhx_evals_sdk.metrics.string_check import StringCheckMetric
from nhx_evals_sdk.metrics.utils import metric_type_name
from nhx_evals_sdk.values.evidence import EVIDENCE_LOGS, EVIDENCE_TRACE
from nhx_evals_sdk.values.metrics import MetricBase
from nhx_evals_sdk.values.protocol import validate_metric_result
from nhx_evals_sdk.values.skill_evidence import SkillContentBundle, SkillScanReport, SkillTestReport
from nhx_evals_sdk.values.skills import SkillSnapshot
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HarborSkillPackage(BaseModel):
    """Digest-verified package already read by the evaluation publisher."""

    model_config = ConfigDict(extra="forbid")
    snapshot: SkillSnapshot
    contents: SkillContentBundle
    inventory: dict[str, str]
    scans: dict[str, SkillScanReport] = Field(default_factory=dict)
    isolated_tests: SkillTestReport | None = None

    @model_validator(mode="after")
    def same_revision(self) -> "HarborSkillPackage":
        if (
            self.contents.artifact_digest != self.snapshot.digest
            or set(self.inventory) != set(self.snapshot.files)
            or self.snapshot.document is None
            or self.contents.files.get("SKILL.md") != self.snapshot.document
            or any(scan.subject_id != self.snapshot.digest for scan in self.scans.values())
            or (self.isolated_tests and self.isolated_tests.artifact_digest != self.snapshot.digest)
        ):
            raise ValueError("package inputs must identify the same skill snapshot")
        return self


def _package_metrics() -> tuple[Metric, ...]:
    return (
        SkillStructureMetric(),
        SkillReferenceIntegrityMetric(),
        SkillContextEfficiencyMetric(),
        SkillStaticCorrectnessMetric(),
        SkillDescriptionQualityMetric(),
        SkillReliabilityReadinessMetric(),
        SkillScriptQualityMetric(),
        SkillUnicodeSafetyMetric(scope="package"),
        SkillSecurityScanMetric(),
        SkillSensitiveDataMetric(),
        SkillCodeRiskMetric(),
        SkillLicenseMetric(),
        SkillDependencyComplianceMetric(),
        SkillPackageConfigurationRiskMetric(),
        SkillBundledTestMetric(),
    )


def _pinned_submission(verifier: Path) -> dict:
    path = verifier / "submission.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() and path.resolve().parent == verifier else {}


class HarborSkillMetrics(MetricBase):
    """Inline skill metrics over Harbor's captured package, verifier and Codex evidence."""

    type: Literal["harbor_skill_metrics"] = "harbor_skill_metrics"
    packages: list[HarborSkillPackage] = Field(min_length=1)
    authoring_directory: str | None = None

    @model_validator(mode="after")
    def unique_names(self) -> "HarborSkillMetrics":
        names = [package.snapshot.directory_name for package in self.packages]
        if len(names) != len(set(names)):
            raise ValueError("skill packages must have distinct names")
        return self

    def _skill_metrics(self, name: str) -> tuple[Metric, ...]:
        return (
            SkillActivationMetric(skill_name=name, should_activate=True),
            SkillRoutingMetric(
                expected_skill=f"Skill:{name}",
                acceptable_skills=[
                    f"Skill:{p.snapshot.directory_name}" for p in self.packages if p.snapshot.directory_name != name
                ],
            ),
        )

    def _trace_metrics(self, staged: Iterable[str]) -> tuple[Metric, ...]:
        components = [f"Skill:{package.snapshot.directory_name}" for package in self.packages]
        metrics: tuple[Metric, ...] = (
            ErrorRecoveryMetric(),
            ToolProductivityMetric(),
            ToolExecutionOutcomeMetric(tool_pattern="*"),
            ComponentUsageMetric(
                declared_components=components,
                unavailable_components=[component for component in components if component[6:] not in staged],
            ),
        )
        if not self.authoring_directory:
            return metrics
        order = SkillOrderRule(before=PUBLISH, after=READBACK)
        return (
            *metrics,
            WorkflowComplianceMetric(rules=[order]),
            ComponentCoordinationMetric(order_rules=[order]),
            ToolSelectionMetric(expected_tools=[PUBLISH, READBACK], acceptable_tools=components),
            ToolArgumentComplianceMetric(
                rules=[
                    SkillArgumentRule(
                        tool=PUBLISH,
                        schema={
                            "required": ["harbor_dir"],
                            "properties": {"harbor_dir": {"const": self.authoring_directory}},
                        },
                    )
                ],
            ),
        )

    def output_spec(self) -> list[MetricOutputSpec]:
        """Declare stable aggregate keys for all package and trial components."""
        groups: list[tuple[str, Iterable[Metric]]] = [
            ("", (AgentPhaseSuccessMetric(), AgentSafetyMetric(), SkillUsedMetric())),
            ("", self._trace_metrics(())),
        ]
        if self.authoring_directory:
            groups.append(("", (StringCheckMetric(operation="contains", left_template="", right_template=""),)))
        for package in self.packages:
            name = package.snapshot.directory_name
            groups.append((f"package.{name}.", _package_metrics()))
            groups.append((f"skill.{name}.", self._skill_metrics(name)))
        return [
            spec.model_copy(update={"name": f"{prefix}{metric_type_name(metric)}.{spec.name}", "required": False})
            for prefix, metrics in groups
            for metric in metrics
            for spec in metric.output_spec()
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        result = MetricResult(outputs=[])
        report: dict[str, str] = {}

        async def score(metric_input: MetricInput, metrics: Iterable[Metric], prefix: str = "") -> None:
            for metric in metrics:
                name = f"{prefix}{metric_type_name(metric)}"
                measured = validate_metric_result(await metric.compute_scores(metric_input), metric.output_spec())
                outputs = {output.name: output.value for output in measured.outputs}
                report[name] = (
                    "unmeasured" if not outputs else "partial" if outputs.get("complete") == 0 else "computed"
                )
                result.outputs.extend(
                    output.model_copy(update={"name": f"{name}.{output.name}"}) for output in measured.outputs
                )
                result.diagnostics.extend(
                    diagnostic.model_copy(update={"details": {**(diagnostic.details or {}), "metric": name}})
                    for diagnostic in measured.diagnostics
                )

        for package in self.packages:
            reference = {
                "skill_snapshot": package.snapshot.model_dump(mode="json"),
                "skill_contents": package.contents.model_dump(mode="json"),
                "skill_scans": {key: scan.model_dump(mode="json") for key, scan in package.scans.items()},
                "skill_test_report": package.isolated_tests.model_dump(mode="json") if package.isolated_tests else None,
            }
            probe = input.model_copy(update={"row": DatasetRow(data={"reference": reference})})
            await score(probe, _package_metrics(), f"package.{package.snapshot.directory_name}.")

        data = input.row.data
        trial = data["trial"]
        metadata = dict(trial["metadata"])
        metadata["agent_ok"] = trial["status"] == "completed" and trial["error"] is None
        candidate = input.candidate.model_copy(update={"metadata": input.candidate.metadata | metadata})
        basic = input.model_copy(update={"candidate": candidate})
        await score(basic, (AgentPhaseSuccessMetric(), AgentSafetyMetric()))

        if candidate.evidence and (descriptor := candidate.evidence.get(EVIDENCE_LOGS)) and descriptor.ref:
            agent = Path(descriptor.ref).resolve()
            if (
                agent.name == "agent"
                and agent.parent == Path(metadata.get("harbor_trial_dir", "")).resolve()
                and agent.is_dir()
            ):
                await self._score_codex(basic, agent, metadata, score)

        for spec in self.output_spec():
            report.setdefault(spec.name.rsplit(".", 1)[0], "unmeasured")
        result.diagnostics.append(
            MetricDiagnostic(
                message="SDK metric coverage for this Harbor trial.",
                details={
                    "metrics": report,
                    "unconfigured": [
                        metric.value
                        for metric in MetricType
                        if metric
                        not in {
                            MetricType.HARBOR_REWARD,
                            MetricType.TOKEN_EFFICIENCY,
                            MetricType.EVIDENCE_PRESENCE,
                            MetricType.LLM_JUDGE,
                        }
                        and not any(name.endswith(f".{metric.value}") or name == metric.value for name in report)
                    ],
                },
            )
        )
        return result

    async def _score_codex(
        self,
        input: MetricInput,
        agent: Path,
        metadata: dict[str, Any],
        score: Callable[[MetricInput, Iterable[Metric], str], Awaitable[None]],
    ) -> None:
        if not (agent / "sessions").is_dir():
            return
        events = list(rollout_events(agent))
        verifier = (agent.parent / "verifier").resolve()
        if verifier.parent != agent.parent:
            raise ValueError("Harbor verifier logs are outside the trial")
        inventory = verifier / "skill-inventory.json"
        observed = (
            json.loads(inventory.read_text(encoding="utf-8"))
            if inventory.is_file() and inventory.resolve().parent == verifier
            else {}
        )
        packages = {p.snapshot.directory_name: p for p in self.packages}
        documents = {name: package.contents.files["SKILL.md"] for name, package in packages.items()}
        observations = {name: observe_codex(events, name, document) for name, document in documents.items()}
        staged = {name: documents[name] for name, observation in observations.items() if observation.skill_listed}
        expected = {name: packages[name].inventory for name in staged}
        if observed != {"staged": expected, "registered": expected}:
            raise ValueError("Harbor skill inventory and the Codex catalog disagree with the pinned archives")
        submission = _pinned_submission(verifier) if self.authoring_directory else {}
        taskset, task = submission.get("taskset_ref", ""), submission.get("task_ref", "")
        trial = input.row.data["trial"]
        trace = tool_trace(events, trial["id"], staged, taskset, task)
        normalized = metadata | {
            "skill_activation": {name: obs.activated for name, obs in observations.items()},
            "skills": [
                {"name": name, "location": str(AGENT_HOME / ".agents" / "skills" / name / "SKILL.md")}
                for name in staged
            ],
            "skill_tool_trace": trace.model_dump(mode="json"),
        }
        candidate = input.candidate.model_copy(update={"metadata": input.candidate.metadata | normalized})
        metric_input = MetricInput(
            row=DatasetRow(data=input.row.data | {"trial": trial | {"metadata": normalized}}), candidate=candidate
        )
        await score(metric_input, self._trace_metrics(staged), "")
        atif = candidate.evidence.get(EVIDENCE_TRACE) if candidate.evidence else None
        if not staged or (atif and atif.ref and Path(atif.ref).resolve().is_relative_to(agent)):
            await score(metric_input, (SkillUsedMetric(),), "")
        if taskset and candidate.output_text:
            reference = metric_input.row.data["reference"] | {"verified_taskset_ref": taskset}
            taskset_input = metric_input.model_copy(
                update={"row": DatasetRow(data=metric_input.row.data | {"reference": reference})}
            )
            await score(
                taskset_input,
                (
                    StringCheckMetric(
                        operation="contains",
                        left_template="{{ sample.output_text }}",
                        right_template="{{ item.reference.verified_taskset_ref }}",
                    ),
                ),
                "",
            )
        for name in documents:
            activation, routing = self._skill_metrics(name)
            await score(metric_input, (activation, routing) if name in staged else (activation,), f"skill.{name}.")
