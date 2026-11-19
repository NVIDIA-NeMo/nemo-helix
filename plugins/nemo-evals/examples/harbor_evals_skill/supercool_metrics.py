# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score the additional SDK skill metrics from saved Supercool evidence."""

import json
import re
import shlex
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from itertools import dropwhile, groupby
from pathlib import Path
from typing import Any, Literal

from codex_evidence import AGENT_HOME, _command_succeeded, _json_object, _proves_tool_read
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.models.client import ModelsClient
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.results import AgentEvalResult
from nhx_evals_sdk.agent_eval.runtimes.harbor.codex_evidence import command_events as _commands
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask, SemanticView
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentOutput
from nhx_evals_sdk.inference import InferenceFn, make_inference_request, new_inference_client
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, Metric, MetricInput
from nhx_evals_sdk.metrics.skill_judges import skill_documentation_rubric_metric, skill_outcome_metric
from nhx_evals_sdk.metrics.skill_readiness import SkillScriptQualityMetric, SkillUnicodeSafetyMetric
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
    SkillHandoffRule,
    SkillOrderRule,
    SkillRoutingMetric,
    ToolArgumentComplianceMetric,
    ToolExecutionOutcomeMetric,
    ToolProductivityMetric,
    ToolSelectionMetric,
    WorkflowComplianceMetric,
)
from nhx_evals_sdk.metrics.utils import metric_type_name
from nhx_evals_sdk.values.models import Model, ModelRef
from nhx_evals_sdk.values.skill_evidence import (
    SkillContentBundle,
    SkillScanReport,
    SkillTestReport,
    SkillToolCall,
    SkillToolTrace,
)
from nhx_evals_sdk.values.skills import SkillSnapshot
from pydantic import BaseModel, ConfigDict, Field

SKILL_NAME = "supercool-agent-eval"
SKILL_COMPONENT = f"Skill:{SKILL_NAME}"
SUBMISSION = "Eval:submit-agent-evaluate"
SUFFIX = "-supercool-7cf95d1e203ab864"
EXPECTED_ALIAS = f"agent-smoke{SUFFIX}"
_SHELL_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
_JSON_RECORD = re.compile(r"(?m)^[ \t]*(\{)")
_SKILL_PATH = str(AGENT_HOME / ".agents" / "skills" / SKILL_NAME / "SKILL.md")


class MetricEvidence(BaseModel):
    """Optional reports from scanners and an isolated test runner."""

    model_config = ConfigDict(extra="forbid")
    skill_scans: dict[
        Literal["security", "sensitive_data", "code_risk", "license", "dependencies", "package_configuration"],
        SkillScanReport,
    ] = Field(default_factory=dict)
    skill_test_report: SkillTestReport | None = None
    trial_safety: dict[str, SkillScanReport] = Field(default_factory=dict)


@dataclass(frozen=True)
class InnerJob:
    id: str
    name: str
    alias: str | None
    mean: float


@dataclass(frozen=True)
class TrialEvidence:
    trial: AgentEvalTrial
    trace: SkillToolTrace
    discovered: bool
    job: InnerJob | None


def read_inner_job(path: Path, trial: AgentEvalTrial) -> InnerJob | None:
    """Use the separate verifier's snapshot when the original Harbor reward verified the inner job."""
    return verified_inner_job(path, trial.metadata["reward_details"])


def verified_inner_job(path: Path, details: Mapping[str, Any]) -> InnerJob | None:
    if details.get("inner_eval_passed") != 1:
        return None
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        job = snapshot["job"]
        job_id, name, alias = job["id"], job["name"], job["run_alias"]
        means = [
            score["mean"] for score in job["summary"]["scores"]["scores"] if score["name"] == "exact-match.exact-match"
        ]
        if (
            snapshot["outcome"] != "collected"
            or job["status"] != "completed"
            or not isinstance(job_id, str)
            or not job_id
            or not isinstance(name, str)
            or not name
            or (alias is not None and not isinstance(alias, str))
            or means != [1.0]
            or (alias == EXPECTED_ALIAS) != bool(details["suffix_applied"])
        ):
            raise ValueError("the inner job snapshot disagrees with the verifier reward")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid evidence for the verified inner agent-evaluation job") from exc
    return InnerJob(job_id, name, alias, float(means[0]))


def _submits_agent_eval(command: str) -> bool:
    # Preserve quotes so a separator printed by echo cannot introduce a command.
    lexer = shlex.shlex(command.replace("\\\n", ""), posix=False, punctuation_chars="&;|<>\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    if "<<" in tokens:
        return False
    for is_separator, segment in groupby(tokens, lambda token: token.strip("\n") in {"", "&&", "||", ";", "|"}):
        if is_separator:
            continue
        args = list(dropwhile(_SHELL_ASSIGNMENT.match, segment))
        if (
            args
            and args[0] in {"nemo", "/app/.venv/bin/nemo"}
            and args[1:3] == ["evals", "agent-evaluate"]
            and any(arg.split("=", 1)[0] in {"--spec", "--spec-file"} for arg in args[3:])
        ):
            return True
    return False


def _receives_job(output: str, job: InnerJob) -> bool:
    _, delimiter, body = output.partition("\nOutput:\n")
    if not delimiter:
        _, _, body = output.partition("\nFinal output:\n")
    # A compound command can print the spec and CLI status before the job receipt.
    decoder = json.JSONDecoder()
    end_of_record = 0
    for match in _JSON_RECORD.finditer(body):
        start = match.start(1)
        if start < end_of_record:
            continue
        try:
            receipt, end_of_record = decoder.raw_decode(body, start)
        except json.JSONDecodeError:
            continue
        if receipt.get("id") == job.id and receipt.get("name") == job.name:
            return True
    return False


def trial_trace(
    events: Iterable[dict[str, Any]], trial_id: str, document: str, *, activated: bool | None, job: InnerJob | None
) -> SkillToolTrace:
    calls = []
    for command in _commands(events):
        name = command.call.get("name")
        if not isinstance(name, str) or not name:
            continue
        shell = name == "exec_command" and command.call.get("namespace") in (None, "functions")
        succeeded = _command_succeeded(command.output) if shell else None
        args = _json_object(command.call.get("arguments"))
        output = command.output or ""
        value = SkillToolCall(id=f"call-{command.started}", step=command.started, tool=name, succeeded=succeeded)
        if (
            activated
            and command.finished is not None
            and _proves_tool_read(command.call, output, _SKILL_PATH, document)
        ):
            # The skill is usable after its read result. Submission order uses command dispatch.
            value.step = command.finished
            value.component, value.succeeded, value.productivity = SKILL_COMPONENT, True, "productive"
            if SUFFIX in output:
                value.observation = {"suffix": SUFFIX}
        elif (
            job is not None
            and shell
            and succeeded is True
            and isinstance(args.get("cmd"), str)
            and _submits_agent_eval(args["cmd"])
            and _receives_job(output, job)
        ):
            value.component, value.productivity = SUBMISSION, "productive"
            value.arguments = {"run_alias": job.alias}
        calls.append(value)
    # Root rollouts omit some tool paths and do not establish retry or initial-context coverage.
    return SkillToolTrace(trial_id=trial_id, complete=False, calls=sorted(calls, key=lambda call: call.step))


async def score_package(
    snapshot: SkillSnapshot, contents: SkillContentBundle, evidence: MetricEvidence
) -> dict[str, Any]:
    inputs = MetricInput(
        row=DatasetRow(
            data={
                "reference": {
                    "skill_snapshot": snapshot.model_dump(mode="json"),
                    "skill_contents": contents.model_dump(mode="json"),
                    **evidence.model_dump(mode="json", include={"skill_scans", "skill_test_report"}, exclude_none=True),
                }
            }
        ),
        candidate=CandidateOutput(),
    )
    return {
        metric_type_name(metric): (await metric.compute_scores(inputs)).model_dump(mode="json")
        for metric in additional_package_metrics()
    }


def additional_package_metrics() -> tuple[Metric, ...]:
    return (
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


def trace_metrics(*, discovered: bool) -> list[Metric]:
    order = SkillOrderRule(before=SKILL_COMPONENT, after=SUBMISSION)
    return [
        SkillRoutingMetric(expected_skill=SKILL_COMPONENT),
        WorkflowComplianceMetric(rules=[order]),
        ErrorRecoveryMetric(),
        ToolProductivityMetric(),
        ToolSelectionMetric(expected_tools=[SUBMISSION], acceptable_tools=["exec_command", SKILL_COMPONENT]),
        ToolArgumentComplianceMetric(
            rules=[
                SkillArgumentRule.model_validate(
                    {
                        "tool": SUBMISSION,
                        "schema": {
                            "type": "object",
                            "required": ["run_alias"],
                            "properties": {"run_alias": {"const": EXPECTED_ALIAS}},
                        },
                    }
                )
            ]
        ),
        ToolExecutionOutcomeMetric(tool_pattern="*"),
        ComponentCoordinationMetric(
            order_rules=[order],
            handoffs=[
                SkillHandoffRule(
                    producer=SKILL_COMPONENT, consumer=SUBMISSION, source_path="suffix", target_path="run_alias"
                )
            ],
        ),
        ComponentUsageMetric(
            declared_components=[SKILL_COMPONENT, SUBMISSION],
            unavailable_components=[] if discovered else [SKILL_COMPONENT],
        ),
        AgentSafetyMetric(),
    ]


def outcome_judge(model: Model | ModelRef) -> tuple[LLMJudgeMetric, dict[str, SemanticView]]:
    return skill_outcome_metric(
        model,
        (
            "States that the agent evaluation completed and identifies its actual job name.",
            "Reports the actual run alias and exact-match mean from the verified job.",
            "Reports the required Supercool run alias.",
        ),
    )


def job_ground_truth(job: InnerJob) -> dict[str, Any]:
    return {
        "job_name": job.name,
        "run_alias": job.alias,
        "status": "completed",
        "metric": "exact-match.exact-match",
        "mean": job.mean,
        "required_run_alias": EXPECTED_ALIAS,
    }


async def score_trial(
    trial: AgentEvalTrial,
    trace: SkillToolTrace,
    *,
    discovered: bool,
    job: InnerJob | None,
    evidence: MetricEvidence,
    instruction: str = "Run and report the stored agent-smoke Taskset evaluation.",
    judge: Model | None = None,
    invoke: InferenceFn | None = None,
) -> dict[str, Any]:
    metrics = trace_metrics(discovered=discovered)
    unmeasured = {
        "skill_overactivation": "The pair does not contain an unrelated task with the skill installed.",
        "expected_script_invocation": "This skill contains no executable script.",
    }
    reference = {}
    views = {}
    if judge is None:
        unmeasured["skill_outcome"] = "Set JUDGE_MODEL_REF to score the final answer."
    elif job is None or not trial.output or not trial.output.output_text:
        unmeasured["skill_outcome"] = "A final answer and a verified inner job are required."
    else:
        judge_metric, views = outcome_judge(judge)
        if invoke is not None:
            judge_metric.set_inference_fn(invoke)
        metrics.append(judge_metric)
        reference = {"ground_truth": job_ground_truth(job)}
    metadata = {**trial.metadata, "skill_tool_trace": trace.model_dump(mode="json", exclude_none=True)}
    if safety := evidence.trial_safety.get(trial.id):
        metadata["skill_scans"] = {"safety": safety.model_dump(mode="json")}
    task = AgentEvalTask(
        id=trial.task_id,
        intent="Evaluate the Supercool agent-evaluation workflow.",
        inputs={"instruction": instruction},
        metrics=metrics,
        views=views,
        reference=reference,
    )
    result = await AgentEvaluator().run(
        tasks=[task],
        trials=[trial.model_copy(update={"metadata": metadata})],
        config=AgentEvalRunConfig(fail_fast=True),
    )
    return {**_result(result), "unmeasured": unmeasured}


async def score_documentation(
    snapshot: SkillSnapshot, judge: Model | None, invoke: InferenceFn | None
) -> dict[str, Any]:
    if judge is None:
        return {"unmeasured": "Set JUDGE_MODEL_REF to score the original skill document."}
    metric, views = skill_documentation_rubric_metric(judge)
    if invoke is not None:
        metric.set_inference_fn(invoke)
    task = AgentEvalTask(
        id="skill-documentation",
        intent="Evaluate the skill's documentation.",
        inputs={"instruction": "Review SKILL.md."},
        reference={"skill_snapshot": snapshot.model_dump(mode="json")},
        metrics=[metric],
        views=views,
    )
    trial = AgentEvalTrial(id=snapshot.digest, task_id=task.id, status="completed", output=AgentOutput())
    return _result(
        await AgentEvaluator().run(
            tasks=[task],
            trials=[trial],
            config=AgentEvalRunConfig(fail_fast=True),
        )
    )


def _result(result: AgentEvalResult) -> dict[str, Any]:
    return {
        "summary": result.summary.model_dump(mode="json"),
        "scores": [score.model_dump(mode="json") for score in result.scores],
    }


async def score_diagnostics(
    client: NemoClient,
    snapshot: SkillSnapshot,
    trials: Mapping[str, TrialEvidence],
    instruction: str,
    evidence: MetricEvidence,
    judge_ref: ModelRef | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if set(evidence.trial_safety) - {item.trial.id for item in trials.values()}:
        raise ValueError("A trial safety report does not belong to either saved trial")
    judge = None
    invoke: InferenceFn | None = None
    async with AsyncExitStack() as stack:
        if judge_ref is not None:
            resolved = ModelsClient.from_client(client).resolve_model_reference(judge_ref.root)
            headers = client.request_headers(url=resolved.url) or {}
            authorization = next((value for name, value in headers.items() if name.casefold() == "authorization"), "")
            scheme, _, key = authorization.partition(" ")
            if authorization and (scheme.casefold() != "bearer" or not key):
                raise ValueError("The judge model requires a Helix bearer-authenticated inference route")
            judge = Model(name=resolved.name, url=resolved.url)
            inference_client = await stack.enter_async_context(new_inference_client(judge, api_key=key or None))

            async def judge_inference(model: Model, request: dict, max_retries: int | None, **options: Any) -> dict:
                options.pop("client", None)
                options.pop("api_key", None)
                return await make_inference_request(model, request, max_retries, client=inference_client, **options)

            invoke = judge_inference
        documentation = await score_documentation(snapshot, judge, invoke)
        scores = {
            name: await score_trial(
                item.trial,
                item.trace,
                discovered=item.discovered,
                job=item.job,
                evidence=evidence,
                instruction=instruction,
                judge=judge,
                invoke=invoke,
            )
            for name, item in trials.items()
        }
    return documentation, scores
