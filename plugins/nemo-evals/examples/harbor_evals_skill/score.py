# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the Codex comparison and replay its saved trials with stored Harbor metrics."""

import argparse
import asyncio
import json
import os
import re
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import supercool_metrics as supercool
from codex_evidence import CodexEvidence, observe_codex, rollout_events
from nemo_evals.api.schemas import HarborTaskDefinition, TaskRef, TasksetInput, TasksetRef
from nemo_evals.sdk.resources import Evaluator
from nemo_helix_plugin.agents.skills import PinnedSkillRef
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.evals.client import EvaluatorClient
from nemo_helix_plugin.jobs.archive import safe_extract_tar
from nhx_evals_sdk.agent_eval.persistence import read_trials
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary
from nhx_evals_sdk.agent_eval.reward_keys import HARBOR_PRIMARY_REWARD_KEY
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import discover_harbor_tasks
from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.tasks import SemanticReducer, SemanticView, ViewSignal
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, TrialMeasurements
from nhx_evals_sdk.enums import MetricType
from nhx_evals_sdk.metrics.protocol import CandidateOutput, DatasetRow, MetricInput
from nhx_evals_sdk.metrics.skill_readiness import (
    SkillDescriptionQualityMetric,
    SkillReliabilityReadinessMetric,
    SkillStaticCorrectnessMetric,
    SkillUnicodeSafetyMetric,
)
from nhx_evals_sdk.metrics.skill_traces import ToolExecutionOutcomeMetric
from nhx_evals_sdk.metrics.skills import (
    SkillActivationMetric,
    SkillContextEfficiencyMetric,
    SkillReferenceIntegrityMetric,
    SkillStructureMetric,
    TokenEfficiencyMetric,
    expected_behavior_metric,
)
from nhx_evals_sdk.values import ModelRef
from nhx_evals_sdk.values.evidence import EVIDENCE_LOGS
from nhx_evals_sdk.values.skill_evidence import SkillToolCall, SkillToolTrace
from nhx_evals_sdk.values.skills import SkillSnapshot
from pydantic import BaseModel, ConfigDict, Field, model_validator
from skill_artifacts import pin_revision as _revision_pin
from skill_artifacts import read_skill_package as _skill_package
from skill_artifacts import split_pin as _split_pin

SKILL_NAME = "nemo-evals-plugin"
AGENT_EVAL_SKILL_NAME = supercool.SKILL_NAME
TASK_BY_SKILL = {SKILL_NAME: "nemo-evals-plugin-001", AGENT_EVAL_SKILL_NAME: "nemo-evals-plugin-002"}
DATASET = Path(__file__).resolve().parent / "dataset"
_ARMS = ("baseline", "with-skill")
_AGENT_EVAL_REPLAY_METRICS = {
    MetricType.HARBOR_REWARD,
    MetricType.SKILL_ACTIVATION,
    MetricType.TOKEN_EFFICIENCY,
}
_REPLAY_METRICS = _AGENT_EVAL_REPLAY_METRICS | {MetricType.TOOL_EXECUTION_OUTCOME, MetricType.LLM_JUDGE}
_AGENT_EVAL_REWARDS = ("inner_eval_passed", "suffix_applied")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PINNED_IMAGE = Annotated[str, Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")]
_ERRORS = (NemoClientError, OSError, RuntimeError, ValueError, tarfile.TarError)


class Arm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    job_name: str = Field(min_length=1)
    reward: float = Field(strict=True, ge=0, le=1)
    agent_image_ref: _PINNED_IMAGE
    verifier_image_ref: _PINNED_IMAGE
    bundle: str = Field(min_length=1)


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taskset_ref: TasksetRef
    skill_ref: PinnedSkillRef
    workspace: str = Field(min_length=1)
    context: str = Field(min_length=1)
    arms: dict[str, Arm]

    @property
    def skill_name(self) -> str:
        return _split_pin(self.skill_ref.root)[1]

    @model_validator(mode="after")
    def validate_pair(self) -> "Comparison":
        taskset_workspace, _, _ = _split_pin(self.taskset_ref.root)
        skill_workspace, skill_name, _ = _split_pin(self.skill_ref.root)
        if taskset_workspace != self.workspace or skill_workspace != self.workspace or skill_name not in TASK_BY_SKILL:
            raise ValueError(
                "Comparison refs must identify the pinned Taskset and Evals skill in the selected workspace"
            )
        if set(self.arms) != set(_ARMS):
            raise ValueError("Comparison needs a baseline and a with-skill arm")
        baseline, treatment = (self.arms[arm] for arm in _ARMS)
        for field in ("agent_image_ref", "verifier_image_ref"):
            if getattr(baseline, field) != getattr(treatment, field):
                raise ValueError(f"Comparison arms used different {field}")
        if any(arm.reward not in {0.0, 1.0} for arm in self.arms.values()):
            raise ValueError("Each completed trial must have a binary Harbor reward")
        if (
            baseline.run_id == treatment.run_id
            or baseline.job_name == treatment.job_name
            or baseline.bundle == treatment.bundle
        ):
            raise ValueError("Comparison needs distinct runs, task Jobs, and bundles")
        return self


def _bundle_root(directory: Path) -> Path:
    matches = list(directory.rglob("trials.jsonl"))
    if len(matches) != 1 or not matches[0].resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Expected one trials.jsonl inside {directory}")
    return matches[0].parent


def _arm_bundle(run_dir: Path, relative: str) -> Path:
    directory = (run_dir / relative).resolve()
    if Path(relative).is_absolute() or not directory.is_relative_to(run_dir) or not directory.is_dir():
        raise ValueError(f"Bundle must be a local directory inside the comparison: {relative}")
    return _bundle_root(directory)


def _json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


async def _package_metrics(snapshot: SkillSnapshot) -> dict[str, Any]:
    metric_input = MetricInput(
        row=DatasetRow(data={"reference": {"skill_snapshot": snapshot.model_dump(mode="json")}}),
        candidate=CandidateOutput(),
    )
    metrics = (
        SkillStructureMetric(),
        SkillReferenceIntegrityMetric(),
        SkillContextEfficiencyMetric(),
        SkillStaticCorrectnessMetric(),
        SkillDescriptionQualityMetric(),
        SkillReliabilityReadinessMetric(),
        SkillUnicodeSafetyMetric(),
    )
    return {
        metric.type.value: (await metric.compute_scores(metric_input)).model_dump(mode="json") for metric in metrics
    }


def _agent_log_dir(bundle: Path, trial: AgentEvalTrial) -> Path:
    descriptor = trial.evidence.descriptors.get(EVIDENCE_LOGS) if trial.evidence is not None else None
    if not descriptor or not descriptor.ref:
        raise ValueError("The saved trial has no Harbor agent log directory")
    agent_dir = Path(descriptor.ref).resolve()
    if agent_dir.name != "agent" or not agent_dir.is_relative_to(bundle.resolve()) or not agent_dir.is_dir():
        raise ValueError("The saved trial has an agent log directory outside the downloaded bundle")
    return agent_dir


def _evidence(
    bundle: Path,
    trial: AgentEvalTrial,
    snapshot: SkillSnapshot,
    expected_inventory: dict[str, dict[str, str]],
    *,
    expect_setup_egress: bool,
    events: Iterable[dict[str, Any]] | None = None,
) -> CodexEvidence:
    agent_dir = _agent_log_dir(bundle, trial)
    log = (agent_dir.parent / "trial.log").read_text(encoding="utf-8")
    if "Codex is already available at the requested version" not in log:
        raise ValueError("Harbor did not record the built-in Codex install being skipped")
    if expect_setup_egress and "Closed agent setup egress" not in log:
        raise ValueError("Harbor did not record the Codex setup egress window closing")
    probe = _json_object(agent_dir / "scope-probe.json")
    if probe.get("enabled") is not False or len(probe) != 1:
        raise ValueError("The scoped key probe must be disabled in a scored arm")
    inventory = _json_object(agent_dir.parent / "artifacts" / "skill-inventory.json")
    if inventory.get("staged") != expected_inventory or inventory.get("registered") != expected_inventory:
        raise ValueError("The staged and registered skill inventories do not match the expected archive files")
    observed = observe_codex(
        rollout_events(agent_dir) if events is None else events, snapshot.directory_name, snapshot.document or ""
    )
    if observed.skill_listed != bool(expected_inventory):
        raise ValueError("The launch skill catalog does not match the arm's expected discovery")
    return observed


def replay_trial(
    trial: AgentEvalTrial,
    expected_reward: float,
    measurements: TrialMeasurements,
    activation: bool | None,
    command_outcomes: tuple[bool | None, ...] = (),
    *,
    skill_name: str = SKILL_NAME,
) -> AgentEvalTrial:
    if trial.error is not None or trial.status != AgentEvalTrialStatus.COMPLETED:
        raise ValueError("The saved Harbor trial did not complete successfully")
    reward_key = trial.metadata.get(HARBOR_PRIMARY_REWARD_KEY)
    reward = trial.metadata.get("reward")
    details = trial.metadata.get("reward_details")
    if (
        reward_key != "reward"
        or type(reward) not in (int, float)
        or reward not in (0, 1)
        or reward != expected_reward
        or not isinstance(details, dict)
        or type(details.get(reward_key)) not in (int, float)
        or details[reward_key] != reward
    ):
        raise ValueError("The saved Harbor reward does not match the comparison")
    if skill_name == AGENT_EVAL_SKILL_NAME:
        components = [details.get(name) for name in _AGENT_EVAL_REWARDS]
        if any(type(value) not in (int, float) or value not in (0, 1) for value in components) or reward != int(
            all(components)
        ):
            raise ValueError("The saved agent-eval reward components do not match its Harbor reward")
    metadata = {HARBOR_PRIMARY_REWARD_KEY: reward_key, "reward": reward, "reward_details": details}
    if activation is not None:
        metadata["skill_activation"] = {skill_name: activation}
    if command_outcomes:
        # The root rollout covers only selected calls. Its event order does not establish tool step boundaries.
        trace = SkillToolTrace(
            trial_id=trial.id,
            complete=False,
            calls=[
                SkillToolCall(id=f"exec-{index}", step=0, tool="exec_command", succeeded=outcome)
                for index, outcome in enumerate(command_outcomes, start=1)
            ],
        )
        metadata["skill_tool_trace"] = trace.model_dump(mode="json", exclude_none=True)
    answer = trial.output.output_text if trial.output is not None else None
    return AgentEvalTrial(
        id=trial.id,
        task_id=trial.task_id,
        status=trial.status,
        output=AgentOutput(output_text=answer if answer and answer.strip() else None),
        measurements=measurements,
        metadata=metadata,
    )


def _scoring_taskset(
    evaluator: Evaluator,
    live_ref: TasksetRef,
    judge_ref: ModelRef | None,
    trials: list[AgentEvalTrial],
    *,
    skill_name: str,
) -> TasksetRef:
    workspace, name, digest = _split_pin(live_ref.root)
    live = evaluator.tasksets.retrieve(name, workspace=workspace, revision=digest)
    if len(live.tasks) != 1:
        raise ValueError("The live Taskset must pin the example's one Harbor task")
    task_workspace, task_name, task_digest = _split_pin(live.tasks[0].root)
    member = evaluator.tasks.retrieve(task_name, workspace=task_workspace, revision=task_digest)
    task_id = TASK_BY_SKILL[skill_name]
    if not isinstance(member.spec, HarborTaskDefinition) or member.spec.native_task_id != task_id:
        raise ValueError("Expected the example's one live Harbor task")
    (source,) = discover_harbor_tasks(DATASET / task_id)
    if source.id != member.spec.native_task_id or any(trial.task_id != source.id for trial in trials):
        raise ValueError(
            "The local task, live Taskset member, and both saved trials must share the native Harbor task ID"
        )
    source.metrics.extend([SkillActivationMetric(skill_name=skill_name, should_activate=True), TokenEfficiencyMetric()])
    if skill_name == AGENT_EVAL_SKILL_NAME:
        source.views["verified_skill_use"] = SemanticView(
            reducer=SemanticReducer.ALL,
            signals=[
                ViewSignal(metric="harbor_reward", output="reward"),
                ViewSignal(metric="skill_activation", output="activated"),
            ],
        )
    else:
        if judge_ref is None:
            raise ValueError("Set JUDGE_MODEL_REF for the offline Evals example")
        judge, views = expected_behavior_metric(
            judge_ref,
            [
                "The final answer states that the offline evaluation job completed successfully.",
                "The final answer gives the name of the evaluation job.",
                "The final answer identifies the exact-match metric and reports its mean of 0.5.",
            ],
        )
        source.metrics.extend([ToolExecutionOutcomeMetric(tool_pattern="exec_command"), judge])
        source.views.update(views)
    prepared = evaluator.tasks.prepare(source, workspace=workspace)
    if (
        not isinstance(prepared.spec, HarborTaskDefinition)
        or prepared.spec.source.files_hash != member.spec.source.files_hash
    ):
        raise ValueError("The scoring task must use the same archive bytes as the pinned live Taskset member")
    scoring_name = f"{task_id}-scoring"
    task = evaluator.tasks.replace(scoring_name, task=prepared, workspace=workspace)
    pin = _revision_pin(evaluator.tasks.list_revisions, workspace, task.name, task.revision)
    scoring = evaluator.tasksets.replace(scoring_name, taskset=TasksetInput(tasks=[TaskRef(pin)]), workspace=workspace)
    return TasksetRef(_revision_pin(evaluator.tasksets.list_revisions, workspace, scoring.name, scoring.revision))


def _replay_results(
    evals_client: EvaluatorClient,
    job_name: str,
    trial: AgentEvalTrial,
    archive_dest: Path,
    *,
    skill_name: str = SKILL_NAME,
) -> tuple[dict[str, Any], list[str]]:
    with tempfile.TemporaryDirectory(dir=archive_dest.parent, prefix=".skill-replay-") as directory:
        root = Path(directory)
        archive_path = root / "bundle.tar.gz"
        response = evals_client.download_agent_eval_job_result(job=job_name, name="agent-eval-results")
        with archive_path.open("wb") as output, response.stream() as chunks:
            for chunk in chunks:
                output.write(chunk)
        with tarfile.open(archive_path) as archive:
            safe_extract_tar(archive, root / "bundle")
        archive_path.replace(archive_dest)
        bundle = _bundle_root(root / "bundle")
        summary = AgentEvalSummary.model_validate_json((bundle / "summary.json").read_text(encoding="utf-8"))
        scores = [
            AgentEvalTaskScore.model_validate_json(line)
            for line in (bundle / "scores.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    rewards = {"reward": trial.metadata["reward"]}
    metrics = _REPLAY_METRICS
    if skill_name == AGENT_EVAL_SKILL_NAME:
        rewards.update({key: trial.metadata["reward_details"][key] for key in _AGENT_EVAL_REWARDS})
        metrics = _AGENT_EVAL_REPLAY_METRICS
    for key, value in rewards.items():
        aggregates = [score for score in summary.scores.scores if score.name == f"harbor_reward.{key}"]
        if len(aggregates) != 1 or aggregates[0].mean != value or aggregates[0].count != 1 or aggregates[0].nan_count:
            raise ValueError(f"The replay did not preserve the saved Harbor reward {key}")
    if (
        summary.trial_count != 1
        or summary.error_count
        or len(scores) != len(metrics)
        or {score.metric_type for score in scores} != metrics
        or any(score.trial_id != trial.id or score.task_id != trial.task_id for score in scores)
    ):
        raise ValueError("The replay results do not match the one saved Harbor trial and reward")
    failed_metrics = [
        score.metric_type
        for score in scores
        if score.status != AgentEvalScoreStatus.COMPLETED and not _blank_answer_diagnostic(score, trial)
    ]
    return {
        "bundle": archive_dest.name,
        "summary": summary.model_dump(mode="json"),
        "scores": [score.model_dump(mode="json") for score in scores],
    }, failed_metrics


def _blank_answer_diagnostic(score: AgentEvalTaskScore, trial: AgentEvalTrial) -> bool:
    return (
        score.metric_type == MetricType.LLM_JUDGE
        and trial.output is not None
        and trial.output.output_text is None
        and len(score.diagnostics) == 1
        and score.diagnostics[0].details.get("exception_type") == "UndefinedError"
        and "output_text" in score.diagnostics[0].message
    )


def _write_summary(run_dir: Path, report: dict[str, Any]) -> None:
    data = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=run_dir, prefix=".skill-eval-", delete=False
    ) as tmp:
        path = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.close()
            path.replace(run_dir / "skill-eval-summary.json")
        finally:
            path.unlink(missing_ok=True)


def _score(run_dir: Path, comparison: Comparison, judge: ModelRef | None, *, expect_setup_egress: bool) -> bool:
    report = comparison.model_dump(mode="json")
    if judge is not None:
        report["judge_model_ref"] = judge.root
    skill_name = comparison.skill_name
    success = True
    try:
        with NemoClient.from_config(context=comparison.context) as context_client:
            client = context_client.with_workspace(comparison.workspace)
            snapshot, inventory, contents = _skill_package(client, comparison.skill_ref)
            report["skill_files_hash"] = snapshot.digest
            report["package_metrics"] = asyncio.run(_package_metrics(snapshot))
            metric_evidence = supercool.MetricEvidence()
            if skill_name == AGENT_EVAL_SKILL_NAME:
                evidence_file = run_dir / "metric-evidence.json"
                if evidence_file.exists():
                    metric_evidence = supercool.MetricEvidence.model_validate_json(evidence_file.read_bytes())
                report["package_metrics"].update(
                    asyncio.run(supercool.score_package(snapshot, contents, metric_evidence))
                )
            trials = {}
            supercool_trials = {}
            for name in _ARMS:
                arm = comparison.arms[name]
                try:
                    bundle = _arm_bundle(run_dir, arm.bundle)
                    saved = read_trials(bundle)
                    if len(saved) != 1:
                        raise ValueError("Expected one saved Harbor trial")
                    expected = {skill_name: inventory} if name == "with-skill" else {}
                    events = None
                    if skill_name == AGENT_EVAL_SKILL_NAME:
                        agent_dir = _agent_log_dir(bundle, saved[0])
                        events = list(rollout_events(agent_dir))
                    evidence = _evidence(
                        bundle, saved[0], snapshot, expected, expect_setup_egress=expect_setup_egress, events=events
                    )
                    trials[name] = replay_trial(
                        saved[0],
                        arm.reward,
                        evidence.measurements,
                        evidence.activated,
                        evidence.command_outcomes if skill_name == SKILL_NAME else (),
                        skill_name=skill_name,
                    )
                    report["arms"][name]["discovery"] = evidence.skill_listed
                    report["arms"][name]["activation"] = evidence.activated
                    report["arms"][name]["measurements"] = evidence.measurements.model_dump(mode="json")
                    if events is not None:
                        job = supercool.read_inner_job(
                            agent_dir.parent / "artifacts" / "job-snapshot.json", trials[name]
                        )
                        trace = supercool.trial_trace(
                            events, trials[name].id, snapshot.document or "", activated=evidence.activated, job=job
                        )
                        supercool_trials[name] = supercool.TrialEvidence(
                            trials[name], trace, evidence.skill_listed, job
                        )
                except _ERRORS as exc:
                    report["arms"][name]["evidence_error"] = str(exc)
                    success = False
            if not success:
                return False

            evaluator = Evaluator.from_client(client)
            scoring = _scoring_taskset(
                evaluator, comparison.taskset_ref, judge, list(trials.values()), skill_name=skill_name
            )
            report["scoring_taskset_ref"] = scoring.root
            evals_client = EvaluatorClient.from_client(client)
            for name in _ARMS:
                arm_report = report["arms"][name]
                try:
                    job = evaluator.submit(tasks=scoring, trials=[trials[name]])
                    arm_report["scoring_job"] = {"name": job.name, "id": job.job.id}
                    _write_summary(run_dir, report)
                    job.wait_until_done()
                    arm_report["scoring_job"]["id"] = job.job.id
                    arm_report["replay"], failed_metrics = _replay_results(
                        evals_client,
                        job.name,
                        trials[name],
                        run_dir / f"{name}-scoring-results.tar.gz",
                        skill_name=skill_name,
                    )
                    if failed_metrics:
                        raise ValueError(f"Replay metrics failed: {', '.join(failed_metrics)}")
                except _ERRORS as exc:
                    arm_report["replay_error"] = str(exc)
                    success = False
            if supercool_trials:
                instruction = (DATASET / TASK_BY_SKILL[skill_name] / "instruction.md").read_text(encoding="utf-8")
                documentation, skill_scores = asyncio.run(
                    supercool.score_diagnostics(client, snapshot, supercool_trials, instruction, metric_evidence, judge)
                )
                report["documentation_rubric"] = documentation
                for name, scores in skill_scores.items():
                    report["arms"][name]["skill_metrics"] = scores
    except _ERRORS as exc:
        report["error"] = str(exc)
        success = False
    finally:
        _write_summary(run_dir, report)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing comparison.json and both downloaded bundles.")
    run_dir = parser.parse_args().run_dir.resolve()
    try:
        comparison = Comparison.model_validate_json((run_dir / "comparison.json").read_text(encoding="utf-8"))
        judge_ref = os.environ.get("JUDGE_MODEL_REF", "")
        judge = ModelRef(judge_ref) if judge_ref or comparison.skill_name == SKILL_NAME else None
        expect_egress = os.environ.get("EXPECT_SETUP_EGRESS", "1")
        if expect_egress not in {"0", "1"}:
            raise ValueError("Set EXPECT_SETUP_EGRESS to 0 for profiles without setup egress, or 1 to check closure")
        success = _score(run_dir, comparison, judge, expect_setup_egress=expect_egress == "1")
    except _ERRORS as exc:
        print(f"Skill evaluation scoring failed: {exc}", file=sys.stderr)
        return 1
    summary = run_dir / "skill-eval-summary.json"
    if not success:
        print(f"Skill evaluation scoring failed. See {summary}", file=sys.stderr)
        return 1
    print(f"Skill evaluation summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
