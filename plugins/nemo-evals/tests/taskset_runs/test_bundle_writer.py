# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merging task bundles gives the summary one run over the same tasks would."""

from datetime import timedelta

import pytest
from nemo_evals.jobs.harbor_scoring import harbor_scoring_task
from nemo_evals.taskset_runs.bundle_writer import load_task_bundle, merge_task_bundles
from nemo_evals.taskset_runs.specs import AcceptedTaskOutput, TasksetBundleSpec
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.persistence import read_trials
from nhx_evals_sdk.agent_eval.runtimes.harbor.scoring import harbor_scoring_metrics
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, AgentOutput

REWARDS = {0: (1.0, 1.0), 1: (0.0, 1.0)}  # task index → (reward, format_ok)


def trial(task_id: str, index: int, n: int = 0) -> AgentEvalTrial:
    reward, format_ok = REWARDS[index]
    return AgentEvalTrial.model_validate(
        {
            "id": f"trial-{n}",
            "task_id": task_id,
            "status": "completed",
            "output": AgentOutput(output_text="done"),
            "metadata": {
                "harbor_primary_reward_key": "reward",
                "reward": reward,
                "reward_details": {"reward": reward, "format_ok": format_ok},
            },
        }
    )


def scoring_task(plan, index, trials):
    task = harbor_scoring_task(plan.spec.tasks[index].spec, reward_key="reward")
    return task.model_copy(update={"metrics": harbor_scoring_metrics(task, trials, reward_key="reward")})


@pytest.fixture
def task_bundles(harbor_plan, tmp_path):
    """What each task Job writes: its own run over one task, persisted in the Docker layout."""
    bundles = []
    for index, resolved in enumerate(harbor_plan.spec.tasks):
        trials = [trial(resolved.id, index, n) for n in range(2)]
        result = AgentEvaluator().run_sync(
            tasks=[scoring_task(harbor_plan, index, trials)],
            trials=trials,
            config=AgentEvalRunConfig(run_id=f"job-run-{index}"),
        )
        directory = tmp_path / f"task-{index}"
        result.persist(directory, write_dashboard=False)
        ref = f"default/job-fileset-{index}#results/att/agent-eval-results"
        bundles.append((directory, AcceptedTaskOutput(task_index=index, job=f"run-x-task-{index}", bundle_ref=ref)))
    return bundles


def spec_for(harbor_plan, harbor_run, accepted):
    return TasksetBundleSpec(
        run_id=harbor_run.name,
        workspace="default",
        plan_ref=harbor_run.plan_ref,
        plan_digest=harbor_run.plan_digest,
        accepted=accepted,
        started_at=harbor_run.accepted_at,
        finished_at=harbor_run.accepted_at + timedelta(minutes=5),
    )


def test_merged_summary_equals_one_run_over_all_tasks(harbor_plan, harbor_run, task_bundles):
    parts = [load_task_bundle(directory, accepted) for directory, accepted in task_bundles]
    merged = merge_task_bundles(spec_for(harbor_plan, harbor_run, [a for _, a in task_bundles]), harbor_plan, parts)

    all_trials = [t for part in parts for t in part.trials]
    single = AgentEvaluator().run_sync(
        tasks=[
            scoring_task(harbor_plan, i, [t for t in all_trials if t.task_id == r.id])
            for i, r in enumerate(harbor_plan.spec.tasks)
        ],
        trials=all_trials,
        config=AgentEvalRunConfig(run_id=harbor_run.name),
    )
    assert merged.summary.model_dump(mode="json") == single.summary.model_dump(mode="json")
    names = {score.name for score in merged.summary.scores.scores}
    assert {"harbor_reward.reward", "harbor_reward.format_ok", "harbor_reward.reward.pass@1"} <= names
    assert merged.summary.scores.score("harbor_reward.reward").mean == 0.5
    assert {score.run_id for score in merged.scores} == {harbor_run.name}
    assert all(score.id.startswith(f"{harbor_run.name}:") for score in merged.scores)


def test_task_without_a_bundle_is_listed_but_adds_no_rows(harbor_plan, harbor_run, task_bundles):
    directory, accepted = task_bundles[0]
    merged = merge_task_bundles(
        spec_for(harbor_plan, harbor_run, [accepted]), harbor_plan, [load_task_bundle(directory, accepted)]
    )
    assert [task.id for task in merged.tasks] == [t.id for t in harbor_plan.spec.tasks]
    assert merged.summary.task_count == 2 and merged.summary.trial_count == 2
    assert {score.task_id for score in merged.scores} == {harbor_plan.spec.tasks[0].id}


def test_merged_bundle_has_the_docker_layout(harbor_plan, harbor_run, task_bundles, tmp_path):
    parts = [load_task_bundle(directory, accepted) for directory, accepted in task_bundles]
    merged = merge_task_bundles(spec_for(harbor_plan, harbor_run, [a for _, a in task_bundles]), harbor_plan, parts)
    out = tmp_path / "merged"
    merged.persist(out, write_dashboard=True)
    assert {p.name for p in out.iterdir()} >= {
        "run.json",
        "metadata.json",
        "tasks.jsonl",
        "trials.jsonl",
        "scores.jsonl",
        "summary.json",
        "report.html",
    }
    assert len(read_trials(out)) == 4
    assert merged.metadata.duration_sec == 300


def test_relative_evidence_is_linked_to_the_task_bundle(harbor_plan, task_bundles):
    directory, accepted = task_bundles[0]
    rows = (directory / "trials.jsonl").read_text().splitlines()
    import json

    row = json.loads(rows[0])
    row["evidence"] = {"descriptors": {"log": {"kind": "file", "ref": "evidence/trial-0/agent.log"}}}
    (directory / "trials.jsonl").write_text("\n".join([json.dumps(row), *rows[1:]]) + "\n")
    part = load_task_bundle(directory, accepted)
    evidence = part.trials[0].evidence
    assert evidence is not None
    ref = evidence.descriptors["log"].ref
    assert ref == f"{accepted.bundle_ref}/evidence/trial-0/agent.log"
