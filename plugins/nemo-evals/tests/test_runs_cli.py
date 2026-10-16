# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals runs submit``: client-side task selection and the run request it sends."""

import json

import httpx
import pytest
from nemo_evals.api.fields import TasksetRef
from nemo_evals.api.schemas import AgentEvalResult
from nemo_evals.runs_cli import (
    Candidate,
    ScoreRollup,
    SelectionError,
    TaskJobView,
    _results_table,
    _status_table,
    needs_image_build,
    rollup_scores,
    run_request,
    run_task_jobs,
    select_tasks,
    selection_suffix,
    submit_run,
)
from nemo_helix_plugin.client.client import NemoClient

CANDIDATES = [
    Candidate(0, "debug-dataset-debug-agent-runtime-error", "hello/debug-agent-runtime-error"),
    Candidate(1, "debug-dataset-greet-universe", "hello/greet-universe"),
    Candidate(2, "debug-dataset-sum-three", "hello/sum-three"),
]


@pytest.mark.parametrize(
    ("tokens", "indexes"),
    [
        (None, [0, 1, 2]),
        ("0,2", [0, 2]),
        ("greet-universe,sum-three", [1, 2]),
        ("hello/sum-three, 1", [2, 1]),
    ],
)
def test_selects_by_index_or_name(tokens, indexes):
    assert [c.index for c in select_tasks(CANDIDATES, tokens)] == indexes


@pytest.mark.parametrize(
    ("tokens", "match"),
    [("3", "out of range"), ("nope", "no task named"), ("2,sum-three", "selected twice"), (",", "selects no task")],
)
def test_bad_selection_fails_before_any_upload(tokens, match):
    with pytest.raises(SelectionError, match=match):
        select_tasks(CANDIDATES, tokens)


def test_ambiguous_name_is_rejected():
    candidates = [Candidate(0, "a-sum", "x/sum"), Candidate(1, "b-sum", "y/sum")]
    with pytest.raises(SelectionError, match="matches several tasks"):
        select_tasks(candidates, "sum")


def test_selection_suffix_ignores_order():
    assert selection_suffix(["a#1", "b#2"]) == selection_suffix(["b#2", "a#1"]) != selection_suffix(["a#1"])


def test_only_dockerfile_tasks_need_builds():
    assert needs_image_build({"environment": {}})
    assert not needs_image_build({"environment": {"docker_image": "public.ecr.aws/x:1"}})


def test_spec_file_keeps_its_settings_and_takes_the_selected_taskset(tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": "default/full", "target": {"kind": "harbor", "source": {"name": "oracle"}}},
                "execution": {"sandbox_policy": "harbor-open", "max_failed_tasks": "10%"},
            }
        )
    )
    request = run_request(
        "default/full-sel-1234abcd#" + "a" * 64,
        spec_file=spec,
        agent="ignored",
        sandbox_policy="ignored",
        dispatcher_profile=None,
    )
    assert isinstance(request.spec.tasks, TasksetRef)
    assert request.spec.tasks.root.startswith("default/full-sel-1234abcd#")
    assert request.execution.sandbox_policy == "harbor-open" and request.execution.max_failed_tasks == "10%"
    assert request.idempotency_key


def test_submit_retries_a_503_with_the_same_body():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(503, json={"detail": "task Jobs not created"})
        return httpx.Response(200, json={"run_id": "run-1", "warnings": []})

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    status = submit_run(
        client, {"idempotency_key": "k"}, workspace="default", echo=lambda _: None, sleep=lambda _: None
    )
    assert status["run_id"] == "run-1" and len(bodies) == 2 and bodies[0] == bodies[1]


# --- runs status / runs results ----------------------------------------------------------------------------

RUN = {
    "run_id": "run-0123456789abcdef",
    "workspace": "default",
    "taskset_ref": "default/debug-1f9c2a7e#" + "e" * 64,
    "phase": "accepted",
    "accepted_at": "2026-10-16T12:00:00Z",
}
NOW = "2026-10-16T12:00:00Z"


def score(name, mean, count, nan=0):
    return {"name": name, "count": count, "nan_count": nan, "mean": mean}


def result_record(job_id, scores, trials=1, errors=0):
    return {
        "id": f"r-{job_id}",
        "name": job_id,
        "workspace": "default",
        "job_id": job_id,
        "scores": {"scores": scores},
        "bundle_ref": f"default/{job_id}#bundle",
        "summary": {"task_count": 1, "trial_count": trials, "score_count": len(scores), "error_count": errors},
        "created_at": NOW,
        "updated_at": NOW,
    }


def job(name, status, task_id):
    return {
        "id": f"id-{name}",
        "attempt_id": "a",
        "name": name,
        "workspace": "default",
        "source": "evals.harbor-task",
        "spec": {"spec": {"tasks": [{"id": task_id}]}},
        "platform_spec": {
            "steps": [{"name": "run", "executor": {"provider": "cpu", "container": {"image": "x"}}, "config": {}}]
        },
        "fileset": "default/f",
        "status": status,
    }


def page(items):
    n = len(items)
    pagination = {"page": 1, "page_size": 100, "current_page_size": n, "total_pages": 1, "total_results": n}
    return {"data": items, "pagination": pagination}


@pytest.fixture
def platform():
    jobs = {
        "run-0123456789abcdef-task-0": job("run-0123456789abcdef-task-0", "completed", "hello/greet-universe"),
        "run-0123456789abcdef-task-1": job("run-0123456789abcdef-task-1", "active", "hello/sum-three"),
    }
    results = {  # keyed and filtered by the Job's name, as the platform stores them
        "run-0123456789abcdef-task-0": result_record(
            "run-0123456789abcdef-task-0",
            [score("harbor_reward.reward", 1.0, 2), score("harbor_reward.style", 0.5, 2, nan=1)],
            trials=2,
        )
    }
    taskset = {
        "id": "ts",
        "name": "debug-1f9c2a7e",
        "workspace": "default",
        "tasks": ["default/a#" + "1" * 64, "default/b#" + "2" * 64, "default/c#" + "3" * 64],
        "revision": 1,
        "tags": {},
        "created_at": NOW,
        "updated_at": NOW,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/agent-evaluate/runs/run-0123456789abcdef"):
            return httpx.Response(200, json=RUN)
        if "/tasksets/debug-1f9c2a7e" in path:
            return httpx.Response(200, json=taskset)
        if "/jobs/" in path:
            name = path.rsplit("/", 1)[1]
            return httpx.Response(200, json=jobs[name]) if name in jobs else httpx.Response(404, json={"detail": "x"})
        if path.endswith("/agent-eval-results"):
            found = [r for r in results.values() if request.url.params.get("filter[job_id]") == r["job_id"]]
            return httpx.Response(200, json=page(found))
        raise AssertionError(f"unexpected {request.method} {request.url}")

    return NemoClient(base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_task_jobs_are_derived_from_the_pinned_taskset(platform):
    views = run_task_jobs(platform, RUN, workspace="default")
    assert [(v.index, v.task, v.status) for v in views] == [
        (0, "hello/greet-universe", "completed"),
        (1, "hello/sum-three", "active"),
        (2, None, "missing"),
    ]


def test_results_read_completed_jobs_only_and_keep_every_reward(platform):
    views = run_task_jobs(platform, RUN, workspace="default", with_results=True)
    assert views[0].result is not None and views[1].result is None
    names = [s.name for s in views[0].result.scores.scores]
    assert names == ["harbor_reward.reward", "harbor_reward.style"]
    table = "\n".join(_results_table(views))
    assert "harbor_reward.style" in table and "1/3 tasks with results, 2 trials" in table


def test_rollup_weights_task_means_by_count():
    first = AgentEvalResult.model_validate(result_record("j0", [score("harbor_reward.reward", 1.0, 3)]))
    second = AgentEvalResult.model_validate(
        result_record("j1", [score("harbor_reward.reward", 0.0, 1, nan=1), score("harbor_reward.extra", 0.5, 1)])
    )
    rollup = {s.name: s for s in rollup_scores([first, second])}
    assert rollup["harbor_reward.reward"] == ScoreRollup("harbor_reward.reward", 0.75, 4, 1, 2)
    assert rollup["harbor_reward.extra"].tasks == 1


def test_status_counts_done_and_failed():
    views = [
        TaskJobView(0, "j0", "a", "completed"),
        TaskJobView(1, "j1", "b", "error"),
        TaskJobView(2, "j2", "c", "active"),
    ]
    assert _status_table(RUN, views)[-1] == "2/3 done (1 completed, 1 failed)"


def test_completed_job_without_a_record_is_flagged():
    views = [TaskJobView(0, "run-x-task-0", "a", "completed")]
    assert _results_table(views)[-2] == "warning: completed without a result record: run-x-task-0"


def test_results_use_the_runs_own_record_when_ready():
    record = AgentEvalResult.model_validate(
        result_record(
            "run-x-bundle-1", [score("harbor_reward.reward", 0.5, 4), score("harbor_reward.reward.pass@1", 0.5, 2)]
        )
        | {"name": "run-x", "bundle_ref": "default/job-fileset-run-x-bundle-1#results/a/agent-eval-results"}
    )
    lines = _results_table([TaskJobView(0, "run-x-task-0", "a", "completed")], record)
    assert lines[-3].startswith("run (merged bundle default/job-fileset-run-x-bundle-1#")
    assert lines[-2] == "  harbor_reward.reward: 0.500 (n=4, nan=0)"


def test_status_shows_result_state_and_bundle():
    run = RUN | {"phase": "completed", "result_state": "ready"}
    lines = _status_table(run, [TaskJobView(0, "j0", "a", "completed")], ("run-x-bundle-1", "completed"))
    assert lines[1] == "phase completed  result ready  bundle run-x-bundle-1 (completed)"
