# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals runs submit``: client-side task selection and the run request it sends."""

import io
import json
import tarfile
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import typer
from nemo_evals.api.fields import TasksetRef
from nemo_evals.api.schemas import AgentEvalResult
from nemo_evals.runs_cli import (
    BundleJobView,
    Candidate,
    ScoreRollup,
    SelectionError,
    TaskJobView,
    _results_table,
    _status_table,
    _workspace_flag,
    add_runs_commands,
    build_images,
    job_error,
    newest_bundle_job,
    rollup_scores,
    run_request,
    run_task_jobs,
    select_tasks,
    selection_suffix,
    submit_run,
)
from nemo_evals.sdk.resources import Evaluator
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.types import HelixJobResponse
from typer.testing import CliRunner

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


class _Tasks:
    """Image builds that finish on the second poll.

    ``fail`` names tasks whose build fails; ``abandon`` names tasks whose first attempt's Job dies, so
    requesting again records it failed and starts attempt 2, as ``request_build`` does.
    """

    def __init__(self, fail: frozenset[str] = frozenset(), abandon: frozenset[str] = frozenset()) -> None:
        self.fail, self.abandon = fail, abandon
        self.requested: list[str] = []
        self.polls: dict[str, int] = {}

    def build_images(self, name, *, revision, workspace):
        self.requested.append(f"{workspace}/{name}#{revision}")
        polls = self.polls[name] = self.polls.get(name, -1) + 1
        if polls == 0:
            return SimpleNamespace(id="1", status="pending", images=[], error=None)
        if name in self.abandon:
            return SimpleNamespace(id="2", status="pending", images=[], error=None)
        if polls < 2:
            return SimpleNamespace(id="1", status="running", images=[], error=None)
        if name in self.fail:
            return SimpleNamespace(id="2", status="pending", images=[], error=None)  # failed, so a new attempt
        return SimpleNamespace(id="1", status="ready", images=[object()], error=None)

    def get_image_build(self, name, build_id, *, revision, workspace):
        assert build_id == "1"
        error = "Job ended (error) without a result" if name in self.abandon else "no Dockerfile"
        return SimpleNamespace(id=build_id, status="failed", images=[], error=error)


def test_build_images_waits_for_every_task_prebuilt_included():
    tasks, lines = _Tasks(), []
    refs = ["default/dockerfile#r1", "default/prebuilt#r2"]
    evaluator = cast(Evaluator, SimpleNamespace(tasks=tasks))
    build_images(evaluator, refs, poll_seconds=0, echo=lines.append, sleep=lambda _: None)
    assert tasks.polls == {"dockerfile": 2, "prebuilt": 2}
    assert lines == ["Image builds: 2 tasks ready (2 images)"]


@pytest.mark.parametrize(("kind", "error"), [("fail", "no Dockerfile"), ("abandon", "without a result")])
def test_build_images_stops_at_the_attempt_that_ended(kind, error):
    """Polling requests again, which repairs a dead Job; a new attempt means the followed one ended."""
    tasks = _Tasks(**{kind: frozenset({"broken"})})
    evaluator = cast(Evaluator, SimpleNamespace(tasks=tasks))
    with pytest.raises(RuntimeError, match=f"default/broken#r1: .*{error}"):
        build_images(evaluator, ["default/broken#r1"], poll_seconds=0, echo=print, sleep=lambda _: None)


def test_spec_file_keeps_its_settings_and_takes_the_selected_taskset(tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": "default/full", "target": {"kind": "harbor", "source": {"name": "oracle"}}},
                "execution": {"sandbox_policy": "harbor-open", "max_concurrent_task_jobs": 3},
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
    assert request.execution.sandbox_policy == "harbor-open" and request.execution.max_concurrent_task_jobs == 3
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


def test_a_failed_submit_names_its_key_so_a_rerun_finishes_the_run():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "task Jobs of run run-1 not created"})

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(RuntimeError, match="rerun the same command with --idempotency-key k-123$"):
        submit_run(client, {"idempotency_key": "k-123"}, workspace="default", echo=lambda _: None, sleep=lambda _: None)


@pytest.mark.parametrize("spec_key", [None, "from-file"])
def test_an_explicit_idempotency_key_wins(tmp_path, spec_key):
    ref = "default/t#" + "a" * 64
    assert (
        run_request(
            ref, spec_file=None, agent="oracle", sandbox_policy="p", dispatcher_profile=None, idempotency_key="k-123"
        ).idempotency_key
        == "k-123"
    )
    spec = tmp_path / "run.json"
    body = {
        "spec": {"tasks": "default/full", "target": {"kind": "harbor"}},
        "execution": {"sandbox_policy": "p"},
        "idempotency_key": spec_key,
    }
    spec.write_text(json.dumps(body))
    explicit = run_request(
        ref, spec_file=spec, agent="", sandbox_policy="", dispatcher_profile=None, idempotency_key="k-123"
    )
    assert explicit.idempotency_key == "k-123"
    default = run_request(ref, spec_file=spec, agent="", sandbox_policy="", dispatcher_profile=None)
    assert default.idempotency_key == (spec_key or default.idempotency_key) and default.idempotency_key


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
    assert lines[-5].startswith("run (merged bundle default/job-fileset-run-x-bundle-1#")
    assert lines[-4] == "  harbor_reward.reward: 0.500 (n=4, nan=0)"
    assert lines[-1] == "Download the merged bundle: nemo evals runs download run-x"
    lines = _results_table([TaskJobView(0, "run-x-task-0", "a", "completed")], record, workspace_flag=" --workspace b")
    assert lines[-1] == "Download the merged bundle: nemo evals runs download run-x --workspace b"


def test_printed_commands_name_the_workspace_only_when_it_is_not_the_configured_one():
    assert _workspace_flag("team", "team") == ""
    assert _workspace_flag("other", "team") == " --workspace other"


def test_status_shows_result_state_and_bundle():
    run = RUN | {"phase": "completed", "result_state": "ready"}
    bundle = BundleJobView("run-x-bundle-1", 1, "completed")
    lines = _status_table(run, [TaskJobView(0, "j0", "a", "completed")], bundle)
    assert lines[3] == "  status    completed; the run's result is ready (phase completed, result ready)"
    assert lines[4] == "  bundle    run-x-bundle-1 (attempt 1 of 3): completed"


MODULE_ERROR = "/app/.venv/bin/python: No module named nemo_evals.tasks.taskset_bundle"


def test_a_failed_bundle_says_tasks_finished_and_shows_its_own_error():
    run = RUN | {"phase": "completed", "result_state": "failed", "error": "bundle Job run-x-bundle-3 ended error"}
    bundle = BundleJobView("run-x-bundle-3", 3, "error", MODULE_ERROR)
    lines = _status_table(run, [TaskJobView(0, "j0", "a", "completed")], bundle)
    assert lines[3:8] == [
        "  status    tasks finished, but merging their results into the run's result failed"
        " (phase completed, result failed)",
        "  bundle    run-x-bundle-3 (attempt 3 of 3): error",
        f"            {MODULE_ERROR}",
        "            logs: nemo jobs get-logs run-x-bundle-3",
        "  completed tasks' scores: nemo evals runs results run-0123456789abcdef",
    ]
    assert not any(line.startswith("  error") for line in lines)  # "bundle Job … ended error" only repeats it


@pytest.mark.parametrize("status", ["active", "cancelled"])
def test_a_deadline_error_shows_whatever_the_bundle_did(status):
    error = "finalization deadline passed before the result was saved"
    run = RUN | {"phase": "completed", "result_state": "failed", "error": error}
    lines = _status_table(run, [TaskJobView(0, "j0", "a", "completed")], BundleJobView("run-x-bundle-1", 1, status))
    assert f"  error     {error}" in lines
    assert not any("no error reported" in line for line in lines)
    assert any(line.strip().startswith("logs:") for line in lines) is (status == "cancelled")


def test_an_accepted_run_whose_tasks_all_finished_reads_as_merging():
    lines = _status_table(RUN, [TaskJobView(0, "j0", "a", "completed")], BundleJobView("run-x-bundle-1", 1, "active"))
    assert lines[3] == "  status    merging task results (phase accepted, result -)"


FAILED_POD = {
    "phase": "Failed",
    "containers": [
        {
            "name": "nemo-job-task",
            "terminated": {
                "exit_code": 1,
                "reason": "Error",
                "message": "2026/10/16 17:31:44 INFO using OTLP log authentication auth_mechanism=x\n"
                "[launcher] 2026/10/16 17:31:44 Running main process: python [-m nemo_evals.tasks.taskset_bundle]\n"
                f"{MODULE_ERROR}\n"
                "[launcher] 2026/10/16 17:31:44 Process exited with error: exit status 1\n",
            },
        }
    ],
}
BACKOFF = {"message": "Job failed: Job has reached the specified backoff limit"}


@pytest.mark.parametrize(
    ("status_details", "expected"),
    [(FAILED_POD, MODULE_ERROR), ({}, BACKOFF["message"])],
    ids=["container", "fallback"],
)
def test_job_error_prefers_the_failed_process_over_the_retry_summary(status_details, expected):
    failed = job("run-x-bundle-1", "error", "t") | {"status_details": status_details, "error_details": BACKOFF}
    assert job_error(HelixJobResponse.model_validate(failed)) == expected


@pytest.mark.parametrize("first", ["error", None], ids=["failed", "unreadable"])
def test_newest_bundle_job_is_the_last_attempt_with_its_error(first):
    """The controller skips a name its create left unreadable (404), so the search goes on past it."""
    jobs = {"run-x-bundle-2": job("run-x-bundle-2", "error", "t") | {"status_details": FAILED_POD}}
    if first is not None:
        jobs["run-x-bundle-1"] = job("run-x-bundle-1", first, "t")

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(200, json=jobs[name]) if name in jobs else httpx.Response(404, json={"detail": "x"})

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert newest_bundle_job(client, "run-x", workspace="default") == BundleJobView(
        "run-x-bundle-2", 2, "error", MODULE_ERROR
    )


@pytest.mark.parametrize(
    "command",
    [
        ["status", "run-0123456789abcdef"],
        ["results", "run-0123456789abcdef"],
        ["download", "run-0123456789abcdef"],
        ["list"],
    ],
)
def test_run_commands_default_to_the_configured_workspace(command):
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(404, json={"detail": "not found"})

    client = NemoClient(
        base_url="http://platform.test",
        workspace="team",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", *command])
    assert result.exit_code == 1
    assert paths and paths[0].startswith("/apis/evals/v2/workspaces/team/")


@pytest.mark.parametrize("command", ["status", "results"])
def test_an_sdk_error_exits_with_a_message_not_a_traceback(command):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/agent-evaluate/runs/run-0123456789abcdef"):
            return httpx.Response(200, json=RUN)
        return httpx.Response(500, json={"detail": "Internal server error"})  # the taskset read, through the SDK

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", command, "run-0123456789abcdef", "--workspace", "default"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    assert result.output.startswith("Error: ")


DIGEST = "1f9c2a7e" + "0" * 56


def listed_run(run_id, phase="running", created_by="user:alice"):
    return {
        "run_id": run_id,
        "workspace": "default",
        "taskset_ref": f"default/debug-dataset#{DIGEST}",
        "phase": phase,
        "result_state": "none",
        "accepted_at": "2026-10-17T12:04:11.123456Z",
        "execution_deadline_at": "2026-10-17T13:04:11Z",
        "finalization_deadline_at": "2026-10-17T14:04:11Z",
        "created_by": created_by,
    }


def run_page(runs, *, page=1, total_pages=1, total=None):
    pagination = {
        "page": page,
        "page_size": 20,
        "current_page_size": len(runs),
        "total_pages": total_pages,
        "total_results": len(runs) if total is None else total,
    }
    return {"data": runs, "pagination": pagination}


def invoke_list(args, response):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return CliRunner().invoke(app, ["runs", "list", "--workspace", "default", *args]), requests


def test_list_passes_its_filters_and_page_to_the_api():
    result, requests = invoke_list(
        ["--active", "--mine", "--page", "2", "--page-size", "5"], httpx.Response(200, json=run_page([]))
    )
    assert result.exit_code == 0
    (request,) = requests
    assert request.url.path == "/apis/evals/v2/workspaces/default/agent-evaluate/runs"
    assert dict(request.url.params) == {"active": "true", "mine": "true", "page": "2", "page_size": "5"}


def test_list_shows_one_row_per_run_and_points_to_the_next_page():
    runs = [listed_run("run-aaaa"), listed_run("run-bbbb", phase="completed", created_by=None)]
    result, _ = invoke_list([], httpx.Response(200, json=run_page(runs, total_pages=3, total=45)))
    lines = result.output.splitlines()
    assert lines[0].split() == ["run", "phase", "result", "taskset", "accepted", "by"]
    assert lines[1].split() == [
        "run-aaaa",
        "running",
        "none",
        "default/debug-dataset#1f9c2a7e",
        "2026-10-17",
        "12:04:11",
        "user:alice",
    ]
    assert lines[2].split()[-1] == "-"  # no creator recorded
    assert lines[3] == "page 1 of 3 (45 runs); next: --page 2"

    result, _ = invoke_list([], httpx.Response(200, json=run_page(runs)))
    assert result.output.splitlines()[-1] == "2 runs"


def test_list_says_so_when_there_are_no_runs():
    result, _ = invoke_list([], httpx.Response(200, json=run_page([])))
    assert result.exit_code == 0 and result.output == "No runs.\n"


def test_list_json_prints_the_api_page_unchanged():
    body = run_page([listed_run("run-aaaa")])
    result, _ = invoke_list(["--json"], httpx.Response(200, json=body))
    assert json.loads(result.output) == body


def test_list_failure_exits_with_the_api_detail():
    result, _ = invoke_list([], httpx.Response(500, json={"detail": "Internal server error"}))
    assert result.exit_code == 1
    assert 'listing runs failed (500): "Internal server error"' in result.output


def bundle_archive(*members: tuple[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


GOOD_BUNDLE = bundle_archive(("agent-eval-results/report.html", b"<html>"), ("agent-eval-results/summary.json", b"{}"))


def downloading(tmp_path, archive, *, result_state="ready", download_status=200):
    """Invoke ``runs download run-x -o tmp_path/run-x`` against a run whose record names ``run-x-bundle-2``."""
    requests: list[str] = []
    run = RUN | {"run_id": "run-x", "phase": "completed", "result_state": result_state}
    record = result_record("run-x-bundle-2", []) | {"name": "run-x"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append(path)
        if path.endswith("/agent-evaluate/runs/run-x"):
            return httpx.Response(200, json=run)
        if path.endswith("/agent-eval-results/run-x"):
            return httpx.Response(200, json=record)
        if path.endswith("/results/agent-eval-results/download"):
            return httpx.Response(download_status, stream=httpx.ByteStream(archive))
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    args = ["runs", "download", "run-x", "--workspace", "default", "-o", str(tmp_path / "run-x")]
    return CliRunner().invoke(app, args), requests


def test_download_unpacks_the_bundle_of_the_job_the_record_names(tmp_path):
    result, requests = downloading(tmp_path, GOOD_BUNDLE)
    assert result.exit_code == 0, result.output
    assert "/apis/jobs/v2/workspaces/default/jobs/run-x-bundle-2/results/agent-eval-results/download" in requests
    bundle = tmp_path / "run-x" / "agent-eval-results"
    assert (bundle / "report.html").read_bytes() == b"<html>"
    assert f"{bundle / 'report.html'}" in result.output
    assert [p.name for p in (tmp_path / "run-x").iterdir()] == ["agent-eval-results"]  # no scratch left


@pytest.mark.parametrize("state", ["failed", "pending"])
def test_download_refuses_a_run_without_a_merged_result(tmp_path, state):
    result, requests = downloading(tmp_path, GOOD_BUNDLE, result_state=state)
    assert result.exit_code == 1
    assert f"has no merged result (phase completed, result {state})" in result.output
    assert not any(path.endswith("/download") for path in requests)


def test_download_never_overwrites_an_existing_bundle(tmp_path):
    existing = tmp_path / "run-x" / "agent-eval-results"
    existing.mkdir(parents=True)
    (existing / "notes.txt").write_text("mine")
    result, _ = downloading(tmp_path, GOOD_BUNDLE)
    assert result.exit_code == 1 and "already exists" in result.output
    assert [p.name for p in existing.iterdir()] == ["notes.txt"]


def test_an_escaping_archive_writes_nothing_and_a_retry_still_works(tmp_path):
    evil = bundle_archive(("agent-eval-results/report.html", b"<html>"), ("../evil", b"x"))
    result, _ = downloading(tmp_path, evil)
    assert result.exit_code == 1 and "unsafe tar member" in result.output
    assert not (tmp_path / "evil").exists()
    assert list((tmp_path / "run-x").iterdir()) == []

    result, _ = downloading(tmp_path, GOOD_BUNDLE)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(("archive", "status"), [(b"not found", 404), (b"not a tarball", 200)], ids=["404", "corrupt"])
def test_a_failed_download_is_an_error_not_a_traceback(tmp_path, archive, status):
    result, _ = downloading(tmp_path, archive, download_status=status)
    assert result.exit_code == 1 and result.output.startswith("Error: ")
    assert not (tmp_path / "run-x" / "agent-eval-results").exists()
