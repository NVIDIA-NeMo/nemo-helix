# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals runs submit``: client-side task selection and the run request it sends."""

import io
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import typer
from nemo_evals.api.fields import TasksetRef
from nemo_evals.api.schemas import AgentEvalResult
from nemo_evals.harbor.remote.profile import SandboxSelection
from nemo_evals.image_builds.models import BuildImage, ImageBuild
from nemo_evals.runs_cli import (
    BundleJobView,
    Candidate,
    CreatedTaskset,
    ScoreRollup,
    SelectionError,
    TaskJobView,
    TaskStateView,
    _progress_line,
    _results_table,
    _select_from_spec,
    _status_table,
    _view_json,
    _workspace_flag,
    add_runs_commands,
    check_git_source,
    download_git_tasks,
    job_error,
    newest_bundle_job,
    rollup_scores,
    run_request,
    run_task_jobs,
    select_tasks,
    selection_suffix,
    submit_run,
    task_state,
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


STORED = "ac9/deep-swe-162c7ea7#" + "2" * 64
MEMBERS: list[str] = [f"ac9/deep-swe-{name}#" + "1" * 64 for name in ("a", "b")]


@pytest.fixture
def submitting(monkeypatch):
    """``runs submit`` over a fake evaluator and submit; returns what it published, requested and sent."""
    seen = SimpleNamespace(bodies=[], published=[], posts=[], created=[])
    builds = SimpleNamespace(
        build_images=lambda name, revision, workspace: seen.posts.append(name) or _ready(name),
        get_image_build=lambda name, build_id, revision, workspace: _ready(name),
    )
    tasks = SimpleNamespace(
        retrieve=lambda name, revision, workspace: SimpleNamespace(revision=1, spec=SimpleNamespace()),
        build_images=builds.build_images,
        get_image_build=builds.get_image_build,
    )
    tasksets = SimpleNamespace(
        retrieve=lambda name, revision, workspace: SimpleNamespace(tasks=MEMBERS),
        replace=lambda name, taskset, workspace: seen.published.append(name),
        list_revisions=lambda name, workspace, page_size: SimpleNamespace(
            data=[SimpleNamespace(content_hash="3" * 64)]
        ),
    )
    fake = SimpleNamespace(tasks=tasks, tasksets=tasksets)
    monkeypatch.setattr("nemo_evals.runs_cli.Evaluator", SimpleNamespace(from_client=lambda _: fake))

    def submit(client, body, *, workspace, echo):
        seen.bodies.append(body)
        if seen.refuse:
            raise RuntimeError('submit failed (422): "tasks without images: ac9/deep-swe-b#..."')
        return {"run_id": "run-1", "warnings": []}

    seen.refuse = False
    monkeypatch.setattr("nemo_evals.runs_cli.submit_run", submit)
    client = SimpleNamespace(resolve_workspace=lambda workspace=None: workspace or "ac9")

    def invoke(*args: str):
        app = typer.Typer()
        add_runs_commands(app, lambda: client)  # ty: ignore[invalid-argument-type]
        return CliRunner().invoke(app, ["runs", "submit", *args])

    seen.invoke = invoke
    return seen


def _image_build(build_id: str = "1", status: str = "ready") -> ImageBuild:
    return ImageBuild(
        id=build_id,
        task="ac9/deep-swe-a",
        revision="1" * 64,
        status=status,  # ty: ignore[invalid-argument-type]
        job=f"task-image-build-{build_id}",
        allowed_registries=[],
        images=[BuildImage(role="main", source="prebuilt")] if status == "ready" else [],
        error="no Dockerfile" if status == "failed" else None,
    )


def _ready(name):
    return _image_build()


def test_submit_with_a_taskset_uploads_nothing_and_requests_no_builds(submitting):
    result = submitting.invoke(
        "--taskset", STORED, "--experiment", "m4", "--dataset", "datacurve/deep-swe@sha256:" + "a" * 64
    )
    assert result.exit_code == 0, result.output
    assert submitting.published == [] and submitting.posts == []
    (body,) = submitting.bodies
    assert body["spec"]["tasks"] == STORED
    assert body["publication"]["intake"] == {
        "experiment": "m4",
        "dataset_name": "datacurve/deep-swe",
        "dataset_version": "sha256:" + "a" * 64,
    }
    assert "Run run-1 accepted: 2 tasks, queued" in result.output


def test_submit_narrows_a_taskset_by_publishing_a_selection_without_uploading_tasks(submitting):
    result = submitting.invoke("--taskset", STORED, "--tasks", "1")
    assert result.exit_code == 0, result.output
    assert len(submitting.published) == 1 and submitting.published[0].startswith("deep-swe-162c7ea7-sel-")
    assert submitting.posts == [] and submitting.bodies[0]["spec"]["tasks"].endswith("#" + "3" * 64)


def test_a_refused_taskset_submit_says_how_to_check_the_builds(submitting):
    submitting.refuse = True
    result = submitting.invoke("--taskset", "deep-swe-162c7ea7#" + "2" * 64)
    assert result.exit_code == 1
    assert f"Check with: nemo evals tasksets build-images {STORED} --status" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["--taskset", STORED, "--harbor-hub", "x/y@1"],
        ["--taskset", STORED, "--dataset", "x/y"],  # no --experiment
        ["--harbor-hub", "x/y@1", "--experiment", "e", "--dataset", "x/y"],  # the source sets the dataset
    ],
)
def test_taskset_flags_that_do_not_combine(submitting, args):
    assert submitting.invoke(*args).exit_code == 2 and submitting.bodies == []


@pytest.mark.parametrize(("failures", "submitted"), [(0, True), (1, True), (2, False)])
def test_the_wrapper_follows_one_retry_and_never_submits_after_a_failed_build(
    submitting, monkeypatch, failures, submitted
):
    created = CreatedTaskset(STORED, MEMBERS[:1], 1, ("datacurve/deep-swe", None))
    monkeypatch.setattr("nemo_evals.runs_cli.create_taskset", lambda *a, **k: created)
    attempts = []

    def build_images(name, revision, workspace):
        attempts.append(name)
        return _image_build(str(len(attempts)), "pending")

    def get_image_build(name, build_id, revision, workspace):
        return _image_build(build_id, "failed" if int(build_id) <= failures else "ready")

    submitting_tasks = SimpleNamespace(build_images=build_images, get_image_build=get_image_build)
    monkeypatch.setattr(
        "nemo_evals.runs_cli.Evaluator",
        SimpleNamespace(from_client=lambda _: SimpleNamespace(tasks=submitting_tasks)),
    )
    result = submitting.invoke("--harbor-hub", "datacurve/deep-swe@latest", "--experiment", "m4", "--poll-seconds", "0")
    assert bool(submitting.bodies) is submitted, result.output
    assert len(attempts) == min(failures, 1) + 1  # at most one retry
    assert "Submit later with: nemo evals runs submit --taskset " + STORED in result.output
    if submitted:
        assert submitting.bodies[0]["publication"]["intake"]["dataset_name"] == "datacurve/deep-swe"
    else:
        assert result.exit_code == 1 and "image builds failed" in result.output


def test_spec_file_keeps_its_settings_and_takes_the_selected_taskset(tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps(
            {
                "spec": {"tasks": "default/full", "target": {"kind": "harbor", "source": {"name": "oracle"}}},
                "execution": {"sandbox": {"profile": "harbor-open"}, "max_concurrent_task_jobs": 3},
            }
        )
    )
    request = run_request(
        "default/full-sel-1234abcd#" + "a" * 64,
        spec_file=spec,
        agent="ignored",
        dispatcher_profile=None,
        sandbox=SandboxSelection(profile="ignored", cpus=64),
    )
    assert isinstance(request.spec.tasks, TasksetRef)
    assert request.spec.tasks.root.startswith("default/full-sel-1234abcd#")
    assert request.execution.sandbox == SandboxSelection(profile="harbor-open")
    assert request.execution.max_concurrent_task_jobs == 3
    assert request.idempotency_key


def test_experiment_publishes_the_run_to_intake(tmp_path):
    plain = run_request("default/t#" + "a" * 64, spec_file=None, agent="codex", dispatcher_profile=None)
    assert plain.publication is None
    request = run_request(
        "default/t#" + "a" * 64,
        spec_file=None,
        agent="codex",
        dispatcher_profile=None,
        experiment="swe-agent-tuning",
    )
    assert request.publication is not None and request.publication.intake.experiment == "swe-agent-tuning"
    spec = tmp_path / "run.json"
    spec.write_text(json.dumps({"spec": {"tasks": "default/t", "target": {"kind": "harbor"}}, "execution": {}}))
    # With --harbor-hub/--harbor-dir the dataset supplies the tasks, so the file may leave spec.tasks out.
    no_tasks = tmp_path / "no-tasks.json"
    no_tasks.write_text(json.dumps({"spec": {"target": {"kind": "harbor"}}, "execution": {}}))
    selected = run_request("default/t#" + "a" * 64, spec_file=no_tasks, agent="", dispatcher_profile=None)
    assert isinstance(selected.spec.tasks, TasksetRef) and selected.spec.tasks.root == "default/t#" + "a" * 64
    # --experiment replaces the file's publication before validation, so an invalid one doesn't block it.
    no_tasks.write_text(
        json.dumps(
            {
                "spec": {"target": {"kind": "harbor"}},
                "execution": {},
                "publication": {"intake": {"experiment": ""}},
            }
        )
    )
    overridden = run_request(
        "default/t#" + "a" * 64,
        spec_file=no_tasks,
        agent="",
        dispatcher_profile=None,
        experiment="x",
    )
    assert overridden.publication is not None and overridden.publication.intake.experiment == "x"
    for bad in ([], {"spec": ["not", "an", "object"]}):
        no_tasks.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="expected a TasksetRunInput JSON object"):
            run_request("default/t#" + "a" * 64, spec_file=no_tasks, agent="", dispatcher_profile=None)
    from_file = run_request("default/t#" + "a" * 64, spec_file=spec, agent="", dispatcher_profile=None, experiment="x")
    assert from_file.publication is not None and from_file.publication.intake.experiment == "x"
    assert from_file.publication.intake.dataset_name is None  # no Hub dataset: the server uses the taskset
    hub = run_request(
        "default/t#" + "a" * 64,
        spec_file=spec,
        agent="",
        dispatcher_profile=None,
        experiment="x",
        dataset=("datacurve/deep-swe", "sha256:" + "b" * 64),
    )
    assert hub.publication is not None
    assert (hub.publication.intake.dataset_name, hub.publication.intake.dataset_version) == (
        "datacurve/deep-swe",
        "sha256:" + "b" * 64,
    )


@pytest.mark.parametrize(
    ("sandbox", "expected"),
    [
        (SandboxSelection(cpus=4, storage_mb=20_480), {"cpus": 4, "storage_mb": 20_480}),
        (SandboxSelection(profile="harbor-open"), {"profile": "harbor-open"}),
        (None, None),
    ],
)
def test_agent_request_carries_the_sandbox_selection(sandbox, expected):
    request = run_request(
        "default/full#" + "a" * 64,
        spec_file=None,
        agent="oracle",
        dispatcher_profile=None,
        sandbox=sandbox,
    )
    assert request.model_dump(mode="json", exclude_none=True)["execution"].get("sandbox") == expected


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--sandbox-profile", "harbor-open"), ("--cpus", "4"), ("--memory-mb", "4"), ("--storage-mb", "4")],
)
def test_sandbox_flags_are_rejected_with_a_spec_file(tmp_path, flag, value):
    spec = tmp_path / "run.json"
    spec.write_text("{}")
    app = typer.Typer()

    def no_client() -> NemoClient:
        raise AssertionError("no request is made")

    add_runs_commands(app, no_client)
    result = CliRunner().invoke(app, ["runs", "submit", "--spec-file", str(spec), flag, value])
    assert result.exit_code == 2
    assert "don't apply with --spec-file" in " ".join(result.output.replace("│", " ").split())


TB = "https://github.com/harbor-framework/terminal-bench.git"
SHA = "2b0442c3" + "0" * 32  # what Harbor's ls-remote gives an annotated tag: the tag object
COMMIT = "9f1e7a55" + "0" * 32  # the commit the download checks out


class FakeGitRegistry:
    """Harbor's ``GitRepoRegistryClient``: three tasks under ``tasks/`` at ``SHA``; records what it downloads."""

    downloads: list[list[str]] = []
    commits: list[str | None] | None = None  # per selected task; default COMMIT for each

    repos: list = []

    def __init__(self, repo):
        self.repo = repo
        self._task_client = self
        FakeGitRegistry.repos.append(repo)

    async def get_dataset_metadata(self, name):
        from harbor.models.task.id import GitTaskId

        names = ["sum-three", "bun-sourcemap-leak", "greet-universe"]
        return SimpleNamespace(
            task_ids=[GitTaskId(git_url=self.repo.git_url, git_commit_id=SHA, path=Path("tasks") / n) for n in names]
        )

    async def download_tasks(self, task_ids, output_dir, export):
        FakeGitRegistry.downloads.append([t.get_name() for t in task_ids])
        paths = [output_dir / t.get_name() for t in task_ids]
        commits = FakeGitRegistry.commits or [COMMIT] * len(paths)
        results = [SimpleNamespace(path=p, resolved_git_commit_id=c) for p, c in zip(paths, commits, strict=True)]
        return SimpleNamespace(paths=paths, results=results)


@pytest.fixture
def git_registry(monkeypatch):
    from harbor.registry.client import git_repo

    FakeGitRegistry.downloads = []
    FakeGitRegistry.repos = []
    FakeGitRegistry.commits = None
    monkeypatch.setattr(git_repo, "GitRepoRegistryClient", FakeGitRegistry)
    return FakeGitRegistry


def test_git_source_downloads_only_the_selected_tasks(git_registry, tmp_path):
    name, sha, dirs, total = download_git_tasks(f"{TB}@v3.0.0", "sum-three,0", tmp_path)
    # The checked-out commit, not the tag object an annotated tag's ls-remote gives.
    assert (name, sha, total) == ("harbor-framework/terminal-bench", COMMIT, 3)
    assert dirs == [tmp_path / "sum-three", tmp_path / "bun-sourcemap-leak"]  # name order: index 0 is bun-…
    assert git_registry.downloads == [["sum-three", "bun-sourcemap-leak"]]


@pytest.mark.parametrize("second", ["a" * 40, None], ids=["another-commit", "unresolved"])
def test_git_source_needs_one_commit_for_every_task(git_registry, tmp_path, second):
    git_registry.commits = [COMMIT, second]
    with pytest.raises(RuntimeError, match="not one commit"):
        download_git_tasks(f"{TB}@v3.0.0", "sum-three,0", tmp_path)


def test_git_source_needs_a_ref():
    check_git_source(f"{TB}@v3.0.0")
    check_git_source("https://github.com/harbor-framework/terminal-bench/tree/2b0442c3/tasks")
    with pytest.raises(ValueError, match="names no tag or commit"):
        check_git_source(TB)
    result = CliRunner().invoke(_runs_app(), ["runs", "submit", "--harbor-git", TB])
    assert result.exit_code == 2
    assert "names no tag or commit" in " ".join(result.output.replace("│", " ").split())


@pytest.mark.parametrize(
    ("other", "flags"),
    [
        (["--harbor-dir", "."], "--harbor-dir and --harbor-git"),
        (["--harbor-hub", "a/b@1"], "--harbor-hub and --harbor-git"),
    ],
)
def test_git_source_excludes_the_other_sources(other, flags):
    result = CliRunner().invoke(_runs_app(), ["runs", "submit", *other, "--harbor-git", f"{TB}@v3.0.0"])
    assert result.exit_code == 2
    assert f"not {flags}" in " ".join(result.output.replace("│", " ").split())


def test_list_tasks_reads_the_git_repo(git_registry):
    client = NemoClient(base_url="http://platform.test", workspace="default")
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "submit", "--harbor-git", f"{TB}@v3.0.0", "--list-tasks"])
    assert result.exit_code == 0, result.output
    assert result.output.split("\n")[1:4] == ["  0  bun-sourcemap-leak", "  1  greet-universe", "  2  sum-three"]
    assert git_registry.downloads == []


def test_a_tree_url_reads_tasks_from_its_directory(git_registry):
    result = CliRunner().invoke(
        _listing_app(),
        ["runs", "submit", "--harbor-git", "https://github.com/acme/evals/tree/v2/suites/smoke", "--list-tasks"],
    )
    assert result.exit_code == 0, result.output
    (repo,) = git_registry.repos
    assert (repo.git_url, repo.ref, repo.subdir) == ("https://github.com/acme/evals", "v2", "suites/smoke")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            subprocess.CalledProcessError(128, ["git", "ls-remote", TB, "v9"], b"", b"fatal: repository not found\n"),
            "Error: git ls-remote failed: fatal: repository not found",
        ),
        (FileNotFoundError(2, "No such file or directory", "git"), "Error: --harbor-git needs git on PATH"),
    ],
)
def test_a_failed_git_command_is_an_error_not_a_traceback(monkeypatch, error, message):
    from harbor.registry.client import git_repo

    class UnreachableRepo(FakeGitRegistry):
        async def get_dataset_metadata(self, name):
            raise error

    monkeypatch.setattr(git_repo, "GitRepoRegistryClient", UnreachableRepo)
    client = NemoClient(base_url="http://platform.test", workspace="default")
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "submit", "--harbor-git", f"{TB}@v9", "--list-tasks"])
    assert result.exit_code == 1
    assert message in result.output
    assert isinstance(result.exception, SystemExit)  # typer.Exit, not the git error


def _listing_app() -> typer.Typer:
    client = NemoClient(base_url="http://platform.test", workspace="default")
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return app


def _runs_app() -> typer.Typer:
    def no_client() -> NemoClient:
        raise AssertionError("no request is made")

    app = typer.Typer()
    add_runs_commands(app, no_client)
    return app


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
            ref, spec_file=None, agent="oracle", dispatcher_profile=None, idempotency_key="k-123"
        ).idempotency_key
        == "k-123"
    )
    spec = tmp_path / "run.json"
    body = {
        "spec": {"tasks": "default/full", "target": {"kind": "harbor"}},
        "execution": {},
        "idempotency_key": spec_key,
    }
    spec.write_text(json.dumps(body))
    explicit = run_request(ref, spec_file=spec, agent="", dispatcher_profile=None, idempotency_key="k-123")
    assert explicit.idempotency_key == "k-123"
    default = run_request(ref, spec_file=spec, agent="", dispatcher_profile=None)
    assert default.idempotency_key == (spec_key or default.idempotency_key) and default.idempotency_key


def test_spec_selection_ignores_a_publication_that_experiment_replaces(tmp_path):
    spec = tmp_path / "run.json"
    body = {
        "spec": {"tasks": "default/suite#" + "e" * 64, "target": {"kind": "harbor"}},
        "execution": {"sandbox_policy": "p"},
        "publication": {"intake": {"experiment": ""}},  # invalid on its own; --experiment replaces it
    }
    spec.write_text(json.dumps(body))
    member = "default/a#" + "1" * 64
    tasks = SimpleNamespace(
        retrieve=lambda name, revision, workspace: SimpleNamespace(revision=1, spec=SimpleNamespace())
    )
    tasksets = SimpleNamespace(retrieve=lambda name, revision, workspace: SimpleNamespace(tasks=[member]))
    evaluator = cast(Evaluator, SimpleNamespace(tasks=tasks, tasksets=tasksets))
    assert _select_from_spec(evaluator, spec, None, "default") == ([member], 1, "suite")


def test_spec_selection_needs_a_taskset(tmp_path):
    spec = tmp_path / "run.json"
    spec.write_text(
        json.dumps({"spec": {"tasks": ["default/a#" + "1" * 64], "target": {"kind": "harbor"}}, "execution": {}})
    )
    with pytest.raises(ValueError, match="must name a taskset"):
        _select_from_spec(cast(Evaluator, SimpleNamespace()), spec, None, "default")


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
        "run-0123456789abcdef-task-0": job("run-0123456789abcdef-task-0", "completed", "hello/greet-universe")
        | {"status_details": {"publication": {"status": "completed", "evaluation_id": "e", "trial_count": 2}}},
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


def test_status_shows_what_each_task_published():
    views = [
        TaskStateView(0, "a", "succeeded", "j0", publication={"status": "completed", "trial_count": 2}),
        TaskStateView(1, "b", "succeeded", "j1", publication={"status": "error", "error": "intake 503"}),
        TaskStateView(2, "c", "registered (active)", "j2"),
    ]
    lines = _status_table(RUN, views)
    assert lines[-5].endswith("published")
    assert lines[-4].endswith("✓ 2 trials") and lines[-3].endswith("✗ intake 503") and lines[-2].endswith("  -")
    failed = TaskJobView(1, "j1", "b", "completed", publication={"status": "error", "error": "intake 503"})
    assert _view_json(failed)["publication"] == {"status": "error", "error": "intake 503"}  # runs results --json


def test_status_counts_done_and_failed():
    views = [
        TaskStateView(0, "a", "succeeded", "j0"),
        TaskStateView(1, "b", "failed", "j1", "job_failed: exit 1"),
        TaskStateView(2, "c", "expired", None),
        TaskStateView(3, "d", "registered (active)", "j3"),
        TaskStateView(4, "e", "queued", None),
    ]
    assert _status_table(RUN, views)[-1] == "3/5 done (1 succeeded, 2 failed)"


@pytest.mark.parametrize(
    ("task", "state"),
    [
        ({"phase": "image_ready"}, "queued"),
        ({"phase": "image_ready", "job": "run-x-task-0"}, "queued (creating)"),
        ({"phase": "registered", "job": "run-x-task-0", "job_status": "active"}, "registered (active)"),
        ({"phase": "registered", "job": "run-x-task-0", "job_status": "pending"}, "registered (pending)"),
        ({"phase": "registered", "job": "run-x-task-0"}, "registered"),  # no status read yet
        ({"phase": "failed", "job": "run-x-task-0", "job_status": "error", "error": "job_failed"}, "failed"),
        ({"phase": "expired"}, "expired"),
        # A settled task shows its Job's status only when the phase doesn't imply it.
        ({"phase": "succeeded", "job": "run-x-task-0", "job_status": "completed"}, "succeeded"),
        ({"phase": "cancelled", "job": "run-x-task-0", "job_status": "completed"}, "cancelled (completed)"),
        ({"phase": "expired", "job": "run-x-task-0", "job_status": "cancelled"}, "expired"),
        (
            {"phase": "failed", "job": "run-x-task-0", "job_status": "completed", "error": "no_result"},
            "failed (completed)",
        ),
    ],
)
def test_task_state_names_the_queue_and_the_jobs_status(task, state):
    assert task_state(task) == state


@pytest.mark.parametrize(
    ("progress", "line"),
    [
        (None, "no tasks yet; the run controller creates them on its next pass"),
        (
            {
                "tasks": 113,
                "members": {"image_ready": 60, "registered": 20, "succeeded": 30, "failed": 3},
                "queued": 60,
                "task_jobs": {"pending": 2, "active": 18},
            },
            "113 tasks: 60 queued, 20 running (18 active, 2 pending), 30 succeeded, 3 failed",
        ),
        (
            {"tasks": 4, "members": {"image_ready": 3, "registered": 1}, "queued": 1, "task_jobs": {}},
            "4 tasks: 1 queued, 2 creating, 1 running",
        ),
        (
            {"tasks": 1, "members": {"cancelled": 1}, "queued": 0, "task_jobs": {}},
            "1 task: 1 cancelled",
        ),
    ],
)
def test_progress_line_counts_queued_creating_running_and_settled(progress, line):
    assert _progress_line(progress) == line


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
    lines = _status_table(run, [TaskStateView(0, "a", "succeeded", "j0")], bundle)
    assert lines[3] == "  status    completed; the run's result is ready (phase completed, result ready)"
    assert lines[5] == "  bundle    run-x-bundle-1 (attempt 1 of 3): completed"


MODULE_ERROR = "/app/.venv/bin/python: No module named nemo_evals.tasks.taskset_bundle"


def test_a_failed_bundle_says_tasks_finished_and_shows_its_own_error():
    progress = {"tasks": 1, "members": {"succeeded": 1}, "queued": 0, "task_jobs": {}}
    error = "bundle Job run-x-bundle-3 ended error"
    run = {**RUN, "phase": "completed", "result_state": "failed", "error": error, "progress": progress}
    bundle = BundleJobView("run-x-bundle-3", 3, "error", MODULE_ERROR)
    lines = _status_table(run, [TaskStateView(0, "a", "succeeded", "j0")], bundle)
    assert lines[3:9] == [
        "  status    tasks finished, but merging their results into the run's result failed"
        " (phase completed, result failed)",
        "  progress  1 task: 1 succeeded",
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
    lines = _status_table(run, [TaskStateView(0, "a", "succeeded", "j0")], BundleJobView("run-x-bundle-1", 1, status))
    assert f"  error     {error}" in lines
    assert not any("no error reported" in line for line in lines)
    assert any(line.strip().startswith("logs:") for line in lines) is (status == "cancelled")


def test_the_status_screen_shows_how_long_the_run_queued():
    started = RUN | {"phase": "running", "started_at": "2026-10-16T12:50:12Z"}
    assert "  started   2026-10-16 12:50:12 (queued 50m12s)" in _status_table(started, [])
    waiting = _status_table(RUN, [], now=datetime(2026, 10, 16, 13, 10, 5, tzinfo=UTC))
    assert waiting[3] == "  queued    1h10m so far"
    never_started = RUN | {"phase": "expired", "error": "no task started within the queue timeout (48 h)"}
    assert not any(line.startswith(("  started", "  queued")) for line in _status_table(never_started, []))


@pytest.mark.parametrize("phase", ["expired", "cancelled"])
@pytest.mark.parametrize(
    "result,words",
    [
        ("ready", "a result over the tasks that finished is ready"),
        ("failed", "merging the finished tasks' results failed"),
        ("none", None),
    ],
)
def test_a_stopped_run_says_whether_its_finished_tasks_have_a_result(phase, result, words):
    run = RUN | {"phase": phase, "result_state": result, "started_at": RUN["accepted_at"]}
    state = f"{phase}; {words}" if words else phase
    assert f"  status    {state} (phase {phase}, result {result})" in _status_table(run, [])


def test_a_running_run_whose_bundle_job_runs_reads_as_merging():
    run = RUN | {"phase": "running", "started_at": RUN["accepted_at"]}
    lines = _status_table(run, [TaskStateView(0, "a", "succeeded", "j0")], BundleJobView("run-x-bundle-1", 1, "active"))
    assert "  status    merging task results (phase running, result -)" in lines


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
        ["tasks", "run-0123456789abcdef"],
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


RUN_ID = "run-0123456789abcdef"
RECORDED_TASKS = [  # GET …/runs/{id}/tasks, as the API returns it (nulls omitted)
    {"task": "task-0", "task_index": 0, "phase": "succeeded", "job": f"{RUN_ID}-task-0", "job_status": "completed"},
    {"task": "task-1", "task_index": 1, "phase": "registered", "job": f"{RUN_ID}-task-1", "job_status": "active"},
    {"task": "task-2", "task_index": 2, "phase": "image_ready", "job": f"{RUN_ID}-task-2"},
    {"task": "task-3", "task_index": 3, "phase": "image_ready"},
    {
        "task": "task-4",
        "task_index": 4,
        "phase": "failed",
        "job": f"{RUN_ID}-task-4",
        "job_status": "error",
        "error": "job_failed: exit 1\ntraceback",
    },
]
RECORDED_PROGRESS = {
    "tasks": 5,
    "members": {"succeeded": 1, "registered": 1, "image_ready": 2, "failed": 1},
    "queued": 1,
    "task_jobs": {"active": 1},
}


def recorded_platform(tasks_page_size=2):
    """Recorded API responses for a running run; returns the client and every request it made."""
    requests: list[httpx.Request] = []
    taskset = {
        "id": "ts",
        "name": "debug-1f9c2a7e",
        "workspace": "default",
        "tasks": [f"default/{name}#" + "1" * 64 for name in ["greet", "sum", "sort", "fizz", "buzz"]],
        "revision": 1,
        "tags": {},
        "created_at": NOW,
        "updated_at": NOW,
    }
    jobs = {
        f"{RUN_ID}-task-0": job(f"{RUN_ID}-task-0", "completed", "greet")
        | {"status_details": {"publication": {"status": "completed", "evaluation_id": "e", "trial_count": 2}}},
        f"{RUN_ID}-task-4": job(f"{RUN_ID}-task-4", "error", "buzz"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith(f"/agent-evaluate/runs/{RUN_ID}"):
            return httpx.Response(200, json=RUN | {"phase": "running", "progress": RECORDED_PROGRESS})
        if path.endswith(f"/agent-evaluate/runs/{RUN_ID}/tasks"):
            number = int(request.url.params["page"])
            data = RECORDED_TASKS[(number - 1) * tasks_page_size : number * tasks_page_size]
            pages = -(-len(RECORDED_TASKS) // tasks_page_size)
            return httpx.Response(
                200, json=page(data) | {"pagination": page(data)["pagination"] | {"page": number, "total_pages": pages}}
            )
        if "/tasksets/debug-1f9c2a7e" in path:
            return httpx.Response(200, json=taskset)
        if "/jobs/" in path:
            name = path.rsplit("/", 1)[1]
            return httpx.Response(200, json=jobs[name]) if name in jobs else httpx.Response(404, json={"detail": "x"})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return client, requests


def test_status_reads_the_run_and_its_tasks_with_no_jobs_call_for_a_queued_or_running_task():
    client, requests = recorded_platform()
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "status", RUN_ID, "--workspace", "default"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[3:5] == [
        "  status    running (phase running, result -)",
        "  progress  5 tasks: 1 queued, 1 creating, 1 running (1 active), 1 succeeded, 1 failed",
    ]
    assert [line.split() for line in lines[6:12]] == [
        ["#", "task", "state", "Job", "published"],
        ["0", "greet", "succeeded", f"{RUN_ID}-task-0", "✓", "2", "trials"],
        ["1", "sum", "registered", "(active)", f"{RUN_ID}-task-1", "-"],
        ["2", "sort", "queued", "(creating)", f"{RUN_ID}-task-2", "-"],
        ["3", "fizz", "queued", "-", "-"],
        ["4", "buzz", "failed", f"{RUN_ID}-task-4", "-", "job_failed:", "exit", "1"],
    ]
    assert lines[-1] == "2/5 done (1 succeeded, 1 failed)"
    task_pages = [r.url.params["page"] for r in requests if r.url.path.endswith("/tasks")]
    assert task_pages == ["1", "2", "3"]
    jobs_read = [r.url.path.rsplit("/", 1)[1] for r in requests if "/jobs/" in r.url.path]
    assert [name for name in jobs_read if "-task-" in name] == [f"{RUN_ID}-task-0", f"{RUN_ID}-task-4"]


def test_status_json_lists_each_tasks_state():
    client, _ = recorded_platform()
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    result = CliRunner().invoke(app, ["runs", "status", RUN_ID, "--workspace", "default", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["run"]["progress"] == RECORDED_PROGRESS
    assert body["tasks"][1]["job_status"] == "active"  # the raw status, beside the state cell
    assert [(t["index"], t["task"], t["state"]) for t in body["tasks"]] == [
        (0, "greet", "succeeded"),
        (1, "sum", "registered (active)"),
        (2, "sort", "queued (creating)"),
        (3, "fizz", "queued"),
        (4, "buzz", "failed"),
    ]


def invoke_tasks(args, response):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return CliRunner().invoke(app, ["runs", "tasks", RUN_ID, "--workspace", "default", *args]), requests


def test_tasks_lists_the_api_page_with_each_tasks_state_and_no_jobs_call():
    body = run_page(RECORDED_TASKS, total_pages=3, total=45)
    result, requests = invoke_tasks(
        ["--phase", "registered", "--page", "1", "--page-size", "5"], httpx.Response(200, json=body)
    )
    assert result.exit_code == 0, result.output
    (request,) = requests
    assert request.url.path == f"/apis/evals/v2/workspaces/default/agent-evaluate/runs/{RUN_ID}/tasks"
    assert dict(request.url.params) == {"phase": "registered", "page": "1", "page_size": "5"}
    lines = result.output.splitlines()
    assert lines[0].split() == ["#", "task", "state", "Job"]
    assert lines[1].split() == ["0", "task-0", "succeeded", f"{RUN_ID}-task-0"]
    assert lines[2].split() == ["1", "task-1", "registered", "(active)", f"{RUN_ID}-task-1"]
    assert lines[3].split() == ["2", "task-2", "queued", "(creating)", f"{RUN_ID}-task-2"]
    assert lines[4].split() == ["3", "task-3", "queued", "-"]
    assert lines[5].split() == ["4", "task-4", "failed", f"{RUN_ID}-task-4", "job_failed:", "exit", "1"]
    assert lines[6] == "page 1 of 3 (45 tasks); next: --page 2"


def test_tasks_without_members_and_as_json():
    result, _ = invoke_tasks([], httpx.Response(200, json=run_page([])))
    assert result.exit_code == 0 and result.output == "No tasks.\n"
    body = run_page(RECORDED_TASKS)
    result, requests = invoke_tasks(["--json"], httpx.Response(200, json=body))
    assert json.loads(result.output) == body
    assert dict(requests[0].url.params) == {"page": "1", "page_size": "100"}


def test_tasks_of_an_unknown_run_exits_with_a_message():
    result, _ = invoke_tasks([], httpx.Response(404, json={"detail": "Taskset run not found"}))
    assert result.exit_code == 1 and result.output == f"Error: run default/{RUN_ID} not found\n"


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


def invoke_cancel(args, response):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    client = NemoClient(
        base_url="http://platform.test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    app = typer.Typer()
    add_runs_commands(app, lambda: client)
    return CliRunner().invoke(app, ["runs", "cancel", "run-aaaa", "--workspace", "ac9", *args]), requests


def test_cancel_posts_the_intent_and_says_how_to_follow_the_run():
    run = listed_run("run-aaaa") | {"cancel_requested_at": "2026-10-17T12:30:00Z"}
    result, requests = invoke_cancel([], httpx.Response(200, json=run))
    assert result.exit_code == 0
    (request,) = requests
    assert request.method == "POST"
    assert request.url.path == "/apis/evals/v2/workspaces/ac9/agent-evaluate/runs/run-aaaa/cancel"
    assert result.output.splitlines() == [
        "Cancel requested for run-aaaa at 2026-10-17T12:30:00Z (phase running).",
        "Follow it with: nemo evals runs status run-aaaa --workspace ac9",
    ]
    result, _ = invoke_cancel(["--json"], httpx.Response(200, json=run))
    assert json.loads(result.output) == run


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404, json={"detail": "Taskset run not found"}), "run ac9/run-aaaa not found"),
        (
            httpx.Response(409, json={"detail": "run run-aaaa already finished: completed"}),
            'cancelling run run-aaaa failed (409): "run run-aaaa already finished: completed"',
        ),
    ],
)
def test_cancel_failure_exits_with_the_reason(response, message):
    result, _ = invoke_cancel([], response)
    assert result.exit_code == 1 and message in result.output


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
