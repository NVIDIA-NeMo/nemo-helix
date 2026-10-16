# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals runs submit``: publish selected Harbor tasks, build their images, and submit a Kubernetes run.

Selection and publication happen here, not in the API: the API would hold a request open for downloads,
uploads and builds. The CLI publishes a taskset of only the selected tasks, pinned by digest, so the run
evaluates exactly what was published even if another publisher writes the same names later.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer
from nemo_evals.api.fields import TaskRef, TasksetRef
from nemo_evals.api.schemas import AgentEvalResult, Task, TasksetInput
from nemo_evals.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evals.api.taskset_run_schemas import HarborTaskJobsExecutionInput, TasksetRunInput
from nemo_evals.image_builds.models import TERMINAL_BUILD_STATUSES, ImageBuild
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, HarborBuiltinAgentSource, HarborRunnerTarget
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.taskset_runs.controller import MAX_BUNDLE_ATTEMPTS
from nemo_evals.taskset_runs.identifiers import bundle_job_name, task_job_name
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.jobs.client import JobsClient

COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml")
RETRY_ATTEMPTS = 5
_NAME_UNSAFE = re.compile(r"[^a-z0-9-]+")


class SelectionError(ValueError):
    """``--tasks`` names a task that isn't there, or the same task twice."""


@dataclass(frozen=True)
class Candidate:
    """A task that can be selected: its position in ref order, its name and its native Harbor id."""

    index: int
    name: str
    native_id: str


def select_tasks(candidates: Sequence[Candidate], tokens: str | None) -> list[Candidate]:
    """Pick tasks by index (``0,5``) or name (``hello/sum-three`` or ``sum-three``); all when ``tokens`` is None.

    Raises:
        SelectionError: An index is out of range, a name is unknown or ambiguous, or a task repeats.
    """
    if tokens is None:
        return list(candidates)
    selected: list[Candidate] = []
    for token in (part.strip() for part in tokens.split(",")):
        if not token:
            continue
        if token.isdigit():
            index = int(token)
            if index >= len(candidates):
                raise SelectionError(f"task index {index} is out of range 0..{len(candidates) - 1}")
            match = candidates[index]
        else:
            matches = [
                c
                for c in candidates
                if token in (c.native_id, c.name)
                or c.native_id.rpartition("/")[2] == token
                or c.name.endswith(f"-{token}")
            ]
            if not matches:
                raise SelectionError(f"no task named {token!r}; run with --list-tasks to see the names")
            if len({c.index for c in matches}) > 1:
                raise SelectionError(f"{token!r} matches several tasks: {', '.join(c.native_id for c in matches)}")
            match = matches[0]
        if match in selected:
            raise SelectionError(f"task {match.native_id} is selected twice")
        selected.append(match)
    if not selected:
        raise SelectionError("--tasks selects no task")
    return selected


def entity_name(text: str) -> str:
    """A task or taskset name the platform accepts: lowercase letters, digits and dashes."""
    return _NAME_UNSAFE.sub("-", text.lower()).strip("-")[:63] or "task"


def selection_suffix(refs: Sequence[str]) -> str:
    """Eight hex characters identifying a set of pinned task refs, independent of their order."""
    return hashlib.sha256("\n".join(sorted(refs)).encode()).hexdigest()[:8]


def is_compose_task(task_dir: Path) -> bool:
    return any((task_dir / "environment" / name).is_file() for name in COMPOSE_FILENAMES)


def needs_image_build(config: dict[str, Any]) -> bool:
    """True when the task builds its agent image from a Dockerfile (no ``[environment] docker_image``)."""
    environment = config.get("environment") or {}
    return not environment.get("docker_image")


def download_hub_tasks(dataset: str, names: str | None, output_dir: Path) -> tuple[str, list[Path]]:
    """Download the selected tasks of Harbor Hub dataset ``org/name@ref`` into ``output_dir``.

    Returns the dataset's name and the task directories, in the dataset's order.
    """
    from harbor.registry.client.package import PackageDatasetClient

    client = PackageDatasetClient()
    metadata = asyncio.run(client.get_dataset_metadata(dataset))
    ids = sorted(metadata.task_ids, key=lambda task_id: task_id.get_name())
    candidates = [Candidate(i, entity_name(t.get_name().rpartition("/")[2]), t.get_name()) for i, t in enumerate(ids)]
    chosen = [ids[c.index] for c in select_tasks(candidates, names)]
    result = asyncio.run(client._task_client.download_tasks(task_ids=chosen, output_dir=output_dir, export=True))
    return metadata.name, list(result.paths)


def list_hub_tasks(dataset: str) -> list[str]:
    from harbor.registry.client.package import PackageDatasetClient

    metadata = asyncio.run(PackageDatasetClient().get_dataset_metadata(dataset))
    return sorted(task_id.get_name() for task_id in metadata.task_ids)


def publish_tasks(
    evaluator: Evaluator, task_dirs: Sequence[Path], *, prefix: str, workspace: str, echo: Callable[[str], None]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Publish each task directory as ``<prefix>-<dir>``; return pinned refs and each task's ``task.toml`` config."""
    from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import discover_harbor_tasks

    refs: list[str] = []
    configs: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs:
        if is_compose_task(task_dir):
            raise SelectionError(f"{task_dir.name} is a compose task; Kubernetes runs don't support them yet")
        (source,) = discover_harbor_tasks(task_dir)
        name = entity_name(f"{prefix}-{task_dir.name}")
        task = evaluator.tasks.replace(name, task=source, workspace=workspace)
        ref = pinned_task_ref(evaluator, workspace, name, task.revision)
        refs.append(ref)
        configs[ref] = _task_config(task)
        echo(f"Published task {ref}")
    return refs, configs


def pinned_task_ref(evaluator: Evaluator, workspace: str, name: str, revision: int) -> str:
    """``workspace/name#<content digest>`` of revision ``revision``."""
    for item in evaluator.tasks.list_revisions(name, workspace=workspace, page_size=100).data:
        if item.revision == revision and item.content_hash:
            return f"{workspace}/{name}#{item.content_hash}"
    raise RuntimeError(f"task {workspace}/{name} has no revision {revision} with a content digest")


def build_images(
    evaluator: Evaluator,
    refs: Sequence[str],
    configs: dict[str, dict[str, Any]],
    *,
    poll_seconds: float,
    echo: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Request (idempotent) and wait for the image build of every task that builds from a Dockerfile.

    Raises:
        RuntimeError: A build failed.
    """
    pending: dict[str, ImageBuild] = {}
    for ref in refs:
        if not needs_image_build(configs.get(ref, {})):
            continue
        workspace, name, digest = _split_ref(ref)
        pending[ref] = evaluator.tasks.build_images(name, revision=digest, workspace=workspace)
    while any(build.status not in TERMINAL_BUILD_STATUSES for build in pending.values()):
        sleep(poll_seconds)
        for ref, build in pending.items():
            if build.status not in TERMINAL_BUILD_STATUSES:
                workspace, name, digest = _split_ref(ref)
                pending[ref] = evaluator.tasks.get_image_build(name, build.id, revision=digest, workspace=workspace)
    failed = {ref: build for ref, build in pending.items() if build.status != "ready"}
    images = sum(len(build.images) for build in pending.values())
    echo(f"Image builds: {len(pending) - len(failed)} tasks ready ({images} images)")
    if failed:
        detail = "; ".join(f"{ref}: {build.error}" for ref, build in failed.items())
        raise RuntimeError(f"image builds failed: {detail}")


def _task_config(task: Task) -> dict[str, Any]:
    return dict(task.spec.config) if isinstance(task.spec, HarborTaskDefinition) else {}


def _split_ref(ref: str) -> tuple[str, str, str]:
    head, _, digest = ref.partition("#")
    workspace, _, name = head.partition("/")
    return workspace, name, digest


def submit_run(
    client: NemoClient,
    body: dict[str, Any],
    *,
    workspace: str,
    echo: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST the run, retrying a 503 (task Jobs not all created yet) with the same idempotency key."""
    url = f"{client.base_url.rstrip('/')}/apis/evals/v2/workspaces/{workspace}/agent-evaluate/runs"
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        response = client.http_client.post(url, json=body, headers=client.request_headers(None, url=url))
        if response.status_code == 503 and attempt < RETRY_ATTEMPTS:
            echo(f"{response.json().get('detail')}; retrying ({attempt}/{RETRY_ATTEMPTS - 1})")
            sleep(2**attempt)
            continue
        if response.status_code >= 400:
            raise RuntimeError(f"submit failed ({response.status_code}): {_detail(response)}")
        return response.json()
    raise AssertionError("unreachable")


def _detail(response: httpx.Response) -> str:
    try:
        return json.dumps(response.json().get("detail"))
    except ValueError:
        return response.text


def run_request(
    taskset_ref: str,
    *,
    spec_file: Path | None,
    agent: str,
    sandbox_policy: str,
    dispatcher_profile: str | None,
) -> TasksetRunInput:
    """The run request: ``--spec-file``'s with its tasks replaced, or ``--agent`` with default execution."""
    if spec_file is not None:
        request = TasksetRunInput.model_validate(json.loads(spec_file.read_text()))
        spec = request.spec.model_copy(update={"tasks": TasksetRef(taskset_ref)})
        data = request.model_dump(mode="json", exclude={"spec"}) | {"spec": spec.model_dump(mode="json")}
        data["idempotency_key"] = data.get("idempotency_key") or str(uuid.uuid4())
        return TasksetRunInput.model_validate(data)
    execution: dict[str, Any] = {"sandbox_policy": sandbox_policy}
    if dispatcher_profile is not None:
        execution["dispatcher_profile"] = dispatcher_profile
    return TasksetRunInput(
        spec=AgentEvalInputSpec(
            tasks=TasksetRef(taskset_ref), target=HarborRunnerTarget(source=HarborBuiltinAgentSource(name=agent))
        ),
        execution=HarborTaskJobsExecutionInput.model_validate(execution),
        idempotency_key=str(uuid.uuid4()),
    )


def add_runs_commands(app: typer.Typer, client_factory: Callable[[], NemoClient]) -> None:
    """Register ``runs submit`` under ``app``."""
    runs = typer.Typer(name="runs", help="Kubernetes taskset runs: one Job per Harbor task.", no_args_is_help=True)
    app.add_typer(runs)
    add_progress_commands(runs, client_factory)

    @runs.command("submit")
    def submit(
        harbor_dir: Path | None = typer.Option(None, "--harbor-dir", help="Local Harbor dataset directory."),
        harbor_hub: str | None = typer.Option(None, "--harbor-hub", help="Harbor Hub dataset, org/name@ref."),
        spec_file: Path | None = typer.Option(None, "--spec-file", help="TasksetRunInput JSON naming a taskset."),
        tasks: str | None = typer.Option(None, "--tasks", help="Indexes or names, comma-separated; default all."),
        agent: str = typer.Option("oracle", "--agent", help="Harbor built-in agent (without --spec-file)."),
        sandbox_policy: str = typer.Option("harbor-standard", "--sandbox-policy", help="Operator sandbox profile."),
        dispatcher_profile: str | None = typer.Option(None, "--dispatcher-profile", help="Jobs profile for task Jobs."),
        no_build_images: bool = typer.Option(False, "--no-build-images", help="Don't request or wait for builds."),
        list_tasks: bool = typer.Option(False, "--list-tasks", help="Print the selectable tasks and exit."),
        workspace: str = typer.Option("default", "--workspace", help="Workspace for tasks, taskset and run."),
        poll_seconds: float = typer.Option(10.0, "--poll-seconds", help="Image-build polling interval."),
    ) -> None:
        """Publish the selected tasks as a taskset, build their images, and submit one task Job per task."""
        if sum(source is not None for source in (harbor_dir, harbor_hub, spec_file)) != 1:
            raise typer.BadParameter("pass exactly one of --harbor-dir, --harbor-hub or --spec-file")
        echo = typer.echo
        try:
            if list_tasks:
                _print_tasks(client_factory, harbor_dir, harbor_hub, spec_file, workspace)
                return
            client = client_factory()
            evaluator = Evaluator.from_client(client)
            with tempfile.TemporaryDirectory(prefix="nemo-runs-") as scratch:
                if spec_file is not None:
                    refs, configs, total, prefix = _select_from_spec(evaluator, spec_file, tasks, workspace)
                else:
                    if harbor_hub is not None:
                        prefix, task_dirs = download_hub_tasks(harbor_hub, tasks, Path(scratch))
                        prefix = prefix.rpartition("/")[2]
                        total = len(list_hub_tasks(harbor_hub))
                    else:
                        assert harbor_dir is not None
                        all_dirs = _task_dirs(harbor_dir)
                        candidates = [Candidate(i, d.name, _native_id(d)) for i, d in enumerate(all_dirs)]
                        task_dirs = [all_dirs[c.index] for c in select_tasks(candidates, tasks)]
                        prefix, total = harbor_dir.resolve().name, len(all_dirs)
                    refs, configs = publish_tasks(evaluator, task_dirs, prefix=prefix, workspace=workspace, echo=echo)
                taskset = entity_name(
                    f"{prefix}-sel-{selection_suffix(refs)}" if spec_file else f"{prefix}-{selection_suffix(refs)}"
                )
                evaluator.tasksets.replace(
                    taskset, taskset=TasksetInput(tasks=[TaskRef(ref) for ref in refs]), workspace=workspace
                )
                revision = evaluator.tasksets.list_revisions(taskset, workspace=workspace, page_size=1).data[0]
                taskset_ref = f"{workspace}/{taskset}#{revision.content_hash}"
                echo(f"Published taskset {workspace}/{taskset} ({len(refs)} of {total} tasks)")
                if not no_build_images:
                    build_images(evaluator, refs, configs, poll_seconds=poll_seconds, echo=echo)
            request = run_request(
                taskset_ref,
                spec_file=spec_file,
                agent=agent,
                sandbox_policy=sandbox_policy,
                dispatcher_profile=dispatcher_profile,
            )
            status = submit_run(
                client, request.model_dump(mode="json", exclude_none=True), workspace=workspace, echo=echo
            )
        except (SelectionError, RuntimeError, ValueError) as exc:
            echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        echo(f"Run {status['run_id']} accepted: {len(refs)} task Jobs")
        for line in status.get("warnings", []):
            echo(f"  {line}")


def _task_dirs(dataset: Path) -> list[Path]:
    if (dataset / "task.toml").is_file():
        return [dataset]
    return sorted(
        (d for d in dataset.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / "task.toml").is_file()),
        key=lambda d: entity_name(f"{dataset.resolve().name}-{d.name}"),
    )


def _native_id(task_dir: Path) -> str:
    import tomllib

    config = tomllib.loads((task_dir / "task.toml").read_text())
    task = config.get("task") or {}
    return str(task.get("name") or config.get("name") or task_dir.name)


def _select_from_spec(
    evaluator: Evaluator, spec_file: Path, tokens: str | None, workspace: str
) -> tuple[list[str], dict[str, dict[str, Any]], int, str]:
    """The pinned refs of ``--spec-file``'s taskset, narrowed by ``--tasks``; tasks are not re-uploaded."""
    request = TasksetRunInput.model_validate(json.loads(spec_file.read_text()))
    assert isinstance(request.spec.tasks, TasksetRef)
    ref = request.spec.tasks.root
    head, _, fragment = ref.partition("#")
    taskset_workspace, _, name = head.rpartition("/")
    taskset = evaluator.tasksets.retrieve(name, revision=fragment or None, workspace=taskset_workspace or workspace)
    members = sorted(str(member.root if isinstance(member, TaskRef) else member) for member in taskset.tasks)
    candidates = []
    configs: dict[str, dict[str, Any]] = {}
    for index, member in enumerate(members):
        task_workspace, task_name, digest = _split_ref(member)
        task = evaluator.tasks.retrieve(task_name, revision=digest or None, workspace=task_workspace or workspace)
        pinned = member if digest else pinned_task_ref(evaluator, task_workspace, task_name, task.revision)
        configs[pinned] = _task_config(task)
        candidates.append(Candidate(index, pinned, getattr(task.spec, "native_task_id", task_name)))
    selected = select_tasks(candidates, tokens)
    return [c.name for c in selected], configs, len(candidates), name


def _print_tasks(
    client_factory: Callable[[], NemoClient],
    harbor_dir: Path | None,
    harbor_hub: str | None,
    spec_file: Path | None,
    workspace: str,
) -> None:
    if harbor_hub is not None:
        rows = list_hub_tasks(harbor_hub)
    elif harbor_dir is not None:
        rows = [_native_id(d) for d in _task_dirs(harbor_dir)]
    else:
        assert spec_file is not None
        rows = _select_from_spec(Evaluator.from_client(client_factory()), spec_file, None, workspace)[0]
    typer.echo("  #  task")
    for index, row in enumerate(rows):
        typer.echo(f"{index:>3}  {row}")


# --- Progress and results (M2 story 0 bonus; M4 and M5 replace the client-side joins) ------------------------


@dataclass(frozen=True)
class TaskJobView:
    """One task Job of a run: where it is, and its result record once it completed."""

    index: int
    job: str
    task: str | None  # the task's id, from the Job's stored spec; None when the Job doesn't exist
    status: str  # Jobs status, or "missing" when submit hasn't created the Job yet
    job_id: str | None = None
    result: AgentEvalResult | None = None


def get_run(client: NemoClient, workspace: str, run_id: str) -> dict[str, Any]:
    """The stored run status from ``GET …/agent-evaluate/runs/{run_id}``."""
    url = f"{client.base_url.rstrip('/')}/apis/evals/v2/workspaces/{workspace}/agent-evaluate/runs/{run_id}"
    response = client.http_client.get(url, headers=client.request_headers(None, url=url))
    if response.status_code == 404:
        raise SelectionError(f"run {workspace}/{run_id} not found")
    if response.status_code >= 400:
        raise RuntimeError(f"reading run {run_id} failed ({response.status_code}): {_detail(response)}")
    return response.json()


def run_task_jobs(
    client: NemoClient, run: dict[str, Any], *, workspace: str, with_results: bool = False
) -> list[TaskJobView]:
    """Every task Job of ``run``, named ``<run_id>-task-<i>`` for each task of its pinned taskset revision."""
    evaluator = Evaluator.from_client(client)
    jobs = JobsClient.from_client(client)
    taskset_workspace, name, digest = _split_ref(run["taskset_ref"])
    taskset = evaluator.tasksets.retrieve(name, revision=digest, workspace=taskset_workspace)
    views: list[TaskJobView] = []
    for index in range(len(taskset.tasks)):
        job_name = task_job_name(run["run_id"], index)
        try:
            job = jobs.get_job(workspace=workspace, name=job_name).data()
        except NotFoundError:
            views.append(TaskJobView(index, job_name, None, "missing"))
            continue
        tasks = job.spec.get("spec", {}).get("tasks") or [{}]
        status = job.status.value if hasattr(job.status, "value") else str(job.status)
        result = None
        if with_results and status == "completed":
            # Result records carry the Job's name as job_id (NEMO_JOB_ID), not its platform id.
            found = evaluator.agent_eval_results.list(workspace=workspace, job_id=job_name, page_size=1).data
            result = found[0] if found else None
        views.append(TaskJobView(index, job_name, tasks[0].get("id"), status, job.id, result))
    return views


@dataclass(frozen=True)
class ScoreRollup:
    """One score over a run's tasks: count-weighted mean of the tasks' means."""

    name: str
    mean: float | None
    count: int
    nan_count: int
    tasks: int


def rollup_scores(results: Sequence[AgentEvalResult]) -> list[ScoreRollup]:
    """Combine each score name across tasks; scalar scores (no count or mean) are left out of the mean."""
    totals: dict[str, list[float]] = {}  # name -> [weighted sum, count, nan_count, tasks]
    for result in results:
        for score in result.scores.scores:
            total = totals.setdefault(score.name, [0.0, 0, 0, 0])
            total[2] += score.nan_count
            total[3] += 1
            if score.mean is not None and score.count:
                total[0] += score.mean * score.count
                total[1] += score.count
    return [
        ScoreRollup(name, total[0] / total[1] if total[1] else None, int(total[1]), int(total[2]), int(total[3]))
        for name, total in totals.items()
    ]


def newest_bundle_job(client: NemoClient, run_id: str, *, workspace: str) -> tuple[str, str] | None:
    """``(name, status)`` of the run's newest bundle Job, or None before the controller created one."""
    jobs = JobsClient.from_client(client)
    newest = None
    for attempt in range(1, MAX_BUNDLE_ATTEMPTS + 1):
        try:
            job = jobs.get_job(workspace=workspace, name=bundle_job_name(run_id, attempt)).data()
        except NotFoundError:
            break
        newest = (job.name, job.status.value if hasattr(job.status, "value") else str(job.status))
    return newest


def _status_table(
    run: dict[str, Any], views: Sequence[TaskJobView], bundle: tuple[str, str] | None = None
) -> list[str]:
    lines = [f"Run {run['run_id']}  taskset {run['taskset_ref']}  accepted {run['accepted_at']}"]
    bundle_text = f"  bundle {bundle[0]} ({bundle[1]})" if bundle else ""
    error = f"  error: {run['error']}" if run.get("error") else ""
    lines.append(f"phase {run['phase']}  result {run.get('result_state', '-')}{bundle_text}{error}")
    width = max([4, *(len(v.task or "") for v in views)])
    lines.append(f"{'#':>3}  {'task':<{width}}  {'Job':<{len(views[0].job) if views else 3}}  status")
    lines += [f"{v.index:>3}  {(v.task or '-'):<{width}}  {v.job}  {v.status}" for v in views]
    done = [v for v in views if v.status in _TERMINAL]
    failed = [v for v in done if v.status != "completed"]
    lines.append(f"{len(done)}/{len(views)} done ({len(done) - len(failed)} completed, {len(failed)} failed)")
    return lines


def _results_table(views: Sequence[TaskJobView], record: AgentEvalResult | None = None) -> list[str]:
    results = [v.result for v in views if v.result is not None]
    names = list(dict.fromkeys(score.name for result in results for score in result.scores.scores))
    width = max([4, *(len(v.task or "") for v in views)])
    header = f"{'#':>3}  {'task':<{width}}  {'status':<10}  trials  errors"
    lines = ["  ".join([header, *names])]
    for view in views:
        row = f"{view.index:>3}  {(view.task or '-'):<{width}}  {view.status:<10}"
        if view.result is None:
            lines.append(f"{row}  {'-':>6}  {'-':>6}")
            continue
        summary = view.result.summary
        by_name = view.result.scores.scores_by_name
        cells = [_cell(by_name[name].mean if name in by_name else None, len(name)) for name in names]
        trials = summary.trial_count if summary else "-"
        errors = summary.error_count if summary else "-"
        lines.append("  ".join([f"{row}  {trials:>6}  {errors:>6}", *cells]))
    unrecorded = [v.job for v in views if v.status == "completed" and v.result is None]
    if unrecorded:
        lines.append(f"warning: completed without a result record: {', '.join(unrecorded)}")
    if record is not None:  # the run's own record, from the merged bundle
        summary = record.summary
        counts = (
            f"{summary.task_count} tasks, {summary.trial_count} trials, {summary.error_count} errors" if summary else ""
        )
        lines.append(f"run (merged bundle {record.bundle_ref}): {counts}")
        for score in record.scores.scores:
            mean = f"{score.mean:.3f}" if score.mean is not None else "n/a"
            lines.append(f"  {score.name}: {mean} (n={score.count}, nan={score.nan_count})")
        return lines
    trials = sum(r.summary.trial_count for r in results if r.summary)
    errors = sum(r.summary.error_count for r in results if r.summary)
    lines.append(
        f"run (combined here; no merged bundle yet): {len(results)}/{len(views)} tasks with results, {trials} trials, {errors} errors"
    )
    for score in rollup_scores(results):
        mean = f"{score.mean:.3f}" if score.mean is not None else "n/a"
        lines.append(f"  {score.name}: {mean} (n={score.count}, nan={score.nan_count}, tasks={score.tasks})")
    return lines


def _cell(value: float | None, width: int) -> str:
    return f"{value:.2f}".rjust(width) if value is not None else "-".rjust(width)


_TERMINAL = frozenset({"completed", "error", "cancelled"})
_RUN_TERMINAL = frozenset({"completed", "failed", "cancelled", "expired"})


def add_progress_commands(runs: typer.Typer, client_factory: Callable[[], NemoClient]) -> None:
    """Register ``runs status`` and ``runs results`` under ``runs``."""

    @runs.command("status")
    def status(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        workspace: str = typer.Option("default", "--workspace", help="Workspace of the run."),
        watch: bool = typer.Option(False, "--watch", help="Refresh until every task Job is done."),
        poll_seconds: float = typer.Option(15.0, "--poll-seconds", help="Refresh interval with --watch."),
        output_json: bool = typer.Option(False, "--json", help="Print JSON."),
    ) -> None:
        """Show each task Job of a run and its status."""
        client = client_factory()
        try:
            run = get_run(client, workspace, run_id)
            while True:
                views = run_task_jobs(client, run, workspace=workspace)
                bundle = newest_bundle_job(client, run_id, workspace=workspace)
                if output_json:
                    body = {"run": run, "bundle": bundle and {"job": bundle[0], "status": bundle[1]}}
                    typer.echo(json.dumps(body | {"tasks": [_view_json(v) for v in views]}, indent=2))
                else:
                    typer.echo("\n".join(_status_table(run, views, bundle)))
                if not watch or run["phase"] in _RUN_TERMINAL:
                    return
                time.sleep(poll_seconds)
                typer.echo("")
                run = get_run(client, workspace, run_id)
        except (SelectionError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    @runs.command("results")
    def results(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        workspace: str = typer.Option("default", "--workspace", help="Workspace of the run."),
        output_json: bool = typer.Option(False, "--json", help="Print every score of every task as JSON."),
    ) -> None:
        """Show each completed task's scores (every Harbor reward) and a run rollup."""
        client = client_factory()
        try:
            run = get_run(client, workspace, run_id)
            views = run_task_jobs(client, run, workspace=workspace, with_results=True)
            record = None
            if run.get("result_state") == "ready":  # the controller saved the run's own record
                record = Evaluator.from_client(client).agent_eval_results.retrieve(run_id, workspace=workspace)
        except (SelectionError, RuntimeError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if output_json:
            body = {
                "run": run,
                "tasks": [_view_json(v) for v in views],
                "rollup": [score.__dict__ for score in rollup_scores([v.result for v in views if v.result])],
                "result": record.model_dump(mode="json") if record is not None else None,
            }
            typer.echo(json.dumps(body, indent=2, default=str))
            return
        typer.echo("\n".join(_results_table(views, record)))


def _view_json(view: TaskJobView) -> dict[str, Any]:
    data: dict[str, Any] = {"index": view.index, "job": view.job, "task": view.task, "status": view.status}
    if view.result is not None:
        data["result"] = view.result.model_dump(mode="json", include={"name", "scores", "summary", "bundle_ref"})
    return data
