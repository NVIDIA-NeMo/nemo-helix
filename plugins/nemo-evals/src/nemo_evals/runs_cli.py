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
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer
from nemo_evals.api.fields import TaskRef, TasksetRef
from nemo_evals.api.schemas import AgentEvalResult, Task, TasksetInput
from nemo_evals.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.harbor.remote.profile import SandboxSelection
from nemo_evals.image_builds.models import TERMINAL_BUILD_STATUSES, ImageBuild
from nemo_evals.jobs.agent_evaluate import DEFAULT_RESULT_NAME, SUMMARY_FILE_NAME
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, HarborBuiltinAgentSource, HarborRunnerTarget
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.taskset_runs.controller import MAX_BUNDLE_ATTEMPTS
from nemo_evals.taskset_runs.identifiers import bundle_job_name, task_job_name
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError
from nemo_helix_plugin.jobs.archive import safe_extract_tar
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import HelixJobResponse

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


def download_hub_tasks(dataset: str, names: str | None, output_dir: Path) -> tuple[str, str | None, list[Path]]:
    """Download the selected tasks of Harbor Hub dataset ``org/name@ref`` into ``output_dir``.

    Returns the dataset's name, its resolved version (``sha256:…``, also without ``@ref``) and the task
    directories, in the dataset's order.
    """
    from harbor.registry.client.package import PackageDatasetClient

    client = PackageDatasetClient()
    metadata = asyncio.run(client.get_dataset_metadata(dataset))
    ids = sorted(metadata.task_ids, key=lambda task_id: task_id.get_name())
    candidates = [Candidate(i, entity_name(t.get_name().rpartition("/")[2]), t.get_name()) for i, t in enumerate(ids)]
    chosen = [ids[c.index] for c in select_tasks(candidates, names)]
    result = asyncio.run(client._task_client.download_tasks(task_ids=chosen, output_dir=output_dir, export=True))
    return metadata.name, metadata.version, list(result.paths)


def list_hub_tasks(dataset: str) -> list[str]:
    from harbor.registry.client.package import PackageDatasetClient

    metadata = asyncio.run(PackageDatasetClient().get_dataset_metadata(dataset))
    return sorted(task_id.get_name() for task_id in metadata.task_ids)


def check_git_source(source: str) -> None:
    """Reject a ``--harbor-git`` source that Harbor can't parse or that names no tag or commit.

    Raises:
        ValueError: The source isn't a git URL, or has no ``@<ref>``.
    """
    from harbor.registry.client.git_repo import resolve_repo_source

    if resolve_repo_source(source).ref is None:
        raise ValueError(f"{source!r} names no tag or commit; pass <url>@<ref> so the run is reproducible")


def download_git_tasks(source: str, names: str | None, output_dir: Path) -> tuple[str, str, list[Path], int]:
    """Download the selected tasks of git repo ``<url>@<ref>`` (its ``tasks/`` directory) into ``output_dir``.

    Returns the repo's ``org/name``, the commit SHA the ref resolved to, the task directories in name order,
    and how many tasks the repo has.
    """
    from harbor.models.task.id import GitTaskId
    from harbor.registry.client.git_repo import GitRepoRegistryClient, resolve_repo_source

    check_git_source(source)
    repo = resolve_repo_source(source)
    client = GitRepoRegistryClient(repo)
    metadata = _run_git(client.get_dataset_metadata(""))  # no dataset name: every task under tasks/
    ids = sorted(metadata.task_ids, key=lambda task_id: task_id.get_name())
    candidates = [Candidate(i, entity_name(t.get_name()), t.get_name()) for i, t in enumerate(ids)]
    chosen = [ids[c.index] for c in select_tasks(candidates, names)]
    result = _run_git(client._task_client.download_tasks(task_ids=chosen, output_dir=output_dir, export=True))
    sha = ids[0].git_commit_id if isinstance(ids[0], GitTaskId) else None  # every id pins the resolved commit
    assert sha is not None
    return f"{repo.org}/{repo.name}", sha, list(result.paths), len(ids)


def list_git_tasks(source: str) -> list[str]:
    from harbor.registry.client.git_repo import GitRepoRegistryClient, resolve_repo_source

    check_git_source(source)
    metadata = _run_git(GitRepoRegistryClient(resolve_repo_source(source)).get_dataset_metadata(""))
    return sorted(task_id.get_name() for task_id in metadata.task_ids)


def _run_git[T](operation: Coroutine[Any, Any, T]) -> T:
    """Run a Harbor git operation; a failed git command becomes a RuntimeError carrying git's own message.

    Raises:
        RuntimeError: A git command Harbor ran failed (unreachable repo, unknown ref, …), or git isn't installed.
    """
    try:
        return asyncio.run(operation)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        raise RuntimeError(f"{' '.join(exc.cmd[:2])} failed: {stderr.strip() or exc}") from exc
    except FileNotFoundError as exc:
        if exc.filename != "git":
            raise
        raise RuntimeError("--harbor-git needs git on PATH; install git and retry") from exc


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
    dispatcher_profile: str | None,
    sandbox: SandboxSelection | None = None,
    experiment: str | None = None,
    dataset: tuple[str, str | None] | None = None,
) -> TasksetRunInput:
    """The run request: ``--spec-file``'s with its tasks replaced, or ``--agent`` with default execution.

    ``--experiment`` sets ``publication.intake``, replacing the spec file's.
    ``dataset`` is the Harbor Hub dataset or git repo the tasks came from, as ``(name, version or commit)``:
    the Evaluation's dataset instead of the taskset. ``dispatcher_profile`` and ``sandbox`` apply
    to ``--agent`` only; a spec file carries its own ``execution``.
    """
    publication: dict[str, Any] | None = None
    if experiment is not None:
        publication = {"intake": {"experiment": experiment}}
        if dataset is not None:
            name, version = dataset
            publication["intake"] |= {"dataset_name": name} | ({"dataset_version": version} if version else {})
    if spec_file is not None:
        # The selected taskset replaces spec.tasks before validation: with a dataset the file may leave it out.
        raw = json.loads(spec_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("spec") or {}, dict):
            raise ValueError(f"{spec_file}: expected a TasksetRunInput JSON object with an object 'spec'")
        raw["spec"] = {**(raw.get("spec") or {}), "tasks": taskset_ref}
        if publication is not None:
            raw["publication"] = publication  # --experiment replaces the file's, before it's validated
        raw["idempotency_key"] = raw.get("idempotency_key") or str(uuid.uuid4())
        return TasksetRunInput.model_validate(raw)
    execution: dict[str, Any] = {}
    if dispatcher_profile is not None:
        execution["dispatcher_profile"] = dispatcher_profile
    if sandbox is not None:
        execution["sandbox"] = sandbox.model_dump(exclude_none=True)
    return TasksetRunInput.model_validate(
        {
            "spec": AgentEvalInputSpec(
                tasks=TasksetRef(taskset_ref), target=HarborRunnerTarget(source=HarborBuiltinAgentSource(name=agent))
            ).model_dump(mode="json", exclude_none=True),
            "execution": execution,
            "publication": publication,
            "idempotency_key": str(uuid.uuid4()),
        }
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
        harbor_git: str | None = typer.Option(
            None, "--harbor-git", help="Public git repo of Harbor tasks under tasks/, <url>@<tag or commit>."
        ),
        spec_file: Path | None = typer.Option(
            None,
            "--spec-file",
            help="TasksetRunInput JSON: its target and execution, and its taskset unless --harbor-dir, --harbor-hub "
            "or --harbor-git supplies the tasks.",
        ),
        tasks: str | None = typer.Option(None, "--tasks", help="Indexes or names, comma-separated; default all."),
        agent: str = typer.Option("oracle", "--agent", help="Harbor built-in agent (without --spec-file)."),
        sandbox_profile: str | None = typer.Option(
            None, "--sandbox-profile", help="Operator sandbox profile; default: the operator's (without --spec-file)."
        ),
        dispatcher_profile: str | None = typer.Option(None, "--dispatcher-profile", help="Jobs profile for task Jobs."),
        cpus: int | None = typer.Option(
            None, "--cpus", min=1, help="CPUs for every task sandbox, over task.toml (without --spec-file)."
        ),
        memory_mb: int | None = typer.Option(
            None, "--memory-mb", min=1, help="Memory in MB for every task sandbox (without --spec-file)."
        ),
        storage_mb: int | None = typer.Option(
            None, "--storage-mb", min=1, help="Disk in MB for every task sandbox (without --spec-file)."
        ),
        no_build_images: bool = typer.Option(False, "--no-build-images", help="Don't request or wait for builds."),
        list_tasks: bool = typer.Option(False, "--list-tasks", help="Print the selectable tasks and exit."),
        workspace: str | None = typer.Option(
            None, "--workspace", help="Workspace for tasks, taskset and run; default: the configured workspace."
        ),
        poll_seconds: float = typer.Option(10.0, "--poll-seconds", help="Image-build polling interval."),
        experiment: str | None = typer.Option(
            None,
            "--experiment",
            help="Intake Experiment (name) to publish the run to: one Evaluation per run, trials as each task ends.",
        ),
    ) -> None:
        """Publish the selected tasks as a taskset, build their images, and submit one task Job per task."""
        sources = [
            flag
            for flag, value in (
                ("--harbor-dir", harbor_dir),
                ("--harbor-hub", harbor_hub),
                ("--harbor-git", harbor_git),
            )
            if value is not None
        ]
        if len(sources) > 1:
            raise typer.BadParameter(
                f"pass one of --harbor-dir, --harbor-hub or --harbor-git, not {' and '.join(sources)}"
            )
        if not sources and spec_file is None:
            raise typer.BadParameter("pass --harbor-dir, --harbor-hub, --harbor-git or --spec-file")
        if harbor_git is not None:
            try:
                check_git_source(harbor_git)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--harbor-git") from exc
        # A spec file alone selects from its own taskset; with a dataset it supplies only the target and settings.
        from_spec = not sources
        if spec_file is not None and any(value is not None for value in (sandbox_profile, cpus, memory_mb, storage_mb)):
            raise typer.BadParameter(
                "--sandbox-profile, --cpus, --memory-mb and --storage-mb don't apply with --spec-file; "
                "set execution.sandbox in it"
            )
        echo = typer.echo
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        try:
            if list_tasks:
                _print_tasks(
                    client_factory, harbor_dir, harbor_hub, harbor_git, spec_file if from_spec else None, workspace
                )
                return
            evaluator = Evaluator.from_client(client)
            dataset: tuple[str, str | None] | None = None
            with tempfile.TemporaryDirectory(prefix="nemo-runs-") as scratch:
                if from_spec:
                    assert spec_file is not None
                    refs, configs, total, prefix = _select_from_spec(evaluator, spec_file, tasks, workspace)
                else:
                    if harbor_hub is not None:
                        hub_name, hub_version, task_dirs = download_hub_tasks(harbor_hub, tasks, Path(scratch))
                        dataset = (hub_name, hub_version)
                        prefix = hub_name.rpartition("/")[2]
                        total = len(list_hub_tasks(harbor_hub))
                    elif harbor_git is not None:
                        repo_name, sha, task_dirs, total = download_git_tasks(harbor_git, tasks, Path(scratch))
                        dataset = (repo_name, sha)
                        prefix = repo_name.rpartition("/")[2]
                    else:
                        assert harbor_dir is not None
                        all_dirs = _task_dirs(harbor_dir)
                        candidates = [Candidate(i, d.name, _native_id(d)) for i, d in enumerate(all_dirs)]
                        task_dirs = [all_dirs[c.index] for c in select_tasks(candidates, tasks)]
                        prefix, total = harbor_dir.resolve().name, len(all_dirs)
                    refs, configs = publish_tasks(evaluator, task_dirs, prefix=prefix, workspace=workspace, echo=echo)
                taskset = entity_name(
                    f"{prefix}-sel-{selection_suffix(refs)}" if from_spec else f"{prefix}-{selection_suffix(refs)}"
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
                dispatcher_profile=dispatcher_profile,
                sandbox=_sandbox_selection(sandbox_profile, cpus, memory_mb, storage_mb),
                experiment=experiment,
                dataset=dataset,
            )
            status = submit_run(
                client, request.model_dump(mode="json", exclude_none=True), workspace=workspace, echo=echo
            )
        except (SelectionError, RuntimeError, NemoClientError, ValueError) as exc:
            echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        echo(f"Run {status['run_id']} accepted: {len(refs)} task Jobs")
        for line in status.get("warnings", []):
            echo(f"  {line}")
        echo("")
        flag = _workspace_flag(workspace, client.resolve_workspace())
        echo(f"View evaluation run: nemo evals runs status {status['run_id']}{flag} --watch")


def _sandbox_selection(
    profile: str | None, cpus: int | None, memory_mb: int | None, storage_mb: int | None
) -> SandboxSelection | None:
    """``execution.sandbox`` from the sandbox flags, or None when none is given."""
    if profile is None and cpus is None and memory_mb is None and storage_mb is None:
        return None
    return SandboxSelection(profile=profile, cpus=cpus, memory_mb=memory_mb, storage_mb=storage_mb)


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
    request = TasksetRunInput.model_validate(json.loads(spec_file.read_text(encoding="utf-8")))
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
    harbor_git: str | None,
    spec_file: Path | None,
    workspace: str,
) -> None:
    if harbor_hub is not None:
        rows = list_hub_tasks(harbor_hub)
    elif harbor_git is not None:
        rows = list_git_tasks(harbor_git)
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


def list_runs(
    client: NemoClient, workspace: str, *, active: bool, mine: bool, page: int, page_size: int
) -> dict[str, Any]:
    """One newest-first page of ``GET …/agent-evaluate/runs``: ``data`` and ``pagination``."""
    url = f"{client.base_url.rstrip('/')}/apis/evals/v2/workspaces/{workspace}/agent-evaluate/runs"
    params = {"active": active, "mine": mine, "page": page, "page_size": page_size}
    response = client.http_client.get(url, params=params, headers=client.request_headers(None, url=url))
    if response.status_code >= 400:
        raise RuntimeError(f"listing runs failed ({response.status_code}): {_detail(response)}")
    return response.json()


def _workspace_flag(workspace: str, configured: str) -> str:
    """`` --workspace <workspace>`` for a printed command, or "" when it's the configured one the command defaults to."""
    return "" if workspace == configured else f" --workspace {workspace}"


def download_run_bundle(
    client: NemoClient, run: dict[str, Any], *, workspace: str, output_dir: Path
) -> tuple[str, Path]:
    """Download and unpack ``run``'s merged bundle into ``output_dir``; return the bundle Job and the bundle's path.

    The bundle Job comes from the run's result record (name = run id), which names the attempt that produced it.
    Unpacking goes through a temporary directory, so a failed download or a rejected archive leaves nothing
    behind that would block a retry.

    Raises:
        SelectionError: The run has no merged result, or the bundle is already there.
    """
    run_id = run["run_id"]
    if run.get("result_state") != "ready":
        raise SelectionError(
            f"run {run_id} has no merged result (phase {run['phase']}, result {run.get('result_state', '-')}); "
            f"see `nemo evals runs status {run_id}{_workspace_flag(workspace, client.resolve_workspace())}`"
        )
    target = output_dir / DEFAULT_RESULT_NAME
    if target.exists():
        raise SelectionError(f"{target} already exists; remove it or pass -o")
    job = Evaluator.from_client(client).agent_eval_results.retrieve(run_id, workspace=workspace).job_id
    if not job:
        raise RuntimeError(f"the result record of run {run_id} names no bundle Job")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{run_id}-", dir=output_dir) as scratch:
        archive = Path(scratch) / "bundle.tar.gz"
        response = JobsClient.from_client(client).download_job_result(
            workspace=workspace, job=job, name=DEFAULT_RESULT_NAME
        )
        with archive.open("wb") as handle, response.stream() as chunks:
            for chunk in chunks:
                handle.write(chunk)
        unpacked = Path(scratch) / "unpacked"
        with tarfile.open(archive) as tar:
            safe_extract_tar(tar, unpacked)
        if not (unpacked / DEFAULT_RESULT_NAME).is_dir():
            raise RuntimeError(f"Job {job}'s {DEFAULT_RESULT_NAME} archive has no {DEFAULT_RESULT_NAME}/ directory")
        shutil.move(unpacked / DEFAULT_RESULT_NAME, target)
    return job, target


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


@dataclass(frozen=True)
class BundleJobView:
    """The run's newest bundle Job: the Job that merges the task results into the run's result."""

    job: str
    attempt: int
    status: str
    error: str | None = None  # why it failed, when it did


def newest_bundle_job(client: NemoClient, run_id: str, *, workspace: str) -> BundleJobView | None:
    """The run's newest bundle Job, or None before the controller created one."""
    jobs = JobsClient.from_client(client)
    newest = None
    for attempt in range(1, MAX_BUNDLE_ATTEMPTS + 1):
        try:
            job = jobs.get_job(workspace=workspace, name=bundle_job_name(run_id, attempt)).data()
        except NotFoundError:
            break
        status = job.status.value if hasattr(job.status, "value") else str(job.status)
        newest = BundleJobView(job.name, attempt, status, job_error(job) if status in _FAILED else None)
    return newest


def job_error(job: HelixJobResponse) -> str | None:
    """Why ``job`` failed, as specifically as Jobs says it.

    A failed container's exit message ends with the process's own error (``No module named …``), while
    ``error_details`` only says the Job ran out of retries; launcher and log-setup lines are dropped.
    """
    for container in job.status_details.get("containers") or []:
        terminated = container.get("terminated") or {}
        if terminated.get("exit_code") in (None, 0) or not terminated.get("message"):
            continue
        lines = [
            line
            for line in terminated["message"].splitlines()
            if line.strip() and not line.startswith("[launcher]") and " INFO " not in line
        ]
        if lines:
            return "\n".join(lines[-3:])
    return (job.error_details or {}).get("message")


def _run_state(run: dict[str, Any], bundle: BundleJobView | None) -> str:
    """The run's state in words; phase and result state alone read as contradictions ("completed", "failed")."""
    phase, result = run["phase"], run.get("result_state")
    # A run whose tasks all finished before the controller's first pass is still `accepted` while it bundles.
    if phase in ("accepted", "running") and bundle is not None and bundle.status not in _TERMINAL:
        return "merging task results"
    if phase in ("accepted", "running"):
        return phase
    if phase == "completed" and result == "ready":
        return "completed; the run's result is ready"
    if phase == "completed" and result == "failed":
        return "tasks finished, but merging their results into the run's result failed"
    if phase == "failed":
        return "failed; no task completed"
    return phase


def _status_table(
    run: dict[str, Any], views: Sequence[TaskJobView], bundle: BundleJobView | None = None, *, workspace_flag: str = ""
) -> list[str]:
    """The status screen; ``workspace_flag`` (see ``_workspace_flag``) completes its printed commands."""
    phase, result = run["phase"], run.get("result_state", "-")
    lines = [
        f"Run {run['run_id']}",
        f"  taskset   {_short_ref(run['taskset_ref'])}",
        f"  accepted  {str(run['accepted_at'])[:19].replace('T', ' ')}",
        f"  status    {_run_state(run, bundle)} (phase {phase}, result {result})",
    ]
    if bundle is not None:
        lines.append(f"  bundle    {bundle.job} (attempt {bundle.attempt} of {MAX_BUNDLE_ATTEMPTS}): {bundle.status}")
    if bundle is not None and bundle.status in _FAILED:
        lines += [f"            {line}" for line in (bundle.error or "").splitlines()]
        lines.append(f"            logs: nemo jobs get-logs {bundle.job}{workspace_flag}")
    error = run.get("error")
    # The controller's "bundle Job … ended …" only repeats the bundle line; any other error (a deadline) is news.
    if error and not (bundle is not None and error.startswith(f"bundle Job {bundle.job} ended")):
        lines.append(f"  error     {error}")
    if result == "failed" and any(v.status == "completed" for v in views):
        lines.append(f"  completed tasks' scores: nemo evals runs results {run['run_id']}{workspace_flag}")
    lines.append("")
    width = max([4, *(len(v.task or "") for v in views)])
    lines.append(f"{'#':>3}  {'task':<{width}}  {'Job':<{len(views[0].job) if views else 3}}  status")
    lines += [f"{v.index:>3}  {(v.task or '-'):<{width}}  {v.job}  {v.status}" for v in views]
    done = [v for v in views if v.status in _TERMINAL]
    failed = [v for v in done if v.status != "completed"]
    lines.append(f"{len(done)}/{len(views)} done ({len(done) - len(failed)} completed, {len(failed)} failed)")
    return lines


def _list_table(page: dict[str, Any]) -> list[str]:
    runs = page["data"]
    if not runs:
        return ["No runs."]
    rows = [
        (
            run["run_id"],
            run["phase"],
            run.get("result_state", "-"),
            _short_ref(run["taskset_ref"]),
            str(run["accepted_at"])[:19].replace("T", " "),
            run.get("created_by") or "-",
        )
        for run in runs
    ]
    header = ("run", "phase", "result", "taskset", "accepted", "by")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [header, *rows]]
    pagination = page.get("pagination") or {}
    total, pages, current = pagination.get("total_results"), pagination.get("total_pages", 1), pagination.get("page", 1)
    if pages > current:
        lines.append(f"page {current} of {pages} ({total} runs); next: --page {current + 1}")
    else:
        count = total if total is not None else len(runs)
        lines.append(f"{count} run{'' if count == 1 else 's'}")
    return lines


def _short_ref(ref: str) -> str:
    """``workspace/name#digest`` with the digest cut to 8 characters; the table has no room for 64."""
    name, sep, digest = ref.partition("#")
    return f"{name}{sep}{digest[:8]}"


def _results_table(
    views: Sequence[TaskJobView], record: AgentEvalResult | None = None, *, workspace_flag: str = ""
) -> list[str]:
    """The results screen; ``workspace_flag`` (see ``_workspace_flag``) completes its printed command."""
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
        lines += ["", f"Download the merged bundle: nemo evals runs download {record.name}{workspace_flag}"]
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
_FAILED = frozenset({"error", "cancelled"})
_RUN_TERMINAL = frozenset({"completed", "failed", "cancelled", "expired"})


def add_progress_commands(runs: typer.Typer, client_factory: Callable[[], NemoClient]) -> None:
    """Register ``runs list``, ``runs status`` and ``runs results`` under ``runs``."""

    @runs.command("list")
    def list_(
        active: bool = typer.Option(False, "--active", help="Only accepted and running runs."),
        mine: bool = typer.Option(False, "--mine", help="Only runs you created."),
        page: int = typer.Option(1, "--page", min=1, help="One-based page number."),
        page_size: int = typer.Option(20, "--page-size", min=1, max=1000, help="Runs per page."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the runs; default: configured."),
        output_json: bool = typer.Option(False, "--json", help="Print the API page as JSON."),
    ) -> None:
        """List a workspace's runs, newest first."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        try:
            body = list_runs(client, workspace, active=active, mine=mine, page=page, page_size=page_size)
        except (RuntimeError, NemoClientError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(body, indent=2) if output_json else "\n".join(_list_table(body)))

    @runs.command("status")
    def status(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the run; default: configured."),
        watch: bool = typer.Option(False, "--watch", help="Refresh until every task Job is done."),
        poll_seconds: float = typer.Option(15.0, "--poll-seconds", help="Refresh interval with --watch."),
        output_json: bool = typer.Option(False, "--json", help="Print JSON."),
    ) -> None:
        """Show each task Job of a run and its status."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        try:
            run = get_run(client, workspace, run_id)
            while True:
                views = run_task_jobs(client, run, workspace=workspace)
                bundle = newest_bundle_job(client, run_id, workspace=workspace)
                if output_json:
                    body = {"run": run, "bundle": bundle and bundle.__dict__}
                    typer.echo(json.dumps(body | {"tasks": [_view_json(v) for v in views]}, indent=2))
                else:
                    flag = _workspace_flag(workspace, client.resolve_workspace())
                    typer.echo("\n".join(_status_table(run, views, bundle, workspace_flag=flag)))
                if not watch or run["phase"] in _RUN_TERMINAL:
                    return
                time.sleep(poll_seconds)
                typer.echo("")
                run = get_run(client, workspace, run_id)
        except (SelectionError, RuntimeError, NemoClientError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    @runs.command("download")
    def download(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        output_dir: Path | None = typer.Option(
            None, "--output-dir", "-o", help="Directory to unpack agent-eval-results/ into; default ./<run_id>."
        ),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the run; default: configured."),
    ) -> None:
        """Download a run's merged result bundle (report.html, summary.json, every trial and score)."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        try:
            run = get_run(client, workspace, run_id)
            job, bundle = download_run_bundle(client, run, workspace=workspace, output_dir=output_dir or Path(run_id))
        except (SelectionError, RuntimeError, NemoClientError, tarfile.TarError, ValueError, OSError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"Downloaded the merged bundle of {run_id} (bundle Job {job})")
        typer.echo(f"  {bundle / 'report.html'}   # open in a browser")
        typer.echo(f"  {bundle / SUMMARY_FILE_NAME}")

    @runs.command("results")
    def results(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the run; default: configured."),
        output_json: bool = typer.Option(False, "--json", help="Print every score of every task as JSON."),
    ) -> None:
        """Show each completed task's scores (every Harbor reward) and a run rollup."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        try:
            run = get_run(client, workspace, run_id)
            views = run_task_jobs(client, run, workspace=workspace, with_results=True)
            record = None
            if run.get("result_state") == "ready":  # the controller saved the run's own record
                record = Evaluator.from_client(client).agent_eval_results.retrieve(run_id, workspace=workspace)
        except (SelectionError, RuntimeError, NemoClientError) as exc:
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
        flag = _workspace_flag(workspace, client.resolve_workspace())
        typer.echo("\n".join(_results_table(views, record, workspace_flag=flag)))


def _view_json(view: TaskJobView) -> dict[str, Any]:
    data: dict[str, Any] = {"index": view.index, "job": view.job, "task": view.task, "status": view.status}
    if view.result is not None:
        data["result"] = view.result.model_dump(mode="json", include={"name", "scores", "summary", "bundle_ref"})
    return data
