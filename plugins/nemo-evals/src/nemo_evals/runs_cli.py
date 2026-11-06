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
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from harbor.registry.client.package import PackageDatasetClient
from nemo_evals.api.fields import TaskRef, TasksetRef
from nemo_evals.api.schemas import AgentEvalResult, TaskInput, TasksetInput
from nemo_evals.api.taskset_run_schemas import TasksetRunInput, TasksetRunStatus, TasksetRunTask
from nemo_evals.harbor.remote.profile import SandboxSelection
from nemo_evals.image_builds.models import TERMINAL_BUILD_STATUSES, ImageBuild
from nemo_evals.jobs.agent_evaluate import DEFAULT_RESULT_NAME, SUMMARY_FILE_NAME
from nemo_evals.jobs.agent_spec import AgentEvalInputSpec, HarborBuiltinAgentSource, HarborRunnerTarget
from nemo_evals.jobs.publication import PublicationOutcome
from nemo_evals.sdk.resources import Evaluator
from nemo_evals.sdk.taskset_run_resources import NamedTask, TasksetRunResults, TasksetRunSubmitError
from nemo_evals.taskset_runs.controller import MAX_BUNDLE_ATTEMPTS
from nemo_evals.taskset_runs.entities import SETTLED_MEMBER_PHASES, MemberPhase
from nemo_evals.taskset_runs.identifiers import bundle_job_name
from nemo_evals.taskset_runs.progress import TasksetRunProgress
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError
from nemo_helix_plugin.jobs.archive import safe_extract_tar
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import HelixJobResponse
from nemo_helix_plugin.schema import Page
from pydantic import BaseModel

_NAME_UNSAFE = re.compile(r"[^a-z0-9-]+")


class SelectionError(ValueError):
    """``--tasks`` has an invalid index/range, an unknown or ambiguous name, or a repeated task."""


@dataclass(frozen=True)
class Candidate:
    """A task that can be selected: its position in ref order, its name and its native Harbor id."""

    index: int
    name: str
    native_id: str


def select_tasks(candidates: Sequence[Candidate], tokens: str | None) -> list[Candidate]:
    """Pick tasks by zero-based index, inclusive range (``0-5``), or name; all when ``tokens`` is None.

    Raises:
        SelectionError: An index/range is invalid, a name is unknown or ambiguous, or a task repeats.
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
            matches = [candidates[index]]
        elif bounds := re.fullmatch(r"([0-9]+)-([0-9]+)", token):
            try:
                start, end = map(int, bounds.groups())
            except ValueError as exc:
                raise SelectionError(
                    f"invalid task range {token!r}: numeric endpoint exceeds the integer limit"
                ) from exc
            if start > end:
                raise SelectionError(f"task range {token!r} is reversed; start must not exceed end")
            if end >= len(candidates):
                raise SelectionError(f"task range {token!r} is out of range 0..{len(candidates) - 1}")
            matches = candidates[start : end + 1]
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
        for match in matches:
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


def download_hub_tasks(dataset: str, names: str | None, output_dir: Path) -> tuple[str, str | None, list[Path]]:
    """Download the selected tasks of Harbor Hub dataset ``org/name@ref`` into ``output_dir``.

    Returns the dataset's name, its resolved version (``sha256:…``, also without ``@ref``) and the task
    directories, in the dataset's order.
    """
    client = PackageDatasetClient()
    metadata = asyncio.run(client.get_dataset_metadata(dataset))
    ids = sorted(metadata.task_ids, key=lambda task_id: task_id.get_name())
    candidates = [Candidate(i, entity_name(t.get_name().rpartition("/")[2]), t.get_name()) for i, t in enumerate(ids)]
    chosen = [ids[c.index] for c in select_tasks(candidates, names)]
    result = asyncio.run(client._task_client.download_tasks(task_ids=chosen, output_dir=output_dir, export=True))
    return metadata.name, metadata.version, list(result.paths)


def list_hub_tasks(dataset: str) -> list[str]:
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
    """Download the selected tasks of git repo ``<url>@<ref>`` (its ``tasks/`` directory, or the ``<dir>`` of a
    ``…/tree/<ref>/<dir>`` URL) into ``output_dir``.

    Returns the repo's ``org/name``, the commit the download checked out, the task directories in name order,
    and how many tasks the repo has.

    Raises:
        RuntimeError: The selected tasks didn't come from one commit.
    """
    from harbor.registry.client.git_repo import GitRepoRegistryClient, resolve_repo_source

    check_git_source(source)
    repo = resolve_repo_source(source)
    client = GitRepoRegistryClient(repo)
    metadata = _run_git(client.get_dataset_metadata(""))  # no dataset name: every task under tasks/
    ids = sorted(metadata.task_ids, key=lambda task_id: task_id.get_name())
    candidates = [Candidate(i, entity_name(t.get_name()), t.get_name()) for i, t in enumerate(ids)]
    chosen = [ids[c.index] for c in select_tasks(candidates, names)]
    result = _run_git(client._task_client.download_tasks(task_ids=chosen, output_dir=output_dir, export=True))
    # Not the ids' git_commit_id: for an annotated tag that's the tag object's SHA, not its commit.
    commits = {r.resolved_git_commit_id for r in result.results}
    commit = next(iter(commits)) if len(commits) == 1 else None
    if commit is None:
        raise RuntimeError(f"Harbor resolved {source} to {sorted(map(str, commits))}, not one commit")
    return f"{repo.org}/{repo.name}", commit, list(result.paths), len(ids)


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
) -> list[str]:
    """Publish each task directory as ``<prefix>-<dir>``; return the pinned refs.

    A task whose stored content already equals what publishing it would write is not published again: its archive
    is packed locally and compared, and its current revision is pinned. Republishing it would upload, read back
    and store the same bytes, on every submit of the same tasks.
    """
    from nemo_evals.sdk.task_preparation import describe_harbor_task
    from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import discover_harbor_tasks

    refs: list[str] = []
    for task_dir in task_dirs:
        (source,) = discover_harbor_tasks(task_dir)
        name = entity_name(f"{prefix}-{task_dir.name}")
        stored = _unchanged_revision(evaluator, workspace, name, describe_harbor_task(source, workspace=workspace))
        if stored is not None:
            ref = pinned_task_ref(evaluator, workspace, name, stored)
            echo(f"Unchanged task {ref}")
        else:
            task = evaluator.tasks.replace(name, task=source, workspace=workspace)
            ref = pinned_task_ref(evaluator, workspace, name, task.revision)
            echo(f"Published task {ref}")
        refs.append(ref)
    return refs


def _unchanged_revision(evaluator: Evaluator, workspace: str, name: str, expected: TaskInput) -> int | None:
    """The stored task's current revision when its content is ``expected``; None when absent or different."""
    try:
        current = evaluator.tasks.retrieve(name, workspace=workspace)
    except NotFoundError:
        return None
    if current.spec == expected.spec and current.metadata == expected.metadata:
        return current.revision
    return None


def pinned_task_ref(evaluator: Evaluator, workspace: str, name: str, revision: int) -> str:
    """``workspace/name#<content digest>`` of revision ``revision``."""
    for item in evaluator.tasks.list_revisions(name, workspace=workspace, page_size=100).data:
        if item.revision == revision and item.content_hash:
            return f"{workspace}/{name}#{item.content_hash}"
    raise RuntimeError(f"task {workspace}/{name} has no revision {revision} with a content digest")


@dataclass(frozen=True)
class CreatedTaskset:
    """A published taskset: its ref, its pinned task refs, the source's task count and its Intake dataset."""

    ref: str
    refs: list[str]
    total: int
    dataset: tuple[str, str | None] | None


def create_taskset(
    evaluator: Evaluator,
    *,
    harbor_dir: Path | None,
    harbor_hub: str | None,
    harbor_git: str | None,
    tasks: str | None,
    workspace: str,
    echo: Callable[[str], None],
) -> CreatedTaskset:
    """Download the selected tasks of one source, publish them, and publish a taskset of exactly those.

    The same content gives the same taskset: its name ends with a hash of the pinned task refs.
    """
    dataset: tuple[str, str | None] | None = None
    with tempfile.TemporaryDirectory(prefix="nemo-runs-") as scratch:
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
        refs = publish_tasks(evaluator, task_dirs, prefix=prefix, workspace=workspace, echo=echo)
    ref = publish_taskset(evaluator, refs, name=entity_name(f"{prefix}-{selection_suffix(refs)}"), workspace=workspace)
    echo(f"Published taskset {ref} ({len(refs)} of {total} tasks)")
    return CreatedTaskset(ref, refs, total, dataset)


def publish_taskset(evaluator: Evaluator, refs: Sequence[str], *, name: str, workspace: str) -> str:
    """Publish taskset ``name`` of ``refs``; return ``workspace/name#<content digest>``."""
    evaluator.tasksets.replace(name, taskset=TasksetInput(tasks=[TaskRef(ref) for ref in refs]), workspace=workspace)
    revision = evaluator.tasksets.list_revisions(name, workspace=workspace, page_size=1).data[0]
    return f"{workspace}/{name}#{revision.content_hash}"


def dataset_flag(dataset: tuple[str, str | None]) -> str:
    """``--intake-dataset`` as ``runs submit --taskset`` takes it: ``name@version``, or the bare name."""
    name, version = dataset
    return f"{name}@{version}" if version else name


def parse_dataset(value: str) -> tuple[str, str | None]:
    """``--intake-dataset NAME[@VERSION]``: the version is everything after the last ``@`` (``sha256:…`` or a commit)."""
    name, at, version = value.rpartition("@")
    return (name, version or None) if at and name else (value, None)


def qualify_ref(ref: str, workspace: str) -> str:
    """``ref`` with ``workspace/`` in front when it names none."""
    head, hash_, digest = ref.partition("#")
    return ref if "/" in head else f"{workspace}/{head}{hash_}{digest}"


def taskset_members(evaluator: Evaluator, ref: str) -> list[str]:
    """The task refs of taskset ``ref`` (``workspace/name[#digest]``), pinned by digest as stored."""
    workspace, name, digest = _split_ref(ref)
    taskset = evaluator.tasksets.retrieve(name, revision=digest or None, workspace=workspace)
    return sorted(member.root for member in taskset.tasks)


# --- Image builds -----------------------------------------------------------------------------------------
# Prebuilt images go through a build too: it copies them into the platform's registry (#2986). A request is
# idempotent per task revision: it returns the build that's ready or in progress, and starts the next attempt
# only after a failure, or once it finds the build's Job ended. Reads never start or repair a build.

BUILD_REPAIR_SECONDS = 300.0
"""How long ``--wait`` follows a build by reads alone before requesting it again, which repairs a dead Job."""
BUILD_RETRIES = 1
"""Attempts ``--wait`` starts after one ends without images, per task."""
BUILD_READ_FAILURES = 5
"""Failed polls in a row after which ``--wait`` gives up."""


def request_builds(evaluator: Evaluator, refs: Sequence[str]) -> dict[str, ImageBuild]:
    """Request every task's image build once; each answer is the build that's ready or in progress."""
    builds: dict[str, ImageBuild] = {}
    for ref in refs:
        workspace, name, digest = _split_ref(ref)
        builds[ref] = evaluator.tasks.build_images(name, revision=digest, workspace=workspace)
    return builds


def read_builds(evaluator: Evaluator, refs: Sequence[str]) -> dict[str, ImageBuild | None]:
    """Each task's newest build for the default platform, the one submit checks; None when it has none."""
    builds: dict[str, ImageBuild | None] = {}
    for ref in refs:
        workspace, name, digest = _split_ref(ref)
        page = evaluator.tasks.list_image_builds(name, revision=digest, workspace=workspace)
        builds[ref] = next((build for build in page.data if build.platform is None), None)
    return builds


def builds_summary(builds: dict[str, ImageBuild] | dict[str, ImageBuild | None]) -> str:
    states = [_build_state(build) for build in builds.values()]
    counts = ", ".join(f"{states.count(state)} {state}" for state in ("ready", "building", "failed"))
    missing = states.count("not requested")
    return f"Image builds for {len(builds)} tasks: {counts}" + (f", {missing} not requested." if missing else ".")


def _build_state(build: ImageBuild | None) -> str:
    if build is None:
        return "not requested"
    return build.status if build.status in TERMINAL_BUILD_STATUSES else "building"


def wait_for_builds(
    evaluator: Evaluator,
    refs: Sequence[str],
    *,
    poll_seconds: float,
    echo: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, ImageBuild]:
    """Request each task's build unless its newest is ready, then follow each by its id until it's ready or failed.

    Returns the last builds. A task whose newest build is already ready gets no request: a request costs the API a
    build check per task, on every submit of the same tasks. Between polls only reads go out. A request goes out again only to start a task's one retry, or after
    ``BUILD_REPAIR_SECONDS`` without a request, since only a request notices a build whose Job died. A request
    that answers with another id means the followed attempt ended. A failed attempt prints its error; a task's
    second failure settles it ``failed``. A read error keeps the last known state, ``BUILD_READ_FAILURES`` in a
    row raise.
    """
    newest = read_builds(evaluator, refs)
    ready = {ref: build for ref, build in newest.items() if build is not None and build.status == "ready"}
    requested = request_builds(evaluator, [ref for ref in refs if ref not in ready])
    followed = {ref: ready[ref] if ref in ready else requested[ref] for ref in refs}
    retries = dict.fromkeys(refs, 0)
    requested_at = dict.fromkeys(refs, clock())
    width = len(str(len(refs)))
    echo(f"{builds_summary(followed)} Ctrl-C is safe; builds keep going.")

    def line(state: str, ref: str, detail: str = "") -> None:
        settled = sum(build.status in TERMINAL_BUILD_STATUSES for build in followed.values())
        echo(f"[{settled:>{width}}/{len(refs)}] {state:<9} {ref}{detail}")

    def ended(ref: str, attempt: ImageBuild, nxt: ImageBuild | None) -> None:
        """``attempt`` ended without images: start the task's retry, or settle the task ``failed``."""
        error = f"  attempt {attempt.id} failed: {attempt.error or 'its Job ended'}"
        if retries[ref] < BUILD_RETRIES:
            workspace, name, digest = _split_ref(ref)
            # Counted once the retry is followed: a request that raises leaves the failed attempt for the next poll.
            followed[ref] = nxt or evaluator.tasks.build_images(name, revision=digest, workspace=workspace)
            retries[ref] += 1
            requested_at[ref] = clock()
            line("retrying", ref, error)
        else:
            followed[ref] = attempt.model_copy(update={"status": "failed"})
            # A repair request that found the Job dead has already started the next attempt: say it isn't followed.
            line("failed", ref, error + (f"; attempt {nxt.id} started, not followed" if nxt is not None else ""))

    read_failures = 0
    while unsettled := [ref for ref in refs if followed[ref].status not in TERMINAL_BUILD_STATUSES]:
        sleep(poll_seconds)
        try:
            for ref in unsettled:
                workspace, name, digest = _split_ref(ref)
                build = followed[ref]
                current = evaluator.tasks.get_image_build(name, build.id, revision=digest, workspace=workspace)
                if (
                    current.status not in TERMINAL_BUILD_STATUSES
                    and clock() - requested_at[ref] >= BUILD_REPAIR_SECONDS
                ):
                    requested_at[ref] = clock()
                    answer = evaluator.tasks.build_images(name, revision=digest, workspace=workspace)
                    if answer.id != build.id:  # the request found this attempt's Job ended and started the next
                        current = evaluator.tasks.get_image_build(name, build.id, revision=digest, workspace=workspace)
                        ended(ref, current, answer)
                        continue
                if current.status == "failed":
                    ended(ref, current, None)
                    continue
                followed[ref] = current
                if current.status != build.status:  # pending → running prints "building": its Job started
                    line(_build_state(current), ref, _ready_detail(current))
            read_failures = 0
        except NemoClientError as exc:
            read_failures += 1
            if read_failures >= BUILD_READ_FAILURES:
                raise
            echo(f"Reading the image builds failed ({read_failures} of {BUILD_READ_FAILURES} in a row): {exc}")
    failed = sum(build.status == "failed" for build in followed.values())
    images = sum(len(build.images) for build in followed.values() if build.status == "ready")
    echo(
        f"Image builds: {len(refs) - failed} of {len(refs)} tasks ready ({images} images)"
        + (f", {failed} failed." if failed else ".")
    )
    return followed


def failed_builds(builds: dict[str, ImageBuild]) -> str | None:
    """``ref: error`` of each failed build, or None when every build is ready."""
    failed = [f"{ref}: {build.error or 'its Job ended'}" for ref, build in builds.items() if build.status != "ready"]
    return "; ".join(failed) or None


def _ready_detail(build: ImageBuild) -> str:
    if build.status != "ready":
        return ""
    took = (
        f", {_duration((build.updated_at - build.created_at).total_seconds())}"
        if build.created_at and build.updated_at
        else ""
    )
    return f" ({len(build.images)} image{'s' if len(build.images) != 1 else ''}{took})"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(max(int(seconds), 0), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _split_ref(ref: str) -> tuple[str, str, str]:
    head, _, digest = ref.partition("#")
    workspace, _, name = head.partition("/")
    return workspace, name, digest


def run_request(
    taskset_ref: str,
    *,
    spec_file: Path | None,
    agent: str,
    dispatcher_profile: str | None,
    sandbox: SandboxSelection | None = None,
    experiment: str | None = None,
    dataset: tuple[str, str | None] | None = None,
    idempotency_key: str | None = None,
    spec: dict[str, Any] | None = None,
) -> TasksetRunInput:
    """The run request: ``--spec-file``'s with its tasks replaced, or ``--agent`` with default execution.

    ``spec`` is ``spec_file`` already read by ``_load_spec``, so a stream such as ``/dev/stdin`` is read once.

    ``--experiment`` sets ``publication.intake``, replacing the spec file's.
    ``dataset`` is the Harbor Hub dataset or git repo the tasks came from, as ``(name, version or commit)``:
    the Evaluation's dataset instead of the taskset. ``dispatcher_profile`` and ``sandbox`` apply
    to ``--agent`` only; a spec file carries its own ``execution``. The idempotency key is
    ``idempotency_key``, else the spec file's, else a new one.
    """
    publication: dict[str, Any] | None = None
    if experiment is not None:
        publication = {"intake": {"experiment": experiment}}
        if dataset is not None:
            name, version = dataset
            publication["intake"] |= {"dataset_name": name} | ({"dataset_version": version} if version else {})
    if spec_file is not None:
        # The selected taskset replaces spec.tasks before validation: with a dataset the file may leave it out.
        raw = dict(spec) if spec is not None else _load_spec(spec_file)
        raw["spec"] = {**(raw.get("spec") or {}), "tasks": taskset_ref}
        if publication is not None:
            raw["publication"] = publication  # --experiment replaces the file's, before it's validated
        raw["idempotency_key"] = idempotency_key or raw.get("idempotency_key") or str(uuid.uuid4())
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
            "idempotency_key": idempotency_key or str(uuid.uuid4()),
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
            None,
            "--harbor-git",
            help="Public git repo of Harbor tasks, pinned to a tag or commit: `<url>@<ref>` reads `tasks/` at that ref. "
            "For another directory, use a GitHub tree URL: `https://github.com/<org>/<repo>/tree/<ref>/<dir>`.",
        ),
        taskset: str | None = typer.Option(
            None,
            "--taskset",
            help="Stored taskset, workspace/name#digest (from `nemo evals tasksets create`): uploads no tasks and "
            "requests no image builds; submit refuses a task whose image isn't built.",
        ),
        spec_file: Path | None = typer.Option(
            None,
            "--spec-file",
            help="TasksetRunInput JSON: its target and execution, and its taskset unless --taskset, --harbor-dir, "
            "--harbor-hub or --harbor-git supplies the tasks.",
        ),
        intake_dataset: str | None = typer.Option(
            None,
            "--intake-dataset",
            help="Intake label, NAME[@VERSION], shown as the dataset of the run's Evaluation; doesn't select tasks. "
            "With --taskset and --experiment only; `tasksets create` prints it. --harbor-hub and --harbor-git set "
            "it themselves.",
        ),
        tasks: str | None = typer.Option(
            None,
            "--tasks",
            help="Zero-based indexes, inclusive ranges, or names, comma-separated (e.g. 0-4,7,anko-example); default all.",
        ),
        agent: str = typer.Option("oracle", "--agent", help="Harbor built-in agent (without --spec-file)."),
        sandbox_profile: str | None = typer.Option(
            None, "--sandbox-profile", help="Operator sandbox profile; default: the operator's (without --spec-file)."
        ),
        dispatcher_profile: str | None = typer.Option(None, "--dispatcher-profile", help="Jobs profile for task Jobs."),
        cpus: int | None = typer.Option(
            None, "--cpus", min=1, help="CPUs for every task's agent sandbox, over task.toml (without --spec-file)."
        ),
        memory_mb: int | None = typer.Option(
            None, "--memory-mb", min=1, help="Memory in MB for every task's agent sandbox (without --spec-file)."
        ),
        storage_mb: int | None = typer.Option(
            None, "--storage-mb", min=1, help="Disk in MB for every task's agent sandbox (without --spec-file)."
        ),
        no_build_images: bool = typer.Option(
            False,
            "--no-build-images",
            help="Don't request or wait for image builds here; submit rejects a task whose image isn't built yet.",
        ),
        idempotency_key: str | None = typer.Option(
            None,
            "--idempotency-key",
            help="Key that makes a resubmit return the same run; default: a new one. A failed submit prints its key.",
        ),
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
        if sources and taskset is not None:
            raise typer.BadParameter(f"--taskset names the tasks already; drop {sources[0]}")
        if not sources and spec_file is None and taskset is None:
            raise typer.BadParameter("pass --taskset, --harbor-dir, --harbor-hub, --harbor-git or --spec-file")
        if intake_dataset is not None and (taskset is None or experiment is None):
            raise typer.BadParameter("--intake-dataset applies with --taskset and --experiment only")
        if harbor_git is not None:
            try:
                check_git_source(harbor_git)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="--harbor-git") from exc
        # --taskset, or a spec file alone, selects from a stored taskset; with a dataset the spec file supplies only
        # the target and settings.
        from_stored = not sources
        if spec_file is not None and any(value is not None for value in (sandbox_profile, cpus, memory_mb, storage_mb)):
            raise typer.BadParameter(
                "--sandbox-profile, --cpus, --memory-mb and --storage-mb don't apply with --spec-file; "
                "set execution.sandbox in it"
            )
        echo = typer.echo
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        stored_ref = qualify_ref(taskset, workspace) if taskset is not None else None
        flag = _workspace_flag(workspace, client.resolve_workspace())
        taskset_ref: str | None = None
        waiting_for_builds = False
        submitting_key: str | None = None
        spec: dict[str, Any] | None = None
        source: tuple[str, str | None] | None = parse_dataset(intake_dataset) if intake_dataset else None

        def rerun(ref: str, *, idempotency_key: str | None = None) -> str:
            """Keep this submit's settings on ``ref``; supply a key only to recover a submission."""
            kept: list[tuple[str, object]] = [
                ("--spec-file", "<spec file>" if spec_file is not None and _is_stream(spec_file) else spec_file),
                ("--agent", None if spec_file is not None or agent == "oracle" else agent),
                ("--sandbox-profile", sandbox_profile),
                ("--cpus", cpus),
                ("--memory-mb", memory_mb),
                ("--storage-mb", storage_mb),
                ("--dispatcher-profile", dispatcher_profile),
                ("--experiment", experiment),
                ("--intake-dataset", "@".join(p for p in source if p) if experiment and source else None),
                ("--idempotency-key", idempotency_key),
            ]
            options = "".join(f" {name} {shlex.quote(str(value))}" for name, value in kept if value is not None)
            return f"nemo evals runs submit --taskset {shlex.quote(ref)}{options}{flag}"

        try:
            if list_tasks:
                _print_tasks(
                    client_factory,
                    harbor_dir,
                    harbor_hub,
                    harbor_git,
                    spec_file if from_stored and stored_ref is None else None,
                    workspace,
                    taskset=stored_ref,
                )
                return
            evaluator = Evaluator.from_client(client)
            spec = _load_spec(spec_file) if spec_file is not None else None
            if stored_ref is not None and tasks is None:
                refs, taskset_ref = taskset_members(evaluator, stored_ref), stored_ref
            elif from_stored:
                base = stored_ref or _spec_taskset(spec_file, spec)
                refs, total, prefix = select_from_taskset(evaluator, base, tasks, workspace)
                taskset_ref = publish_taskset(
                    evaluator, refs, name=entity_name(f"{prefix}-sel-{selection_suffix(refs)}"), workspace=workspace
                )
                echo(f"Published taskset {taskset_ref} ({len(refs)} of {total} tasks)")
            else:
                created = create_taskset(
                    evaluator,
                    harbor_dir=harbor_dir,
                    harbor_hub=harbor_hub,
                    harbor_git=harbor_git,
                    tasks=tasks,
                    workspace=workspace,
                    echo=echo,
                )
                refs, taskset_ref, source = created.refs, created.ref, created.dataset
                if not no_build_images:
                    waiting_for_builds = True
                    builds = wait_for_builds(evaluator, refs, poll_seconds=poll_seconds, echo=echo)
                    waiting_for_builds = False
                    if (failed := failed_builds(builds)) is not None:
                        raise RuntimeError(f"image builds failed: {failed}")
            request = run_request(
                taskset_ref,
                spec=spec,
                spec_file=spec_file,
                agent=agent,
                dispatcher_profile=dispatcher_profile,
                sandbox=_sandbox_selection(sandbox_profile, cpus, memory_mb, storage_mb),
                experiment=experiment,
                dataset=source,
                idempotency_key=idempotency_key,
            )
            submitting_key = request.idempotency_key
            status = evaluator.taskset_runs.submit(request, workspace=workspace)
        except KeyboardInterrupt:
            # The run may already be accepted: only the same key avoids a second one
            if submitting_key is not None and taskset_ref is not None:
                echo(
                    "Interrupted during submit; the run may still be accepted. To check or finish it without starting "
                    # the published ref, not the source: a changed source would send a different request (409)
                    f"a second run, submit with the same key: {rerun(taskset_ref, idempotency_key=submitting_key)}"
                )
                raise typer.Exit(code=130) from None
            if waiting_for_builds and taskset_ref:  # --taskset doesn't wait: it rejects tasks still building
                echo(
                    "Stopped waiting; the image builds keep going. Check them with: "
                    f"nemo evals tasksets build-images {shlex.quote(taskset_ref)} --status"  # the ref names the workspace
                )
                echo(f"Once they're ready, submit with: {rerun(taskset_ref, idempotency_key=idempotency_key)}")
            else:
                echo(
                    f"Interrupted. Submit with: {rerun(taskset_ref, idempotency_key=idempotency_key)}"
                    if taskset_ref
                    else "Interrupted."
                )
            raise typer.Exit(code=130) from None
        except (SelectionError, RuntimeError, NemoClientError, ValueError) as exc:
            echo(f"Error: {exc}", err=True)
            # May be accepted: no answer, a 5xx, or a 403 from a downstream check partway through the submit.
            # Any other status (400, 401, 404, 409, 422, 429) means the server refused it.
            if isinstance(exc, TasksetRunSubmitError) and taskset_ref and _may_be_accepted(exc.status_code):
                echo(
                    "The run may already be accepted. To check or finish it without starting a second run, submit "
                    f"with the same key: {rerun(taskset_ref, idempotency_key=exc.idempotency_key)}",
                    err=True,
                )
            if stored_ref is not None and "without images" in str(exc):
                # The ref names its workspace, and build-images takes no --workspace.
                echo(f"Check with: nemo evals tasksets build-images {shlex.quote(stored_ref)} --status", err=True)
            raise typer.Exit(code=1) from exc
        if status.phase == "accepted":
            echo(f"Run {status.run_id} accepted: {len(refs)} tasks, queued; the run controller starts their task Jobs")
        else:  # the key replayed a run that already started or ended
            echo(f"Run {status.run_id} was already submitted with this key and is {status.phase}; nothing new started")
        for line in status.warnings:
            echo(f"  {line}")
        if taskset_ref != stored_ref:  # a new taskset: reruns can skip the upload and the builds
            echo(f"Rerun these tasks: {rerun(taskset_ref)}")
            if spec is not None and spec.get("idempotency_key"):  # that command falls back to the file's key
                echo("  It uses the spec file's idempotency_key; pass a new --idempotency-key to start another run")
        echo("")
        echo(f"View evaluation run: nemo evals runs status {status.run_id}{flag} --watch")


def _is_stream(path: Path) -> bool:
    """A path that's gone once this command exits: ``/dev/stdin`` (even redirected from a file), ``<(…)``, a pipe."""
    parts = path.parts
    return (
        parts[:3] in {("/", "dev", "stdin"), ("/", "dev", "fd")}
        or (parts[:2] == ("/", "proc") and parts[3:4] == ("fd",))  # /proc/<pid or self>/fd/<n>
        or not path.is_file()
    )


def _load_spec(spec_file: Path) -> dict[str, Any]:
    """``--spec-file``'s JSON object; read it once, since it may be a stream."""
    raw = json.loads(spec_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("spec") or {}, dict):
        raise ValueError(f"{spec_file}: expected a TasksetRunInput JSON object with an object 'spec'")
    return raw


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
) -> tuple[list[str], int, str]:
    """The pinned refs of ``--spec-file``'s taskset, narrowed by ``--tasks``; tasks are not re-uploaded."""
    return select_from_taskset(evaluator, _spec_taskset(spec_file), tokens, workspace)


def _spec_taskset(spec_file: Path | None, raw: dict[str, Any] | None = None) -> str:
    """The taskset ref ``--spec-file`` names in ``spec.tasks``; ``raw`` is the file already read by ``_load_spec``."""
    assert spec_file is not None
    # Only spec.tasks matters here: run_request applies --experiment before the whole file is validated.
    raw = raw if raw is not None else _load_spec(spec_file)
    if not isinstance(raw.get("spec"), dict):
        raise ValueError(f"{spec_file}: expected a TasksetRunInput JSON object with an object 'spec'")
    spec = AgentEvalInputSpec.model_validate(raw["spec"])
    if not isinstance(spec.tasks, TasksetRef):
        raise ValueError(f"{spec_file}: spec.tasks must name a taskset (workspace/name#revision)")
    return spec.tasks.root


def select_from_taskset(
    evaluator: Evaluator, ref: str, tokens: str | None, workspace: str
) -> tuple[list[str], int, str]:
    """The pinned refs of taskset ``ref``, narrowed by ``--tasks``, its size and its name; nothing is uploaded."""
    head, _, fragment = ref.partition("#")
    taskset_workspace, _, name = head.rpartition("/")
    taskset = evaluator.tasksets.retrieve(name, revision=fragment or None, workspace=taskset_workspace or workspace)
    members = sorted(member.root for member in taskset.tasks)
    candidates = []
    for index, member in enumerate(members):
        task_workspace, task_name, digest = _split_ref(member)
        task = evaluator.tasks.retrieve(task_name, revision=digest or None, workspace=task_workspace or workspace)
        pinned = member if digest else pinned_task_ref(evaluator, task_workspace, task_name, task.revision)
        candidates.append(Candidate(index, pinned, getattr(task.spec, "native_task_id", task_name)))
    selected = select_tasks(candidates, tokens)
    return [c.name for c in selected], len(candidates), name


def _print_tasks(
    client_factory: Callable[[], NemoClient],
    harbor_dir: Path | None,
    harbor_hub: str | None,
    harbor_git: str | None,
    spec_file: Path | None,
    workspace: str,
    *,
    taskset: str | None = None,
) -> None:
    if harbor_hub is not None:
        rows = list_hub_tasks(harbor_hub)
    elif harbor_git is not None:
        rows = list_git_tasks(harbor_git)
    elif harbor_dir is not None:
        rows = [_native_id(d) for d in _task_dirs(harbor_dir)]
    else:
        evaluator = Evaluator.from_client(client_factory())
        rows = select_from_taskset(evaluator, taskset or _spec_taskset(spec_file), None, workspace)[0]
    typer.echo("  #  task")
    for index, row in enumerate(rows):
        typer.echo(f"{index:>3}  {row}")


# --- Progress and results ----------------------------------------------------------------------------------
# Runs are read through ``Evaluator.taskset_runs``; only the bundle Job and the bundle download read Jobs.


# The Job status each settled phase implies; the state cell shows another one, e.g. a cancel after the Job finished.
_EXPECTED_JOB_STATUS = {
    MemberPhase.SUCCEEDED: "completed",
    MemberPhase.FAILED: "error",
    MemberPhase.CANCELLED: "cancelled",
    MemberPhase.EXPIRED: "cancelled",
}


def _text(value: object) -> str:
    """An enum's value, or the text of anything else: ``f"{HelixJobStatus.ACTIVE}"`` would print its name."""
    return str(value.value) if isinstance(value, Enum) else str(value)


def task_state(task: TasksetRunTask) -> str:
    """The state cell of one ``/tasks`` row: ``queued``, ``queued (creating)``, ``registered (active)``, or its phase.

    A queued task with a Job name has a create in flight: it holds a slot, but its Job may not exist yet. A
    settled task shows its Job's status when the phase doesn't imply it: ``cancelled (completed)``.
    """
    phase, job_status = _text(task.phase), _text(task.job_status) if task.job_status else None
    if phase == MemberPhase.IMAGE_READY:
        return "queued (creating)" if task.job else "queued"
    if phase == MemberPhase.REGISTERED and job_status:
        return f"registered ({job_status})"
    if job_status and _EXPECTED_JOB_STATUS.get(phase, job_status) != job_status:
        return f"{phase} ({job_status})"
    return phase


def _state_cell(named: NamedTask, run: TasksetRunStatus | None) -> str:
    """The one state cell of ``runs status`` and ``runs results``: a task with no row is ``waiting`` or ``not run``."""
    if named.task is None:
        return "not run" if run is not None and run.phase in _RUN_TERMINAL else "waiting"
    return task_state(named.task)


def _task_json(named: NamedTask, run: TasksetRunStatus) -> dict[str, Any]:
    """One task in ``runs status --json`` and ``runs results --json``: the SDK's task plus its state cell."""
    return named.model_dump(mode="json", exclude_none=True) | {"state": _state_cell(named, run)}


def _published(publication: PublicationOutcome | None) -> str:
    """The ``published`` cell: what the task's Job published to Intake."""
    if publication is None:
        return "-"
    if _text(publication.status) == "completed":
        return f"✓ {publication.trial_count} trials"
    return f"✗ {publication.error or 'failed'}"


def _may_be_accepted(status_code: int | None) -> bool:
    """Whether a failed submit may still have been accepted: no answer, a 5xx, or a downstream 403."""
    return status_code is None or status_code == 403 or status_code >= 500


def _not_found(workspace: str, run_id: str) -> SelectionError:
    return SelectionError(f"run {workspace}/{run_id} not found")


def _not_found_message(exc: NotFoundError, workspace: str, run_id: str) -> str:
    """``run ws/id not found`` when the 404 came from the run's own URLs; otherwise the server's message."""
    _, _, rest = exc.http_response.request.url.path.partition("/agent-evaluate/runs/")
    return str(_not_found(workspace, run_id) if rest.split("/")[0] == run_id else exc)


def _workspace_flag(workspace: str, configured: str) -> str:
    """`` --workspace <workspace>`` for a printed command, or "" when it's the configured one the command defaults to."""
    return "" if workspace == configured else f" --workspace {workspace}"


def download_run_bundle(
    client: NemoClient, run: TasksetRunStatus, *, workspace: str, output_dir: Path
) -> tuple[str, Path]:
    """Download and unpack ``run``'s merged bundle into ``output_dir``; return the bundle Job and the bundle's path.

    The bundle Job comes from the run's result record (name = run id), which names the attempt that produced it.
    Unpacking goes through a temporary directory, so a failed download or a rejected archive leaves nothing
    behind that would block a retry.

    Raises:
        SelectionError: The run has no merged result, or the bundle is already there.
    """
    run_id = run.run_id
    if run.result_state != "ready":
        raise SelectionError(
            f"run {run_id} has no merged result (phase {run.phase}, result {run.result_state}); "
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
    """The run's newest bundle Job, or None before the controller created one.

    A missing name doesn't end the search: the controller skips a name whose create left it unreadable.
    """
    jobs = JobsClient.from_client(client)
    newest = None
    for attempt in range(1, MAX_BUNDLE_ATTEMPTS + 1):
        try:
            job = jobs.get_job(workspace=workspace, name=bundle_job_name(run_id, attempt)).data()
        except NotFoundError:
            continue
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


def _run_state(run: TasksetRunStatus, bundle: BundleJobView | None) -> str:
    """The run's state in words; phase and result state alone read as contradictions ("completed", "failed")."""
    phase, result = _text(run.phase), _text(run.result_state)
    if phase == "running" and bundle is not None and bundle.status not in _TERMINAL:
        return "merging task results"
    if phase in ("accepted", "running"):
        return phase
    if phase == "completed" and result == "ready":
        return "completed; the run's result is ready"
    if phase == "completed" and result == "failed":
        return "tasks finished, but merging their results into the run's result failed"
    if phase == "failed":
        return "failed; no task completed"
    if phase in ("expired", "cancelled") and result == "ready":
        return f"{phase}; a result over the tasks that finished is ready"
    if phase in ("expired", "cancelled") and result == "failed":
        return f"{phase}; merging the finished tasks' results failed"
    return phase


def _instant(value: datetime) -> datetime:
    """A status timestamp as an aware datetime; a naive one is UTC."""
    return value if value.utcoffset() is not None else value.replace(tzinfo=UTC)


def _clock(value: datetime) -> str:
    """``2026-10-16 12:00:00``: a status timestamp to the second, without its zone."""
    return value.isoformat()[:19].replace("T", " ")


def _queue_line(run: TasksetRunStatus, now: datetime | None) -> str | None:
    """How long the run waited for its first task slot; its clock starts there, not at acceptance."""
    accepted = _instant(run.accepted_at)
    if run.started_at is not None:
        waited = _duration((_instant(run.started_at) - accepted).total_seconds())
        return f"  started   {_clock(run.started_at)} (queued {waited})"
    if run.phase == "accepted":
        return f"  queued    {_duration(((now or datetime.now(UTC)) - accepted).total_seconds())} so far"
    return None


def _status_table(
    run: TasksetRunStatus,
    tasks: Sequence[NamedTask],
    bundle: BundleJobView | None = None,
    *,
    workspace_flag: str = "",
    now: datetime | None = None,
) -> list[str]:
    """The status screen; ``workspace_flag`` (see ``_workspace_flag``) completes its printed commands."""
    phase, result = _text(run.phase), _text(run.result_state)
    queue = _queue_line(run, now)
    lines = [
        f"Run {run.run_id}",
        f"  taskset   {_short_ref(run.taskset_ref)}",
        f"  accepted  {_clock(run.accepted_at)}",
        *([queue] if queue else []),
        f"  status    {_run_state(run, bundle)} (phase {phase}, result {result})",
        f"  progress  {_progress_line(run.progress)}",
    ]
    if bundle is not None:
        lines.append(f"  bundle    {bundle.job} (attempt {bundle.attempt} of {MAX_BUNDLE_ATTEMPTS}): {bundle.status}")
    if bundle is not None and bundle.status in _FAILED:
        lines += [f"            {line}" for line in (bundle.error or "").splitlines()]
        lines.append(f"            logs: nemo jobs get-logs {bundle.job}{workspace_flag}")
    error = run.error
    # The controller's "bundle Job … ended …" only repeats the bundle line; any other error (a deadline) is news.
    if error and not (bundle is not None and error.startswith(f"bundle Job {bundle.job} ended")):
        lines.append(f"  error     {error}")
    phases = [_text(t.task.phase) if t.task is not None else None for t in tasks]
    if result == "failed" and MemberPhase.SUCCEEDED in phases:
        lines.append(f"  completed tasks' scores: nemo evals runs results {run.run_id}{workspace_flag}")
    lines.append("")
    if not tasks:
        return [*lines, "no tasks yet"]
    lines += _task_lines(tasks, run)
    done = [p for p in phases if p in SETTLED_MEMBER_PHASES]
    counts = {p: done.count(p) for p in SETTLED_MEMBER_PHASES}
    stopped = "".join(f", {counts[p]} {p}" for p in (MemberPhase.CANCELLED, MemberPhase.EXPIRED) if counts[p])
    succeeded, failed = counts[MemberPhase.SUCCEEDED], counts[MemberPhase.FAILED]
    lines.append(f"{len(done)}/{len(tasks)} done ({succeeded} succeeded, {failed} failed{stopped})")
    return lines


def _progress_line(progress: TasksetRunProgress | None) -> str:
    """The ``progress`` cell: tasks queued, being created, running (by Job status), then settled by phase."""
    if progress is None:
        return "no tasks yet; the run controller creates them on its next pass"
    members, queued = progress.members, progress.queued
    parts = [f"{queued} queued"] if queued else []
    creating = members.get(MemberPhase.IMAGE_READY, 0) - queued
    if creating:
        parts.append(f"{creating} creating")
    if registered := members.get(MemberPhase.REGISTERED, 0):
        jobs = sorted(((_text(s), n) for s, n in progress.task_jobs.items()), key=lambda item: (-item[1], item[0]))
        by_status = f" ({', '.join(f'{count} {status}' for status, count in jobs)})" if jobs else ""
        parts.append(f"{registered} running{by_status}")
    skip = (MemberPhase.IMAGE_READY, MemberPhase.REGISTERED)
    parts += [f"{members[phase]} {phase}" for phase in MemberPhase if phase not in skip and members.get(phase)]
    tasks = progress.tasks
    return f"{tasks} task{'' if tasks == 1 else 's'}: {', '.join(parts)}"


def _task_lines(
    tasks: Sequence[NamedTask], run: TasksetRunStatus | None, *, show_publication: bool = True
) -> list[str]:
    """The task table: one row per task with its state and Job, then what it published and its error, when set.

    The ``published`` column appears only when ``show_publication`` is set and some task published.
    """
    states = [_state_cell(t, run) for t in tasks]
    jobs = [(t.task.job if t.task else None) or "-" for t in tasks]
    width = max([4, *(len(t.name) for t in tasks)])
    state_width = max([5, *(len(state) for state in states)])
    job_width = max([3, *(len(job) for job in jobs)])
    publications = [t.task.publication if t.task else None for t in tasks]
    shown = show_publication and any(p is not None for p in publications)
    published = [_published(p) for p in publications] if shown else []
    published_width = max([9, *(len(cell) for cell in published)])
    header = f"{'#':>3}  {'task':<{width}}  {'state':<{state_width}}  {'Job':<{job_width}}"
    lines = [f"{header}  published" if published else header.rstrip()]
    for i, task in enumerate(tasks):
        row = f"{task.index:>3}  {task.name:<{width}}  {states[i]:<{state_width}}  {jobs[i]:<{job_width}}"
        if published:
            row += f"  {published[i]:<{published_width}}"
        if task.task is not None and task.task.error:
            row += f"  {task.task.error.splitlines()[0]}"
        lines.append(row.rstrip())
    return lines


def _page_footer(page: Page[Any], noun: str) -> str:
    """``page 1 of 3 (45 runs); next: --page 2``, or the count on the last page."""
    pagination = page.pagination
    total = pagination.total_results if pagination else None
    pages, current = (pagination.total_pages, pagination.page) if pagination else (1, 1)
    if pages > current:
        return f"page {current} of {pages} ({total} {noun}s); next: --page {current + 1}"
    count = total if total is not None else len(page.data)
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _tasks_table(page: Page[TasksetRunTask]) -> list[str]:
    """The ``runs tasks`` screen: one ``/tasks`` page, by member name."""
    if not page.data:
        return ["No tasks."]
    tasks = [NamedTask(index=t.task_index, name=t.task, task=t) for t in page.data]
    return [*_task_lines(tasks, None, show_publication=False), _page_footer(page, "task")]


def _list_table(page: Page[TasksetRunStatus]) -> list[str]:
    if not page.data:
        return ["No runs."]
    rows = [
        (
            run.run_id,
            _text(run.phase),
            _text(run.result_state),
            _short_ref(run.taskset_ref),
            _clock(run.accepted_at),
            run.created_by or "-",
        )
        for run in page.data
    ]
    header = ("run", "phase", "result", "taskset", "accepted", "by")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [header, *rows]]
    return [*lines, _page_footer(page, "run")]


def _short_ref(ref: str) -> str:
    """``workspace/name#digest`` with the digest cut to 8 characters; the table has no room for 64."""
    name, sep, digest = ref.partition("#")
    return f"{name}{sep}{digest[:8]}"


def _results_table(results: TasksetRunResults, *, workspace_flag: str = "") -> list[str]:
    """The results screen; ``workspace_flag`` (see ``_workspace_flag``) completes its printed command."""
    tasks, record = results.tasks, results.result
    if not tasks:
        return ["no tasks yet"]
    found = [t.result for t in tasks if t.result is not None]
    names = list(dict.fromkeys(score.name for result in found for score in result.scores.scores))
    states = [_state_cell(t, results.run) for t in tasks]
    width = max([4, *(len(t.name) for t in tasks)])
    state_width = max([10, *(len(state) for state in states)])
    header = f"{'#':>3}  {'task':<{width}}  {'status':<{state_width}}  trials  errors"
    lines = ["  ".join([header, *names])]
    for task, state in zip(tasks, states):
        row = f"{task.index:>3}  {task.name:<{width}}  {state:<{state_width}}"
        if task.result is None:
            lines.append(f"{row}  {'-':>6}  {'-':>6}")
            continue
        summary = task.result.summary
        by_name = task.result.scores.scores_by_name
        cells = [_cell(by_name[name].mean if name in by_name else None, len(name)) for name in names]
        trials = summary.trial_count if summary else "-"
        errors = summary.error_count if summary else "-"
        lines.append("  ".join([f"{row}  {trials:>6}  {errors:>6}", *cells]))
    unrecorded = [
        t.task.job or t.task.task
        for t in tasks
        if t.result is None and t.task is not None and _text(t.task.job_status) == "completed"
    ]
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
    trials = sum(r.summary.trial_count for r in found if r.summary)
    errors = sum(r.summary.error_count for r in found if r.summary)
    lines.append(
        f"run (combined here; no merged bundle yet): {len(found)}/{len(tasks)} tasks with results, {trials} trials, {errors} errors"
    )
    for score in rollup_scores(found):
        mean = f"{score.mean:.3f}" if score.mean is not None else "n/a"
        lines.append(f"  {score.name}: {mean} (n={score.count}, nan={score.nan_count}, tasks={score.tasks})")
    return lines


def _cell(value: float | None, width: int) -> str:
    return f"{value:.2f}".rjust(width) if value is not None else "-".rjust(width)


def _dump(model: BaseModel) -> str:
    """A model as indented JSON, without its unset fields, as the API prints it."""
    return json.dumps(model.model_dump(mode="json", exclude_none=True), indent=2)


_TERMINAL = frozenset({"completed", "error", "cancelled"})
_FAILED = frozenset({"error", "cancelled"})
_RUN_TERMINAL = frozenset({"completed", "failed", "cancelled", "expired"})


def add_progress_commands(runs: typer.Typer, client_factory: Callable[[], NemoClient]) -> None:
    """Register ``runs list``, ``runs status``, ``runs tasks``, ``runs cancel`` and ``runs results`` under ``runs``."""

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
        taskset_runs = Evaluator.from_client(client).taskset_runs
        try:
            body = taskset_runs.list(workspace=workspace, active=active, mine=mine, page=page, page_size=page_size)
        except NemoClientError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(_dump(body) if output_json else "\n".join(_list_table(body)))

    @runs.command("status")
    def status(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the run; default: configured."),
        watch: bool = typer.Option(False, "--watch", help="Refresh until every task Job is done."),
        poll_seconds: float = typer.Option(15.0, "--poll-seconds", help="Refresh interval with --watch."),
        output_json: bool = typer.Option(False, "--json", help="Print JSON."),
    ) -> None:
        """Show a run's progress and each task's state."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        taskset_runs = Evaluator.from_client(client).taskset_runs
        try:
            while True:
                run = taskset_runs.retrieve(run_id, workspace=workspace)  # once per refresh: header and rows agree
                tasks = list(taskset_runs.iter_tasks(run))
                bundle = newest_bundle_job(client, run_id, workspace=workspace)
                if output_json:
                    body = {
                        "run": run.model_dump(mode="json", exclude_none=True),
                        "bundle": bundle and bundle.__dict__,
                        "tasks": [_task_json(task, run) for task in tasks],
                    }
                    typer.echo(json.dumps(body, indent=2))
                else:
                    flag = _workspace_flag(workspace, client.resolve_workspace())
                    typer.echo("\n".join(_status_table(run, tasks, bundle, workspace_flag=flag)))
                if not watch or run.phase in _RUN_TERMINAL:
                    return
                time.sleep(poll_seconds)
                typer.echo("")
        except NotFoundError as exc:
            typer.echo(f"Error: {_not_found(workspace, run_id)}", err=True)
            raise typer.Exit(code=1) from exc
        except NemoClientError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    @runs.command("tasks")
    def tasks(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        phase: MemberPhase | None = typer.Option(None, "--phase", help="Only tasks in this phase."),
        page: int = typer.Option(1, "--page", min=1, help="One-based page number."),
        page_size: int = typer.Option(100, "--page-size", min=1, max=1000, help="Tasks per page."),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the run; default: configured."),
        output_json: bool = typer.Option(False, "--json", help="Print the API page as JSON."),
    ) -> None:
        """List a run's tasks with their state and Job."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        taskset_runs = Evaluator.from_client(client).taskset_runs
        try:
            body = taskset_runs.tasks(run_id, workspace=workspace, phase=phase, page=page, page_size=page_size)
        except NotFoundError as exc:
            typer.echo(f"Error: {_not_found(workspace, run_id)}", err=True)
            raise typer.Exit(code=1) from exc
        except NemoClientError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(_dump(body) if output_json else "\n".join(_tasks_table(body)))

    @runs.command("cancel")
    def cancel(
        run_id: str = typer.Argument(..., help="Run name, e.g. run-7f3a…"),
        workspace: str | None = typer.Option(None, "--workspace", help="Workspace of the run; default: configured."),
        output_json: bool = typer.Option(False, "--json", help="Print the run's status as JSON."),
    ) -> None:
        """Request a run's cancel; the run controller acts on it, and the run keeps its phase until then."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        try:
            run = Evaluator.from_client(client).taskset_runs.cancel(run_id, workspace=workspace)
        except NotFoundError as exc:
            typer.echo(f"Error: {_not_found(workspace, run_id)}", err=True)
            raise typer.Exit(code=1) from exc
        except NemoClientError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if output_json:
            typer.echo(_dump(run))
            return
        flag = _workspace_flag(workspace, client.resolve_workspace())
        requested = run.cancel_requested_at.isoformat() if run.cancel_requested_at else "-"
        typer.echo(f"Cancel requested for {run_id} at {requested} (phase {_text(run.phase)}).")
        typer.echo(f"Follow it with: nemo evals runs status {run_id}{flag}")

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
            try:
                run = Evaluator.from_client(client).taskset_runs.retrieve(run_id, workspace=workspace)
            except NotFoundError as exc:
                raise _not_found(workspace, run_id) from exc
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
            found = Evaluator.from_client(client).taskset_runs.results(run_id, workspace=workspace)
        except NotFoundError as exc:  # the run, or its merged record
            typer.echo(f"Error: {_not_found_message(exc, workspace, run_id)}", err=True)
            raise typer.Exit(code=1) from exc
        except NemoClientError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if output_json:
            body = found.model_dump(mode="json", exclude_none=True)
            body["tasks"] = [_task_json(task, found.run) for task in found.tasks]
            typer.echo(json.dumps(body, indent=2))
            return
        flag = _workspace_flag(workspace, client.resolve_workspace())
        typer.echo("\n".join(_results_table(found, workspace_flag=flag)))
