# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator run controller: notice when a run's task Jobs finish, merge their bundles, save its result.

M2 story 0's slice of M4 story 2 (the controller) and M5 stories 6-8 (result record, bundle Job).
One pass per open run, every step safe to repeat:

- some task Job still running → the run is ``running``;
- some task Job unfinished or missing at ``execution_deadline_at`` → cancel the unfinished ones;
  the run ends ``expired``, ``result_state=none``;
- every task Job finished, none completed → ``failed``, ``result_state=none``;
- otherwise → create (or adopt) the bundle Job; once it completes, save the run's result record
  (name = run id) and mark the run ``completed``, ``result_state=ready``;
- no result saved by ``finalization_deadline_at`` → cancel the bundle Job; ``result_state=failed``.

A bundle Job whose create failed midway never runs, so the next attempt's name is used (see
``STALLED_CREATE_SECONDS``).

Task and bundle Job names are predictable, so a Job under one counts only when it runs exactly what this
run would create there: someone else's Job is never bundled, adopted or cancelled.

Expiry keeps no partial result. TODO(M4 stories 5, 7): merge the tasks that finished.

Task Jobs save their result bundle as a Job result and, being agent-evaluate Jobs, also a per-task
result record named after the Job. The bundle Job saves files only. This controller writes the run's
state and its result record (name = run id), and does not publish to Intake.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import ClassVar, Protocol, cast

from filesets import AsyncFilesetFileSystem, parse_fileset_ref
from nemo_evals.entities import AgentEvalResultEntity
from nemo_evals.jobs.agent_evaluate import DEFAULT_RESULT_NAME, SUMMARY_FILE_NAME
from nemo_evals.jobs.result_persistence import agent_eval_result_entity
from nemo_evals.jobs.taskset_bundle import BUNDLE_JOB_SOURCE, TasksetBundleJob
from nemo_evals.taskset_runs.entities import ResultState, TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.identifiers import bundle_job_name
from nemo_evals.taskset_runs.specs import AcceptedTaskOutput, TasksetBundleSpec, TasksetRunPlan
from nemo_evals.taskset_runs.store import TasksetRunStore
from nemo_evals.taskset_runs.task_jobs import runs_request, task_job_request
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import EntityBase, EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.file_manager import AsyncFilesetFileManager
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary

logger = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})
FAILED_JOB_STATUSES = frozenset({"error", "cancelled"})
NOT_STARTED_JOB_STATUSES = frozenset({"created", "pending"})
MAX_BUNDLE_ATTEMPTS = 3
STALLED_CREATE_SECONDS = 60
"""A bundle Job still ``created`` without a step this long after it was saved will never run.

Jobs saves a Job, then its attempt, then the attempt's first step, with no rollback, and schedules only steps.
A create that failed after the first write leaves a Job that either shows no step or isn't shown at all.
"""
DEVELOPMENT_CALLER = "user:development"


class RunJobs(Protocol):
    """The Jobs, Files and record operations a completion pass needs, acting as the run's creator."""

    async def get(self, name: str) -> HelixJobResponse | None: ...

    async def result_url(self, job: str, result: str) -> str | None: ...

    async def create(self, request: CreateHelixJobRequest) -> None: ...

    async def has_steps(self, name: str) -> bool:
        """True when Job ``name``'s current attempt has at least one step."""
        ...

    async def read_summary(self, bundle_ref: str) -> AgentEvalSummary: ...

    async def save_record(self, entity: EntityBase) -> None: ...

    async def get_record(self, name: str) -> AgentEvalResultEntity | None: ...

    async def cancel(self, name: str) -> None:
        """Cancel Job ``name``; a missing or already finished Job is not an error."""
        ...


@dataclass(frozen=True)
class PassOutcome:
    """What one pass did, for logs and tests."""

    action: str  # waiting | running | expired | failed | bundling | completed | bundle_failed | finalization_missed
    detail: str = ""


async def complete_run(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    jobs: RunJobs,
    store: TasksetRunStore,
    *,
    now: datetime | None = None,
) -> PassOutcome:
    """One completion pass over ``run`` at ``now`` (default: current UTC); safe to repeat, and to run twice at once."""
    now = now or datetime.now(timezone.utc)
    task_jobs = [await _own_task_job(run, plan, i, jobs) for i in range(len(plan.spec.tasks))]
    if any(job is None or _status(job) not in TERMINAL_JOB_STATUSES for job in task_jobs):
        if now >= _utc(run.execution_deadline_at):
            return await _expire(run, task_jobs, jobs, store)
        # A missing Job is left to submit's 503 retry, which creates it from the plan.
        started = any(job is not None and _status(job) not in NOT_STARTED_JOB_STATUSES for job in task_jobs)
        if run.phase is TasksetRunPhase.ACCEPTED and started:
            await store.set_state(run, phase=TasksetRunPhase.RUNNING)
            return PassOutcome("running")
        return PassOutcome("waiting")

    accepted: list[AcceptedTaskOutput] = []
    for index, job in enumerate(task_jobs):
        assert job is not None
        if _status(job) == "completed" and (ref := await jobs.result_url(job.name, DEFAULT_RESULT_NAME)):
            accepted.append(AcceptedTaskOutput(task_index=index, job=job.name, bundle_ref=ref))
    if not accepted:
        await store.set_state(
            run, phase=TasksetRunPhase.FAILED, result_state=ResultState.NONE, error="no task Job completed"
        )
        return PassOutcome("failed", "no task Job completed")
    if now >= _utc(run.finalization_deadline_at):
        return await _miss_finalization(run, plan, accepted, jobs, store)

    try:
        bundle = await _ensure_bundle_job(run, plan, accepted, task_jobs, jobs, now)
    except ForeignBundleJobError as exc:
        # Left open, not failed: deleting the Job lets the next pass bundle; the finalization deadline caps the wait.
        logger.warning("Run %s: %s", run.name, exc)
        return PassOutcome("bundling", str(exc))
    if bundle is None or _status(bundle) not in TERMINAL_JOB_STATUSES:
        return PassOutcome("bundling", bundle.name if bundle else "")
    if _status(bundle) == "completed":
        bundle_ref = await jobs.result_url(bundle.name, DEFAULT_RESULT_NAME)
        if bundle_ref is None:
            return PassOutcome("bundling", f"{bundle.name} completed without {DEFAULT_RESULT_NAME}")
        summary = await jobs.read_summary(bundle_ref)
        record = agent_eval_result_entity(
            summary,
            name=run.name,
            workspace=run.workspace,
            job_id=bundle.name,
            target=plan.spec.target,
            bundle_ref=bundle_ref,
        )
        try:
            await jobs.save_record(record)
        except NemoEntityConflictError:
            # An earlier pass saved this bundle's record, then failed before marking the run done.
            existing = await jobs.get_record(run.name)
            if existing is None or existing.job_id != bundle.name:
                raise
        await store.set_state(run, phase=TasksetRunPhase.COMPLETED, result_state=ResultState.READY)
        return PassOutcome("completed", bundle_ref)
    # The newest attempt failed and no further attempt is allowed.
    error = f"bundle Job {bundle.name} ended {_status(bundle)}"
    await store.set_state(run, phase=TasksetRunPhase.COMPLETED, result_state=ResultState.FAILED, error=error)
    return PassOutcome("bundle_failed", error)


async def _expire(
    run: TasksetRunEntity, task_jobs: list[HelixJobResponse | None], jobs: RunJobs, store: TasksetRunStore
) -> PassOutcome:
    """Cancel the unfinished task Jobs, then end the run ``expired`` without a result.

    Cancels first, so a run never ends with Jobs left running. If the state write then fails, the next
    pass sees only finished Jobs and ends the run ``failed`` (or bundles what completed) instead.

    TODO(M4 story 8): delete the run's sandboxes by their ``nemo-run`` label. A dispatcher that is
    cancelled, OOM-killed or evicted leaves its sandboxes until OpenSandbox's ``max_sandbox_lifetime_s`` TTL.
    """
    unfinished = 0
    for job in task_jobs:
        if job is None or _status(job) not in TERMINAL_JOB_STATUSES:
            unfinished += 1
            if job is not None:
                await jobs.cancel(job.name)
    error = f"execution deadline passed with {unfinished} of {len(task_jobs)} task Jobs unfinished"
    await store.set_state(run, phase=TasksetRunPhase.EXPIRED, result_state=ResultState.NONE, error=error)
    return PassOutcome("expired", error)


async def _own_task_job(
    run: TasksetRunEntity, plan: TasksetRunPlan, index: int, jobs: RunJobs
) -> HelixJobResponse | None:
    """Task ``index``'s Job, or None when it is missing or runs something else.

    Someone else's Job under the name counts as missing, like submit's check: its result never reaches the
    bundle and the execution deadline never cancels it.
    """
    request = await task_job_request(plan, run, index, plan.task_environments[index])
    assert request.name is not None
    job = await jobs.get(request.name)
    if job is not None and not runs_request(job, request):
        logger.warning("Run %s: Job %s runs something else; treated as missing", run.name, job.name)
        return None
    return job


async def _miss_finalization(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    jobs: RunJobs,
    store: TasksetRunStore,
) -> PassOutcome:
    """Past the finalization deadline without a saved result: cancel the bundle Job and fail the result."""
    names = [bundle_job_name(run.name, n) for n in range(1, MAX_BUNDLE_ATTEMPTS + 1)]
    record = await jobs.get_record(run.name)
    if record is not None and record.job_id in names:
        # An earlier pass saved the result, then failed before marking the run done.
        await store.set_state(run, phase=TasksetRunPhase.COMPLETED, result_state=ResultState.READY)
        return PassOutcome("completed", record.bundle_ref)
    for name in names:
        bundle = await jobs.get(name)
        if bundle is None or _status(bundle) in TERMINAL_JOB_STATUSES:
            continue
        try:
            await _own_bundle_job(bundle, run, plan, accepted)
        except ForeignBundleJobError as exc:
            logger.warning("Run %s: not cancelling: %s", run.name, exc)
            continue
        await jobs.cancel(name)
    error = "finalization deadline passed before the result was saved"
    await store.set_state(run, phase=TasksetRunPhase.COMPLETED, result_state=ResultState.FAILED, error=error)
    return PassOutcome("finalization_missed", error)


class ForeignBundleJobError(Exception):
    """A Job under a bundle Job's name runs something other than this run's bundle."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Job {name} exists and runs something else; delete it with `nemo jobs delete`")


async def _ensure_bundle_job(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    task_jobs: list[HelixJobResponse | None],
    jobs: RunJobs,
    now: datetime,
) -> HelixJobResponse | None:
    """The newest bundle Job; a new attempt starts only when there is none or the newest failed or never runs.

    Bundle Job names are predictable, so a Job found under one is adopted only when it runs this run's
    bundle: anyone who can create Jobs in the workspace could otherwise supply the run's result.

    A name whose create failed midway is skipped: its Job is stalled without a step, or Jobs answers its
    create with 409 yet doesn't show it.

    Why: Jobs saves a Job, its attempt and the attempt's first step as separate writes, with no rollback,
    and schedules only steps. A failure after the first write leaves the name taken by a Job that never runs:

    - attempt not saved or not linked: GET answers 404, yet every create of the name answers 409;
    - first step not saved: the Job stays ``created`` forever.

    Adopting such a Job would keep the run ``bundling`` until its finalization deadline and fail its result
    although every task completed, so the next attempt's name is used instead.

    TODO: AIRCORE-1379 makes Jobs create atomic. Once it ships, remove these workarounds: the stalled and
    unreadable-name skips here, ``_stalled``, ``STALLED_CREATE_SECONDS``, ``RunJobs.has_steps``, and submit's
    steps check in ``task_jobs._adoption_refused``.

    Raises:
        ForeignBundleJobError: A Job under a bundle Job's name runs something else.
    """
    newest: HelixJobResponse | None = None
    for n in range(1, MAX_BUNDLE_ATTEMPTS + 1):
        name = bundle_job_name(run.name, n)
        found = await jobs.get(name)
        if found is None:
            found = await _create_bundle_job(run, plan, accepted, task_jobs, jobs, name)
            if found is not None:
                return await _own_bundle_job(found, run, plan, accepted)
            # 409 on create, 404 on GET: Jobs saved the Job but not its attempt, so the name is lost for good.
            logger.warning("Run %s: bundle Job %s exists but can't be read; trying the next attempt", run.name, name)
            continue
        newest = await _own_bundle_job(found, run, plan, accepted)
        # Saved without its first step, so nothing will ever schedule it: treat it like a failed attempt.
        if await _stalled(newest, jobs, now):
            logger.warning("Run %s: bundle Job %s was saved without a step; trying the next attempt", run.name, name)
        elif _status(newest) not in FAILED_JOB_STATUSES:
            return newest
    return newest


async def _create_bundle_job(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    task_jobs: list[HelixJobResponse | None],
    jobs: RunJobs,
    name: str,
) -> HelixJobResponse | None:
    """Create bundle Job ``name``; return the Job now under that name, if Jobs shows one."""
    finished_at = max(_utc(job.updated_at) for job in task_jobs if job is not None)
    request = await _bundle_request(run, plan, accepted, name, finished_at)
    try:
        await jobs.create(request)
        logger.info("Created bundle Job %s for run %s (%d tasks)", name, run.name, len(accepted))
    except ConflictError:
        pass  # another pass created it, or someone else did; the caller checks
    return await jobs.get(name)


async def _stalled(job: HelixJobResponse, jobs: RunJobs, now: datetime) -> bool:
    """True when ``job`` is still ``created`` without a step ``STALLED_CREATE_SECONDS`` after it was saved."""
    return (
        _status(job) == "created"
        and now - _utc(job.created_at) >= timedelta(seconds=STALLED_CREATE_SECONDS)
        and not await jobs.has_steps(job.name)
    )


async def _bundle_request(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    name: str,
    finished_at: datetime,
) -> CreateHelixJobRequest:
    """The create request for bundle Job ``name``."""
    spec = TasksetBundleSpec(
        run_id=run.name,
        workspace=run.workspace,
        plan_ref=run.plan_ref,
        plan_digest=run.plan_digest,
        accepted=accepted,
        started_at=run.accepted_at,
        finished_at=finished_at,
    )
    return CreateHelixJobRequest(
        name=name,
        description=f"Merged result bundle of taskset run {run.name}",
        spec=spec.model_dump(mode="json"),
        platform_spec=await TasksetBundleJob.compile(
            spec=spec, profile=plan.dispatcher_profile, image=plan.dispatcher_image
        ),
        source=BUNDLE_JOB_SOURCE,
    )


async def _own_bundle_job(
    job: HelixJobResponse, run: TasksetRunEntity, plan: TasksetRunPlan, accepted: list[AcceptedTaskOutput]
) -> HelixJobResponse:
    """``job`` when it is the bundle Job this run would create under its name.

    ``finished_at`` comes from the Job itself: the task Jobs' ``updated_at`` it was derived from may move
    between passes, and a Job differing only there still merges exactly this run's accepted tasks.

    Raises:
        ForeignBundleJobError: ``job`` runs something else.
    """
    finished_at = job.spec.get("finished_at")
    if not isinstance(finished_at, str):
        raise ForeignBundleJobError(job.name)
    try:
        expected = await _bundle_request(run, plan, accepted, job.name, datetime.fromisoformat(finished_at))
    except ValueError as exc:
        raise ForeignBundleJobError(job.name) from exc
    if not runs_request(job, expected):
        raise ForeignBundleJobError(job.name)
    return job


def _utc(value: datetime) -> datetime:
    """Jobs returns naive timestamps; they are UTC."""
    return value if value.utcoffset() is not None else value.replace(tzinfo=timezone.utc)


def _status(job: HelixJobResponse) -> str:
    status = job.status
    return status.value if hasattr(status, "value") else str(status)


class JobsServiceRunJobs:
    """:class:`RunJobs` over the platform, through one client acting for the run's creator."""

    def __init__(self, client: AsyncNemoClient, workspace: str) -> None:
        self._jobs = AsyncJobsClient.from_client(client)
        self._files = AsyncFilesClient.from_client(client)
        self._entities = EntityClient(AsyncEntitiesClient.from_client(client))
        self._workspace = workspace

    async def get(self, name: str) -> HelixJobResponse | None:
        try:
            return (await self._jobs.get_job(workspace=self._workspace, name=name)).data()
        except NotFoundError:
            return None

    async def result_url(self, job: str, result: str) -> str | None:
        try:
            return (
                (await self._jobs.get_job_result(workspace=self._workspace, job=job, name=result)).data().artifact_url
            )
        except NotFoundError:
            return None

    async def create(self, request: CreateHelixJobRequest) -> None:
        await self._jobs.create_job(workspace=self._workspace, body=request)

    async def has_steps(self, name: str) -> bool:
        response = await self._jobs.list_steps(workspace=self._workspace, name=name, query_params={"page_size": 1})
        return bool(response.page().items)

    async def read_summary(self, bundle_ref: str) -> AgentEvalSummary:
        import tempfile

        workspace, fileset, _ = parse_fileset_ref(bundle_ref, workspace_fallback=self._workspace)
        manager = AsyncFilesetFileManager(
            workspace=workspace or self._workspace,
            fileset_name=fileset or "",
            filesystem=AsyncFilesetFileSystem(client=self._files),
            ensure_fileset_exists=False,
        )
        with tempfile.TemporaryDirectory(prefix="run-summary-") as scratch:
            path = (await manager.download_from_url(f"{bundle_ref}/{SUMMARY_FILE_NAME}", local_dir=scratch)).path
            return AgentEvalSummary.model_validate_json(path.read_text(encoding="utf-8"))

    async def save_record(self, entity: EntityBase) -> None:
        await self._entities.save(entity)

    async def get_record(self, name: str) -> AgentEvalResultEntity | None:
        try:
            return await self._entities.get(AgentEvalResultEntity, name=name, workspace=self._workspace)
        except NemoEntityNotFoundError:
            return None

    async def cancel(self, name: str) -> None:
        try:
            await self._jobs.cancel_job(workspace=self._workspace, name=name)
        except NotFoundError:
            pass
        except ConflictError:
            # 409 also covers a Job still running: a rerun replaced its step, or it is pausing. Raise for
            # those so the run stays open and the next pass cancels again; a Job that finished is done.
            job = await self.get(name)
            if job is not None and _status(job) not in TERMINAL_JOB_STATUSES:
                raise


ClientFactory = Callable[[str | None], AbstractAsyncContextManager[AsyncNemoClient]]


@asynccontextmanager
async def _service_client(on_behalf_of: str | None) -> AsyncGenerator[AsyncNemoClient]:
    from nemo_helix_plugin.client_provider import get_async_nemo_client

    async with get_async_nemo_client(as_service="evals", internal=True, on_behalf_of=on_behalf_of) as client:
        yield client


class EvaluatorRunController(NemoController):
    """Complete Kubernetes taskset runs once their task Jobs finish (see module docstring)."""

    name: ClassVar[str] = "evals-taskset-runs"
    dependencies: ClassVar[list[str]] = ["entities", "files", "jobs"]

    def __init__(self, client_factory: ClientFactory = _service_client) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._store: TasksetRunStore | None = None
        self._service_context: AbstractAsyncContextManager[AsyncNemoClient] | None = None

    @property
    def interval_seconds(self) -> float:
        return 15.0

    async def on_startup(self) -> None:
        self._service_context = self._client_factory(None)
        client = await self._service_context.__aenter__()
        self._store = TasksetRunStore(AsyncEntitiesClient.from_client(client), AsyncFilesClient.from_client(client))

    async def on_shutdown(self) -> None:
        if self._service_context is not None:
            await self._service_context.__aexit__(None, None, None)

    @property
    def store(self) -> TasksetRunStore:
        if self._store is None:
            raise RuntimeError("EvaluatorRunController used before on_startup()")
        return self._store

    async def list_objects(self) -> list:
        try:
            return await self.store.list_open_runs()
        except Exception:
            logger.exception("Listing open taskset runs failed")
            return []

    async def reconcile_one(self, obj: object) -> None:
        run = cast(TasksetRunEntity, obj)
        try:
            plan = await self.store.load_plan(run)
            async with self._client_factory(run.created_by or DEVELOPMENT_CALLER) as client:
                outcome = await complete_run(run, plan, JobsServiceRunJobs(client, run.workspace), self.store)
            if outcome.action not in {"waiting", "bundling"}:
                logger.info("Run %s/%s: %s %s", run.workspace, run.name, outcome.action, outcome.detail)
        except NemoEntityConflictError:
            logger.debug("Run %s changed during the pass; next pass re-reads it", run.name)
