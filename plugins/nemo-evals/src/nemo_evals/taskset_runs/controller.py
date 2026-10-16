# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator run controller: notice when a run's task Jobs finish, merge their bundles, save its result.

M2 story 0's slice of M4 story 2 (the controller) and M5 stories 6-8 (result record, bundle Job).
One pass per open run, every step safe to repeat:

- some task Job still running → the run is ``running``;
- every task Job finished, none completed → ``failed``, ``result_state=none``;
- otherwise → create (or adopt) the bundle Job; once it completes, save the run's result record
  (name = run id) and mark the run ``completed``, ``result_state=ready``.

Workers write files only; this controller writes the records.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar, Protocol, cast

from filesets import AsyncFilesetFileSystem, parse_fileset_ref
from nemo_evals.entities import AgentEvalResultEntity
from nemo_evals.jobs.agent_evaluate import DEFAULT_RESULT_NAME, SUMMARY_FILE_NAME
from nemo_evals.jobs.result_persistence import agent_eval_result_entity
from nemo_evals.jobs.taskset_bundle import BUNDLE_JOB_SOURCE, TasksetBundleJob
from nemo_evals.taskset_runs.entities import ResultState, TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.identifiers import bundle_job_name, task_job_name
from nemo_evals.taskset_runs.specs import AcceptedTaskOutput, TasksetBundleSpec, TasksetRunPlan
from nemo_evals.taskset_runs.store import TasksetRunStore
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
DEVELOPMENT_CALLER = "user:development"


class RunJobs(Protocol):
    """The Jobs, Files and record operations a completion pass needs, acting as the run's creator."""

    async def get(self, name: str) -> HelixJobResponse | None: ...

    async def result_url(self, job: str, result: str) -> str | None: ...

    async def create(self, request: CreateHelixJobRequest) -> None: ...

    async def read_summary(self, bundle_ref: str) -> AgentEvalSummary: ...

    async def save_record(self, entity: EntityBase) -> None: ...

    async def get_record(self, name: str) -> AgentEvalResultEntity | None: ...


@dataclass(frozen=True)
class PassOutcome:
    """What one pass did, for logs and tests."""

    action: str  # waiting | running | failed | bundling | completed | bundle_failed
    detail: str = ""


async def complete_run(
    run: TasksetRunEntity, plan: TasksetRunPlan, jobs: RunJobs, store: TasksetRunStore
) -> PassOutcome:
    """One completion pass over ``run``: safe to repeat, and to run twice at once."""
    task_jobs = [await jobs.get(task_job_name(run.name, i)) for i in range(len(plan.spec.tasks))]
    if any(job is None or _status(job) not in TERMINAL_JOB_STATUSES for job in task_jobs):
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

    bundle = await _ensure_bundle_job(run, plan, accepted, task_jobs, jobs)
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


async def _ensure_bundle_job(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    task_jobs: list[HelixJobResponse | None],
    jobs: RunJobs,
) -> HelixJobResponse | None:
    """The newest bundle Job; a new attempt starts only when there is none or the newest failed."""
    newest: HelixJobResponse | None = None
    attempt = 0
    for n in range(1, MAX_BUNDLE_ATTEMPTS + 1):
        found = await jobs.get(bundle_job_name(run.name, n))
        if found is None:
            break
        newest, attempt = found, n
    if newest is not None and (_status(newest) not in FAILED_JOB_STATUSES or attempt >= MAX_BUNDLE_ATTEMPTS):
        return newest
    name = bundle_job_name(run.name, attempt + 1)
    spec = TasksetBundleSpec(
        run_id=run.name,
        workspace=run.workspace,
        plan_ref=run.plan_ref,
        plan_digest=run.plan_digest,
        accepted=accepted,
        started_at=run.accepted_at,
        finished_at=max(_utc(job.updated_at) for job in task_jobs if job is not None),
    )
    request = CreateHelixJobRequest(
        name=name,
        description=f"Merged result bundle of taskset run {run.name}",
        spec=spec.model_dump(mode="json"),
        platform_spec=await TasksetBundleJob.compile(
            spec=spec, profile=plan.dispatcher_profile, image=plan.dispatcher_image
        ),
        source=BUNDLE_JOB_SOURCE,
    )
    try:
        await jobs.create(request)
        logger.info("Created bundle Job %s for run %s (%d tasks)", name, run.name, len(accepted))
    except ConflictError:
        pass  # another pass created it; read it below
    return await jobs.get(name)


def _utc(value: datetime) -> datetime:
    """Jobs returns naive timestamps; they are UTC."""
    return value if value.utcoffset() is not None else value.replace(tzinfo=timezone.utc)


def _status(job: HelixJobResponse) -> str:
    status = job.status
    return status.value if hasattr(status, "value") else str(status)


class PlatformRunJobs:
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
                outcome = await complete_run(run, plan, PlatformRunJobs(client, run.workspace), self.store)
            if outcome.action not in {"waiting", "bundling"}:
                logger.info("Run %s/%s: %s %s", run.workspace, run.name, outcome.action, outcome.detail)
        except NemoEntityConflictError:
            logger.debug("Run %s changed during the pass; next pass re-reads it", run.name)
