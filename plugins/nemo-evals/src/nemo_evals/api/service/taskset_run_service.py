# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-scoped run acceptance and stored-record reads without execution joins."""

import asyncio
import dataclasses
import logging
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException
from nemo_evals.api.fields import TasksetRef, parse_subentity_ref
from nemo_evals.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evals.api.taskset_run_schemas import (
    TasksetRunInput,
    TasksetRunStatus,
    TasksetRunSubmitResponse,
    TasksetRunTask,
    request_digest,
    to_status,
    to_task,
)
from nemo_evals.config import EvaluatorConfig, HarborTaskJobsConfig, platform_config
from nemo_evals.entities import TasksetEntity, TasksetRevisionEntity
from nemo_evals.harbor.remote.preparation import (
    ImagePreparer,
    PreparationClients,
    TaskPreparation,
    prepare_task_images,
)
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy
from nemo_evals.harbor.remote.resources import defaulted_resources_notice
from nemo_evals.harbor.remote.sandbox_policy import SandboxPolicyError, resolve_sandbox_policy
from nemo_evals.image_builds.service import ImageBuildService, JobsServiceBuildJobs, builder_authorizer
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalSpec, ResolvedTask, registered_agent_source, target_agent_identity
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment
from nemo_evals.revisions import RevisionNotFoundError, get_revision
from nemo_evals.source_authorization import remember_allowed
from nemo_evals.taskset_runs.entities import TERMINAL_RUN_PHASES, MemberPhase, TasksetRunEntity
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.plan_storage import delete_fileset_ref
from nemo_evals.taskset_runs.run_evaluation import (
    ForeignEvaluationError,
    IntakeRunEvaluations,
    RunEvaluations,
    ensure_run_evaluation,
)
from nemo_evals.taskset_runs.specs import RunIntakePublication, TasksetRunPlan
from nemo_evals.taskset_runs.store import PlanStorageError, RunAcceptance, RunRequestConflict, TasksetRunStore
from nemo_evals.taskset_runs.submission import check_timeout, resolve_deadlines
from nemo_evals.taskset_runs.task_jobs import JobsServiceTaskJobs, TaskJobs, TaskJobsIncompleteError, create_task_jobs
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import AuthenticationError, NemoClientError, PermissionDeniedError
from nemo_helix_plugin.config import NemoHelixConfig
from nemo_helix_plugin.dependencies import RequestAuthorizer
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.schema import Page

DEVELOPMENT_CALLER = "user:development"
"""Stable creator identity for headerless requests when authentication is disabled."""

logger = logging.getLogger(__name__)

MAX_CONCURRENT_IMAGE_PREPARATIONS = 10
"""Image checks or preparations one submit runs at once (each reads a task archive or an image build)."""


@dataclass(frozen=True)
class SubmitOutcome:
    """Report the authoritative status and whether this request created the run.

    Attributes:
        status: Public run status with submission-only warnings.
        created: True for a new acceptance, including recovery of its lost create reply.
    """

    status: TasksetRunSubmitResponse
    created: bool


def normalize_run_caller(caller: str) -> str:
    """Keep delegation and run identity consistent for an empty development caller.

    Args:
        caller: Effective request principal ID; empty in headerless development requests.

    Returns:
        The supplied ID, or a stable non-empty development identity.
    """
    return caller or DEVELOPMENT_CALLER


async def _all[T](calls: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """Run every call at once and return their results in order; the first failure cancels the rest."""
    try:
        async with asyncio.TaskGroup() as group:
            running = [group.create_task(call) for call in calls]
    except ExceptionGroup as failed:
        raise failed.exceptions[0]  # the error itself, so submit maps it as before
    return [task.result() for task in running]


def _unprepared(task: ResolvedTask, preparation: TaskPreparation) -> str:
    if preparation.error is not None:
        return f"{task.id}: {preparation.error.code}: {preparation.error.detail}"
    return f"{task.id}: image preparation is still pending"


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp for acceptance budgets."""
    return datetime.now(timezone.utc)


class TasksetRunResolutionError(ValueError):
    """Represent an expected missing or incompatible request input during resolution."""


class IntakeUnavailableError(RuntimeError):
    """Intake failed while submit looked up the Experiment; nothing was accepted, so a retry is safe."""


class RunEvaluationError(RuntimeError):
    """The accepted run's Intake Evaluation couldn't be created; no task Jobs were started."""

    def __init__(self, run_id: str, reason: str) -> None:
        super().__init__(f"run {run_id}: {reason}")
        self.run_id = run_id


async def pin_taskset_ref(entity_client: EntityClient, ref: TasksetRef, workspace: str) -> TasksetRef:
    """Resolve a taskset selector to its verified, fully qualified content digest.

    Args:
        entity_client: Caller-delegated client for stored heads and revisions.
        ref: Bare, qualified, tagged or content-pinned taskset reference.
        workspace: Submission workspace used for an unqualified reference.

    Returns:
        An immutable workspace/name#64-hex-content-digest reference.

    Raises:
        TasksetRunResolutionError: The taskset or selected revision is missing.
        RevisionContentMismatchError: Stored content fails its integrity check.
    """
    source_workspace, name, fragment = parse_subentity_ref(ref.root, workspace)
    try:
        head = await entity_client.get(TasksetEntity, name=name, workspace=source_workspace)
        revision = await get_revision(entity_client, TasksetRevisionEntity, head, fragment)
    except (NemoEntityNotFoundError, RevisionNotFoundError) as exc:
        raise TasksetRunResolutionError(str(exc)) from exc
    return TasksetRef(f"{source_workspace}/{name}#{revision.content_hash}")


class TasksetRunService:
    """Accept Harbor submissions and project stored runs/members into public DTOs.

    The shared request-scoped clients supply acceptance and read storage. Reads
    return persisted state without resolving sources, loading plans or inspecting Jobs.
    """

    def __init__(
        self,
        store: TasksetRunStore,
        entity_client: EntityClient,
        async_client: AsyncNemoClient,
        authorize: RequestAuthorizer,
        config: EvaluatorConfig,
        clock: Callable[[], datetime] = utc_now,
        *,
        platform: NemoHelixConfig = platform_config,
        preparer: ImagePreparer = prepare_task_images,
        task_jobs: TaskJobs | None = None,
        evaluations: RunEvaluations | None = None,
    ) -> None:
        """Bind request-scoped storage, identity and budget dependencies.

        Args:
            store: Run acceptance and replay boundary.
            entity_client: Caller-delegated entity resolver.
            async_client: Caller-delegated platform client for spec resolution, image reads and task Jobs.
            authorize: Caller-scoped source read authorizer, independent of service privileges.
            config: Operator configuration including run budgets and ``runs.harbor``.
            clock: Aware acceptance clock, invoked after resolution.
            platform: Platform configuration naming the OpenSandbox server.
            preparer: Checks every task, then prepares each one's images, before acceptance.
            task_jobs: Jobs operations for fan-out; defaults to the Jobs service as the caller.
            evaluations: Intake operations for a run's Evaluation; defaults to Intake as the caller.
        """
        self._store = store
        self._entities = entity_client
        self._async_client = async_client
        self._authorizer = authorize
        self._config = config
        self._clock = clock
        self._platform = platform
        self._preparer = preparer
        self._task_jobs = task_jobs or JobsServiceTaskJobs(AsyncJobsClient.from_client(async_client))
        self._evaluations = evaluations or IntakeRunEvaluations(AsyncIntakeClient.from_client(async_client))

    async def get(self, workspace: str, run_id: str) -> TasksetRunStatus | None:
        """Read one stored run without joining members, plans or Jobs.

        Args:
            workspace: Workspace that scopes the record lookup.
            run_id: Public run name in that workspace.

        Returns:
            Stored public status, or None only when the record is absent. Storage
            and projection failures propagate for the route to report as errors.
        """
        run = await self._store.find(workspace, run_id)
        return to_status(run) if run is not None else None

    async def list_runs(
        self,
        workspace: str,
        *,
        caller: str,
        active: bool = False,
        mine: bool = False,
        page: int = 1,
        page_size: int = 100,
    ) -> Page[TasksetRunStatus]:
        """Project one independently filtered workspace run page into public statuses.

        Args:
            workspace: Workspace containing the records.
            caller: Effective request principal, normalized as for submit.
            active: Restrict to accepted/running runs when True.
            mine: Restrict to the effective caller's runs when True.
            page: One-based page number.
            page_size: Maximum records on the requested page.

        Returns:
            Newest-first status page preserving the store's pagination and sort.
            Storage and projection failures propagate rather than producing an empty page.
        """
        result = await self._store.list_runs(
            workspace,
            caller=normalize_run_caller(caller) if mine else None,
            active=active,
            page=page,
            page_size=page_size,
        )
        return Page(
            data=[to_status(run) for run in result.data],
            pagination=result.pagination,
            sort=result.sort,
            filter=result.filter,
        )

    async def list_tasks(
        self,
        workspace: str,
        run_id: str,
        *,
        phase: MemberPhase | None,
        page: int,
        page_size: int,
    ) -> Page[TasksetRunTask] | None:
        """Project one member page only after verifying its workspace-scoped parent.

        Args:
            workspace: Workspace containing the run and its children.
            run_id: Public run name whose persisted entity ID scopes the members.
            phase: Optional member phase filter; None includes every phase.
            page: One-based page number.
            page_size: Maximum records on the requested page.

        Returns:
            Public task page preserving pagination/sort, including an empty page
            when the run has no members; None only for an absent parent. Storage
            and projection failures propagate to the route.
        """
        run = await self._store.find(workspace, run_id)
        if run is None:
            return None
        result = await self._store.list_members(run, phase=phase, page=page, page_size=page_size)
        return Page(
            data=[to_task(member) for member in result.data],
            pagination=result.pagination,
            sort=result.sort,
            filter=result.filter,
        )

    async def submit(self, req: TasksetRunInput, *, workspace: str, caller: str) -> SubmitOutcome:
        """Authorize, freeze and atomically accept a run, or replay its original status.

        Args:
            req: Validated Kubernetes Harbor taskset submission, with its idempotency_key set (the route
                generates one when the caller omits it).
            workspace: Workspace that owns the new run and plan.
            caller: Effective principal ID used for delegation and idempotency scope.

        Returns:
            The authoritative run status and whether this request created it.

        Raises:
            TimeoutAboveMaximumError: The current operator maximum is exceeded.
            HTTPException: The caller cannot read a source.
            TasksetRunResolutionError: A source is missing or incompatible.
            RunRequestConflict: The key already belongs to another logical request.
            ValueError: ``req`` has no idempotency_key.
            Exception: Unexpected storage and integrity failures propagate.
        """
        caller = normalize_run_caller(caller)
        # These checks run before the replay lookup on purpose: they decide whether a retry may start more
        # task Jobs, using today's operator config and source access. A lowered timeout maximum, a removed
        # dispatcher or sandbox profile, disabled runs or revoked access stops new Jobs at once, instead of an
        # old idempotency_key starting them. How the Jobs run (images, TTL, egress, deadlines) stays frozen in
        # the accepted plan. Run status stays readable via GET.
        timeout = check_timeout(req.timeout_seconds, self._config.runs)
        harbor, policy, dispatcher_profile = await self._check_execution(req)
        await self._authorize(req, workspace)
        if req.publication is not None:
            await self._authorize_publication(workspace)
        digest = request_digest(req)
        if req.idempotency_key is None:
            raise ValueError("submit needs an idempotency_key; the route generates one when the caller omits it")
        run_id = derive_run_id(workspace, caller, req.idempotency_key)
        replay = await self._store.replay(workspace, run_id, digest)
        if replay is not None:
            # A retry completes the task Jobs an earlier attempt didn't create.
            warnings = await self._start_task_jobs(replay)
            status = TasksetRunSubmitResponse(**to_status(replay).model_dump(), warnings=warnings)
            return SubmitOutcome(status, created=False)

        pinned, spec = await self._resolve(req, workspace)
        source = registered_agent_source(spec.target)
        snapshot = source.files if source is not None else None
        acceptance_started = False
        try:
            if req.execution.max_concurrent_task_jobs is not None and (
                len(spec.tasks) > req.execution.max_concurrent_task_jobs
            ):
                raise TasksetRunResolutionError(
                    f"this deployment starts every task Job at once: {len(spec.tasks)} tasks exceed "
                    f"execution.max_concurrent_task_jobs={req.execution.max_concurrent_task_jobs}; select fewer tasks"
                )
            intake = await self._resolve_publication(req, workspace, spec, taskset_ref=pinned.root)
            prepared = await self._prepare_images(spec.tasks, policy, caller=caller)
            accepted_at = self._clock()
            execution, finalization = resolve_deadlines(accepted_at, timeout, self._config.runs)
            plan = TasksetRunPlan(
                run_id=run_id,
                workspace=workspace,
                taskset_ref=pinned.root,
                spec=spec,
                max_concurrent_task_jobs=req.execution.max_concurrent_task_jobs,
                sandbox_policy=policy,
                dispatcher_image=harbor.dispatcher_image,
                dispatcher_profile=dispatcher_profile,
                task_environments=prepared,
                intake=intake,
            )
            acceptance = RunAcceptance(
                plan=plan,
                idempotency_key=req.idempotency_key,
                request_digest=digest,
                accepted_at=accepted_at,
                execution_deadline_at=execution,
                finalization_deadline_at=finalization,
                discard_on_loss=(snapshot.root,) if snapshot is not None else (),
            )
            acceptance_started = True
            result = await self._store.accept(acceptance)
        except RunRequestConflict:
            # The store already discarded the resources belonging to a proven loser.
            raise
        except Exception as exc:
            if snapshot is not None and (not acceptance_started or isinstance(exc, PlanStorageError)):
                await delete_fileset_ref(self._store.files_client, snapshot.root)
            # Even a subsequent absent lookup cannot prove a create did not commit.
            raise

        warnings = await self._duplicate_warnings(result.run, caller) if result.created else []
        warnings += await self._start_task_jobs(result.run)
        status = TasksetRunSubmitResponse(**to_status(result.run).model_dump(), warnings=warnings)
        return SubmitOutcome(status, created=result.created)

    async def _check_execution(
        self, req: TasksetRunInput
    ) -> tuple[HarborTaskJobsConfig, ResolvedHarborSandboxPolicy, str]:
        """Check the deployment can run the request's execution settings; freeze its sandbox and dispatcher profiles.

        The sandbox profile is frozen with the request's resource overrides. An omitted dispatcher profile
        is the operator's first ``dispatcher_profiles`` entry.

        Raises:
            TasksetRunResolutionError: Runs are off, the profile or dispatcher profile can't be used, or a
                resource override exceeds the profile's ceilings.
        """
        harbor = self._config.runs.harbor
        if harbor is None:
            raise TasksetRunResolutionError("Kubernetes taskset runs aren't configured in this deployment")
        try:
            policy = resolve_sandbox_policy(harbor, self._platform, req.execution.sandbox)
        except SandboxPolicyError as exc:
            raise TasksetRunResolutionError(str(exc)) from exc
        profile = req.execution.dispatcher_profile or harbor.dispatcher_profiles[0]
        if profile not in harbor.dispatcher_profiles:
            allowed = ", ".join(harbor.dispatcher_profiles)
            raise TasksetRunResolutionError(f"dispatcher_profile {profile!r} isn't allowed; use one of: {allowed}")
        if await self._task_jobs.is_subprocess_profile(profile):
            raise TasksetRunResolutionError(
                f"dispatcher_profile {profile!r} runs as a host subprocess; task Jobs need a Kubernetes profile"
            )
        return harbor, policy, profile

    async def _prepare_images(
        self, tasks: list[ResolvedTask], policy: ResolvedHarborSandboxPolicy, *, caller: str
    ) -> list[PreparedHarborEnvironment]:
        """Each task's prepared images, in task order.

        Every task is checked before any is prepared, so a task the run can't take never leaves image builds
        behind for the others. Up to ``MAX_CONCURRENT_IMAGE_PREPARATIONS`` run at once, so submit time doesn't
        grow task by task. All or nothing: an exception from any task cancels the others and fails the submit.

        Raises:
            TasksetRunResolutionError: Some task has no images; the message names each one and why.
        """
        clients = PreparationClients(
            client=self._async_client, files=self._store.files_client, image_builds=self._image_builds
        )
        limit = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_PREPARATIONS)

        async def check(task: ResolvedTask) -> str | None:
            async with limit:
                error = await self._preparer.check(task, policy=policy, clients=clients)
            return f"{task.id}: {error.code}: {error.detail}" if error is not None else None

        async def prepare(task: ResolvedTask) -> TaskPreparation:
            async with limit:
                return await self._preparer(
                    task, None, policy=policy, caller=caller, clients=clients, deadline=self._clock()
                )

        rejected = [reason for reason in await _all(check(task) for task in tasks) if reason is not None]
        if rejected:
            raise TasksetRunResolutionError(f"tasks without images: {'; '.join(rejected)}")
        preparations = await _all(prepare(task) for task in tasks)
        rejected = [_unprepared(task, p) for task, p in zip(tasks, preparations, strict=True) if p.environment is None]
        if rejected:
            raise TasksetRunResolutionError(f"tasks without images: {'; '.join(rejected)}")
        return [p.environment for p in preparations if p.environment is not None]

    def _image_builds(self, workspace: str) -> ImageBuildService:
        """The image build service for tasks in ``workspace``, starting builds as the caller."""
        jobs = JobsServiceBuildJobs(AsyncJobsClient.from_client(self._async_client), workspace=workspace)
        return ImageBuildService(self._entities, jobs, builder_authorizer(self._authorizer))

    async def _resolve_publication(
        self, req: TasksetRunInput, workspace: str, spec: AgentEvalSpec, *, taskset_ref: str
    ) -> RunIntakePublication | None:
        """Resolve ``publication.intake`` for the plan: the Experiment's id, the agent and the dataset.

        The dataset defaults to the pinned taskset: its name, and its content digest as the version.

        Raises:
            TasksetRunResolutionError: The Experiment doesn't exist, or no agent name can be derived.
            HTTPException: 403 when the caller can't read the Experiment.
            IntakeUnavailableError: Any other Intake failure; a retry is safe.
        """
        if req.publication is None:
            return None
        requested = req.publication.intake
        unreadable = f"you can't read Intake Experiment {workspace}/{requested.experiment}"
        # The lookup below runs as the evals service, so check the caller's own read first.
        try:
            await self._authorizer(
                "GET",
                f"/apis/intake/v2/workspaces/{quote(workspace, safe='')}"
                f"/experiments/{quote(requested.experiment, safe='')}",
            )
        except HTTPException as exc:
            if exc.status_code != 403:
                raise
            raise HTTPException(status_code=403, detail=unreadable) from exc
        try:
            experiment_id = await self._evaluations.experiment_id(workspace, requested.experiment)
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise HTTPException(status_code=403, detail=unreadable) from exc
        except NemoClientError as exc:
            raise IntakeUnavailableError(
                f"looking up Intake Experiment {workspace}/{requested.experiment} failed ({type(exc).__name__}); "
                "nothing was accepted, retry the submit"
            ) from exc
        if experiment_id is None:
            raise TasksetRunResolutionError(
                f"Intake Experiment {workspace}/{requested.experiment} doesn't exist; create it first "
                f"(nemo experiments create {requested.experiment} --workspace {workspace})"
            )
        taskset_name, _, taskset_digest = taskset_ref.partition("#")
        agent_name = requested.agent_name or target_agent_identity(spec.target)[0]
        if not agent_name:
            raise TasksetRunResolutionError(
                "publication.intake.agent_name is required: the Harbor target names no agent to derive it from"
            )
        return RunIntakePublication(
            experiment=requested.experiment,
            experiment_id=experiment_id,
            agent_name=agent_name,
            dataset_name=requested.dataset_name or taskset_name,
            dataset_version=requested.dataset_version or taskset_digest,
        )

    async def _authorize_publication(self, workspace: str) -> None:
        """Check, before accepting, that the caller may create the run's Intake Evaluation.

        A run that asked to be published shouldn't start unpublished. This runs before the replay lookup too,
        so a caller who has since lost the permission gets the 403 instead of their existing run.

        Raises:
            HTTPException: 403 when the caller may not create Intake Evaluations in ``workspace``.
        """
        try:
            await self._authorizer("POST", f"/apis/intake/v2/workspaces/{quote(workspace, safe='')}/evaluations")
        except HTTPException as exc:
            if exc.status_code != 403:
                raise
            raise HTTPException(
                status_code=403, detail=f"you may not create Intake Evaluations in workspace {workspace}"
            ) from exc

    async def _ensure_run_evaluation(self, run: TasksetRunEntity, plan: TasksetRunPlan) -> list[str]:
        """Create (or adopt) the run's Intake Evaluation before any task Job can publish to it.

        Returns:
            Notices for the submit response.

        Raises:
            RunEvaluationError: The Evaluation isn't the run's, or Intake refused or failed to create it.
                No task Job has started; a retry with the run's idempotency_key creates it and then the Jobs.
        """
        assert plan.intake is not None
        try:
            await ensure_run_evaluation(self._evaluations, run, plan)
        except ForeignEvaluationError as exc:
            raise RunEvaluationError(run.name, str(exc)) from exc
        except NemoClientError as exc:
            # Includes a 403: submit checked the permission before accepting, so it was revoked since.
            raise RunEvaluationError(run.name, f"creating its Intake Evaluation failed: {exc}") from exc
        return [f"Intake: Evaluation {run.name} in Experiment {plan.intake.experiment}"]

    async def _start_task_jobs(self, run: TasksetRunEntity) -> list[str]:
        """Create or adopt every task Job of ``run`` from its stored plan; one notice per task Job.

        Images come from the plan, never prepared again: a retry must give each Job the images it was accepted with.
        An ended run (a replay, or a concurrent winner's run returned by accept) gets no Jobs and no notices,
        nor does one past its execution deadline: the controller is expiring it. A Job this misses because
        the run expires mid fan-out refuses to start (``check_execution_deadline``).
        """
        if run.phase in TERMINAL_RUN_PHASES or self._clock() >= run.execution_deadline_at:
            return []
        plan = await self._store.load_plan(run)
        harbor = self._config.runs.harbor
        if harbor is None or plan.dispatcher_profile not in harbor.dispatcher_profiles:
            # A retry that omits dispatcher_profile passes submit's check on today's default, so check the
            # profile the plan froze: an operator who removed it has stopped new Jobs on it.
            raise TasksetRunResolutionError(
                f"run {run.name}'s dispatcher_profile {plan.dispatcher_profile!r} is no longer allowed"
            )
        notices = await self._ensure_run_evaluation(run, plan) if plan.intake is not None else []
        configs = [(t.id, t.spec.config) for t in plan.spec.tasks if isinstance(t.spec, ResolvedHarborTaskDefinition)]
        if (resources := defaulted_resources_notice(configs, plan.sandbox_policy)) is not None:
            notices.append(resources)
        try:
            names = await create_task_jobs(self._task_jobs, plan, run, plan.task_environments)
        except TaskJobsIncompleteError as exc:
            raise dataclasses.replace(exc, notices=(*exc.notices, *notices)) from None
        return [
            *(f"task {i} ({task.id}): Job {name}" for i, (task, name) in enumerate(zip(plan.spec.tasks, names))),
            *notices,
        ]

    async def _resolve(self, req: TasksetRunInput, workspace: str) -> tuple[TasksetRef, AgentEvalSpec]:
        """Pin before resolving so a concurrent publication cannot replace task members.

        Args:
            req: Request whose logical digest has already been computed.
            workspace: Default workspace for unqualified inputs.

        Returns:
            Verified taskset digest reference and existing resolved evaluation spec.

        Raises:
            TasksetRunResolutionError: Resolution rejects a missing or incompatible input.
            HTTPException: Registered agent resolution reports a denied read.
            Exception: Integrity and service failures propagate unchanged.
        """
        assert isinstance(req.spec.tasks, TasksetRef)
        pinned = await pin_taskset_ref(self._entities, req.spec.tasks, workspace)
        input_spec = req.spec.model_copy(deep=True, update={"tasks": pinned})
        try:
            spec = await AgentEvalJob.to_spec(
                input_spec,
                workspace=workspace,
                entity_client=self._entities,
                async_sdk=self._async_client,
                is_local=False,
                # The resolver reads with service credentials; check each nested source as the caller first.
                authorize=remember_allowed(self._authorizer),
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValueError, NemoEntityNotFoundError, RevisionNotFoundError) as exc:
            raise TasksetRunResolutionError(str(exc)) from exc
        if not isinstance(spec, AgentEvalSpec):
            raise TypeError("AgentEvalJob.to_spec returned an unexpected spec type")
        return pinned, spec

    async def _duplicate_warnings(self, run: TasksetRunEntity, caller: str) -> list[str]:
        """Find advisory overlaps without allowing a lookup failure to undo acceptance.

        Args:
            run: Newly accepted authoritative record, including its pinned revision.
            caller: Normalized creator identity used by the run store's filter.

        Returns:
            Notices for other accepted/running runs on the same revision, or an empty
            list when lookup fails. Ordinary failures are logged.
        """
        try:
            active = await self._store.list_active(run.workspace, caller, run.taskset_ref)
            return [f"{other.name} is already running {run.taskset_ref}" for other in active if other.name != run.name]
        except Exception:
            logger.warning("Failed to look up overlapping taskset runs for %s", run.name, exc_info=True)
            return []

    async def _authorize(self, req: TasksetRunInput, workspace: str) -> None:
        """Check source reads using the caller's grants before service-backed resolution.

        Args:
            req: Validated Harbor taskset request.
            workspace: Submission workspace for unqualified source references.

        Returns:
            None after the taskset and any registered agent are authorized. Member tasks, metrics and
            metric models are checked as the caller during resolution, before each one is read.

        Raises:
            HTTPException: A caller cannot read a referenced source.
        """
        assert isinstance(req.spec.tasks, TasksetRef)
        source_workspace, name, _ = parse_subentity_ref(req.spec.tasks.root, workspace)
        await self._authorizer(
            "GET", f"/apis/evals/v2/workspaces/{quote(source_workspace, safe='')}/tasksets/{quote(name, safe='')}"
        )
        source = registered_agent_source(req.spec.target)
        if source is not None:
            agent_workspace, name, _ = parse_subentity_ref(source.agent.root, workspace)
            await self._authorizer(
                "GET", f"/apis/agents/v2/workspaces/{quote(agent_workspace, safe='')}/agents/{quote(name, safe='')}"
            )
