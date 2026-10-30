# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-scoped run acceptance and stored-record reads without execution joins."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException
from nemo_evals.agent_layers.models import AgentLayer, AgentLayerRequest
from nemo_evals.agent_layers.run_agents import agent_layer_request
from nemo_evals.agent_layers.service import AgentLayerService, agent_layer_key
from nemo_evals.api.fields import TasksetRef, parse_subentity_ref
from nemo_evals.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evals.api.task_definitions.provenance import TaskProvenance
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
from nemo_evals.image_builds.planning import ImagePlanError
from nemo_evals.image_builds.service import ImageBuildService, JobsServiceBuildJobs, builder_authorizer
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalSpec, ResolvedTask, registered_agent_source, target_agent_identity
from nemo_evals.jobs.gateway_models import require_closed_book_route
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment
from nemo_evals.revisions import RevisionNotFoundError, get_revision
from nemo_evals.source_authorization import remember_allowed
from nemo_evals.task_identity import UnsupportedTaskKindError
from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunEntity
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.plan_storage import delete_fileset_ref
from nemo_evals.taskset_runs.progress import build_progress
from nemo_evals.taskset_runs.run_evaluation import (
    EvaluationDatasetMismatchError,
    ForeignEvaluationError,
    IntakeRunEvaluations,
    RunEvaluations,
    ensure_run_evaluation,
)
from nemo_evals.taskset_runs.specs import RunIntakePublication, TasksetRunPlan
from nemo_evals.taskset_runs.store import PlanStorageError, RunAcceptance, RunRequestConflict, TasksetRunStore
from nemo_evals.taskset_runs.submission import check_timeout, resolve_deadlines
from nemo_evals.taskset_runs.task_jobs import JobsServiceTaskJobs, TaskJobs
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import AuthenticationError, NemoClientError, PermissionDeniedError
from nemo_helix_plugin.config import NemoHelixConfig
from nemo_helix_plugin.dependencies import RequestAuthorizer
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.image import get_qualified_image
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
    """The run's Intake Evaluation couldn't be created; the run wasn't accepted."""

    def __init__(self, run_id: str, reason: str) -> None:
        super().__init__(f"run {run_id}: {reason}")
        self.run_id = run_id


class RunEvaluationDeniedError(RunEvaluationError):
    """Intake refused the caller the run's Evaluation; a retry fails the same way until the caller may create it."""


class RunEvaluationMismatchError(RunEvaluationError):
    """The run's Evaluation exists with another dataset; a retry of this request fails the same way."""


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
        auth_context: AuthContext | None = None,
    ) -> None:
        """Bind request-scoped storage, identity and budget dependencies.

        Args:
            store: Run acceptance and replay boundary.
            entity_client: Caller-delegated entity resolver.
            async_client: Caller-delegated platform client for spec resolution, image reads and Jobs checks.
            authorize: Caller-scoped source read authorizer, independent of service privileges.
            config: Operator configuration including run budgets and ``runs.harbor``.
            clock: Aware acceptance clock, invoked after resolution.
            platform: Platform configuration naming the OpenSandbox server and the default dispatcher image.
            preparer: Checks every task, then reads each one's images, before acceptance.
            task_jobs: Jobs operations submit's checks need; defaults to the Jobs service as the caller.
            evaluations: Intake operations for a run's Evaluation; defaults to Intake as the caller.
            auth_context: The caller's full principal, stored on each new run for the run controller to
                create on the caller's behalf; None with auth off.
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
        self._auth_context = auth_context

    async def get(self, workspace: str, run_id: str) -> TasksetRunStatus | None:
        """Read one stored run with its progress counted from its members, without loading plans or reading Jobs.

        Args:
            workspace: Workspace that scopes the record lookup.
            run_id: Public run name in that workspace.

        Returns:
            Stored public status with ``progress`` (None while the run has no members), or None only
            when the record is absent. Storage and projection failures propagate for the route to report as errors.
        """
        run = await self._store.find(workspace, run_id)
        if run is None:
            return None
        members = await self._store.list_all_members(run)
        return to_status(run).model_copy(update={"progress": build_progress(members) if members else None})

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

    async def request_agent_layers(self, req: TasksetRunInput, *, workspace: str) -> list[AgentLayer]:
        """Request each of the run's tasks' agent layers for its agent, as submit would resolve them; accepts nothing.

        ``nemo evals runs submit`` calls this before it submits and waits for the layers, so the run's first trials
        already start from them. Returns the layers requested, one per distinct task revision; none when layers are
        off or the agent can't be layered. A task that can't be given a layer is left out: its trials install the
        agent in setup.

        Raises:
            HTTPException: The caller cannot read a source.
            TasksetRunResolutionError: A source is missing or incompatible.
            ImageBuildForbiddenError: The caller may not submit Builder builds in a task's workspace.
        """
        harbor, _, _ = await self._check_execution(req)
        await self._authorize(req, workspace)
        if not harbor.agent_layers:
            return []
        _, spec = await self._resolve(req, workspace)
        source = registered_agent_source(spec.target)
        snapshot = source.files if source is not None else None
        try:
            agent = agent_layer_request(spec.target)
            if agent is None:
                return []
            revisions = dict.fromkeys(
                task.spec.provenance for task in spec.tasks if isinstance(task.spec, ResolvedHarborTaskDefinition)
            )
            limit = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_PREPARATIONS)

            async def request(provenance: TaskProvenance) -> AgentLayer | None:
                task_workspace, _, name = provenance.entity_name.partition("/")
                jobs = JobsServiceBuildJobs(AsyncJobsClient.from_client(self._async_client), workspace=task_workspace)
                service = AgentLayerService(self._entities, jobs, builder_authorizer(self._authorizer))
                async with limit:
                    try:
                        return await service.request_layer(task_workspace, name, provenance.revision_digest, agent)
                    except (ImagePlanError, UnsupportedTaskKindError) as exc:
                        logger.info("Task %s gets no agent layer: %s", provenance.entity_name, exc)
                        return None

            return [layer for layer in await _all(request(p) for p in revisions) if layer is not None]
        finally:
            # Resolving a registered agent snapshots its files for the run; no run is accepted here.
            if snapshot is not None:
                await delete_fileset_ref(self._store.files_client, snapshot.root)

    async def submit(self, req: TasksetRunInput, *, workspace: str, caller: str) -> SubmitOutcome:
        """Authorize, freeze and atomically accept a run, or replay its stored status.

        Submit starts no task Jobs and no image builds: the controller starts the task Jobs, and every task's
        image must be built already.

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
            RunEvaluationError: The run's Intake Evaluation couldn't be created; nothing was accepted.
            ValueError: ``req`` has no idempotency_key.
            Exception: Unexpected storage and integrity failures propagate.
        """
        caller = normalize_run_caller(caller)
        # These checks run before the replay lookup on purpose: a lowered timeout maximum, a removed dispatcher
        # or sandbox profile, disabled runs or revoked access refuse a retry as they would a new request.
        # How the Jobs run (images, TTL, egress, deadlines) stays frozen in the accepted plan. Run status stays
        # readable via GET.
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
            return SubmitOutcome(TasksetRunSubmitResponse(**to_status(replay).model_dump()), created=False)

        pinned, spec = await self._resolve(req, workspace)
        source = registered_agent_source(spec.target)
        snapshot = source.files if source is not None else None
        acceptance_started = False
        try:
            configs = [(t.id, t.spec.config) for t in spec.tasks if isinstance(t.spec, ResolvedHarborTaskDefinition)]
            try:
                await require_closed_book_route(
                    spec.target, configs, async_sdk=self._async_client, authorize=self._authorizer
                )
            except ValueError as exc:
                raise TasksetRunResolutionError(str(exc)) from exc
            intake = await self._resolve_publication(req, workspace, spec, taskset_ref=pinned.root)
            agent = agent_layer_request(spec.target) if harbor.agent_layers else None
            prepared = await self._prepare_images(spec.tasks, policy, caller=caller, agent=agent)
            plan = TasksetRunPlan(
                run_id=run_id,
                workspace=workspace,
                taskset_ref=pinned.root,
                spec=spec,
                max_concurrent_task_jobs=req.execution.max_concurrent_task_jobs,
                sandbox_policy=policy,
                dispatcher_image=harbor.dispatcher_image or self._default_dispatcher_image(),
                dispatcher_profile=dispatcher_profile,
                task_environments=prepared,
                intake=intake,
            )
            # Before accept, so every stored run has its Evaluation. An accept that fails after this leaves the
            # Evaluation behind; a retry with the same idempotency_key derives the same run id and adopts it.
            notices = await self._ensure_run_evaluation(plan) if intake is not None else []
            # Until its first task gets a slot, the run's deadlines count the queue timeout from acceptance; the
            # run controller restarts them from that slot with the execution budget (TasksetRunStore.start).
            runs = self._config.runs
            accepted_at = self._clock()
            execution, finalization = resolve_deadlines(
                accepted_at, runs.queue_timeout_seconds, runs.finalization_grace_seconds
            )
            acceptance = RunAcceptance(
                plan=plan,
                idempotency_key=req.idempotency_key,
                request_digest=digest,
                accepted_at=accepted_at,
                timeout_seconds=timeout,
                execution_deadline_at=execution,
                finalization_deadline_at=finalization,
                discard_on_loss=(snapshot.root,) if snapshot is not None else (),
                auth_context=self._auth_context,
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
        configs = [(t.id, t.spec.config) for t in spec.tasks if isinstance(t.spec, ResolvedHarborTaskDefinition)]
        if (resources := defaulted_resources_notice(configs, policy)) is not None:
            notices.append(resources)
        status = TasksetRunSubmitResponse(**to_status(result.run).model_dump(), warnings=[*warnings, *notices])
        return SubmitOutcome(status, created=result.created)

    def _default_dispatcher_image(self) -> str:
        """The platform's nhx-tasks image: the dispatcher a run gets when ``runs.harbor`` doesn't pin one."""
        return get_qualified_image("nhx-tasks", tag=self._platform.image_tag, registry=self._platform.image_registry)

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
        self,
        tasks: list[ResolvedTask],
        policy: ResolvedHarborSandboxPolicy,
        *,
        caller: str,
        agent: AgentLayerRequest | None = None,
    ) -> list[PreparedHarborEnvironment]:
        """Each task's prepared images, in task order.

        Every task is checked before any task's builds are read. Up to ``MAX_CONCURRENT_IMAGE_PREPARATIONS`` run
        at once, so submit time doesn't grow task by task. All or nothing: an exception from any task cancels the others and fails the submit.

        Raises:
            TasksetRunResolutionError: Some task has no images; the message names each one and why.
        """
        clients = PreparationClients(
            client=self._async_client,
            files=self._store.files_client,
            image_builds=self._image_builds,
            agent_layer=self._agent_layers_for(agent) if agent is not None else None,
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

    def _agent_layers_for(self, agent: AgentLayerRequest) -> Callable[[str, str, str], Awaitable[AgentLayer | None]]:
        """Each task's ready layer for ``agent``, which the preparer starts the task's trials from; reads only."""
        key = agent_layer_key(agent)

        async def ready(workspace: str, name: str, revision: str) -> AgentLayer | None:
            jobs = JobsServiceBuildJobs(AsyncJobsClient.from_client(self._async_client), workspace=workspace)
            service = AgentLayerService(self._entities, jobs, builder_authorizer(self._authorizer))
            return await service.ready_layer(workspace, name, revision, key)

        return ready

    def _image_builds(self, workspace: str) -> ImageBuildService:
        """The image build service for tasks in ``workspace``; the preparer only reads builds through it."""
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

    async def _ensure_run_evaluation(self, plan: TasksetRunPlan) -> list[str]:
        """Create (or adopt) the run's Intake Evaluation before the run is accepted.

        Returns:
            Notices for the submit response.

        Raises:
            RunEvaluationDeniedError: Intake refused the caller the Evaluation. Nothing was accepted.
            RunEvaluationMismatchError: The run's Evaluation exists with another dataset. Nothing was accepted.
            RunEvaluationError: The Evaluation isn't the run's, or Intake failed to create it. Nothing was
                accepted; a retry with the same idempotency_key adopts an Evaluation this attempt created.
        """
        assert plan.intake is not None
        try:
            await ensure_run_evaluation(self._evaluations, plan.workspace, plan.run_id, plan)
        except ForeignEvaluationError as exc:
            raise RunEvaluationError(plan.run_id, str(exc)) from exc
        except EvaluationDatasetMismatchError as exc:
            raise RunEvaluationMismatchError(plan.run_id, str(exc)) from exc
        except PermissionDeniedError as exc:
            # Submit checked the permission just before, so it was revoked since; retrying can't help.
            raise RunEvaluationDeniedError(plan.run_id, "Intake refused to create its Evaluation") from exc
        except NemoClientError as exc:
            raise RunEvaluationError(plan.run_id, f"creating its Intake Evaluation failed: {exc}") from exc
        return [f"Intake: Evaluation {plan.run_id} in Experiment {plan.intake.experiment}"]

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
