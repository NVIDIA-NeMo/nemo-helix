# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-scoped run acceptance and stored-record reads without execution joins."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote

from fastapi import HTTPException
from nemo_evals.api.fields import TasksetRef, parse_subentity_ref
from nemo_evals.api.taskset_run_schemas import (
    TasksetRunInput,
    TasksetRunStatus,
    TasksetRunSubmitResponse,
    TasksetRunTask,
    request_digest,
    to_status,
    to_task,
)
from nemo_evals.config import EvaluatorConfig
from nemo_evals.entities import TasksetEntity, TasksetRevisionEntity
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalSpec, registered_agent_source
from nemo_evals.revisions import RevisionNotFoundError, get_revision
from nemo_evals.source_authorization import remember_allowed
from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunEntity
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_evals.taskset_runs.plan_storage import delete_fileset_ref
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import PlanStorageError, RunAcceptance, RunRequestConflict, TasksetRunStore
from nemo_evals.taskset_runs.submission import check_timeout, resolve_deadlines
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import RequestAuthorizer
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.schema import Page

DEVELOPMENT_CALLER = "user:development"
"""Stable creator identity for headerless requests when authentication is disabled."""

logger = logging.getLogger(__name__)


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


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp for acceptance budgets."""
    return datetime.now(timezone.utc)


class TasksetRunResolutionError(ValueError):
    """Represent an expected missing or incompatible request input during resolution."""


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
    ) -> None:
        """Bind request-scoped storage, identity and budget dependencies.

        Args:
            store: Run acceptance and replay boundary.
            entity_client: Caller-delegated entity resolver.
            async_client: Caller-delegated platform client for existing spec resolution.
            authorize: Caller-scoped source read authorizer, independent of service privileges.
            config: Operator configuration including run budgets.
            clock: Aware acceptance clock, invoked after resolution.
        """
        self._store = store
        self._entities = entity_client
        self._async_client = async_client
        self._authorizer = authorize
        self._config = config
        self._clock = clock

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
            req: Validated Kubernetes Harbor taskset submission.
            workspace: Workspace that owns the new run and plan.
            caller: Effective principal ID used for delegation and idempotency scope.

        Returns:
            The authoritative run status and whether this request created it.

        Raises:
            TimeoutAboveMaximumError: The current operator maximum is exceeded.
            HTTPException: The caller cannot read a source.
            TasksetRunResolutionError: A source is missing or incompatible.
            RunRequestConflict: The key already belongs to another logical request.
            Exception: Unexpected storage and integrity failures propagate.
        """
        caller = normalize_run_caller(caller)
        timeout = check_timeout(req.timeout_seconds, self._config.runs)
        await self._authorize(req, workspace)
        digest = request_digest(req)
        run_id = derive_run_id(workspace, caller, req.idempotency_key)
        if req.idempotency_key is not None:
            replay = await self._store.replay(workspace, run_id, digest)
            if replay is not None:
                return SubmitOutcome(TasksetRunSubmitResponse(**to_status(replay).model_dump()), created=False)

        pinned, spec = await self._resolve(req, workspace)
        source = registered_agent_source(spec.target)
        snapshot = source.files if source is not None else None
        acceptance_started = False
        try:
            accepted_at = self._clock()
            execution, finalization = resolve_deadlines(accepted_at, timeout, self._config.runs)
            # M1 intentionally omits sandbox_policy and dispatcher_profile from the plan.
            # M2 Story 3 freezes them with the dispatcher image before enabling execution;
            # earlier development runs are deleted when those required fields are added.
            plan = TasksetRunPlan(
                run_id=run_id,
                workspace=workspace,
                taskset_ref=pinned.root,
                spec=spec,
                missing_prebuilt_image=req.execution.missing_prebuilt_image,
                max_concurrent_task_jobs=req.execution.max_concurrent_task_jobs,
                max_failed_tasks=req.execution.resolve_max_failed_tasks(len(spec.tasks)),
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
        status = TasksetRunSubmitResponse(**to_status(result.run).model_dump(), warnings=warnings)
        return SubmitOutcome(status, created=result.created)

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
