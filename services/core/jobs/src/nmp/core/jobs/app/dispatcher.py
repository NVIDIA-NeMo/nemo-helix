# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import weakref
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.files.types import CreateFilesetRequest
from nemo_platform_plugin.secrets.client import AsyncSecretsClient
from nmp.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nmp.common.api.in_memory_filter import InMemoryFilterRepository
from nmp.common.api.parsed_filter import ParsedFilter
from nmp.common.auth import AuthContext
from nmp.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nmp.common.jobs.schemas import (
    PlatformJobStatus,
    PlatformJobStatusResponse,
    PlatformJobStepStatusResponse,
    PlatformJobTaskStatusResponse,
)
from nmp.common.observability import create_counter
from nmp.common.sdk_factory import get_entity_parts
from nmp.core.jobs.api.v2.jobs.schemas import (
    CreatePlatformJobRequest,
    PlatformJobListSortField,
    PlatformJobResponse,
    PlatformJobSortField,
    PlatformJobStepsListFilter,
    PlatformJobStepWithContext,
    PlatformJobTaskUpdate,
    get_model_id,
    job_artifact_base_path,
)
from nmp.core.jobs.app.schemas import (
    PlatformJobSpec,
)
from nmp.core.jobs.entities import (
    STEP_SPEC_NAME_CONFIG_KEY,
    PlatformJob,
    PlatformJobAttempt,
    PlatformJobResult,
    PlatformJobStep,
    PlatformJobTask,
    get_step_spec_name,
)
from opentelemetry import metrics, trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class StateTransitionConflictError(Exception):
    """Exception raised when a state transition is invalid."""


class JobAlreadyExistsError(ValueError):
    """Exception raised when creating a job whose name is already in use."""


class JobDeletionConflictError(ValueError):
    """Exception raised when deleting a job that is not safe to hard-delete."""


class JobStatusUpdateSkippedError(Exception):
    """Raised when a status update loses a race with job deletion."""


class JobSecretValidationError(ValueError):
    """Exception raised when a job's secret references cannot be validated."""


class JobOutputLocationError(ValueError):
    """Exception raised when a job's output_location references a fileset that does not exist."""


operations_counter = create_counter(
    meter=meter,
    subsystem="jobs",
    name="dispatch.operations.total",
    description="Total number of job dispatcher operations performed",
)

_DELETE_PAGE_SIZE = 1000
_JOB_MUTATION_LOCKS: weakref.WeakValueDictionary[tuple[str, str], asyncio.Lock] = weakref.WeakValueDictionary()
EntityT = TypeVar("EntityT")


def _get_job_mutation_lock(job_name: str, workspace: str) -> asyncio.Lock:
    """Return the process-local lock for a job's mutable state.

    This serializes delete, rerun, status updates, and same-process associated
    record writes inside one API worker. It does not coordinate across API
    replicas; multi-replica deployments need a store-level lease or conditional
    delete to close that gap durably.
    """
    key = (workspace, job_name)
    lock = _JOB_MUTATION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _JOB_MUTATION_LOCKS[key] = lock
    return lock


def create_platform_job_response(job: PlatformJob, attempt: PlatformJobAttempt) -> PlatformJobResponse:
    """Helper to create PlatformJobResponse from job and attempt entities."""
    ownership = job.ownership
    if hasattr(ownership, "model_dump"):
        ownership = ownership.model_dump()  # type: ignore
    return PlatformJobResponse(
        id=job.id,
        attempt_id=attempt.id,
        name=job.name,
        description=job.description,
        workspace=job.workspace,
        project=job.project,
        created_at=job.created_at,  # type: ignore
        updated_at=job.updated_at,  # type: ignore
        source=job.source,
        spec=job.spec,
        platform_spec=job.platform_spec,
        fileset=job.fileset,
        output_location=job.output_location,
        status=attempt.status,
        status_details=attempt.status_details,
        error_details=attempt.error_details,
        ownership=ownership,
        custom_fields=job.custom_fields,
    )


def _format_status_for_message(status: PlatformJobStatus | str) -> str:
    if isinstance(status, PlatformJobStatus):
        return status.value
    return str(status)


# Status lives on PlatformJobAttempt, not PlatformJob, so it cannot be
# resolved by the PlatformJob entity-store query. Instead, the full filter tree
# is evaluated in-memory (op.apply(InMemoryFilterRepository(virtual_job))) against
# a "virtual job" entity that carries the attempt's status. The store query
# receives only a status-free *superset* of the filter (see _status_free_superset)
# so it never drops a row the in-memory pass would accept; that pass then narrows
# exactly.
_STATUS_FIELD = "data.status"


def _references_status(operation: FilterOperation | None) -> bool:
    """Whether the operation tree contains any comparison on the status field."""
    if operation is None:
        return False
    if isinstance(operation, ComparisonOperation):
        return operation.field == _STATUS_FIELD
    if isinstance(operation, LogicalOperation):
        return any(_references_status(child) for child in operation.operations)
    return False


def _status_free_superset(operation: FilterOperation | None) -> FilterOperation | None:
    """Build a status-free store filter that accepts a SUPERSET of ``operation``.

    Status lives on the attempt, not the job, so it cannot be pushed to the
    PlatformJob store query. We push down a relaxed, status-independent filter
    that is guaranteed to keep every row the in-memory evaluation would accept;
    that evaluation then narrows the candidate set exactly.

    ``None`` means "no store constraint" (accept all jobs in scope) — always a
    valid superset. The relaxation rules below preserve the superset property:

    - Status comparison -> None. Replacing a constraint with "accept all" only
      widens.
    - Non-status comparison -> itself. Status-independent, so it is exact and
      safe to push verbatim.
    - ``$and`` -> AND of the children's supersets (children that relax to None
      are dropped). Any row satisfying the original AND satisfies every child,
      hence every child-superset, hence their AND. Dropping a None child only
      removes a constraint.
    - ``$or`` -> OR of the children's supersets, UNLESS any child relaxes to
      None, in which case the whole OR relaxes to None (an unconstrained branch
      makes the union unconstrained). Each child-superset is a superset of its
      child, so the OR of supersets is a superset of the OR of children.
    - ``$not`` -> kept verbatim only when its operand is status-free (then the
      whole negation is exact and status-independent). If the operand references
      status, negation can invert sub/superset relationships, so we relax the
      entire ``$not`` to None and let the in-memory pass do the work.
    """
    if operation is None:
        return None
    if isinstance(operation, ComparisonOperation):
        return None if operation.field == _STATUS_FIELD else operation
    if isinstance(operation, LogicalOperation):
        if operation.operator == FilterOperator.NOT:
            # not X is status-independent only when X is; otherwise relax to None.
            return None if _references_status(operation) else operation

        relaxed = [_status_free_superset(child) for child in operation.operations]

        if operation.operator == FilterOperator.OR:
            # An unconstrained branch makes the union unconstrained.
            if any(child is None for child in relaxed):
                return None
            kept = [child for child in relaxed if child is not None]
        else:  # AND: dropping a None child only removes a constraint.
            kept = [child for child in relaxed if child is not None]

        if not kept:
            return None
        if len(kept) == 1:
            return kept[0]
        return LogicalOperation(operator=operation.operator, operations=kept)
    return operation


def _build_virtual_job_entity(job: PlatformJob, attempt: PlatformJobAttempt) -> dict[str, Any]:
    """Build the virtual entity that ``InMemoryFilterRepository`` evaluates.

    The filter tree addresses fields the way the entity store stores them: base
    columns (``name``, ``project``, ``workspace``, ...) as plain attributes and
    everything else under ``data.<field>`` (e.g. ``data.source``). This dict
    mirrors that DBEntity row shape so the in-memory repository resolves every
    field the same way the SQL repository would.

    Status is the join: it lives on the attempt, so it is injected as
    ``data.status``. The attempt status enum is stored as its string value
    (e.g. ``"active"``) so it compares equal to the string the filter carries.
    """
    data = dict(job._get_data_fields())
    data[_STATUS_FIELD.split(".", 1)[1]] = attempt.status.value
    return {
        "id": job.id,
        "name": job.name,
        "workspace": job.workspace,
        "project": job.project,
        "entity_type": PlatformJob.__entity_type__,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "data": data,
    }


class JobDispatcher:
    """Job Dispatcher for managing job lifecycle and interactions with job related repositories."""

    def __init__(
        self,
        store: EntityClient,
        sdk: AsyncNeMoPlatform,
    ):
        self.store = store
        self.sdk = sdk

    # =========================================================================
    # Job Operations
    # =========================================================================

    async def validate_job_secrets(
        self, job_spec: PlatformJobSpec, job_workspace: str, sdk: Optional[AsyncNeMoPlatform] = None
    ) -> None:
        # Ensure that any referenced secrets in steps exist and the user has access (user-scoped sdk).
        sdk_to_use = sdk if sdk is not None else self.sdk
        for step in job_spec.steps:
            if not step.environment:
                continue
            for env_var in step.environment:
                if env_var.from_secret:
                    workspace, secret_name = get_entity_parts(env_var.from_secret.name, default_workspace=job_workspace)
                    secrets = client_from_platform(sdk_to_use, AsyncSecretsClient)
                    try:
                        await secrets.get_secret(name=secret_name, workspace=workspace)
                    except ClientNotFoundError as exc:
                        raise JobSecretValidationError(f"Secret '{workspace}/{secret_name}' not found.") from exc
                    except ClientPermissionDeniedError as exc:
                        raise JobSecretValidationError(
                            f"User does not have access to secret '{workspace}/{secret_name}'."
                        ) from exc
                    except Exception as exc:
                        logger.exception(
                            "Error validating secret", extra={"secret_name": secret_name, "workspace": workspace}
                        )
                        raise JobSecretValidationError(
                            f"Unknown error when validating secret '{workspace}/{secret_name}'."
                        ) from exc

    async def create_job(
        self,
        job_req: CreatePlatformJobRequest,
        workspace: str,
        auth_context: Optional[AuthContext] = None,
        sdk: Optional[AsyncNeMoPlatform] = None,
    ) -> PlatformJobResponse:
        """Create a new job and its first step."""
        if job_req.name is None:
            return await self._create_job(job_req, workspace, auth_context=auth_context, sdk=sdk)

        async with _get_job_mutation_lock(job_req.name, workspace):
            return await self._create_job(job_req, workspace, auth_context=auth_context, sdk=sdk)

    async def _create_job(
        self,
        job_req: CreatePlatformJobRequest,
        workspace: str,
        auth_context: Optional[AuthContext] = None,
        sdk: Optional[AsyncNeMoPlatform] = None,
    ) -> PlatformJobResponse:
        """Create a new job after the caller has acquired any needed name lock."""
        job_name = job_req.name
        if job_name is not None:
            # Check if a job with the same name already exists
            try:
                existing_job = await self.store.get(PlatformJob, job_name, workspace=workspace)
                if existing_job:
                    raise JobAlreadyExistsError(
                        f"Job with name '{job_name}' already exists in workspace '{workspace}'."
                    )
            except EntityNotFoundError:
                pass  # Job does not exist, proceed to create

        try:
            platform_spec = job_req.platform_spec

            await self.validate_job_secrets(platform_spec, workspace, sdk=sdk)

            # Generate a reference ID for naming (job entity ID is assigned by store)
            # Generate auto-name if not provided, ensuring it fits 32 char limit
            # Format: {source[:20]}-{short_id} where short_id is last 8 chars of job_ref_id
            if not job_name:
                job_ref_id = get_model_id("job")
                short_id = job_ref_id[-8:].lower()
                source_prefix = job_req.source[:20]
                job_name = f"{source_prefix}-{short_id}"

            # Resolve the fileset for job artifacts (caller-supplied output_location or auto-created).
            files = client_from_platform(self.sdk, AsyncFilesClient)
            if job_req.output_location is not None:
                try:
                    await files.get_fileset(name=job_req.output_location, workspace=workspace)
                except (ClientNotFoundError, ClientPermissionDeniedError) as exc:
                    raise JobOutputLocationError(
                        f"fileset '{job_req.output_location}' not found or not accessible in workspace '{workspace}'"
                    ) from exc
                fileset_name = job_req.output_location
            else:
                fileset_resp = await files.create_fileset(
                    body=CreateFilesetRequest(name=f"job-fileset-{job_name}"),
                    workspace=workspace,
                )
                fileset_name = fileset_resp.data().name

            # Create job (ID is assigned by entity store)
            job = await self.store.create(
                PlatformJob(
                    name=job_name,
                    workspace=workspace,
                    project=job_req.project,
                    description=job_req.description,
                    source=job_req.source,
                    spec=job_req.spec,
                    platform_spec=platform_spec,
                    fileset=fileset_name,
                    output_location=job_req.output_location,
                    ownership=job_req.ownership,
                    custom_fields=job_req.custom_fields,
                )
                # Add auth_context separately, it's a PrivateAttr, so it's not set in the constructor
                .with_auth_context(auth_context)
            )

            # Create first attempt
            # Use short ID prefix to stay within 32 char name limit
            attempt_ref_id = get_model_id("att")
            attempt = await self.store.create(
                PlatformJobAttempt(
                    name=attempt_ref_id,
                    workspace=workspace,
                    job=job.id,
                    seq=0,
                    status=PlatformJobStatus.CREATED,
                    spec=job.spec,
                    platform_spec=job.platform_spec,
                )
            )

            # Update job with attempt ID.
            # Restore auth_context: the create response may have sanitized it for
            # non-service principals, but we need the original value persisted.
            job.current_attempt_id = attempt.id
            job.with_auth_context(auth_context)
            job = await self.store.update(job)
            result = create_platform_job_response(job, attempt)
            logger.info("Created new job", extra={"job": job.name, "workspace": job.workspace})
            await self._start_attempt(job, attempt)

            operations_counter.add(1, attributes={"operation": "create_job"})
            return result

        except Exception as e:
            raise e

    async def get_job(self, job_name: str, workspace: str) -> PlatformJobResponse | None:
        """Get a platform job by ID with its current attempt."""
        try:
            job_entity = await self.store.get(PlatformJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            return None
        try:
            attempt = await self.store.get_by_id(PlatformJobAttempt, job_entity.current_attempt_id)
        except EntityNotFoundError:
            return None
        return create_platform_job_response(job_entity, attempt)

    async def list_jobs(
        self,
        parsed: ParsedFilter,
        workspace: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[PlatformJobListSortField] = None,
    ) -> Tuple[List[PlatformJobResponse], int]:
        """List platform jobs with their current attempts."""
        # Status lives on PlatformJobAttempt, not PlatformJob, so the full filter
        # tree is evaluated in-memory against a virtual job entity that carries the
        # attempt status. The store query receives a status-free SUPERSET of the
        # filter so it never drops a row the in-memory pass would accept (see
        # _status_free_superset); that pass then narrows the page exactly.
        # The operation is already entity-translated by make_filter_dep.
        store_operation = _status_free_superset(parsed.operation)

        # Calculate page from offset
        page = 1
        page_size = limit or 100
        if offset is not None and limit is not None:
            page = (offset // limit) + 1

        sort_str = sort.value if sort else "-created_at"

        response = await self.store.list(
            PlatformJob,
            filter_operation=store_operation,
            page=page,
            page_size=page_size,
            workspace=workspace,
            sort=sort_str,
        )

        # IN-MEMORY JOIN: load each job's current attempt, then evaluate the full
        # (status-aware) filter tree against a virtual entity carrying the status.
        job_outputs: list[PlatformJobResponse] = []
        for job in response.data:
            if not job.current_attempt_id:
                continue

            try:
                attempt = await self.store.get_by_id(PlatformJobAttempt, job.current_attempt_id)
            except EntityNotFoundError:
                logger.warning(f"Attempt {job.current_attempt_id} not found for job {job.id}")
                continue

            if parsed.operation is not None:
                virtual_entity = _build_virtual_job_entity(job, attempt)
                if not parsed.operation.apply(InMemoryFilterRepository(virtual_entity)):
                    continue

            platform_job = create_platform_job_response(job, attempt)
            job_outputs.append(platform_job)

        # Sort in memory
        if sort:
            reverse = sort.get_sort_direction() == "desc"
            job_outputs.sort(key=lambda j: getattr(j, sort.get_field_name()), reverse=reverse)

        return job_outputs, response.pagination.total_results

    async def delete_job(self, job_name: str, workspace: str) -> bool:
        """Delete a job and all of its associated data (steps, tasks, results, logs).

        Returns:
            True if job was deleted, False if job was not found.
        """
        async with _get_job_mutation_lock(job_name, workspace):
            return await self._delete_job_locked(job_name, workspace)

    async def _delete_job_locked(self, job_name: str, workspace: str) -> bool:
        """Delete a terminal job while holding the per-job mutation lock."""
        try:
            job_entity = await self.store.get(PlatformJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            return False

        extras = {"job": job_entity.name, "workspace": job_entity.workspace}

        try:
            # Get all attempts
            attempts = await self._list_all_entities(
                PlatformJobAttempt,
                filter_obj={"job": job_entity.id},
                workspace=workspace,
            )
            steps_by_attempt: dict[str, list[PlatformJobStep]] = {}
            for attempt in attempts:
                if not attempt.status.is_terminal():
                    raise JobDeletionConflictError(
                        f"Cannot delete job '{job_entity.name}' while it is "
                        f"'{_format_status_for_message(attempt.status)}'. Cancel the job and wait for it to reach a "
                        "terminal state before deleting."
                    )

                steps = await self._list_all_entities(
                    PlatformJobStep,
                    filter_obj={"attempt_id": attempt.id},
                    workspace=workspace,
                )
                steps_by_attempt[attempt.id] = steps

                for step in steps:
                    if not step.status.is_terminal():
                        raise JobDeletionConflictError(
                            f"Cannot delete job '{job_entity.name}' while step '{step.name}' is "
                            f"'{_format_status_for_message(step.status)}'. Cancel the job and wait for it to reach a "
                            "terminal state before deleting."
                        )

            # Get all results
            results = await self._list_all_entities(
                PlatformJobResult,
                filter_obj={"job": job_entity.id},
                workspace=workspace,
            )

            for result in results:
                await self.store.delete_by_id(PlatformJobResult, result.id)

            # Delete steps and tasks for each attempt
            for attempt in attempts:
                for step in steps_by_attempt[attempt.id]:
                    # Delete tasks
                    tasks = await self._list_all_entities(
                        PlatformJobTask,
                        filter_obj={"step_id": step.id},
                        workspace=workspace,
                    )
                    for task in tasks:
                        await self.store.delete_by_id(PlatformJobTask, task.id)

                    # Delete step
                    await self.store.delete_by_id(PlatformJobStep, step.id)

                # Delete attempt
                await self.store.delete_by_id(PlatformJobAttempt, attempt.id)

            # Delete job
            await self.store.delete_by_id(PlatformJob, job_entity.id)

            # Delete only the fileset the job owns (auto-created); a caller-supplied output_location
            # fileset is left for the caller to manage. Ownership is recorded at create time, so a
            # caller cannot fool this by naming their fileset like an auto-created one. Tolerate the
            # owned fileset already being gone.
            if job_entity.output_location is None:
                try:
                    files = client_from_platform(self.sdk, AsyncFilesClient)
                    await files.delete_fileset(name=job_entity.fileset, workspace=workspace)
                except ClientNotFoundError:
                    logger.warning(
                        "Job fileset not found during deletion, may have been cleaned up already", extra=extras
                    )

            logger.info(
                "Deleted job and all associated data for job",
                extra=extras,
            )
            operations_counter.add(1, attributes={"operation": "delete_job"})
            return True

        except JobDeletionConflictError:
            logger.info("Refusing to delete non-terminal job", extra=extras)
            raise
        except Exception as e:
            logger.exception("Error deleting job", extra=extras)
            raise e

    async def _list_all_entities(
        self,
        entity_type: type[EntityT],
        *,
        filter_obj: dict[str, Any],
        workspace: str,
    ) -> list[EntityT]:
        """Return every page for a filtered entity query."""
        page = 1
        entities: list[EntityT] = []
        while True:
            response = await self.store.list(
                entity_type,
                filter_obj=filter_obj,
                page=page,
                page_size=_DELETE_PAGE_SIZE,
                workspace=workspace,
            )
            entities.extend(response.data)
            if response.pagination.page >= response.pagination.total_pages:
                return entities
            page = response.pagination.page + 1

    # =========================================================================
    # Attempt Operations
    # =========================================================================

    async def _start_attempt(self, job: PlatformJob, attempt: PlatformJobAttempt) -> None:
        """Create the first step of the new attempt."""
        first_step = attempt.get_first_step()
        # With parent-scoped uniqueness, step names are unique per attempt (parent)
        # Store original spec name in config for reference
        step_config = dict(first_step.config) if first_step.config else {}
        step_config[STEP_SPEC_NAME_CONFIG_KEY] = first_step.name
        await self.store.create(
            PlatformJobStep(
                name=first_step.name,  # Simple name, unique per attempt via parent-scoped uniqueness
                workspace=job.workspace,
                attempt_id=attempt.id,
                config=step_config,
                status=PlatformJobStatus.CREATED,
            )
        )

        operations_counter.add(1, attributes={"operation": "start_attempt"})
        logger.info(
            "Created first step for job", extra={"job": job.name, "workspace": job.workspace, "step": first_step.name}
        )

    async def get_attempt(self, attempt_id: str) -> Optional[PlatformJobAttempt]:
        """Get an attempt by ID."""
        try:
            return await self.store.get_by_id(PlatformJobAttempt, attempt_id)
        except EntityNotFoundError:
            return None

    async def _get_job_by_id_optional(self, job_id: str) -> Optional[PlatformJob]:
        """Get a job by ID, returning None if not found."""
        try:
            return await self.store.get_by_id(PlatformJob, job_id)
        except EntityNotFoundError:
            return None

    async def _gather_attempts(self, attempt_ids: set[str]) -> list[Optional[PlatformJobAttempt]]:
        """Fetch all attempts by ID concurrently using TaskGroup."""
        if not attempt_ids:
            return []
        aid_list = list(attempt_ids)
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self.get_attempt(aid)) for aid in aid_list]
        return [t.result() for t in tasks]

    async def _gather_jobs(self, job_ids: set[str]) -> list[Optional[PlatformJob]]:
        """Fetch all jobs by ID concurrently using TaskGroup."""
        if not job_ids:
            return []
        jid_list = list(job_ids)
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._get_job_by_id_optional(jid)) for jid in jid_list]
        return [t.result() for t in tasks]

    async def get_current_attempt(self, job_name: str, workspace: str) -> Optional[PlatformJobAttempt]:
        """Get the current attempt for a job."""

        job_entity = await self.store.get(PlatformJob, job_name, workspace=workspace)
        response = await self.store.list(
            PlatformJobAttempt,
            filter_obj={"job": job_entity.id},
            page_size=1000,
            workspace=workspace,
        )

        if not response.data:
            return None

        # Sort by seq descending and return first
        sorted_attempts = sorted(response.data, key=lambda a: a.seq, reverse=True)
        return sorted_attempts[0] if sorted_attempts else None

    # =========================================================================
    # Step Operations
    # =========================================================================

    async def get_current_job_step_by_name(
        self, job_name: str, step_name: str, workspace: str
    ) -> Optional[PlatformJobStep]:
        """Get the current job step by name.

        With parent-scoped uniqueness, step names are exact matches (no suffix needed).
        The job_name parameter is the job's name (not entity ID), as received from API endpoints.
        If workspace is None, searches across all workspaces.
        """
        try:
            attempt = await self.get_current_attempt(job_name, workspace=workspace)
        except EntityNotFoundError:
            return None
        operations_counter.add(1, attributes={"operation": "get_current_job_step_by_name"})
        if not attempt:
            return None

        # Get all steps for the attempt and find the one matching the step name
        response = await self.store.list(
            PlatformJobStep,
            filter_obj={"attempt_id": attempt.id},
            page_size=100,
            workspace=workspace,
        )
        # With parent-scoped uniqueness, step names are exact matches
        for step in response.data:
            if step.name == step_name:
                return step
        return None

    async def _get_step_by_status(
        self, job: PlatformJob, workspace: str, possible_statuses: list[PlatformJobStatus]
    ) -> Optional[PlatformJobStep]:
        """Get the step for a job with a given possible status."""

        if not job.current_attempt_id:
            return None

        response = await self.store.list(
            PlatformJobStep,
            filter_obj={"attempt_id": job.current_attempt_id},
            page_size=1000,
            workspace=workspace,
        )

        for step in response.data:
            if step.status in possible_statuses:
                return step

        return None

    async def list_steps(
        self,
        filter: PlatformJobStepsListFilter,
        sort: PlatformJobSortField,
        limit: int,
        offset: int,
        workspace: str,
    ) -> tuple[list[PlatformJobStepWithContext], int]:
        """Poll steps based on the provided filter, sort, limit, and offset."""
        filter_str: Optional[str] = None
        if filter.status:
            filter_str = json.dumps({"data.status": {"$in": filter.status}})
        sort_str = sort.value

        use_store_pagination = (not filter.job or filter.job == "-") and not filter.source

        if use_store_pagination:
            page = (offset // limit) + 1 if limit > 0 else 1
            response = await self.store.list(
                PlatformJobStep,
                workspace=workspace,
                filter_str=filter_str,
                sort=sort_str,
                page=page,
                page_size=limit,
            )
            all_steps = list(response.data)
            total_count = response.pagination.total_results
        else:
            response = await self.store.list(
                PlatformJobStep,
                workspace=workspace,
                filter_str=filter_str,
                sort=sort_str,
                page_size=1000,
            )
            all_steps = list(response.data)
            while response.pagination.page < response.pagination.total_pages:
                response = await self.store.list(
                    PlatformJobStep,
                    workspace=workspace,
                    filter_str=filter_str,
                    sort=sort_str,
                    page=response.pagination.page + 1,
                    page_size=1000,
                )
                all_steps.extend(response.data)
            total_count = 0  # set after in-memory filter below

        if not all_steps:
            operations_counter.add(1, attributes={"operation": "list_steps"})
            return [], total_count if use_store_pagination else 0

        attempt_ids = {s.attempt_id for s in all_steps}
        attempt_results = await self._gather_attempts(attempt_ids)
        attempt_by_id = {aid: a for aid, a in zip(attempt_ids, attempt_results) if a is not None}
        job_ids = {a.job for a in attempt_by_id.values()}
        job_results = await self._gather_jobs(job_ids)
        job_by_id = {jid: j for jid, j in zip(job_ids, job_results) if j is not None}

        result = []
        for step in all_steps:
            attempt = attempt_by_id.get(step.attempt_id)
            if not attempt:
                continue
            job = job_by_id.get(attempt.job)
            if not job:
                continue
            if not use_store_pagination:
                if filter.job and filter.job != "-" and job.name != filter.job:
                    continue
                if filter.status and step.status not in filter.status:
                    continue
                if filter.source and job.source != filter.source:
                    continue
            enriched_step = PlatformJobStepWithContext(
                id=step.id,
                job=job.name,
                attempt_id=step.attempt_id,
                workspace=step.workspace,
                fileset=job.fileset,
                artifact_base_path=job_artifact_base_path(job.name, job.output_location),
                name=step.name,
                step_spec=attempt.get_step_spec(step.name),
                status=step.status,
                status_details=step.status_details,
                error_details=step.error_details,
                auth_context=job.auth_context,
                created_at=step.created_at,
                updated_at=step.updated_at,
            )
            result.append(enriched_step)

        if use_store_pagination:
            operations_counter.add(1, attributes={"operation": "list_steps"})
            return result, total_count
        total_count = len(result)
        reverse = sort.get_sort_direction() == "desc"
        result.sort(key=lambda s: getattr(s, sort.get_field_name()), reverse=reverse)
        start_idx = offset
        end_idx = start_idx + limit
        paginated = result[start_idx:end_idx]
        operations_counter.add(1, attributes={"operation": "list_steps"})
        return paginated, total_count

    # =========================================================================
    # Status Operations
    # =========================================================================

    async def get_job_status(self, job_name: str, workspace: str) -> Optional[PlatformJobStatusResponse]:
        """Get the overall status for a job, indexed by step and task."""
        job = await self.get_job(job_name, workspace)
        if job is None:
            return None

        # Get steps for current attempt, sorted by creation order to match platform_spec
        steps_response = await self.store.list(
            PlatformJobStep,
            workspace=workspace,
            filter_obj={"attempt_id": job.attempt_id},
            sort="created_at",
            page_size=1000,
        )

        step_statuses = []
        for step in steps_response.data:
            # Get tasks for step
            tasks_response = await self.store.list(
                PlatformJobTask,
                workspace=workspace,
                filter_obj={"step_id": step.id},
                page_size=1000,
            )

            task_statuses = [
                PlatformJobTaskStatusResponse(
                    id=task.id,
                    name=task.name,
                    status=task.status,
                    status_details=task.status_details,
                    error_details=task.error_details or {},
                    error_stack=task.error_stack,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
                for task in tasks_response.data
            ]

            step_statuses.append(
                PlatformJobStepStatusResponse(
                    id=step.id,
                    name=step.name,
                    status=step.status,
                    status_details=step.status_details,
                    error_details=step.error_details or {},
                    tasks=task_statuses,
                    created_at=step.created_at,
                    updated_at=step.updated_at,
                )
            )

        operations_counter.add(1, attributes={"operation": "get_job_status"})
        return PlatformJobStatusResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            status_details=job.status_details,
            error_details=job.error_details,
            steps=step_statuses,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    def _step_update_would_be_noop(
        self,
        step: PlatformJobStep,
        status: PlatformJobStatus,
        status_details: Optional[Dict[str, Any]],
        error_details: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True if applying the update would leave the persisted step unchanged."""
        if step.status != status:
            return False
        if status_details:
            merged_sd = self._update_status_details_object(dict(step.status_details), status_details)
        else:
            merged_sd = step.status_details
        if merged_sd != step.status_details:
            return False
        effective_ed = error_details if error_details else step.error_details
        if (effective_ed or {}) != (step.error_details or {}):
            return False
        return True

    async def update_job_status_from_step(
        self,
        step: PlatformJobStep,
        status: PlatformJobStatus,
        status_details: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> tuple[PlatformJobStep, PlatformJobAttempt]:
        """Update a job's status based on the step's status.

        On EntityConflictError (e.g. reconciler updated the step concurrently), refetches
        the step and retries once if the requested transition is still valid.
        """
        attempt = await self.get_attempt(step.attempt_id)
        if attempt is None:
            raise JobStatusUpdateSkippedError(f"Attempt does not exist: {step.attempt_id}")
        try:
            job = await self.store.get_by_id(PlatformJob, attempt.job)
        except EntityNotFoundError as exc:
            raise JobStatusUpdateSkippedError(f"Job does not exist: {attempt.job}") from exc

        async with _get_job_mutation_lock(job.name, job.workspace):
            return await self._update_job_status_from_step_locked(step, status, status_details, error_details)

    async def _update_job_status_from_step_locked(
        self,
        step: PlatformJobStep,
        status: PlatformJobStatus,
        status_details: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> tuple[PlatformJobStep, PlatformJobAttempt]:
        """Update a job from a step while holding the per-job mutation lock."""
        step_to_save = step
        saved_step: PlatformJobStep | None = None
        for attempt in range(2):
            if not step_to_save.status.can_transition_to(status):
                raise StateTransitionConflictError(
                    f"Invalid status transition from {step_to_save.status} to {status} for step {step_to_save.id}"
                )

            if self._step_update_would_be_noop(step_to_save, status, status_details, error_details):
                saved_step = step_to_save
                break

            # Apply updates to a copy of fields we persist
            if status != step_to_save.status:
                step_to_save.status = status
            if error_details:
                step_to_save.error_details = error_details
            if status_details:
                step_to_save.status_details = self._update_status_details_object(
                    step_to_save.status_details, status_details
                )
            try:
                saved_step = await self.store.update(step_to_save)
                break
            except EntityConflictError as e:
                if attempt == 0:
                    # Refetch step and retry once if transition still valid
                    try:
                        refetched = await self.store.get_by_id(PlatformJobStep, step_to_save.id)
                    except EntityNotFoundError:
                        raise e from e
                    if refetched.status.can_transition_to(status):
                        step_to_save = refetched
                        continue
                raise e from e

        if saved_step is None:
            raise RuntimeError("update_job_status_from_step did not produce a saved step")

        # Update job / attempt status from step
        attempt = await self.get_attempt(saved_step.attempt_id)
        if attempt is None:
            raise JobStatusUpdateSkippedError(f"Attempt does not exist: {saved_step.attempt_id}")

        # Determine new attempt status
        new_attempt_status = attempt.status

        # Get the original step spec name from config (step entity names have suffixes for uniqueness)
        step_spec_name = get_step_spec_name(saved_step.config, saved_step.name) or saved_step.name

        if (
            saved_step.status == PlatformJobStatus.PENDING
            and attempt.status != PlatformJobStatus.PENDING
            and step_spec_name == attempt.platform_spec.steps[0].name
        ):
            new_attempt_status = PlatformJobStatus.PENDING
            logger.info(
                "Job is pending",
                extra={
                    "job": attempt.job,
                    "attempt": attempt.id,
                    "step": saved_step.id,
                    "workspace": attempt.workspace,
                },
            )
        elif saved_step.status == PlatformJobStatus.ACTIVE and attempt.status != PlatformJobStatus.ACTIVE:
            new_attempt_status = PlatformJobStatus.ACTIVE
            logger.info(
                "Job is active",
                extra={
                    "job": attempt.job,
                    "attempt": attempt.id,
                    "step": saved_step.id,
                    "workspace": attempt.workspace,
                },
            )
        elif saved_step.status == PlatformJobStatus.COMPLETED:
            # Use the step spec name from config to find the next step
            next_step = attempt.get_next_step_spec(step_spec_name)
            if next_step:
                # With parent-scoped uniqueness, use simple step name (unique per attempt)
                next_step_config = dict(next_step.config) if next_step.config else {}
                next_step_config[STEP_SPEC_NAME_CONFIG_KEY] = next_step.name
                try:
                    await self.store.create(
                        PlatformJobStep(
                            name=next_step.name,  # Simple name, unique per attempt via parent-scoped uniqueness
                            workspace=attempt.workspace,
                            attempt_id=attempt.id,
                            config=next_step_config,
                        )
                    )
                    logger.info(
                        "Scheduled job step",
                        extra={
                            "job": attempt.job,
                            "attempt": attempt.id,
                            "step": next_step.name,
                            "workspace": attempt.workspace,
                        },
                    )
                except EntityConflictError:
                    logger.debug(
                        "Step already exists, skipping creation (idempotent)",
                        extra={
                            "job": attempt.job,
                            "attempt": attempt.id,
                            "step": next_step.name,
                            "workspace": attempt.workspace,
                        },
                    )
            else:
                new_attempt_status = PlatformJobStatus.COMPLETED
                logger.info(
                    "Job completed all steps",
                    extra={"job": attempt.job, "attempt": attempt.id, "workspace": attempt.workspace},
                )
        elif saved_step.status in (
            PlatformJobStatus.CANCELLING,
            PlatformJobStatus.CANCELLED,
            PlatformJobStatus.ERROR,
            PlatformJobStatus.PAUSING,
            PlatformJobStatus.PAUSED,
            PlatformJobStatus.RESUMING,
        ):
            new_attempt_status = saved_step.status

        # Update attempt if status changed, respecting state machine transitions to prevent
        # backward transitions from concurrent reconciler/scheduler updates
        if new_attempt_status != attempt.status:
            if not attempt.status.can_transition_to(new_attempt_status):
                logger.debug(
                    f"Skipping attempt {attempt.id} status update from '{attempt.status}' "
                    f"to '{new_attempt_status}': transition not valid (concurrent update "
                    f"advanced the attempt past the desired state)"
                )
            else:
                attempt.status = new_attempt_status
                if attempt.status == PlatformJobStatus.ERROR:
                    attempt.error_details = saved_step.error_details
                attempt = await self.store.update(attempt)
                logger.info(
                    "Updated job attempt status",
                    extra={
                        "job": attempt.job,
                        "attempt": attempt.id,
                        "status": attempt.status,
                        "workspace": attempt.workspace,
                    },
                )

        operations_counter.add(1, attributes={"operation": "update_job_status_from_step"})
        return saved_step, attempt

    async def update_job_status_details(
        self, job_name: str, workspace: str, status_details: Dict[str, Any]
    ) -> PlatformJobResponse | None:
        """Update the status details of a job attempt."""
        job = await self.store.get(PlatformJob, job_name, workspace=workspace)
        if job is None:
            return None

        attempt = await self.get_current_attempt(job.name, workspace=workspace)
        if attempt is None:
            raise Exception(f"Job attempt with Job ID {job.id} does not exist.")

        attempt.status_details = self._update_status_details_object(attempt.status_details, status_details)
        attempt = await self.store.update(attempt)

        operations_counter.add(1, attributes={"operation": "update_job_status_details"})
        return create_platform_job_response(job, attempt)

    @staticmethod
    def _update_status_details_object(existing: Optional[Dict[str, Any]], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for updating status details."""
        if existing is None:
            return updates
        for key, value in updates.items():
            existing[key] = value
        return existing

    # =========================================================================
    # Job Control Operations
    # =========================================================================

    async def cancel_job(self, job_name: str, workspace: str) -> PlatformJobResponse | None:
        """Cancel a job."""
        try:
            job_entity = await self.store.get(PlatformJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            return None

        # Check if the job has any created steps, and if so, check if all of them are in created state.
        # If so, cancel the job immediately without going through the normal cancellation process.
        if job_entity.current_attempt_id:
            response = await self.store.list(
                PlatformJobStep,
                filter_obj={"attempt_id": job_entity.current_attempt_id},
                workspace=workspace,
            )
            steps = response.data
            if steps and all(s.status == PlatformJobStatus.CREATED for s in steps):
                _, attempt = await self.update_job_status_from_step(
                    steps[0],
                    PlatformJobStatus.CANCELLED,
                )
                operations_counter.add(1, attributes={"operation": "cancel_job"})
                return create_platform_job_response(job_entity, attempt)

        non_terminal_step = await self._get_step_by_status(job_entity, workspace, PlatformJobStatus.non_terminals())
        if non_terminal_step:
            _, attempt = await self.update_job_status_from_step(
                non_terminal_step,
                PlatformJobStatus.CANCELLING,
            )
        else:
            attempt = await self.get_current_attempt(job_entity.name, workspace=job_entity.workspace)
            if attempt is None:
                raise Exception(f"Job Attempt with Job ID {job_entity.id} does not exist.")

        operations_counter.add(1, attributes={"operation": "cancel_job"})
        return create_platform_job_response(job_entity, attempt)

    async def rerun_job(self, job_name: str, workspace: str) -> PlatformJobResponse | None:
        """Re-run a job."""
        async with _get_job_mutation_lock(job_name, workspace):
            return await self._rerun_job_locked(job_name, workspace)

    async def _rerun_job_locked(self, job_name: str, workspace: str) -> PlatformJobResponse | None:
        """Re-run a terminal job while holding the per-job mutation lock."""
        try:
            job_entity = await self.store.get(PlatformJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            return None

        attempt = await self.get_current_attempt(job_entity.name, workspace=job_entity.workspace)
        if attempt is None:
            return None

        # Only allow re-run if the attempt is in a terminal state
        if attempt.status.is_terminal():
            seq = attempt.seq + 1
            # Use short ID prefix to stay within 32 char name limit
            attempt_ref_id = get_model_id("att")
            next_attempt = await self.store.create(
                PlatformJobAttempt(
                    name=attempt_ref_id,
                    workspace=job_entity.workspace,
                    job=job_entity.id,
                    seq=seq,
                    status=PlatformJobStatus.CREATED,
                    spec=job_entity.spec,
                    platform_spec=job_entity.platform_spec,
                )
            )
            # Update job with new attempt ID
            job_entity.current_attempt_id = next_attempt.id
            job_entity = await self.store.update(job_entity)
            await self._start_attempt(job_entity, next_attempt)
            operations_counter.add(1, attributes={"operation": "rerun_job"})
            return create_platform_job_response(job_entity, next_attempt)
        else:
            logger.warning(
                "Attempt to re-run job not in terminal state",
                extra={
                    "job": job_entity.name,
                    "attempt": attempt.id,
                    "status": attempt.status,
                    "workspace": job_entity.workspace,
                },
            )
            operations_counter.add(1, attributes={"operation": "rerun_job"})
            return create_platform_job_response(job_entity, attempt)

    async def pause_job(self, job_name: str, workspace: str) -> PlatformJobResponse | None:
        """Pause a job."""
        try:
            job_entity = await self.store.get(PlatformJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            return None

        active_or_pending_step = await self._get_step_by_status(
            job_entity, workspace, [PlatformJobStatus.CREATED, PlatformJobStatus.PENDING, PlatformJobStatus.ACTIVE]
        )
        if active_or_pending_step:
            _, attempt = await self.update_job_status_from_step(
                active_or_pending_step,
                PlatformJobStatus.PAUSING,
            )
        else:
            attempt = await self.get_current_attempt(job_entity.name, workspace=job_entity.workspace)
            if attempt is None:
                raise Exception(f"Job Attempt with Job ID {job_entity.id} does not exist.")
        operations_counter.add(1, attributes={"operation": "pause_job"})
        return create_platform_job_response(job_entity, attempt)

    async def resume_job(self, job_name: str, workspace: str) -> PlatformJobResponse | None:
        """Resume a job."""
        try:
            job = await self.store.get(PlatformJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            return None

        # Only allow resume if the job is in a paused state
        paused_step = await self._get_step_by_status(job, workspace, [PlatformJobStatus.PAUSED])
        if paused_step:
            await self.update_job_status_from_step(
                paused_step,
                PlatformJobStatus.RESUMING,
            )
        operations_counter.add(1, attributes={"operation": "resume_job"})
        return create_platform_job_response(job, await self.get_current_attempt(job.name, workspace))  # type: ignore

    # =========================================================================
    # Task Operations
    # =========================================================================

    async def get_task(self, step_id: str, task_name: str, workspace: str) -> Optional[PlatformJobTask]:
        """Get a platform job task by task name within a step.

        Note: task_id is actually the task NAME (not entity ID). Tasks are
        parent-scoped entities, meaning task names are unique within their
        parent step. This method looks up tasks by name within the given step.
        """
        try:
            task = await self.store.get(PlatformJobTask, task_name, workspace=workspace, parent=step_id)
            if task.step_id == step_id:
                return task
            return None
        except EntityNotFoundError:
            return None

    async def list_tasks(self, step_id: str, workspace: str) -> list[PlatformJobTask]:
        """List all platform job tasks for a specific step."""
        response = await self.store.list(
            PlatformJobTask,
            workspace=workspace,
            filter_obj={"step_id": step_id},
            page_size=1000,
        )
        return list(response.data)

    async def create_or_update_task(
        self, job_name: str, task_name: str, workspace: str, task_update: PlatformJobTaskUpdate, step: PlatformJobStep
    ) -> PlatformJobTask:
        """Create or update a task for a job step."""
        async with _get_job_mutation_lock(job_name, workspace):
            return await self._create_or_update_task_locked(job_name, task_name, workspace, task_update, step)

    async def _create_or_update_task_locked(
        self, job_name: str, task_name: str, workspace: str, task_update: PlatformJobTaskUpdate, step: PlatformJobStep
    ) -> PlatformJobTask:
        """Create or update a task while holding the per-job mutation lock."""
        await self.store.get(PlatformJob, job_name, workspace=workspace)
        await self.store.get_by_id(PlatformJobStep, step.id)

        task = await self.get_task(step.id, task_name, workspace=workspace)

        if not task:
            task = await self.store.create(
                PlatformJobTask(
                    name=task_name,
                    workspace=step.workspace,
                    step_id=step.id,
                    status=task_update.status,
                    status_details=task_update.status_details or {},
                    error_details=task_update.error_details or {},
                    error_stack=task_update.error_stack,
                )
            )
            logger.info(
                "Creating new job task",
                extra={"task": task.id, "job": job_name, "step": step.name, "workspace": step.workspace},
            )
        else:
            current_status = PlatformJobStatus(task.status)
            new_status = PlatformJobStatus(task_update.status)
            if not current_status.can_transition_to(new_status):
                logger.info(
                    "Ignoring invalid job task status transition",
                    extra={
                        "task": task.id,
                        "job": job_name,
                        "step": step.name,
                        "workspace": step.workspace,
                        "current_status": current_status,
                        "new_status": new_status,
                    },
                )
                operations_counter.add(1, attributes={"operation": "create_or_update_task"})
                return task

            task.status = new_status

            if task_update.error_details:
                task.error_details = task_update.error_details
            if task_update.status_details:
                task.status_details = self._update_status_details_object(
                    task.status_details, task_update.status_details
                )
            if task_update.error_stack:
                task.error_stack = task_update.error_stack

            task = await self.store.update(task)
            logger.info(
                "Updating job task",
                extra={
                    "task": task.id,
                    "job": job_name,
                    "step": step.name,
                    "workspace": step.workspace,
                    "status": task.status,
                },
            )

        # Propagate task status_details to parent job for progress tracking.
        # Training callbacks report progress (percentage_done, epoch, step, loss) at the task level,
        # but users query job status via sdk.customization.jobs.retrieve(). Without this propagation,
        # job.status_details would be empty and users couldn't see training progress.
        if task_update.status_details:
            try:
                await self.update_job_status_details(job_name, workspace, task_update.status_details)
                logger.debug("Propagated task status_details to job", extra={"job": job_name, "workspace": workspace})
            except Exception as e:
                # Log but don't fail the task update if job status_details update fails
                logger.warning(
                    "Failed to propagate task status_details to job",
                    extra={"job": job_name, "workspace": workspace, "error": e},
                )

        operations_counter.add(1, attributes={"operation": "create_or_update_task"})
        return task

    def is_step_cancelled(self, step: PlatformJobStep) -> bool:
        """Determine if a step is in a cancelled state."""
        return step.status in (PlatformJobStatus.CANCELLING, PlatformJobStatus.CANCELLED)

    # =========================================================================
    # Result Operations
    # =========================================================================

    async def create_result(
        self,
        job_id: str,
        job_name: str,
        result_name: str,
        artifact_url: str,
        artifact_storage_type: Any,
        workspace: str,
    ) -> PlatformJobResult:
        """Create a job result.

        Args:
            job_id: Job entity ID (used as parent reference for FK)
            result_name: Name of the result
            artifact_url: URL to the artifact
            artifact_storage_type: Type of artifact storage
            workspace: Workspace for the result
        """
        async with _get_job_mutation_lock(job_name, workspace):
            job = await self.store.get(PlatformJob, job_name, workspace=workspace)
            if job.id != job_id:
                raise EntityNotFoundError(f"Job ID mismatch for {workspace}/{job_name}")
            result = PlatformJobResult(
                name=result_name,
                workspace=workspace,
                job=job_id,  # Use entity ID for parent FK
                artifact_url=artifact_url,
                artifact_storage_type=artifact_storage_type,
            )
            return await self.store.create(result)

    async def get_result(self, job_name: str, result_name: str, workspace: str) -> Optional[PlatformJobResult]:
        """Get a platform job result."""
        try:
            job_entity = await self.store.get(PlatformJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            logger.warning(f"Job '{job_name}' not found in workspace '{workspace}'")
            return None
        try:
            return await self.store.get(PlatformJobResult, result_name, workspace=workspace, parent=job_entity.id)
        except EntityNotFoundError:
            logger.warning(f"Result '{result_name}' not found for job '{job_name}' in workspace '{workspace}'")
            return None

    async def list_results(
        self, job_id: str, workspace: str, sort: Optional[PlatformJobSortField] = None
    ) -> tuple[list[PlatformJobResult], int]:
        """List platform job results with optional filtering and pagination."""
        response = await self.store.list(
            PlatformJobResult,
            filter_obj={"job": job_id},
            page_size=1000,
            workspace=workspace,
        )

        results = list(response.data)
        if sort:
            reverse = sort.get_sort_direction() == "desc"
            results.sort(key=lambda r: getattr(r, sort.get_field_name()), reverse=reverse)

        return results, response.pagination.total_results
