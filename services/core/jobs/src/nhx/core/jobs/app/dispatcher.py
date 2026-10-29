# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import datetime
import json
import logging
import re
import weakref
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from nemo_helix_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_helix_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.types import JOB_FILESET_NAME_PREFIX, JobControl
from nemo_helix_plugin.refs import parse_entity_ref
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.common.auth import AuthContext
from nhx.common.entities import SYSTEM_WORKSPACE
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.jobs.schemas import (
    HelixJobStatus,
    HelixJobStatusResponse,
    HelixJobStepStatusResponse,
    HelixJobTaskStatusResponse,
    JobGroupStatus,
    aggregate_job_group_status,
    aggregate_job_group_status_from_counts,
)
from nhx.common.observability import create_counter
from nhx.core.jobs.api.v2.jobs.schemas import (
    CreateHelixJobRequest,
    CreateJobGroupRequest,
    HelixJobAttemptResponse,
    HelixJobAttemptSortField,
    HelixJobListSortField,
    HelixJobResponse,
    HelixJobSortField,
    HelixJobStepsListFilter,
    HelixJobStepWithContext,
    HelixJobTaskUpdate,
    JobGroupResponse,
    UpdateHelixJobRequest,
    UpdateJobGroupRequest,
    get_model_id,
    job_artifact_base_path,
)
from nhx.core.jobs.app.constants import (
    ERROR_CAUSE,
    IMAGE_DIGEST_AT_SAVE,
    IMAGE_DIGEST_STEP,
    JOB_TASK_CLEANUP_PENDING_KEY,
    STORAGE_RECLAIMED_AT,
)
from nhx.core.jobs.app.lifecycle import (
    attempt_can_transition,
    clear_image_comparison,
    rerun_status_details,
    saved_image_digest_for_step,
    storage_targets,
    with_lifecycle_timestamps,
)
from nhx.core.jobs.app.schemas import (
    HelixJobSpec,
)
from nhx.core.jobs.config import config as jobs_config
from nhx.core.jobs.entities import (
    STEP_SPEC_NAME_CONFIG_KEY,
    HelixJob,
    HelixJobAttempt,
    HelixJobResult,
    HelixJobStep,
    HelixJobStorageReclaim,
    HelixJobTask,
    JobGroup,
    get_step_spec_name,
)
from nhx.core.jobs.telemetry import build_job_run_telemetry, emit_job_run_event
from opentelemetry import metrics, trace
from pydantic import ValidationError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class StateTransitionConflictError(Exception):
    """Exception raised when a state transition is invalid."""


class JobOperationConflictError(Exception):
    """Exception raised when a job operation cannot proceed in the job's current state."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class PauseTTLLimitError(Exception):
    """Exception raised when a pause window is longer than the platform maximum."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class JobAlreadyExistsError(ValueError):
    """Exception raised when creating a job whose name is already in use."""


class JobStorageReclaimPendingError(ValueError):
    """Exception raised when a deleted job with the same name still has storage to reclaim."""


class JobDeletionConflictError(ValueError):
    """Exception raised when deleting a job that is not safe to hard-delete."""


class JobStatusUpdateSkippedError(Exception):
    """Raised when a status update loses a race with job deletion."""


class JobSecretValidationError(ValueError):
    """Exception raised when a job's secret references cannot be validated."""


class JobOutputLocationError(ValueError):
    """Exception raised when a job's output_location references a fileset that does not exist."""


_AUTO_NAME_SOURCE_MAX_LENGTH = 20
_AUTO_NAME_FALLBACK_PREFIX = "job"


def generate_job_name(source: str, short_id: str) -> str:
    """Return ``{source prefix}-{short_id}``, valid both as a job name and inside ``job-fileset-{name}``."""
    prefix = re.sub(r"[^a-z0-9@.+_]+", "-", source.lower())
    prefix = re.sub(r"^[^a-z]+", "", prefix)[:_AUTO_NAME_SOURCE_MAX_LENGTH].rstrip("-")
    return f"{prefix or _AUTO_NAME_FALLBACK_PREFIX}-{short_id}"


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into a copy of ``base``.

    Nested dicts merge key-by-key; every other value (including lists) is
    replaced wholesale by the patch. The inputs are not mutated.
    """
    merged = dict(base)
    for key, value in patch.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _effective_member_timeout(
    member_override: int | None,
    group_default: int | None,
) -> int | None:
    """Resolve the per-step staleness timeout for a member.

    The member's own override wins; otherwise the group default applies; when
    neither is set the template's own per-step value is left untouched.
    """
    if member_override is not None:
        return member_override
    return group_default


def _expand_member_spec(
    template: HelixJobSpec,
    config_patch: Mapping[str, Any],
    timeout_seconds: int | None,
) -> HelixJobSpec:
    """Build one member's full job spec from the shared template.

    Deep-merges ``config_patch`` into every step's ``config`` and, when
    ``timeout_seconds`` is set, applies it to every step's
    ``lifecycle.staleness_timeout_seconds``. The template is copied, never
    mutated.
    """
    member_spec = template.model_copy(deep=True)
    for step in member_spec.steps:
        if config_patch:
            step.config = _deep_merge(step.config, config_patch)
        if timeout_seconds is not None:
            step.lifecycle.staleness_timeout_seconds = timeout_seconds
    return member_spec


class JobHasNoAttemptError(ValueError):
    """Raised when the reserved current selector is used before a job has run."""


operations_counter = create_counter(
    meter=meter,
    subsystem="jobs",
    name="dispatch.operations.total",
    description="Total number of job dispatcher operations performed",
)

_DELETE_PAGE_SIZE = 1000
# IDs per store query when fetching the attempts and jobs of a page of steps. The IDs travel in the
# query string of a GET, so the batch stays small enough for proxies and ingress URL limits.
_ENRICHMENT_BATCH_SIZE = 100
_JOB_MUTATION_LOCKS: weakref.WeakValueDictionary[tuple[str, str], asyncio.Lock] = weakref.WeakValueDictionary()
_JOB_RUN_TELEMETRY_TERMINAL_STATUSES = frozenset(
    {
        HelixJobStatus.COMPLETED,
        HelixJobStatus.ERROR,
        HelixJobStatus.CANCELLED,
        HelixJobStatus.PAUSED,
    }
)
EntityT = TypeVar("EntityT")


def _should_emit_job_run_telemetry(
    previous_status: HelixJobStatus,
    new_status: HelixJobStatus,
) -> bool:
    return (
        previous_status not in _JOB_RUN_TELEMETRY_TERMINAL_STATUSES
        and new_status in _JOB_RUN_TELEMETRY_TERMINAL_STATUSES
    )


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


def _reject_pause_ttl_over_maximum(pause_ttl_seconds: int | None) -> None:
    """Reject a pause window longer than the platform maximum."""
    if pause_ttl_seconds is None:
        return
    maximum = jobs_config.storage.maximum_pause_ttl_seconds
    if pause_ttl_seconds > maximum:
        raise PauseTTLLimitError(f"pause_ttl_seconds cannot exceed the platform maximum of {maximum} seconds.")


def _keep_reported_cause(reported: dict[str, Any] | None, update: dict[str, Any]) -> dict[str, Any]:
    """Return the error details after ``update`` arrives for an error already ``reported``.

    A task that classified its own failure reports a ``cause`` key, None when no rule matched. A later error
    for the same failure, such as the controller's when the process exits, only sees the exit, so it adds
    keys without replacing what the task reported.
    """
    if reported and ERROR_CAUSE in reported:
        return {**update, **reported}
    return update


def create_platform_job_response(job: HelixJob, attempt: HelixJobAttempt) -> HelixJobResponse:
    """Helper to create HelixJobResponse from job and attempt entities."""
    ownership = job.ownership
    if hasattr(ownership, "model_dump"):
        ownership = ownership.model_dump()  # type: ignore
    return HelixJobResponse(
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
        control=job.control,
        fileset=job.fileset,
        output_location=job.output_location,
        status=attempt.status,
        status_details=attempt.status_details,
        error_details=attempt.error_details,
        ownership=ownership,
        custom_fields=job.custom_fields,
    )


def create_attempt_response(attempt: HelixJobAttempt, *, current_attempt_id: str | None) -> HelixJobAttemptResponse:
    """Convert an attempt entity into the public attempt response."""
    if attempt.created_at is None or attempt.updated_at is None:
        raise ValueError(f"Attempt '{attempt.id}' is missing persistence timestamps.")
    return HelixJobAttemptResponse(
        id=attempt.id,
        name=attempt.name,
        job=attempt.job,
        workspace=attempt.workspace,
        seq=attempt.seq,
        spec=attempt.spec,
        platform_spec=attempt.platform_spec,
        status=attempt.status,
        status_details=attempt.status_details,
        error_details=attempt.error_details,
        created_at=attempt.created_at,
        updated_at=attempt.updated_at,
        is_current=attempt.id == current_attempt_id,
    )


def _format_status_for_message(status: HelixJobStatus | str) -> str:
    if isinstance(status, HelixJobStatus):
        return status.value
    return str(status)


class JobDispatcher:
    """Job Dispatcher for managing job lifecycle and interactions with job related repositories."""

    def __init__(
        self,
        store: EntityClient,
        files: AsyncFilesClient,
        secrets: AsyncSecretsClient,
        system_store: EntityClient | None = None,
    ):
        self.store = store
        # Storage reclaim records live in the system workspace, which callers can only read.
        # Write them as the jobs service. Tests pass a single store for both.
        self.system_store = system_store if system_store is not None else store
        self.files = files
        self.secrets = secrets

    # =========================================================================
    # Job Operations
    # =========================================================================

    async def validate_job_secrets(self, job_spec: HelixJobSpec, job_workspace: str) -> None:
        # Ensure that any referenced secrets in steps exist and the user has access (user-scoped client).
        for step in job_spec.steps:
            if not step.environment:
                continue
            for env_var in step.environment:
                if env_var.from_secret:
                    secret_ref = parse_entity_ref(env_var.from_secret.name, default_workspace=job_workspace)
                    workspace, secret_name = secret_ref.workspace, secret_ref.name
                    try:
                        await self.secrets.get_secret(name=secret_name, workspace=workspace)
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
        job_req: CreateHelixJobRequest,
        workspace: str,
        auth_context: Optional[AuthContext] = None,
        parent: Optional[str] = None,
    ) -> HelixJobResponse:
        """Create a new job and its first step.

        ``parent`` parents the new job to another entity (e.g. a job group) via
        the Entity Store parent relationship, making it unique within
        ``(workspace, platform_job, parent, name)``.
        """
        if job_req.name is None:
            return await self._create_job(job_req, workspace, auth_context=auth_context, parent=parent)

        async with _get_job_mutation_lock(job_req.name, workspace):
            return await self._create_job(job_req, workspace, auth_context=auth_context, parent=parent)

    async def _create_job(
        self,
        job_req: CreateHelixJobRequest,
        workspace: str,
        auth_context: Optional[AuthContext] = None,
        parent: Optional[str] = None,
    ) -> HelixJobResponse:
        """Create a new job after the caller has acquired any needed name lock."""
        job_name = job_req.name
        if job_name is not None:
            # Check if a job with the same name already exists
            try:
                existing_job = await self.store.get(HelixJob, job_name, workspace=workspace, parent=parent)
                if existing_job:
                    raise JobAlreadyExistsError(
                        f"Job with name '{job_name}' already exists in workspace '{workspace}'."
                    )
            except EntityNotFoundError:
                pass  # Job does not exist, proceed to create

        try:
            platform_spec = job_req.platform_spec
            control = job_req.control or JobControl()
            _reject_pause_ttl_over_maximum(control.pause_ttl_seconds)

            await self.validate_job_secrets(platform_spec, workspace)

            if not job_name:
                job_name = generate_job_name(job_req.source, get_model_id("job")[-8:])
            await self._reject_pending_storage_reclaim(job_name, workspace)

            # Resolve the fileset for job artifacts (caller-supplied output_location or auto-created).
            if job_req.output_location is not None:
                try:
                    await self.files.get_fileset(name=job_req.output_location, workspace=workspace)
                except (ClientNotFoundError, ClientPermissionDeniedError) as exc:
                    raise JobOutputLocationError(
                        f"fileset '{job_req.output_location}' not found or not accessible in workspace '{workspace}'"
                    ) from exc
                fileset_name = job_req.output_location
            else:
                try:
                    fileset_request = CreateFilesetRequest(name=f"{JOB_FILESET_NAME_PREFIX}{job_name}")
                except ValidationError as exc:
                    raise RuntimeError(f"job name '{job_name}' cannot name its artifact fileset") from exc
                fileset_resp = await self.files.create_fileset(body=fileset_request, workspace=workspace)
                fileset_name = fileset_resp.data().name

            # Create job (ID is assigned by entity store)
            new_job = HelixJob(
                name=job_name,
                workspace=workspace,
                project=job_req.project,
                description=job_req.description,
                source=job_req.source,
                spec=job_req.spec,
                platform_spec=platform_spec,
                control=control,
                fileset=fileset_name,
                output_location=job_req.output_location,
                status=HelixJobStatus.CREATED,
                ownership=job_req.ownership,
                custom_fields=job_req.custom_fields,
            )
            # Parent the job to its group (if any) for parent-scoped membership.
            if parent is not None:
                new_job._parent = parent
            # Add auth_context separately, it's a PrivateAttr, so it's not set in the constructor
            new_job.with_auth_context(auth_context)
            job = await self.store.create(new_job)

            # Create first attempt
            # Use short ID prefix to stay within 32 char name limit
            attempt_ref_id = get_model_id("att")
            attempt = await self.store.create(
                HelixJobAttempt(
                    name=attempt_ref_id,
                    workspace=workspace,
                    job=job.id,
                    seq=0,
                    status=HelixJobStatus.CREATED,
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

    async def get_job(self, identifier: str, workspace: str) -> HelixJobResponse | None:
        """Get a platform job by name or ID, with its current attempt.

        ``nemo jobs list`` leads with the job ``id`` (the ``platform-job-...``
        value), while jobs are stored keyed by ``name``. Resolve the stable ID
        first, then fall back to a workspace-scoped name lookup. The ID lookup
        is workspace-agnostic, so its result is accepted only when it lives in
        the requested workspace.
        """
        job_entity = await self._resolve_job(identifier, workspace)
        if job_entity is None or job_entity.current_attempt_id is None:
            return None
        try:
            attempt = await self.store.get_by_id(HelixJobAttempt, job_entity.current_attempt_id)
        except EntityNotFoundError:
            return None
        return create_platform_job_response(job_entity, attempt)

    async def _resolve_job(self, identifier: str, workspace: str) -> Optional[HelixJob]:
        """Resolve a job by ID, falling back to its name within the workspace."""
        job_entity = await self._get_job_by_id_optional(identifier)
        if job_entity is not None and job_entity.workspace == workspace:
            return job_entity
        try:
            return await self.store.get(HelixJob, identifier, workspace=workspace)
        except EntityNotFoundError:
            return None

    async def create_job_group(
        self,
        group_req: CreateJobGroupRequest,
        workspace: str,
        auth_context: Optional[AuthContext] = None,
    ) -> JobGroupResponse:
        """Create a job group and expand its members into parented jobs.

        Each member is built by deep-merging its ``config_patch`` into every step
        of the shared ``template`` and applying the effective per-step staleness
        timeout (member override, else group default, else the template value).
        Every expanded job is parented to the group via the Entity Store parent
        relationship, so the group's members are exactly the jobs whose
        ``parent`` is the group ID.

        If any member fails to create, the group and the members already created
        are rolled back so a partial group is never left behind.
        """
        group_name = group_req.name or generate_job_name(group_req.source, get_model_id("jobgroup")[-8:])

        async with _get_job_mutation_lock(f"group:{group_name}", workspace):
            try:
                existing = await self.store.get(JobGroup, group_name, workspace=workspace)
                if existing:
                    raise JobAlreadyExistsError(
                        f"Job group with name '{group_name}' already exists in workspace '{workspace}'."
                    )
            except EntityNotFoundError:
                pass

            new_group = JobGroup(
                name=group_name,
                workspace=workspace,
                project=group_req.project,
                description=group_req.description,
                source=group_req.source,
                max_concurrent_jobs=group_req.max_concurrent_jobs,
                fail_fast=group_req.fail_fast,
                default_timeout_seconds=group_req.default_timeout_seconds,
                ownership=group_req.ownership,
                custom_fields=group_req.custom_fields,
            )
            group = await self.store.create(new_group)

            created_jobs: list[HelixJobResponse] = []
            try:
                for member in group_req.members:
                    member_spec = _expand_member_spec(
                        template=group_req.template,
                        config_patch=member.config_patch,
                        timeout_seconds=_effective_member_timeout(
                            member_override=member.timeout_seconds,
                            group_default=group_req.default_timeout_seconds,
                        ),
                    )
                    member_req = CreateHelixJobRequest(
                        name=member.name,
                        description=group_req.description,
                        project=group_req.project,
                        spec=dict(group_req.spec),
                        platform_spec=member_spec,
                        source=group_req.source,
                        ownership=group_req.ownership,
                        custom_fields=group_req.custom_fields,
                    )
                    created = await self.create_job(member_req, workspace, auth_context=auth_context, parent=group.id)
                    created_jobs.append(created)
            except Exception:
                # Roll back the partial group so a failed expansion leaves nothing behind.
                for created in created_jobs:
                    try:
                        await self.delete_job(created.id, workspace)
                    except Exception:
                        logger.warning(
                            "Failed to roll back member job during job group create failure",
                            extra={"job": created.id, "group": group.name, "workspace": workspace},
                        )
                try:
                    await self.store.delete_by_id(JobGroup, group.id)
                except Exception:
                    logger.warning(
                        "Failed to roll back job group after member expansion failure",
                        extra={"group": group.name, "workspace": workspace},
                    )
                raise

            operations_counter.add(1, attributes={"operation": "create_job_group"})
            logger.info(
                "Created job group",
                extra={"group": group.name, "workspace": workspace, "members": len(created_jobs)},
            )

            # Persist the group's initial status from the members just expanded. The status is a
            # denormalized field (not recomputed per read); the controller's group roll-up keeps
            # it current as members change thereafter.
            group.status = aggregate_job_group_status([self._member_status(created) for created in created_jobs])
            group = await self.store.update(group)

            return self._build_job_group_response(group, member_count=len(created_jobs))

    async def get_job_group(self, identifier: str, workspace: str) -> JobGroupResponse | None:
        """Get a job group by name or ID, reading its persisted aggregate status."""
        group = await self._resolve_job_group(identifier, workspace)
        if group is None:
            return None
        member_count = await self._count_group_members(group.id, workspace)
        return self._build_job_group_response(group, member_count=member_count)

    async def list_job_groups(
        self,
        workspace: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[HelixJobListSortField] = None,
    ) -> Tuple[List[JobGroupResponse], int]:
        """List job groups with their persisted aggregate status."""
        page = 1
        page_size = limit or 100
        if offset is not None and limit is not None:
            page = (offset // limit) + 1
        sort_str = sort.value if sort else "-created_at"

        response = await self.store.list(
            JobGroup,
            page=page,
            page_size=page_size,
            workspace=workspace,
            sort=sort_str,
        )
        groups = [
            self._build_job_group_response(group, member_count=await self._count_group_members(group.id, workspace))
            for group in response.data
        ]
        return groups, response.pagination.total_results

    async def update_job_group(
        self,
        identifier: str,
        workspace: str,
        update: UpdateJobGroupRequest,
    ) -> JobGroupResponse | None:
        """Apply a validated partial update to a job group's mutable metadata."""
        resolved = await self._resolve_job_group(identifier, workspace)
        if resolved is None:
            return None
        async with _get_job_mutation_lock(f"group:{resolved.name}", workspace):
            try:
                group = await self.store.get(JobGroup, resolved.name, workspace=workspace)
            except EntityNotFoundError:
                return None
            if group.id != resolved.id:
                return None

            fields = update.model_fields_set
            if "description" in fields:
                group.description = update.description
            if "max_concurrent_jobs" in fields:
                group.max_concurrent_jobs = update.max_concurrent_jobs
            if "fail_fast" in fields and update.fail_fast is not None:
                group.fail_fast = update.fail_fast
            if "default_timeout_seconds" in fields:
                group.default_timeout_seconds = update.default_timeout_seconds
            if "custom_fields" in fields:
                group.custom_fields = update.custom_fields

            group = await self.store.update(group)
            operations_counter.add(1, attributes={"operation": "update_job_group"})
            member_count = await self._count_group_members(group.id, workspace)
            return self._build_job_group_response(group, member_count=member_count)

    async def delete_job_group(self, identifier: str, workspace: str) -> bool:
        """Delete a job group and all of its member jobs.

        Each member is deleted through the normal job-deletion path, so a group
        with a non-terminal member raises the same ``JobDeletionConflictError``
        that deleting that job directly would.
        """
        group = await self._resolve_job_group(identifier, workspace)
        if group is None:
            return False
        async with _get_job_mutation_lock(f"group:{group.name}", workspace):
            members = await self._list_group_members(group.id, workspace)
            for member in members:
                await self.delete_job(member.id, workspace)
            try:
                await self.store.delete_by_id(JobGroup, group.id)
            except EntityNotFoundError:
                return False
            operations_counter.add(1, attributes={"operation": "delete_job_group"})
            return True

    async def _resolve_job_group(self, identifier: str, workspace: str) -> Optional[JobGroup]:
        """Resolve a job group by ID, falling back to its name within the workspace."""
        try:
            group = await self.store.get_by_id(JobGroup, identifier)
            if group.workspace == workspace:
                return group
        except EntityNotFoundError:
            pass
        try:
            return await self.store.get(JobGroup, identifier, workspace=workspace)
        except EntityNotFoundError:
            return None

    async def _list_group_members(self, group_id: str, workspace: str) -> list[HelixJob]:
        """Return every member job parented to the group.

        Membership is the Entity Store ``parent`` relationship, so members are
        the jobs whose top-level ``parent`` column equals the group ID. ``parent``
        is a base column (not a ``data.*`` field), so it is filtered directly
        rather than via ``filter_obj`` (which would namespace it under ``data.``).
        """
        members: list[HelixJob] = []
        page = 1
        parent_filter = ComparisonOperation(operator=FilterOperator.EQ, field="parent", value=group_id)
        while True:
            response = await self.store.list(
                HelixJob,
                filter_operation=parent_filter,
                page=page,
                page_size=_DELETE_PAGE_SIZE,
                workspace=workspace,
            )
            members.extend(response.data)
            if response.pagination.page >= response.pagination.total_pages:
                return members
            page = response.pagination.page + 1

    def _build_job_group_response(self, group: JobGroup, member_count: int) -> JobGroupResponse:
        """Assemble a group response from the group entity's own fields.

        The aggregate ``status`` is read straight off the persisted entity rather than
        recomputed from the members -- deriving it here would fan out a read over every
        member, which does not scale to groups with thousands of members. The controller's
        group roll-up keeps the stored value current.
        """
        return JobGroupResponse(
            id=group.id,
            name=group.name,
            workspace=group.workspace,
            project=group.project,
            description=group.description,
            source=group.source,
            max_concurrent_jobs=group.max_concurrent_jobs,
            fail_fast=group.fail_fast,
            default_timeout_seconds=group.default_timeout_seconds,
            status=group.status,
            member_count=member_count,
            created_at=group.created_at,  # type: ignore[arg-type]
            updated_at=group.updated_at,  # type: ignore[arg-type]
            ownership=group.ownership,
            custom_fields=group.custom_fields,
        )

    async def _count_group_members(self, group_id: str, workspace: str) -> int:
        """Return how many jobs are parented to the group without fetching them all.

        Asks for a single-item page and reads the total off the pagination envelope, so
        rendering a group response never fans a full read out over every member.
        """
        response = await self.store.list(
            HelixJob,
            filter_operation=ComparisonOperation(operator=FilterOperator.EQ, field="parent", value=group_id),
            page=1,
            page_size=1,
            workspace=workspace,
        )
        return response.pagination.total_results

    @staticmethod
    def _member_status(member: HelixJobResponse) -> HelixJobStatus:
        """The status a freshly-created member contributes to the group's initial rollup."""
        return member.status

    async def _count_group_members_by_status(
        self, group_id: str, workspace: str, statuses: list[HelixJobStatus]
    ) -> int:
        """Count group members whose job status is one of ``statuses`` -- without fetching them.

        ANDs the base-column ``parent`` filter with a ``data.status`` ``$in`` filter and reads the
        count off the pagination envelope (``page_size=1``), so a caller learns how many members
        are in a given status set with a single one-row query rather than a member scan. ``status``
        is copied onto each member job by :meth:`_save_job_status`, so it reflects the current attempt.
        """
        combined = LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="parent", value=group_id),
                ComparisonOperation(
                    operator=FilterOperator.IN,
                    field="data.status",
                    value=[s.value for s in statuses],
                ),
            ],
        )
        response = await self.store.list(HelixJob, filter_operation=combined, page=1, page_size=1, workspace=workspace)
        return response.pagination.total_results

    async def _rolled_up_group_status(self, group_id: str, workspace: str) -> JobGroupStatus:
        """Compute a group's aggregate status from cheap per-status COUNTS, not a member scan.

        O(1) in group size: a handful of count queries (each a ``page_size=1`` list) feed
        :func:`aggregate_job_group_status_from_counts`, which shares its precedence rules with
        the list-based aggregate. Probes are ordered so the common cases short-circuit early:

        - total==0 -> PENDING (no members).
        - all members terminal -> COMPLETED / FAILED / CANCELLED, disambiguated by the error and
          cancelled bucket counts.
        - otherwise non-terminal: a cancel in flight -> CANCELLING; nothing past CREATED -> PENDING;
          else RUNNING. Only the buckets needed to decide are probed.
        """
        total = await self._count_group_members(group_id, workspace)
        if total == 0:
            return JobGroupStatus.PENDING
        terminal_count = await self._count_group_members_by_status(group_id, workspace, HelixJobStatus.terminals())
        if terminal_count >= total:
            # Every member is terminal; the error/cancelled mix decides FAILED vs CANCELLED vs COMPLETED.
            error_count = await self._count_group_members_by_status(group_id, workspace, [HelixJobStatus.ERROR])
            cancelled_count = await self._count_group_members_by_status(group_id, workspace, [HelixJobStatus.CANCELLED])
            return aggregate_job_group_status_from_counts(
                total,
                {
                    HelixJobStatus.ERROR: error_count,
                    HelixJobStatus.CANCELLED: cancelled_count,
                    HelixJobStatus.COMPLETED: total - error_count - cancelled_count,
                },
            )
        cancel_in_flight = await self._count_group_members_by_status(
            group_id, workspace, [HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED]
        )
        if cancel_in_flight > 0:
            # A cancel is sweeping the group while members still run.
            return aggregate_job_group_status_from_counts(
                total, {HelixJobStatus.CANCELLING: cancel_in_flight, HelixJobStatus.ACTIVE: total - cancel_in_flight}
            )
        created_count = await self._count_group_members_by_status(group_id, workspace, [HelixJobStatus.CREATED])
        if created_count >= total:
            return JobGroupStatus.PENDING
        return JobGroupStatus.RUNNING

    async def _persist_group_status_for_member(self, member_job_id: str, workspace: str) -> None:
        """Recompute and persist a member's parent group's status after the member changed.

        The group ``status`` is a denormalized field (1431): rather than deriving it per read,
        each member transition rolls it up here and writes it through. Runs AFTER the member's
        own per-job lock is released, and takes only the short ``group:`` lock around the
        recompute-and-write so two members transitioning concurrently cannot lose-update the
        group row. No member cancel happens under this lock (fail_fast fans out separately),
        so it never nests a second per-job lock and cannot reintroduce the fan-out deadlock.
        """
        try:
            member = await self.store.get_by_id(HelixJob, member_job_id)
        except EntityNotFoundError:
            return
        group_id = member.parent
        if group_id is None:
            return
        try:
            group = await self.store.get_by_id(JobGroup, group_id)
        except EntityNotFoundError:
            return
        async with _get_job_mutation_lock(f"group:{group.name}", workspace):
            # Re-read under the lock: a concurrent roll-up may have advanced the stored status.
            try:
                group = await self.store.get_by_id(JobGroup, group_id)
            except EntityNotFoundError:
                return
            new_status = await self._rolled_up_group_status(group_id, workspace)
            if new_status in (JobGroupStatus.CANCELLING, JobGroupStatus.CANCELLED):
                # Cancel lifecycle is 1433's: this bottom-up member roll-up owns
                # pending/running/completed/failed only and must never write a cancel
                # state (a single-member cancel must not flip the whole group). 1433
                # writes the cancelling intent and the terminal cancelled top-down.
                return
            if group.status == new_status:
                return
            group.status = new_status
            try:
                await self.store.update(group)
            except EntityConflictError:
                logger.debug(
                    "Group status write lost a version race; a concurrent roll-up will re-persist",
                    extra={"group": group_id, "workspace": workspace},
                )
                return
            logger.info(
                "Rolled up JobGroup status from member change",
                extra={"group": group_id, "status": new_status.value, "workspace": workspace},
            )

    async def _member_current_status(self, member: HelixJob) -> HelixJobStatus:
        """Return a member job's effective status from its current attempt.

        A member with no attempt yet is treated as ``CREATED``, mirroring how
        :meth:`_build_job_group_response` derives the group's aggregate status.
        """
        if member.current_attempt_id is None:
            return HelixJobStatus.CREATED
        try:
            attempt = await self.store.get_by_id(HelixJobAttempt, member.current_attempt_id)
        except EntityNotFoundError:
            return HelixJobStatus.CREATED
        return attempt.status

    async def cancel_group_members(
        self, group_id: str, workspace: str, *, exclude_job_id: str | None = None
    ) -> list[str]:
        """Cancel every cancellable member of a group, returning the IDs cancelled.

        Shared fan-out used both by bottom-up fail-fast (a member reaching
        ``ERROR`` sweeps its siblings) and, later, by a top-down group-cancel
        request. Only members that are still non-terminal and not already
        cancelling are cancelled, so repeated calls while cancels are in flight
        are naturally idempotent. ``exclude_job_id`` skips the member that
        triggered the sweep (it is already terminal and must not cancel itself).

        Cancellation reuses :meth:`cancel_job`, which acquires each member's own
        mutation lock. Members are distinct jobs from any lock the caller holds,
        so the per-member fan-out does not deadlock against the caller.
        """
        cancelled: list[str] = []
        skip_statuses = (HelixJobStatus.CANCELLING, *HelixJobStatus.terminals())
        for member in await self._list_group_members(group_id, workspace):
            if member.id == exclude_job_id:
                continue
            if await self._member_current_status(member) in skip_statuses:
                continue
            try:
                await self.cancel_job(member.id, workspace)
                cancelled.append(member.id)
            except (JobOperationConflictError, StateTransitionConflictError):
                # The member advanced to a terminal/cancelling state between the
                # status read and the cancel; nothing left to do for it.
                logger.debug(
                    "Skipping group-member cancel; member already advanced",
                    extra={"job": member.id, "group": group_id, "workspace": workspace},
                )
            except Exception:
                # Best-effort sweep: an unexpected failure cancelling one member
                # must not abandon the remaining siblings (nor fail the caller's
                # own status write, when this runs off a member-ERROR roll-up).
                logger.warning(
                    "Failed to cancel group member during sweep; continuing",
                    exc_info=True,
                    extra={"job": member.id, "group": group_id, "workspace": workspace},
                )
        return cancelled

    async def _apply_fail_fast_on_member_error(self, member_job_id: str, workspace: str) -> None:
        """Fan a fail-fast cancel across a group when a member reaches ``ERROR``.

        Invoked AFTER the failed member's own mutation lock is released (so the
        sibling fan-out cannot deadlock against a concurrent same-group member
        failure). Because the lock is already gone, every precondition is
        re-asserted here against FRESH state -- the member, the group, and the
        group's aggregate status can all move in the post-lock gap:

        - the member must still exist, still be parented to a group, and still be
          in ``ERROR`` (a rerun could have moved it on);
        - the group must still exist and still have ``fail_fast`` set (an update
          could have cleared it);
        - the group must not already be terminal (another member's deferred sweep
          or a group-level cancel may have finished first).

        The trigger is ``ERROR`` only: a user cancelling a single member must not
        sweep the group (group-wide cancel is a separate path). The sweep itself
        (:meth:`cancel_group_members`) only touches non-terminal, not-already-
        cancelling siblings, so a redundant invocation is a harmless no-op.
        """
        try:
            member = await self.store.get_by_id(HelixJob, member_job_id)
        except EntityNotFoundError:
            return
        group_id = member.parent
        if group_id is None or await self._member_current_status(member) != HelixJobStatus.ERROR:
            return
        try:
            group = await self.store.get_by_id(JobGroup, group_id)
        except EntityNotFoundError:
            return
        if not isinstance(group, JobGroup) or not group.fail_fast:
            return

        members = await self._list_group_members(group_id, workspace)
        member_statuses = [await self._member_current_status(m) for m in members]
        if aggregate_job_group_status(member_statuses).is_terminal():
            # The group already settled (e.g. a concurrent sweep finished); nothing to do.
            return

        cancelled = await self.cancel_group_members(group_id, workspace, exclude_job_id=member.id)
        if cancelled:
            logger.info(
                "Applied JobGroup fail_fast: cancelled siblings of failed member",
                extra={
                    "group": group_id,
                    "failed_member": member.id,
                    "cancelled_count": len(cancelled),
                    "workspace": workspace,
                },
            )

    async def update_job(
        self,
        identifier: str,
        workspace: str,
        update: UpdateHelixJobRequest,
        auth_context: AuthContext | None = None,
    ) -> HelixJobResponse | None:
        """Apply a validated partial update to a job definition or metadata."""
        resolved = await self._resolve_job(identifier, workspace)
        if resolved is None:
            return None
        async with _get_job_mutation_lock(resolved.name, workspace):
            try:
                job = await self.store.get(HelixJob, resolved.name, workspace=workspace)
            except EntityNotFoundError:
                return None
            if job.id != resolved.id:
                return None

            fields = update.model_fields_set
            if "spec" in fields:
                assert update.platform_spec is not None
                await self.validate_job_secrets(update.platform_spec, workspace)

            if job.current_attempt_id is None:
                raise JobHasNoAttemptError(f"Job '{job.name}' has not run yet.")
            attempt = await self.get_attempt(job.current_attempt_id)
            if attempt is None or attempt.job != job.id or attempt.workspace != workspace:
                raise JobHasNoAttemptError(f"Current attempt for job '{job.name}' does not exist.")

            if "description" in fields:
                job.description = update.description
            if "spec" in fields:
                assert update.spec is not None
                assert update.platform_spec is not None
                job.spec = update.spec
                job.platform_spec = update.platform_spec

            if auth_context is not None:
                job.with_auth_context(auth_context)
            job = await self.store.update(job)
            if auth_context is not None:
                job.with_auth_context(auth_context)
            operations_counter.add(1, attributes={"operation": "update_job"})
            return create_platform_job_response(job, attempt)

    async def list_jobs(
        self,
        parsed: ParsedFilter,
        workspace: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[HelixJobListSortField] = None,
    ) -> Tuple[List[HelixJobResponse], int]:
        """List platform jobs with their current attempts."""
        # Calculate page from offset
        page = 1
        page_size = limit or 100
        if offset is not None and limit is not None:
            page = (offset // limit) + 1

        sort_str = sort.value if sort else "-created_at"

        response = await self.store.list(
            HelixJob,
            filter_operation=parsed.operation,
            page=page,
            page_size=page_size,
            workspace=workspace,
            sort=sort_str,
        )

        attempts_by_id = await self._list_current_attempts(response.data, workspace)
        job_outputs: list[HelixJobResponse] = []
        for job in response.data:
            if not job.current_attempt_id:
                continue

            attempt = attempts_by_id.get(job.current_attempt_id)
            if attempt is None:
                logger.warning(f"Attempt {job.current_attempt_id} not found for job {job.id}")
                continue

            platform_job = create_platform_job_response(job, attempt)
            job_outputs.append(platform_job)

        return job_outputs, response.pagination.total_results

    async def delete_job(self, identifier: str, workspace: str) -> bool:
        """Delete a job (by name or ID) and all of its associated data (steps, tasks, results, logs).

        Returns:
            True if job was deleted, False if job was not found.
        """
        job_entity = await self._resolve_job(identifier, workspace)
        if job_entity is None:
            return False
        async with _get_job_mutation_lock(job_entity.name, workspace):
            return await self._delete_job_locked(
                job_entity.name, workspace, expected_id=job_entity.id, parent=job_entity.parent
            )

    async def _delete_job_locked(
        self, job_name: str, workspace: str, expected_id: str | None = None, parent: str | None = None
    ) -> bool:
        """Delete a terminal job while holding the per-job mutation lock.

        ``expected_id`` guards against a same-name replacement: the identifier was
        resolved before the lock was taken, so re-verify the re-fetched job is
        still the one we resolved before deleting anything.

        ``parent`` scopes the name re-fetch to the job's parent (e.g. a job group).
        A member job is parent-scoped, so a name lookup without it would resolve
        ``parent IS NULL`` and miss the member entirely.
        """
        try:
            job_entity = await self.store.get(HelixJob, job_name, workspace=workspace, parent=parent)
        except EntityNotFoundError:
            return False
        if expected_id is not None and job_entity.id != expected_id:
            # A same-name job was recreated between resolve and lock acquisition;
            # the originally-resolved job is gone. Do not touch the replacement.
            return False

        extras = {"job": job_entity.name, "workspace": job_entity.workspace}

        try:
            # Get all attempts
            attempts = await self._list_all_entities(
                HelixJobAttempt,
                filter_obj={"job": job_entity.id},
                workspace=workspace,
            )
            steps_by_attempt: dict[str, list[HelixJobStep]] = {}
            for attempt in attempts:
                if not attempt.status.is_terminal():
                    raise JobDeletionConflictError(
                        f"Cannot delete job '{job_entity.name}' while it is "
                        f"'{_format_status_for_message(attempt.status)}'. Cancel the job and wait for it to reach a "
                        "terminal state before deleting."
                    )

                steps = await self._list_all_entities(
                    HelixJobStep,
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

            await self._queue_storage_reclaim(job_entity, attempts)

            # Get all results
            results = await self._list_all_entities(
                HelixJobResult,
                filter_obj={"job": job_entity.id},
                workspace=workspace,
            )

            for result in results:
                await self.store.delete_by_id(HelixJobResult, result.id)

            # Delete steps and tasks for each attempt
            for attempt in attempts:
                for step in steps_by_attempt[attempt.id]:
                    # Delete tasks
                    tasks = await self._list_all_entities(
                        HelixJobTask,
                        filter_obj={"step_id": step.id},
                        workspace=workspace,
                    )
                    for task in tasks:
                        await self.store.delete_by_id(HelixJobTask, task.id)

                    # Delete step
                    await self.store.delete_by_id(HelixJobStep, step.id)

                # Delete attempt
                await self.store.delete_by_id(HelixJobAttempt, attempt.id)

            # Delete job
            await self.store.delete_by_id(HelixJob, job_entity.id)

            # Delete only the fileset the job owns (auto-created); a caller-supplied output_location
            # fileset is left for the caller to manage. Ownership is recorded at create time, so a
            # caller cannot fool this by naming their fileset like an auto-created one. Tolerate the
            # owned fileset already being gone.
            if job_entity.output_location is None:
                try:
                    await self.files.delete_fileset(name=job_entity.fileset, workspace=workspace)
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

    async def _queue_storage_reclaim(self, job: HelixJob, attempts: list[HelixJobAttempt]) -> None:
        """Record the job's storage for the controller to reclaim once the job is gone.

        Skipped when the job never mounted storage. Written before any entity is deleted,
        and a failure to write it fails the delete: without the record, a new job with the
        same name would mount this job's storage. A job the storage sweep already reclaimed
        is recorded too, since the sweep may have stopped before queueing it.

        The current attempt is marked reclaimed first. A delete that fails after queueing
        leaves the job in place while the controller deletes its storage, and the mark makes
        resume, rerun and control updates refuse it.
        """
        targets = storage_targets(job.platform_spec)
        if not targets:
            return
        extras = {"job": job.name, "workspace": job.workspace}
        current = next((attempt for attempt in attempts if attempt.id == job.current_attempt_id), None)
        if current is not None and not (current.status_details or {}).get(STORAGE_RECLAIMED_AT):
            current.status_details = self._update_status_details_object(
                current.status_details, {STORAGE_RECLAIMED_AT: datetime.datetime.now(datetime.timezone.utc).isoformat()}
            )
            await self.store.update(current)
        try:
            await self.system_store.create(
                HelixJobStorageReclaim(
                    name=HelixJobStorageReclaim.name_for_job(job.id),
                    workspace=SYSTEM_WORKSPACE,
                    job_id=job.id,
                    job_workspace=job.workspace,
                    job_name=job.name,
                    targets=targets,
                )
            )
        except EntityConflictError:
            # A previous delete of this job failed after queueing the reclaim.
            pass
        except Exception:
            logger.exception("Could not queue storage reclaim for deleted job", extra=extras)
            raise
        logger.info("Queued storage reclaim for deleted job", extra=extras)

    async def _reject_pending_storage_reclaim(self, job_name: str, workspace: str) -> None:
        """Refuse a job name whose deleted predecessor's storage is not reclaimed yet.

        Job storage is keyed by name, so the pending reclaim would delete the new
        job's storage. The controller clears the record once the storage is gone. A
        failed check refuses the name too, since a record may still exist.
        """
        try:
            pending = await self.system_store.list(
                HelixJobStorageReclaim,
                filter_obj={"job_workspace": workspace, "job_name": job_name},
                workspace=SYSTEM_WORKSPACE,
            )
        except Exception as exc:
            logger.warning(
                "Could not check for a pending storage reclaim",
                extra={"job": job_name, "workspace": workspace},
                exc_info=True,
            )
            raise JobStorageReclaimPendingError(
                f"Could not confirm that storage of an earlier job named '{job_name}' in workspace '{workspace}' "
                "was reclaimed. Retry in a few seconds."
            ) from exc
        if pending.data:
            raise JobStorageReclaimPendingError(
                f"Storage of a deleted job named '{job_name}' in workspace '{workspace}' is still being reclaimed. "
                "Retry in a few seconds."
            )

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

    async def _start_attempt(self, job: HelixJob, attempt: HelixJobAttempt) -> None:
        """Create the first step of the new attempt."""
        first_step = attempt.get_first_step()
        # With parent-scoped uniqueness, step names are unique per attempt (parent)
        # Store original spec name in config for reference
        step_config = dict(first_step.config) if first_step.config else {}
        step_config[STEP_SPEC_NAME_CONFIG_KEY] = first_step.name
        await self.store.create(
            HelixJobStep(
                name=first_step.name,  # Simple name, unique per attempt via parent-scoped uniqueness
                workspace=job.workspace,
                attempt_id=attempt.id,
                config=step_config,
                status=HelixJobStatus.CREATED,
                status_details=saved_image_digest_for_step(attempt.status_details, first_step.name),
            )
        )

        operations_counter.add(1, attributes={"operation": "start_attempt"})
        logger.info(
            "Created first step for job", extra={"job": job.name, "workspace": job.workspace, "step": first_step.name}
        )

    async def get_attempt(self, attempt_id: str) -> Optional[HelixJobAttempt]:
        """Get an attempt by ID."""
        try:
            return await self.store.get_by_id(HelixJobAttempt, attempt_id)
        except EntityNotFoundError:
            return None

    async def resolve_attempt(
        self,
        job_identifier: str,
        attempt_identifier: str,
        workspace: str,
    ) -> tuple[HelixJob, HelixJobAttempt] | None:
        """Resolve an attempt by ID or the reserved current selector."""
        job = await self._resolve_job(job_identifier, workspace)
        if job is None:
            return None

        if attempt_identifier == "current":
            if job.current_attempt_id is None:
                raise JobHasNoAttemptError(f"Job '{job.name}' has not run yet.")
            attempt = await self.get_attempt(job.current_attempt_id)
        else:
            attempt = await self.get_attempt(attempt_identifier)

        if attempt is None or attempt.workspace != workspace or attempt.job != job.id:
            return None
        return job, attempt

    async def list_attempts(
        self,
        job_identifier: str,
        workspace: str,
        parsed: ParsedFilter,
        *,
        page: int,
        page_size: int,
        sort: HelixJobAttemptSortField,
    ) -> tuple[list[HelixJobAttemptResponse], int] | None:
        """List attempts belonging to a job with store-backed pagination."""
        job = await self._resolve_job(job_identifier, workspace)
        if job is None:
            return None
        parsed.and_with(ComparisonOperation(operator=FilterOperator.EQ, field="data.job", value=job.id))
        response = await self.store.list(
            HelixJobAttempt,
            workspace=workspace,
            filter_operation=parsed.operation,
            sort=sort.value,
            page=page,
            page_size=page_size,
        )
        attempts = [
            create_attempt_response(attempt, current_attempt_id=job.current_attempt_id) for attempt in response.data
        ]
        operations_counter.add(1, attributes={"operation": "list_attempts"})
        return attempts, response.pagination.total_results

    async def _get_job_by_id_optional(self, job_id: str) -> Optional[HelixJob]:
        """Get a job by ID, returning None if not found."""
        try:
            return await self.store.get_by_id(HelixJob, job_id)
        except EntityNotFoundError:
            return None

    async def _list_by_ids(self, entity_type: type[EntityT], ids: set[str], workspace: str) -> dict[str, EntityT]:
        """Fetch entities by ID, a page of IDs per store query, keyed by ID. IDs with no entity are absent.

        One query per page replaces a request per ID. Each of those requests is an authorized call back
        into the platform, so a page of N steps cost about 2N of them.
        """
        found: dict[str, EntityT] = {}
        id_list = sorted(ids)
        for start in range(0, len(id_list), _ENRICHMENT_BATCH_SIZE):
            batch = id_list[start : start + _ENRICHMENT_BATCH_SIZE]
            response = await self.store.list(
                entity_type,
                filter_operation=ComparisonOperation(field="id", operator=FilterOperator.IN, value=batch),
                page_size=len(batch),
                workspace=workspace,
            )
            found.update({entity.id: entity for entity in response.data})
        return found

    async def _list_current_attempts(self, jobs: list[HelixJob], workspace: str) -> dict[str, HelixJobAttempt]:
        """Fetch the current attempt of every job in one store query, keyed by attempt ID."""
        attempt_ids = [job.current_attempt_id for job in jobs if job.current_attempt_id]
        if not attempt_ids:
            return {}
        response = await self.store.list(
            HelixJobAttempt,
            filter_operation=ComparisonOperation(field="id", operator=FilterOperator.IN, value=attempt_ids),
            page_size=len(attempt_ids),
            workspace=workspace,
        )
        return {attempt.id: attempt for attempt in response.data}

    async def get_current_attempt(self, job_name: str, workspace: str) -> Optional[HelixJobAttempt]:
        """Get the current attempt for a job."""
        job_entity = await self._resolve_job(job_name, workspace)
        if job_entity is None or job_entity.current_attempt_id is None:
            return None
        attempt = await self.get_attempt(job_entity.current_attempt_id)
        if attempt is None or attempt.job != job_entity.id or attempt.workspace != workspace:
            return None
        return attempt

    # =========================================================================
    # Step Operations
    # =========================================================================

    async def get_current_job_step_by_name(
        self, job_name: str, step_name: str, workspace: str
    ) -> Optional[HelixJobStep]:
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
            HelixJobStep,
            filter_obj={"attempt_id": attempt.id},
            page_size=100,
            workspace=workspace,
        )
        # With parent-scoped uniqueness, step names are exact matches
        for step in response.data:
            if step.name == step_name:
                return step
        return None

    async def get_attempt_step_by_name(
        self,
        job_identifier: str,
        attempt_identifier: str,
        step_name: str,
        workspace: str,
    ) -> Optional[HelixJobStep]:
        """Get a step after validating its complete job/attempt parent chain."""
        resolved = await self.resolve_attempt(job_identifier, attempt_identifier, workspace)
        if resolved is None:
            return None
        _, attempt = resolved
        try:
            step = await self.store.get(HelixJobStep, step_name, workspace=workspace, parent=attempt.id)
        except EntityNotFoundError:
            return None
        return step if step.attempt_id == attempt.id else None

    async def list_attempt_steps(
        self,
        job_identifier: str,
        attempt_identifier: str,
        workspace: str,
        parsed: ParsedFilter,
        *,
        page: int,
        page_size: int,
        sort: HelixJobSortField,
    ) -> tuple[list[HelixJobStepWithContext], int] | None:
        """List steps for one resolved attempt with full parent validation."""
        resolved = await self.resolve_attempt(job_identifier, attempt_identifier, workspace)
        if resolved is None:
            return None
        job, attempt = resolved
        parsed.and_with(ComparisonOperation(operator=FilterOperator.EQ, field="data.attempt_id", value=attempt.id))
        response = await self.store.list(
            HelixJobStep,
            workspace=workspace,
            filter_operation=parsed.operation,
            sort=sort.value,
            page=page,
            page_size=page_size,
        )
        steps: list[HelixJobStepWithContext] = []
        for step in response.data:
            if step.created_at is None or step.updated_at is None:
                raise ValueError(f"Job step '{step.id}' is missing persistence timestamps.")
            steps.append(
                HelixJobStepWithContext(
                    id=step.id,
                    job=job.name,
                    attempt_id=attempt.id,
                    workspace=step.workspace,
                    fileset=job.fileset,
                    artifact_base_path=job_artifact_base_path(job.name, job.output_location),
                    name=step.name,
                    step_spec=attempt.get_step_spec(get_step_spec_name(step.config, step.name) or step.name),
                    status=step.status,
                    status_details=step.status_details,
                    error_details=step.error_details,
                    auth_context=job.auth_context,
                    created_at=step.created_at,
                    updated_at=step.updated_at,
                )
            )
        operations_counter.add(1, attributes={"operation": "list_attempt_steps"})
        return steps, response.pagination.total_results

    async def _get_step_by_status(
        self, job: HelixJob, workspace: str, possible_statuses: list[HelixJobStatus]
    ) -> Optional[HelixJobStep]:
        """Get the step for a job with a given possible status."""

        if not job.current_attempt_id:
            return None

        response = await self.store.list(
            HelixJobStep,
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
        filter: HelixJobStepsListFilter,
        sort: HelixJobSortField,
        limit: int,
        offset: int,
        workspace: str,
    ) -> tuple[list[HelixJobStepWithContext], int]:
        """Poll steps based on the provided filter, sort, limit, and offset."""
        filter_str: Optional[str] = None
        if filter.status:
            filter_str = json.dumps({"data.status": {"$in": filter.status}})
        if filter.task_cleanup_pending is not None:
            cleanup_filter = {"data.status_details." + JOB_TASK_CLEANUP_PENDING_KEY: filter.task_cleanup_pending}
            filter_str = json.dumps(
                {"$and": [json.loads(filter_str), cleanup_filter]} if filter_str else cleanup_filter
            )
        sort_str = sort.value

        use_store_pagination = (not filter.job or filter.job == "-") and not filter.source

        if use_store_pagination:
            page = (offset // limit) + 1 if limit > 0 else 1
            response = await self.store.list(
                HelixJobStep,
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
                HelixJobStep,
                workspace=workspace,
                filter_str=filter_str,
                sort=sort_str,
                page_size=1000,
            )
            all_steps = list(response.data)
            while response.pagination.page < response.pagination.total_pages:
                response = await self.store.list(
                    HelixJobStep,
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

        attempt_by_id = await self._list_by_ids(HelixJobAttempt, {s.attempt_id for s in all_steps}, workspace)
        job_by_id = await self._list_by_ids(HelixJob, {a.job for a in attempt_by_id.values()}, workspace)

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
            enriched_step = HelixJobStepWithContext(
                id=step.id,
                job=job.name,
                job_id=job.id,
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

    async def get_job_status(self, job_name: str, workspace: str) -> Optional[HelixJobStatusResponse]:
        """Get the overall status for a job, indexed by step and task."""
        job = await self.get_job(job_name, workspace)
        if job is None:
            return None

        # Get steps for current attempt, sorted by creation order to match platform_spec
        steps_response = await self.store.list(
            HelixJobStep,
            workspace=workspace,
            filter_obj={"attempt_id": job.attempt_id},
            sort="created_at",
            page_size=1000,
        )

        step_statuses = []
        for step in steps_response.data:
            tasks = await self.list_tasks(step.id, workspace)

            task_statuses = [
                HelixJobTaskStatusResponse(
                    id=task.id,
                    name=task.name,
                    status=task.status,
                    status_details=task.status_details,
                    error_details=task.error_details or {},
                    error_stack=task.error_stack,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
                for task in tasks
            ]

            step_statuses.append(
                HelixJobStepStatusResponse(
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
        return HelixJobStatusResponse(
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
        step: HelixJobStep,
        status: HelixJobStatus,
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
        step: HelixJobStep,
        status: HelixJobStatus,
        status_details: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
        expected_step_id: str | None = None,
    ) -> tuple[HelixJobStep, HelixJobAttempt]:
        """Update a job's status based on the step's status.

        On EntityConflictError (e.g. reconciler updated the step concurrently), refetches
        the step and retries once if the requested transition is still valid.
        """
        if expected_step_id is not None and expected_step_id != step.id:
            raise StateTransitionConflictError("Job step was replaced by a new attempt")
        attempt = await self.get_attempt(step.attempt_id)
        if attempt is None:
            raise JobStatusUpdateSkippedError(f"Attempt does not exist: {step.attempt_id}")
        try:
            job = await self.store.get_by_id(HelixJob, attempt.job)
        except EntityNotFoundError as exc:
            raise JobStatusUpdateSkippedError(f"Job does not exist: {attempt.job}") from exc

        # Captured before the lock so the post-lock sweep can tell a fresh
        # transition INTO error apart from a replayed write that was already error.
        attempt_status_before = attempt.status

        async with _get_job_mutation_lock(job.name, job.workspace):
            if status == HelixJobStatus.RESUMING:
                # Checked again under the lock the storage sweep holds while it marks storage reclaimed.
                current = await self.get_attempt(step.attempt_id)
                if current is not None and (current.status_details or {}).get(STORAGE_RECLAIMED_AT):
                    raise JobOperationConflictError(
                        f"Cannot resume job '{job.name}': storage has been reclaimed. Create a new job."
                    )
            saved_step, attempt = await self._update_job_status_from_step_locked(
                step, status, status_details, error_details
            )

        # JobGroup roll-ups ride the member status write but run AFTER the member's own lock is
        # released. Two things happen here, both only when the member actually changed status:
        #   1. fail_fast fan-out (fresh ERROR): cancelling a sibling takes that sibling's per-job
        #      lock; holding the member's lock while doing so would let two same-group members
        #      failing concurrently deadlock (A holds a, waits b; B holds b, waits a). Releasing
        #      first makes that ordering impossible. "Entered ERROR" is computed from the pre-lock
        #      status; the sweep re-asserts fresh state since the group/member can move in the gap.
        #   2. group status roll-up (any transition): the group's denormalized status is re-rolled
        #      from the members and persisted, so created/pending/running/completed/error stay
        #      current without deriving on read. Cancel is 1433's; this never writes cancelling.
        member_status_changed = attempt.status != attempt_status_before
        if member_status_changed:
            if attempt.status == HelixJobStatus.ERROR and attempt_status_before != HelixJobStatus.ERROR:
                await self._apply_fail_fast_on_member_error(attempt.job, job.workspace)
            await self._persist_group_status_for_member(attempt.job, job.workspace)
        return saved_step, attempt

    async def _update_job_status_from_step_locked(
        self,
        step: HelixJobStep,
        status: HelixJobStatus,
        status_details: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> tuple[HelixJobStep, HelixJobAttempt]:
        """Update a job from a step while holding the per-job mutation lock."""
        step_to_save = await self.store.get_by_id(HelixJobStep, step.id)
        if status == HelixJobStatus.ERROR:
            error_details = await self._with_task_cause(step_to_save, error_details)
        saved_step: HelixJobStep | None = None
        requested_details = status_details
        for attempt in range(2):
            previous_step_status = step_to_save.status
            if not step_to_save.status.can_transition_to(status):
                raise StateTransitionConflictError(
                    f"Invalid status transition from {step_to_save.status} to {status} for step {step_to_save.id}"
                )

            # Stamp against the step as it is now, so a retry after a refetch does not carry
            # a clock stamped for a transition that another writer already made.
            status_details = with_lifecycle_timestamps(step_to_save, status, requested_details)
            if status in (HelixJobStatus.CANCELLED, HelixJobStatus.ERROR):
                await self._finalize_step_tasks(
                    step_to_save.model_copy(
                        update={"status": status, "error_details": error_details or step_to_save.error_details}
                    )
                )
                if status != step_to_save.status or step_to_save.status_details.get(JOB_TASK_CLEANUP_PENDING_KEY):
                    status_details = {**(status_details or {}), JOB_TASK_CLEANUP_PENDING_KEY: True}

            if self._step_update_would_be_noop(step_to_save, status, status_details, error_details):
                saved_step = step_to_save
                break

            # Apply updates to a copy of fields we persist
            if status != step_to_save.status:
                step_to_save.status = status
            if error_details:
                step_to_save.error_details = error_details
            if status == HelixJobStatus.RESUMING:
                clear_image_comparison(step_to_save.status_details)
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
                        refetched = await self.store.get_by_id(HelixJobStep, step_to_save.id)
                    except EntityNotFoundError:
                        raise e from e
                    if refetched.status.can_transition_to(status):
                        step_to_save = refetched
                        continue
                raise e from e

        if saved_step is None:
            raise RuntimeError("update_job_status_from_step did not produce a saved step")

        if saved_step.status in (HelixJobStatus.CANCELLED, HelixJobStatus.ERROR):
            await self._finalize_step_tasks(saved_step)

        # Update job / attempt status from step
        attempt = await self.get_attempt(saved_step.attempt_id)
        if attempt is None:
            raise JobStatusUpdateSkippedError(f"Attempt does not exist: {saved_step.attempt_id}")

        # Determine new attempt status
        previous_attempt_status = attempt.status
        new_attempt_status = attempt.status

        # Get the original step spec name from config (step entity names have suffixes for uniqueness)
        step_spec_name = get_step_spec_name(saved_step.config, saved_step.name) or saved_step.name

        if (
            saved_step.status == HelixJobStatus.PENDING
            and attempt.status != HelixJobStatus.PENDING
            and step_spec_name == attempt.platform_spec.steps[0].name
        ):
            new_attempt_status = HelixJobStatus.PENDING
            logger.info(
                "Job is pending",
                extra={
                    "job": attempt.job,
                    "attempt": attempt.id,
                    "step": saved_step.id,
                    "workspace": attempt.workspace,
                },
            )
        elif saved_step.status == HelixJobStatus.ACTIVE and attempt.status != HelixJobStatus.ACTIVE:
            new_attempt_status = HelixJobStatus.ACTIVE
            logger.info(
                "Job is active",
                extra={
                    "job": attempt.job,
                    "attempt": attempt.id,
                    "step": saved_step.id,
                    "workspace": attempt.workspace,
                },
            )
        elif saved_step.status == HelixJobStatus.COMPLETED:
            # Use the step spec name from config to find the next step
            next_step = attempt.get_next_step_spec(step_spec_name)
            if next_step:
                # With parent-scoped uniqueness, use simple step name (unique per attempt)
                next_step_config = dict(next_step.config) if next_step.config else {}
                next_step_config[STEP_SPEC_NAME_CONFIG_KEY] = next_step.name
                try:
                    await self.store.create(
                        HelixJobStep(
                            name=next_step.name,  # Simple name, unique per attempt via parent-scoped uniqueness
                            workspace=attempt.workspace,
                            attempt_id=attempt.id,
                            config=next_step_config,
                            status_details=saved_image_digest_for_step(attempt.status_details, next_step.name),
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
                # The step being paused finished before it saw the pause. The pause no
                # longer applies and the pipeline moves on, so the job is running again.
                # Only that step's own pausing -> completed move counts: a replayed update
                # for a step that had already completed must not undo a later step's pause.
                if attempt.status == HelixJobStatus.PAUSING and previous_step_status == HelixJobStatus.PAUSING:
                    new_attempt_status = HelixJobStatus.ACTIVE
            else:
                new_attempt_status = HelixJobStatus.COMPLETED
                logger.info(
                    "Job completed all steps",
                    extra={"job": attempt.job, "attempt": attempt.id, "workspace": attempt.workspace},
                )
        elif saved_step.status in (
            HelixJobStatus.CANCELLING,
            HelixJobStatus.CANCELLED,
            HelixJobStatus.ERROR,
            HelixJobStatus.PAUSING,
            HelixJobStatus.PAUSED,
            HelixJobStatus.RESUMING,
        ):
            new_attempt_status = saved_step.status

        # Update attempt if status changed, respecting state machine transitions to prevent
        # backward transitions from concurrent reconciler/scheduler updates
        if new_attempt_status != attempt.status:
            attempt, changed = await self._save_attempt_status(attempt, new_attempt_status, saved_step, step_spec_name)
            if changed:
                if _should_emit_job_run_telemetry(previous_attempt_status, attempt.status):
                    try:
                        job = await self.store.get_by_id(HelixJob, attempt.job)
                        status_details = dict(attempt.status_details or {})
                        status_details.update(saved_step.status_details or {})
                        telemetry_event = build_job_run_telemetry(
                            source=job.source,
                            status=attempt.status.value,
                            status_details=status_details,
                            custom_fields=job.custom_fields,
                            created_at=attempt.created_at,
                            updated_at=attempt.updated_at,
                        )
                        if telemetry_event is not None:
                            emit_job_run_event(telemetry_event)
                    except Exception:
                        logger.debug("Failed to queue job_run telemetry", exc_info=True)
                logger.info(
                    "Updated job attempt status",
                    extra={
                        "job": attempt.job,
                        "attempt": attempt.id,
                        "status": attempt.status,
                        "workspace": attempt.workspace,
                    },
                )

        if saved_step.status_details.get(JOB_TASK_CLEANUP_PENDING_KEY):
            for retry in range(2):
                try:
                    saved_step = await self.store.update(
                        saved_step.model_copy(
                            update={
                                "status_details": {**saved_step.status_details, JOB_TASK_CLEANUP_PENDING_KEY: False}
                            }
                        )
                    )
                    break
                except EntityConflictError:
                    if retry:
                        raise
                    saved_step = await self.store.get_by_id(HelixJobStep, saved_step.id)
                    if not saved_step.status_details.get(JOB_TASK_CLEANUP_PENDING_KEY):
                        break

        operations_counter.add(1, attributes={"operation": "update_job_status_from_step"})
        return saved_step, attempt

    async def _with_task_cause(self, step: HelixJobStep, error_details: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return the step's error details, led by the error a failed task classified for itself, if any."""
        for task in await self.list_tasks(step.id, step.workspace):
            if task.status == HelixJobStatus.ERROR and ERROR_CAUSE in (task.error_details or {}):
                return _keep_reported_cause(task.error_details, error_details or step.error_details or {})
        return error_details

    async def _save_attempt_status(
        self,
        attempt: HelixJobAttempt,
        status: HelixJobStatus,
        step: HelixJobStep,
        step_spec_name: str,
    ) -> tuple[HelixJobAttempt, bool]:
        """Move the attempt to ``status`` after ``step`` changed. Return the attempt and whether it moved.

        Tasks write progress into the attempt without the job lock, so the save can
        lose a version race. Like the step update, refetch and retry once while the
        transition is still valid, so the attempt does not fall behind its step.
        """
        for try_number in range(2):
            if attempt.status == status:
                return attempt, False
            if not attempt_can_transition(attempt.status, status):
                logger.debug(
                    f"Skipping attempt {attempt.id} status update from '{attempt.status}' "
                    f"to '{status}': transition not valid (concurrent update "
                    f"advanced the attempt past the desired state)"
                )
                return attempt, False
            attempt.status = status
            if status == HelixJobStatus.ERROR:
                attempt.error_details = step.error_details
            if status in (HelixJobStatus.PAUSED, HelixJobStatus.ERROR):
                # A rerun compares images against this, on the step that saved it.
                saved_digest = (step.status_details or {}).get(IMAGE_DIGEST_AT_SAVE)
                if isinstance(saved_digest, str) and saved_digest:
                    attempt.status_details = self._update_status_details_object(
                        attempt.status_details,
                        {IMAGE_DIGEST_AT_SAVE: saved_digest, IMAGE_DIGEST_STEP: step_spec_name},
                    )
            # Job first: if either save fails, the retried step update still finds the attempt behind and redoes both.
            await self._save_job_status(attempt)
            try:
                return await self.store.update(attempt), True
            except EntityConflictError as e:
                if try_number > 0:
                    raise
                try:
                    attempt = await self.store.get_by_id(HelixJobAttempt, attempt.id)
                except EntityNotFoundError:
                    raise e from e
        raise RuntimeError("_save_attempt_status did not save or skip the attempt")

    async def _save_job_status(self, attempt: HelixJobAttempt) -> None:
        """Copy the attempt's status onto its job so list filters, sorts and paging run in the store."""
        # A caller-scoped read strips the stored auth context, and an update would write that back.
        store = self.store.as_service("jobs", internal=True)
        for try_number in range(2):
            job = await store.get_by_id(HelixJob, attempt.job)
            if job.current_attempt_id != attempt.id or job.status == attempt.status:
                return
            job.status = attempt.status
            try:
                await store.update(job)
                return
            except EntityConflictError:
                if try_number > 0:
                    raise

    async def update_attempt_status_details(
        self,
        job_identifier: str,
        attempt_identifier: str,
        workspace: str,
        status_details: Dict[str, Any],
        expected_step_id: str | None = None,
        expected_step_status: HelixJobStatus | None = None,
    ) -> HelixJobResponse | None:
        """Update status details for one concrete, job-owned attempt.

        With ``expected_step_id``, the update applies only while that step is in this attempt, the
        attempt is the job's current one, and the step has ``expected_step_status``. Otherwise this
        raises ``JobOperationConflictError``. The check and the write hold the job's lock, as resume
        and rerun do, so the storage sweep cannot mark a job that one of them took over.
        """
        resolved = await self.resolve_attempt(job_identifier, attempt_identifier, workspace)
        if resolved is None:
            return None
        job, attempt = resolved
        async with _get_job_mutation_lock(job.name, workspace):
            if expected_step_id is not None:
                await self._require_current_step(job, attempt, expected_step_id, expected_step_status)
            attempt = await self._update_attempt_status_details_locked(job, attempt, workspace, status_details)
            if attempt is None:
                return None

        operations_counter.add(1, attributes={"operation": "update_attempt_status_details"})
        return create_platform_job_response(job, attempt)

    async def _require_current_step(
        self,
        job: HelixJob,
        attempt: HelixJobAttempt,
        step_id: str,
        step_status: HelixJobStatus | None,
    ) -> None:
        """Raise ``JobOperationConflictError`` unless the step is still in the job's current attempt, with ``step_status``.

        The caller holds the job's lock. The job is read again under it, because a rerun changes its
        current attempt.
        """
        try:
            current_attempt_id = (await self.store.get_by_id(HelixJob, job.id)).current_attempt_id
            step = await self.store.get_by_id(HelixJobStep, step_id)
        except EntityNotFoundError:
            current_attempt_id, step = None, None
        if (
            step is None
            or step.attempt_id != attempt.id
            or current_attempt_id != attempt.id
            or (step_status is not None and step.status != step_status)
        ):
            expected = f" and {step_status.value}" if step_status is not None else ""
            raise JobOperationConflictError(
                f"Job '{job.name}' changed: step '{step_id}' is no longer in its current attempt{expected}."
            )

    async def _update_attempt_status_details_locked(
        self,
        job: HelixJob,
        attempt: HelixJobAttempt,
        workspace: str,
        status_details: Dict[str, Any],
    ) -> HelixJobAttempt | None:
        """Update one attempt while the caller holds the job mutation lock."""
        attempt = await self.store.get_by_id(HelixJobAttempt, attempt.id)
        if attempt.job != job.id or attempt.workspace != workspace:
            return None
        attempt.status_details = self._update_status_details_object(dict(attempt.status_details), status_details)
        return await self.store.update(attempt)

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

    async def cancel_job(self, identifier: str, workspace: str) -> HelixJobResponse | None:
        """Cancel a job (by name or ID)."""
        job_entity = await self._resolve_job(identifier, workspace)
        if job_entity is None:
            return None

        # Check if the job has any created steps, and if so, check if all of them are in created state.
        # If so, cancel the job immediately without going through the normal cancellation process.
        if job_entity.current_attempt_id:
            steps = await self._list_all_entities(
                HelixJobStep,
                filter_obj={"attempt_id": job_entity.current_attempt_id},
                workspace=workspace,
            )
            if steps and all(s.status == HelixJobStatus.CREATED for s in steps):
                for step in steps:
                    _, attempt = await self.update_job_status_from_step(step, HelixJobStatus.CANCELLED)
                operations_counter.add(1, attributes={"operation": "cancel_job"})
                return create_platform_job_response(job_entity, attempt)

        non_terminal_step = await self._get_step_by_status(job_entity, workspace, HelixJobStatus.non_terminals())
        if non_terminal_step:
            _, attempt = await self.update_job_status_from_step(
                non_terminal_step,
                HelixJobStatus.CANCELLING,
            )
        else:
            attempt = await self.get_current_attempt(job_entity.name, workspace=job_entity.workspace)
            if attempt is None:
                raise JobOperationConflictError(f"Cannot cancel job '{job_entity.name}': it has no current attempt.")
            steps = await self._list_all_entities(
                HelixJobStep,
                filter_obj={"attempt_id": attempt.id},
                workspace=workspace,
            )
            for step in steps:
                if step.status in (HelixJobStatus.CANCELLED, HelixJobStatus.ERROR):
                    _, attempt = await self.update_job_status_from_step(step, step.status)

        operations_counter.add(1, attributes={"operation": "cancel_job"})
        return create_platform_job_response(job_entity, attempt)

    async def rerun_job(
        self, identifier: str, workspace: str, auth_context: AuthContext | None = None
    ) -> HelixJobResponse | None:
        """Re-run a job (by name or ID)."""
        job_entity = await self._resolve_job(identifier, workspace)
        if job_entity is None:
            return None
        async with _get_job_mutation_lock(job_entity.name, workspace):
            return await self._rerun_job_locked(
                job_entity.name, workspace, expected_id=job_entity.id, auth_context=auth_context
            )

    async def _rerun_job_locked(
        self,
        job_name: str,
        workspace: str,
        expected_id: str | None = None,
        auth_context: AuthContext | None = None,
    ) -> HelixJobResponse | None:
        """Re-run a terminal job while holding the per-job mutation lock.

        ``expected_id`` guards against a same-name replacement created between the
        pre-lock resolve and lock acquisition (see ``_delete_job_locked``).
        """
        try:
            job_entity = await self.store.get(HelixJob, job_name, workspace=workspace)
        except EntityNotFoundError:
            return None
        if expected_id is not None and job_entity.id != expected_id:
            # The originally-resolved job was replaced by a same-name job; don't rerun it.
            return None

        attempt = await self.get_current_attempt(job_entity.name, workspace=job_entity.workspace)
        if attempt is None:
            return None

        # Rerun continues a failed attempt from surviving storage. Other statuses
        # are a different operation: paused jobs resume, and successful or cancelled
        # jobs are finished.
        if attempt.status != HelixJobStatus.ERROR:
            if attempt.status == HelixJobStatus.PAUSED:
                detail = f"Cannot rerun job '{job_entity.name}': this job is paused, use resume."
            else:
                detail = (
                    f"Cannot rerun job '{job_entity.name}': rerun applies to a job that ended in error "
                    f"(current status is '{attempt.status.value}')."
                )
            raise JobOperationConflictError(detail)

        previous_details = dict(attempt.status_details or {})
        if previous_details.get(STORAGE_RECLAIMED_AT):
            raise JobOperationConflictError(
                f"Cannot rerun job '{job_entity.name}': storage has been reclaimed. Create a new job."
            )

        for step in await self._list_all_entities(
            HelixJobStep, filter_obj={"attempt_id": attempt.id}, workspace=workspace
        ):
            if step.status_details.get(JOB_TASK_CLEANUP_PENDING_KEY):
                await self._update_job_status_from_step_locked(step, step.status)
        seq = attempt.seq + 1
        # Use short ID prefix to stay within 32 char name limit
        attempt_ref_id = get_model_id("att")
        next_details = rerun_status_details(previous_details)
        next_attempt = await self.store.create(
            HelixJobAttempt(
                name=attempt_ref_id,
                workspace=job_entity.workspace,
                job=job_entity.id,
                seq=seq,
                status=HelixJobStatus.CREATED,
                spec=job_entity.spec,
                platform_spec=job_entity.platform_spec,
                status_details=next_details,
            )
        )
        # Update job with new attempt ID. Re-apply the caller's auth context the
        # same way create does, so the rerun runs as the person who requested it.
        job_entity.current_attempt_id = next_attempt.id
        job_entity.status = next_attempt.status
        if auth_context is None:
            # The caller's reads strip the stored auth context, and an update replaces the
            # job's data, so without a caller's context, write back the job's own.
            stored = await self.store.as_service("jobs", internal=True).get_by_id(HelixJob, job_entity.id)
            auth_context = stored.auth_context
        job_entity.with_auth_context(auth_context)
        job_entity = await self.store.update(job_entity)
        job_entity.with_auth_context(auth_context)
        await self._start_attempt(job_entity, next_attempt)
        operations_counter.add(1, attributes={"operation": "rerun_job"})
        return create_platform_job_response(job_entity, next_attempt)

    async def pause_job(self, identifier: str, workspace: str) -> HelixJobResponse | None:
        """Pause a job (by name or ID)."""
        job_entity = await self._resolve_job(identifier, workspace)
        if job_entity is None:
            return None

        attempt = await self.get_current_attempt(job_entity.name, workspace=job_entity.workspace)
        if attempt is None:
            raise JobOperationConflictError(f"Cannot pause job '{job_entity.name}': it has no current attempt.")

        # A live step is paused even when another step is already pausing or paused.
        # The idempotent return applies only when nothing is still created, pending, or active.
        active_or_pending_step = await self._get_step_by_status(
            job_entity, workspace, [HelixJobStatus.CREATED, HelixJobStatus.PENDING, HelixJobStatus.ACTIVE]
        )
        if active_or_pending_step is None:
            already_pausing = await self._get_step_by_status(
                job_entity, workspace, [HelixJobStatus.PAUSING, HelixJobStatus.PAUSED]
            )
            if already_pausing is not None:
                operations_counter.add(1, attributes={"operation": "pause_job"})
                return create_platform_job_response(job_entity, attempt)
            raise JobOperationConflictError(f"Cannot pause job '{job_entity.name}': the job is {attempt.status.value}.")

        step_spec_name = get_step_spec_name(active_or_pending_step.config, active_or_pending_step.name)
        step_spec = next((spec for spec in attempt.platform_spec.steps if spec.name == step_spec_name), None)
        deadline = step_spec.lifecycle.pause_deadline_seconds if step_spec is not None else 0
        if deadline <= 0:
            raise JobOperationConflictError(
                f"Cannot pause job '{job_entity.name}': the running step '{step_spec_name}' does not support pause."
            )

        # The lifecycle stamp writes pause_requested_at, which starts the pause deadline.
        _, updated_attempt = await self.update_job_status_from_step(active_or_pending_step, HelixJobStatus.PAUSING)
        operations_counter.add(1, attributes={"operation": "pause_job"})
        return create_platform_job_response(job_entity, updated_attempt)

    async def resume_job(self, identifier: str, workspace: str) -> HelixJobResponse | None:
        """Resume a paused job (by name or ID)."""
        job = await self._resolve_job(identifier, workspace)
        if job is None:
            return None

        attempt = await self.get_current_attempt(job.name, workspace=job.workspace)
        if attempt is None:
            raise JobOperationConflictError(f"Cannot resume job '{job.name}': it has no current attempt.")

        if attempt.status in (HelixJobStatus.ACTIVE, HelixJobStatus.RESUMING):
            operations_counter.add(1, attributes={"operation": "resume_job"})
            return create_platform_job_response(job, attempt)

        if attempt.status == HelixJobStatus.PAUSED:
            if (attempt.status_details or {}).get(STORAGE_RECLAIMED_AT):
                raise JobOperationConflictError(
                    f"Cannot resume job '{job.name}': storage has been reclaimed. Create a new job."
                )
            paused_step = await self._get_step_by_status(job, workspace, [HelixJobStatus.PAUSED])
            if paused_step is None:
                raise JobOperationConflictError(f"Cannot resume job '{job.name}': no paused step was found.")
            await self.update_job_status_from_step(paused_step, HelixJobStatus.RESUMING)
            operations_counter.add(1, attributes={"operation": "resume_job"})
            return create_platform_job_response(job, await self.get_current_attempt(job.name, workspace))  # type: ignore

        raise JobOperationConflictError(self._resume_conflict_detail(job.name, attempt.status))

    @staticmethod
    def _resume_conflict_detail(job_name: str, status: HelixJobStatus) -> str:
        if status == HelixJobStatus.ERROR:
            return f"Cannot resume job '{job_name}': this job failed rather than paused, use rerun."
        if status == HelixJobStatus.COMPLETED:
            return f"Cannot resume job '{job_name}': the job succeeded."
        if status == HelixJobStatus.CANCELLED:
            return f"Cannot resume job '{job_name}': cancel is final."
        if status in (HelixJobStatus.PENDING, HelixJobStatus.CREATED):
            return f"Cannot resume job '{job_name}': the job has not been paused."
        if status == HelixJobStatus.PAUSING:
            return f"Cannot resume job '{job_name}': a pause is in flight."
        return f"Cannot resume job '{job_name}' from status '{status.value}'."

    async def update_job_control(
        self,
        identifier: str,
        workspace: str,
        update: JobControl,
    ) -> HelixJobResponse | None:
        """Change mutable job behavior without rewriting the attempt spec.

        Only the values set in ``update`` change. An omitted or null value keeps the
        job's current value. The pause window is measured from the paused step's
        ``stopped_at``, so a new value does not move the start. The job keeps the
        identity it runs as: a settings change by another user does not replace it.
        """
        changes = update.model_dump(exclude_none=True)
        _reject_pause_ttl_over_maximum(changes.get("pause_ttl_seconds"))

        job = await self._resolve_job(identifier, workspace)
        if job is None:
            return None

        attempt = await self.get_current_attempt(job.name, workspace=job.workspace)
        if attempt is None:
            raise JobOperationConflictError(f"Cannot update control for job '{job.name}': it has no current attempt.")
        if (attempt.status_details or {}).get(STORAGE_RECLAIMED_AT):
            raise JobOperationConflictError(
                f"Cannot update control for job '{job.name}': storage has been reclaimed. Create a new job."
            )

        job.control = job.control.model_copy(update=changes)
        # The caller's reads strip the stored auth context, and an update replaces the
        # job's data, so read the context as the jobs service and write it back unchanged.
        stored = await self.store.as_service("jobs", internal=True).get_by_id(HelixJob, job.id)
        job.with_auth_context(stored.auth_context)
        job = await self.store.update(job)
        operations_counter.add(1, attributes={"operation": "update_job_control"})
        return create_platform_job_response(job, attempt)

    # =========================================================================
    # Task Operations
    # =========================================================================

    async def get_task(self, step_id: str, task_name: str, workspace: str) -> Optional[HelixJobTask]:
        """Get a platform job task by task name within a step.

        Note: task_id is actually the task NAME (not entity ID). Tasks are
        parent-scoped entities, meaning task names are unique within their
        parent step. This method looks up tasks by name within the given step.
        """
        try:
            task = await self.store.get(HelixJobTask, task_name, workspace=workspace, parent=step_id)
            if task.step_id == step_id:
                return task
            return None
        except EntityNotFoundError:
            return None

    async def list_tasks(self, step_id: str, workspace: str) -> list[HelixJobTask]:
        """List all platform job tasks for a specific step."""
        return await self._list_all_entities(
            HelixJobTask,
            workspace=workspace,
            filter_obj={"step_id": step_id},
        )

    async def _finalize_step_tasks(self, step: HelixJobStep) -> None:
        """Finalize nonterminal children to match a cancelled or errored step."""
        for task in await self.list_tasks(step.id, step.workspace):
            await self._finalize_task(task, step)

    async def _finalize_task(self, task: HelixJobTask, step: HelixJobStep) -> HelixJobTask:
        """Preserve terminal winners and retry one conflicting task write."""
        for retry in range(2):
            if task.status.is_terminal():
                break
            updates = {
                "status": step.status,
                "status_details": {
                    **task.status_details,
                    "message": f"Task finalized because job step is {step.status.value}",
                },
            }
            if step.status == HelixJobStatus.ERROR:
                updates["error_details"] = task.error_details or step.error_details or {"message": "Job step failed"}
            try:
                task = await self.store.update(task.model_copy(update=updates))
                break
            except EntityConflictError:
                if retry:
                    raise
                task = await self.store.get_by_id(HelixJobTask, task.id)
        return task

    async def create_or_update_task(
        self, job_name: str, task_name: str, workspace: str, task_update: HelixJobTaskUpdate, step: HelixJobStep
    ) -> HelixJobTask:
        """Create or update a task for a job step."""
        async with _get_job_mutation_lock(job_name, workspace):
            return await self._create_or_update_task_locked(job_name, task_name, workspace, task_update, step)

    async def _create_or_update_task_locked(
        self, job_name: str, task_name: str, workspace: str, task_update: HelixJobTaskUpdate, step: HelixJobStep
    ) -> HelixJobTask:
        """Create or update a task while holding the per-job mutation lock."""
        job = await self.store.get(HelixJob, job_name, workspace=workspace)
        step = await self.store.get_by_id(HelixJobStep, step.id)

        task = await self.get_task(step.id, task_name, workspace=workspace)

        if step.status in (HelixJobStatus.CANCELLED, HelixJobStatus.ERROR):
            if task is not None and task.status.is_terminal():
                return task
            if not task_update.status.is_terminal() or (
                task is not None and not task.status.can_transition_to(task_update.status)
            ):
                if task is not None:
                    return await self._finalize_task(task, step)
                task_update = HelixJobTaskUpdate(
                    status=step.status,
                    status_details={"message": f"Task finalized because job step is {step.status.value}"},
                    error_details=step.error_details if step.status == HelixJobStatus.ERROR else None,
                )

        if not task:
            task = await self.store.create(
                HelixJobTask(
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
            current_status = HelixJobStatus(task.status)
            new_status = HelixJobStatus(task_update.status)
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
                # Pod-level error detection races the container, so progress arriving after ERROR
                # means the workload is alive: record it, leave the status terminal. Late reports
                # against the deliberate terminals, COMPLETED and CANCELLED, are stale.
                live_workload_report = (
                    current_status == HelixJobStatus.ERROR
                    and new_status != HelixJobStatus.ERROR
                    and bool(task_update.status_details)
                )
                if not live_workload_report:
                    operations_counter.add(1, attributes={"operation": "create_or_update_task"})
                    return task
            else:
                task.status = new_status

            if task_update.error_details:
                task.error_details = _keep_reported_cause(task.error_details, task_update.error_details)
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

        current_step = await self.store.get_by_id(HelixJobStep, step.id)
        if current_step.status in (HelixJobStatus.CANCELLED, HelixJobStatus.ERROR):
            if not task.status.is_terminal():
                task = await self._finalize_task(task, current_step)
            return task

        # Propagate task status_details to parent job for progress tracking.
        # Training callbacks report progress (percentage_done, epoch, step, loss) at the task level,
        # but users query job status via the jobs API. Without this propagation,
        # job.status_details would be empty and users couldn't see training progress.
        if task_update.status_details:
            try:
                attempt = await self.get_attempt(step.attempt_id)
                if attempt is not None:
                    await self._update_attempt_status_details_locked(
                        job,
                        attempt,
                        workspace,
                        task_update.status_details,
                    )
                logger.debug("Propagated task status_details to job", extra={"job": job_name, "workspace": workspace})
            except Exception as e:
                # Log but don't fail the task update if job status_details update fails
                logger.warning(
                    "Failed to propagate task status_details to job",
                    extra={"job": job_name, "workspace": workspace, "error": e},
                )

        operations_counter.add(1, attributes={"operation": "create_or_update_task"})
        return task

    def is_step_cancelled(self, step: HelixJobStep) -> bool:
        """Determine if a step is in a cancelled state."""
        return step.status in (HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED)

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
        attempt_id: str,
    ) -> HelixJobResult:
        """Create a job result.

        Args:
            job_id: Job entity ID (used as parent reference for FK)
            result_name: Name of the result
            artifact_url: URL to the artifact
            artifact_storage_type: Type of artifact storage
            workspace: Workspace for the result
        """
        async with _get_job_mutation_lock(job_name, workspace):
            job = await self.store.get(HelixJob, job_name, workspace=workspace)
            if job.id != job_id:
                raise EntityNotFoundError(f"Job ID mismatch for {workspace}/{job_name}")
            attempt = await self.get_attempt(attempt_id)
            if attempt is None or attempt.job != job.id or attempt.workspace != workspace:
                raise EntityNotFoundError(f"Attempt '{attempt_id}' does not belong to job '{job_name}'")
            result = HelixJobResult(
                name=result_name,
                workspace=workspace,
                job=job_id,
                attempt_id=attempt.id,
                artifact_url=artifact_url,
                artifact_storage_type=artifact_storage_type,
            )
            return await self.store.create(result)

    async def get_result(self, job_name: str, result_name: str, workspace: str) -> Optional[HelixJobResult]:
        """Get a job result by its job-wide unique name."""
        job_entity = await self._resolve_job(job_name, workspace)
        if job_entity is None:
            logger.warning(f"Job '{job_name}' not found in workspace '{workspace}'")
            return None
        try:
            return await self.store.get(HelixJobResult, result_name, workspace=workspace, parent=job_entity.id)
        except EntityNotFoundError:
            logger.warning(f"Result '{result_name}' not found for job '{job_name}' in workspace '{workspace}'")
            return None

    async def list_results(
        self,
        job_id: str,
        workspace: str,
        sort: Optional[HelixJobSortField] = None,
        attempt_id: str | None = None,
    ) -> tuple[list[HelixJobResult], int]:
        """List job-owned results, optionally filtered by producing attempt."""
        filter_obj = {"job": job_id}
        if attempt_id is not None:
            filter_obj["attempt_id"] = attempt_id
        response = await self.store.list(
            HelixJobResult,
            filter_obj=filter_obj,
            page_size=1000,
            workspace=workspace,
        )
        results = list(response.data)
        if sort:
            reverse = sort.get_sort_direction() == "desc"
            results.sort(key=lambda r: getattr(r, sort.get_field_name()), reverse=reverse)

        return results, response.pagination.total_results
