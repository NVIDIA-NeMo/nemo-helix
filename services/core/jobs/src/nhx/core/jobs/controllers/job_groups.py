# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile persisted template submissions through the ordinary Jobs lifecycle."""

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from itertools import islice

from nemo_helix_plugin._spec_flags import deep_merge
from nemo_helix_plugin.client.errors import (
    ConflictError,
    InternalServerError,
    NemoTransportError,
    PermissionDeniedError,
    RateLimitError,
)
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.common.auth import AuthClient, AuthContext
from nhx.common.client_factory import get_async_nemo_client
from nhx.common.config import get_auth_config
from nhx.common.entities import ALL_WORKSPACES
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError, EntityValidationError
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import get_model_id
from nhx.core.jobs.app.dispatcher import (
    JobDeletionConflictError,
    JobDispatcher,
    JobOperationConflictError,
    JobSecretValidationError,
)
from nhx.core.jobs.entities import HelixJob
from nhx.core.jobs.job_groups import (
    HelixJobGroup,
    JobGroupMember,
    JobGroupMemberOutcome,
    JobGroupMemberState,
    JobGroupStatus,
    log_unreadable_job_group,
)

logger = logging.getLogger(__name__)
NO_WRITE_RETRY = RetryPolicy(max_retries=0)
PAGE_SIZE = 16
MEMBERS_PER_PASS = 16


def member_request(group: HelixJobGroup, member: JobGroupMember, name: str | None = None) -> CreateHelixJobRequest:
    submission = group.submission
    template = submission.template.model_copy(deep=True)
    timeout = member.timeout_seconds
    if timeout is None:
        timeout = submission.default_timeout_seconds
    for step in template.steps:
        step.config = deep_merge(step.config or {}, member.config_patch)
        if timeout is not None:
            step.lifecycle.staleness_timeout_seconds = timeout
    return CreateHelixJobRequest(
        name=name,
        source=submission.source,
        project=group.project,
        spec=submission.spec,
        platform_spec=template,
    )


class JobGroupReconciler:
    def __init__(self) -> None:
        self.page = 1
        self.member_offsets: dict[str, int] = {}
        self.deletion_pages: dict[str, int] = {}

    async def step(self, *, stopped: Callable[[], bool], on_progress: Callable[[], None] | None = None) -> None:
        async with get_async_nemo_client(as_service="jobs", internal=True, retry=NO_WRITE_RETRY) as client:
            store = EntityClient(AsyncEntitiesClient.from_client(client))
            response = await store.list(
                HelixJobGroup,
                workspace=ALL_WORKSPACES,
                filter_obj={
                    "controller_state.status": {
                        "$in": [
                            JobGroupStatus.PENDING,
                            JobGroupStatus.ACTIVE,
                            JobGroupStatus.CANCELLING,
                            JobGroupStatus.DELETION_PENDING,
                        ]
                    }
                },
                sort="created_at",
                page=self.page,
                page_size=PAGE_SIZE,
                on_decode_error=log_unreadable_job_group,
            )
            self.page = self.page + 1 if self.page < response.pagination.total_pages else 1
            for group in response.data:
                if stopped():
                    return
                try:
                    dispatcher = JobDispatcher(
                        store, AsyncFilesClient.from_client(client), AsyncSecretsClient.from_client(client)
                    )
                    await self.reconcile_group(group, dispatcher, on_progress=on_progress)
                except (EntityConflictError, EntityNotFoundError):
                    # Another writer changed the group or one of its Jobs. Its next pass rereads the state.
                    continue
                except Exception:
                    logger.exception("Could not reconcile JobGroup", extra={"group": group.id})
                finally:
                    if on_progress is not None:
                        on_progress()

    async def reconcile_group(
        self, group: HelixJobGroup, dispatcher: JobDispatcher, *, on_progress: Callable[[], None] | None = None
    ) -> None:
        state = group.controller_state
        if state.status == JobGroupStatus.DELETION_PENDING:
            await self.delete_group(group, dispatcher, on_progress=on_progress)
            return
        if state.status.terminal:
            return

        store = dispatcher.store
        stopping = state.status == JobGroupStatus.CANCELLING
        before = state.model_copy(deep=True)
        member_count = len(group.submission.members)
        if stopping:
            for index in state.pending_members(member_count):
                state.finish_member(index, JobGroupMemberOutcome.CANCELLED)

        in_flight = state.members_in_flight(member_count)
        if in_flight:
            # Rotate the bounded poll so slow Jobs cannot hide later member outcomes.
            offset = self.member_offsets.get(group.id, 0) % len(in_flight)
            self.member_offsets[group.id] = (offset + MEMBERS_PER_PASS) % len(in_flight)
            for index in (in_flight[offset:] + in_flight[:offset])[:MEMBERS_PER_PASS]:
                await self.reconcile_member(group, index, dispatcher, stopping=stopping)
                if on_progress is not None:
                    on_progress()
                if (
                    not stopping
                    and group.submission.fail_fast
                    and state.member(index).outcome == JobGroupMemberOutcome.ERROR
                ):
                    state.cancel()
                    await store.update(group)
                    return

        if not stopping:
            available = state.available_claims(group.submission.max_concurrent_jobs, MEMBERS_PER_PASS)
            for index in islice(state.pending_members(member_count), available):
                state.claim_member(index, get_model_id("att"))

        if state.settle(member_count):
            self.member_offsets.pop(group.id, None)
        if state != before:
            await store.update(group)

    async def find_owned_job(self, group: HelixJobGroup, job_id: str, store: EntityClient) -> HelixJob | None:
        try:
            job = await store.get_by_id(HelixJob, job_id)
        except EntityNotFoundError:
            return None
        if job.workspace != group.workspace or job.parent != group.id:
            raise JobOperationConflictError("The recorded Job is owned by a different group or workspace.")
        return job

    async def recover_member_job(
        self, group: HelixJobGroup, index: int, job_name: str, store: EntityClient
    ) -> HelixJob | None:
        try:
            job = await store.get(HelixJob, job_name, workspace=group.workspace, parent=group.id)
        except EntityNotFoundError:
            return None
        group.controller_state.record_job(index, job.id)
        return job

    async def reconcile_member(
        self, group: HelixJobGroup, index: int, dispatcher: JobDispatcher, *, stopping: bool
    ) -> None:
        try:
            await self.progress_member(group, index, dispatcher, stopping=stopping)
        except (
            EntityConflictError,
            InternalServerError,
            NemoTransportError,
            RateLimitError,
        ):
            logger.warning(
                "Could not progress JobGroup member; will retry",
                extra={"group": group.id, "member_index": index},
            )
        except (
            JobOperationConflictError,
            JobSecretValidationError,
            PermissionDeniedError,
            EntityValidationError,
        ) as exc:
            group.controller_state.finish_member(index, JobGroupMemberOutcome.ERROR, str(exc))

    async def progress_member(
        self,
        group: HelixJobGroup,
        index: int,
        dispatcher: JobDispatcher,
        *,
        stopping: bool,
    ) -> None:
        state = group.controller_state
        submitted = group.submission.members[index]
        member = state.member(index)
        if member.job_id is None:
            name = group.member_job_name(index)
            job = await self.recover_member_job(group, index, name, dispatcher.store)
        else:
            job = await self.find_owned_job(group, member.job_id, dispatcher.store)
        if job is None:
            if member.reason:
                state.finish_missing_job(index, JobGroupMemberOutcome.ERROR, member.reason)
                return
            if stopping:
                state.finish_missing_job(index, JobGroupMemberOutcome.CANCELLED)
                return
            if member.job_id is not None:
                state.finish_member(
                    index, JobGroupMemberOutcome.ERROR, "The owned Job disappeared before the member finished."
                )
                return
        elif job.status == HelixJobStatus.DELETION_PENDING:
            return
        elif job.status == HelixJobStatus.CREATION_PENDING and member.attempt_id is None:
            if stopping or member.reason:
                await dispatcher.delete_job(job.id, group.workspace, group_id=group.id)
                return
            if job.output_location is None and job.fileset_id is None:
                return

        if job is not None:
            await self.progress_attempt(group, index, member, job, dispatcher, stopping=stopping)
            return

        name = group.member_job_name(index)
        request = member_request(group, submitted, name)
        async with self.delegated_dispatcher(group, dispatcher.store) as delegated:
            try:
                response = await delegated.create_group_member(
                    request, group.workspace, group_id=group.id, auth_context=self.auth_context(group)
                )
            except EntityConflictError:
                if await self.recover_member_job(group, index, name, dispatcher.store) is not None:
                    return
                raise
            except (ConflictError, PermissionDeniedError, EntityValidationError) as exc:
                job = await self.recover_member_job(group, index, name, dispatcher.store)
                if job is None:
                    if isinstance(exc, ConflictError):
                        raise JobOperationConflictError("Could not allocate the member Job's Fileset.") from exc
                    raise
                state.record_creation_failure(index, job.id, str(exc))
                return
        state.record_job(index, response.id)

    async def progress_attempt(
        self,
        group: HelixJobGroup,
        index: int,
        member: JobGroupMemberState,
        job: HelixJob,
        dispatcher: JobDispatcher,
        *,
        stopping: bool,
    ) -> None:
        state = group.controller_state
        if member.attempt_id is not None:
            await self.observe_attempt(group, index, member.attempt_id, job, dispatcher, stopping=stopping)
            return

        attempt_name = member.attempt_name
        if attempt_name is None:
            raise JobOperationConflictError("The group member has no reserved Attempt name.")
        attempt = await dispatcher.find_prepared_group_attempt(job, attempt_name)
        if attempt is None:
            if stopping:
                state.finish_member(index, JobGroupMemberOutcome.CANCELLED)
                return
            request = member_request(group, group.submission.members[index])
            async with self.delegated_dispatcher(group, dispatcher.store) as delegated:
                attempt = await delegated.prepare_group_attempt(job, request, attempt_name)
        state.record_attempt(index, job.id, attempt.id)

    @asynccontextmanager
    async def delegated_dispatcher(self, group: HelixJobGroup, store: EntityClient) -> AsyncGenerator[JobDispatcher]:
        principal = self.auth_context(group).to_principal()
        auth = AuthClient(principal=principal, config=get_auth_config(), service_name="jobs")
        if auth.auth_enabled and not await self.can_create_jobs(group, auth):
            raise JobOperationConflictError("Could not verify the submitter's current permission to create Jobs.")
        async with get_async_nemo_client(
            as_service="jobs",
            internal=True,
            on_behalf_of=principal,
            retry=NO_WRITE_RETRY,
        ) as scoped:
            yield JobDispatcher(
                EntityClient(AsyncEntitiesClient.from_client(scoped)),
                AsyncFilesClient.from_client(scoped),
                AsyncSecretsClient.from_client(scoped),
                system_store=store,
            )

    @staticmethod
    def auth_context(group: HelixJobGroup) -> AuthContext:
        if group.auth_context is None:
            raise JobOperationConflictError("The saved submitter identity is unavailable.")
        # The controller can recheck policy for the saved identity. It has no
        # fresh source for the submitter's external group membership.
        return group.auth_context.model_copy(update={"principal_groups": [], "principal_on_behalf_of_groups": None})

    @staticmethod
    async def can_create_jobs(group: HelixJobGroup, auth: AuthClient) -> bool:
        if auth.principal.on_behalf_of:
            return await auth.check_permissions(group.workspace, subject=["jobs.create"])
        return await auth.check_permissions(group.workspace, actor=["jobs.create"])

    async def observe_attempt(
        self,
        group: HelixJobGroup,
        index: int,
        attempt_id: str,
        job: HelixJob,
        dispatcher: JobDispatcher,
        *,
        stopping: bool,
    ) -> None:
        attempt = await dispatcher.get_attempt(attempt_id)
        if attempt is None or attempt.job != job.id or attempt.workspace != job.workspace:
            raise JobOperationConflictError("The group Attempt no longer exists.")
        if attempt.status.is_terminal():
            if job.current_attempt_id != attempt.id:
                await dispatcher.publish_group_attempt(job, attempt, self.auth_context(group))
            message = attempt.error_details.get("message") if attempt.error_details else None
            group.controller_state.finish_member(
                index, JobGroupMemberOutcome(attempt.status.value), str(message) if message else None
            )
        elif stopping:
            await dispatcher.cancel_prepared_group_attempt(job, attempt)
        elif attempt.status == HelixJobStatus.CREATED:
            await dispatcher.publish_group_attempt(job, attempt, self.auth_context(group))

    async def delete_group(
        self, group: HelixJobGroup, dispatcher: JobDispatcher, *, on_progress: Callable[[], None] | None = None
    ) -> None:
        page = self.deletion_pages.get(group.id, 1)
        jobs = await dispatcher.store.list(
            HelixJob,
            workspace=group.workspace,
            filter_operation=ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=group.id),
            page=page,
            page_size=MEMBERS_PER_PASS,
        )
        self.deletion_pages[group.id] = page + 1 if page < jobs.pagination.total_pages else 1
        for job in jobs.data:
            try:
                await dispatcher.delete_job(job.id, group.workspace, group_id=group.id)
            except (EntityConflictError, EntityNotFoundError, JobDeletionConflictError):
                continue
            finally:
                if on_progress is not None:
                    on_progress()
        if jobs.pagination.total_results == 0:
            await dispatcher.store.delete_by_id(HelixJobGroup, group.id, expected_db_version=group.db_version)
            self.deletion_pages.pop(group.id, None)
            self.member_offsets.pop(group.id, None)
