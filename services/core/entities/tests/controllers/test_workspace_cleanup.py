# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nhx.core.entities.controllers.workspace_cleanup import (
    WorkspaceCleanup,
    WorkspaceJobCleanupError,
    WorkspaceResourceCleanupError,
)
from nhx.core.entities.entities import Workspace, WorkspaceDeletionStage


def _make_workspace(
    name: str = "test-workspace",
    *,
    updated_at: datetime | None = None,
    deletion_stage: WorkspaceDeletionStage = WorkspaceDeletionStage.PENDING,
) -> Workspace:
    now = datetime.now(tz=timezone.utc)
    workspace = Workspace(
        id="ws-123",
        name=name,
        description="test",
        created_at=now,
        updated_at=updated_at or now,
    )
    workspace._deletion_stage = deletion_stage
    return workspace


class _MockAsyncPaginatedResponse:
    """Mock for AsyncNemoPaginatedResponse that exposes .items() as an async generator."""

    def __init__(self, items):
        self._items = items

    async def items(self):
        for item in self._items:
            yield item


def _make_mock_files_client(filesets: list | None = None) -> AsyncMock:
    """Build a mock AsyncFilesClient with list_filesets/delete_fileset."""
    mock_files = AsyncMock()
    mock_files.list_filesets = AsyncMock(return_value=_MockAsyncPaginatedResponse(filesets or []))
    mock_files.delete_fileset = AsyncMock()
    return mock_files


def _make_response(data) -> MagicMock:
    response = MagicMock()
    response.data.return_value = data
    return response


def _make_job_status(status: str) -> MagicMock:
    status_response = MagicMock()
    status_response.status = status
    return status_response


def _make_jobs_client(jobs: list | None = None) -> MagicMock:
    """Build a mock typed AsyncJobsClient.

    Production routes jobs calls through ``AsyncJobsClient.from_client(client)``
    and iterates ``(await jobs_client.list_jobs(...)).items()``. So ``list_jobs`` is an
    ``AsyncMock`` returning a paginated response whose ``.items()`` yields an async
    iterator over the jobs.
    """
    jobs_client = MagicMock()
    jobs_client.list_jobs = AsyncMock(return_value=_MockAsyncPaginatedResponse(jobs or []))
    jobs_client.cancel_job = AsyncMock()
    jobs_client.delete_job = AsyncMock()
    jobs_client.get_job_status = AsyncMock(return_value=_make_response(_make_job_status("cancelled")))
    return jobs_client


def _make_models_client(
    deployments: list | None = None,
    models: list | None = None,
    adapters: list | None = None,
) -> MagicMock:
    """Build a mock typed AsyncModelsClient.

    Production routes deployment, model, and adapter cleanup through
    ``AsyncModelsClient.from_client(client)``.
    """
    models_client = MagicMock()
    models_client.list_deployments = AsyncMock(return_value=_MockAsyncPaginatedResponse(deployments or []))
    models_client.delete_deployment = AsyncMock()
    models_client.list_models = AsyncMock(return_value=_MockAsyncPaginatedResponse(models or []))
    models_client.delete_model = AsyncMock()
    models_client.list_adapters = AsyncMock(return_value=_MockAsyncPaginatedResponse(adapters or []))
    models_client.delete_adapter = AsyncMock()
    return models_client


def _patch_jobs_client(jobs_client: MagicMock):
    """Patch the typed ``from_client`` classmethods so each client class resolves to its mock.

    Returns *jobs_client* for ``AsyncJobsClient`` and safe empty mocks for the
    deployment/fileset clients, which keeps ``_async_step`` tests correct even
    when ``_cleanup_jobs`` succeeds and execution proceeds to
    ``_cleanup_deployments`` and ``_cleanup_filesets``.
    """
    return _patch_clients(jobs_client, _make_mock_files_client([]), _make_models_client([]))


@contextmanager
def _patch_clients(jobs_client: MagicMock, files_client: MagicMock, models_client: MagicMock | None = None):
    """Patch the typed ``from_client`` classmethods so each client class resolves to its mock.

    ``_async_step`` cleans up jobs, deployments, models/adapters, and filesets,
    so it derives ``AsyncJobsClient``, ``AsyncModelsClient``, and
    ``AsyncFilesClient`` from the platform client; return the matching mock.
    """
    models = models_client if models_client is not None else _make_models_client([])

    with (
        patch.object(AsyncFilesClient, "from_client", return_value=files_client),
        patch.object(AsyncJobsClient, "from_client", return_value=jobs_client),
        patch.object(AsyncModelsClient, "from_client", return_value=models),
    ):
        yield


def _make_job(name: str, status: str = "completed") -> MagicMock:
    job = MagicMock()
    job.name = name
    job.status = status
    return job


def _named(name: str) -> MagicMock:
    item = MagicMock()
    item.name = name
    return item


def _not_found() -> NotFoundError:
    return NotFoundError(httpx.Response(404, request=httpx.Request("DELETE", "http://platform.test/resource")))


def _make_controller(
    workspace_repo: AsyncMock | None = None,
    client: MagicMock | None = None,
) -> WorkspaceCleanup:
    if workspace_repo is None:
        workspace_repo = AsyncMock()
    if client is None:
        client = MagicMock()

    return WorkspaceCleanup(
        client=client,
        workspace_repository=workspace_repo,
    )


class TestWorkspaceCleanupStep:
    def test_step_skips_when_stop_signal_set(self):
        import threading

        stop = threading.Event()
        stop.set()
        repo = AsyncMock()
        controller = WorkspaceCleanup(
            client=MagicMock(),
            workspace_repository=repo,
            stop_signal=stop,
        )

        controller.step()

        repo.list_workspaces.assert_not_called()

    def test_step_uses_provided_loop(self):
        loop = asyncio.new_event_loop()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([], None)
        controller = WorkspaceCleanup(
            client=MagicMock(),
            workspace_repository=repo,
            loop=loop,
        )

        assert controller._loop is loop
        controller.step()
        assert controller.is_healthy
        assert repo.list_workspaces.await_count == 3
        loop.close()

    def test_step_sets_healthy_on_success(self):
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([], None)
        controller = _make_controller(workspace_repo=repo)

        assert not controller.is_healthy
        controller.step()
        assert controller.is_healthy

    def test_step_sets_unhealthy_on_failure(self):
        repo = AsyncMock()
        repo.list_workspaces.side_effect = Exception("db error")
        controller = _make_controller(workspace_repo=repo)

        controller.step()
        assert not controller.is_healthy


class TestWorkspaceCleanupAsyncStep:
    @pytest.mark.asyncio
    async def test_no_pending_workspaces(self):
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([], None)
        controller = _make_controller(workspace_repo=repo)

        await controller._async_step()

        repo.claim_workspace_for_deletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_workspace_deletion(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.claim_workspace_for_deletion.return_value = workspace
        repo.delete_workspace_if_claimed.return_value = True

        sdk = MagicMock()
        mock_files = _make_mock_files_client([])
        controller = _make_controller(workspace_repo=repo, client=sdk)

        with _patch_clients(_make_jobs_client([]), mock_files, _make_models_client([])):
            await controller._async_step()

        repo.claim_workspace_for_deletion.assert_awaited_once_with(
            name="test-workspace",
            expected_stage=WorkspaceDeletionStage.PENDING,
            expected_updated_at=workspace.updated_at,
            lease_expired_before=None,
        )
        repo.delete_workspace_if_claimed.assert_awaited_once_with(
            name="test-workspace",
            expected_updated_at=workspace.updated_at,
        )

    @pytest.mark.asyncio
    async def test_deletes_models_and_adapters_before_filesets(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.claim_workspace_for_deletion.return_value = workspace
        repo.delete_workspace_if_claimed.return_value = True

        adapter = MagicMock()
        adapter.name = "adapter"
        model = MagicMock()
        model.name = "model"
        fileset = MagicMock()
        fileset.name = "weights"
        call_order: list[str] = []
        models_client = _make_models_client(models=[model], adapters=[adapter])
        models_client.delete_adapter = AsyncMock(side_effect=lambda **_: call_order.append("adapter"))
        models_client.delete_model = AsyncMock(side_effect=lambda **_: call_order.append("model"))
        mock_files = _make_mock_files_client([fileset])
        mock_files.delete_fileset = AsyncMock(side_effect=lambda **_: call_order.append("fileset"))
        controller = _make_controller(workspace_repo=repo)

        with _patch_clients(_make_jobs_client([]), mock_files, models_client):
            await controller._async_step()

        assert call_order == ["adapter", "model", "fileset"]
        repo.delete_workspace_if_claimed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_workspace_already_being_processed(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.claim_workspace_for_deletion.return_value = None

        controller = _make_controller(workspace_repo=repo)

        await controller._async_step()

        repo.delete_workspace_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_failure_marks_workspace_failed(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.claim_workspace_for_deletion.return_value = workspace
        repo.fail_workspace_deletion_if_claimed.return_value = True

        jobs_client = MagicMock()
        jobs_client.list_jobs = AsyncMock(side_effect=Exception("jobs service down"))

        controller = _make_controller(workspace_repo=repo)

        with _patch_jobs_client(jobs_client):
            await controller._async_step()

        repo.fail_workspace_deletion_if_claimed.assert_awaited_once_with(
            name="test-workspace",
            expected_updated_at=workspace.updated_at,
        )
        repo.delete_workspace_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_failed_workspace_after_backoff(self):
        workspace = _make_workspace(
            updated_at=datetime.now(tz=timezone.utc) - timedelta(seconds=120),
            deletion_stage=WorkspaceDeletionStage.FAILED,
        )
        repo = AsyncMock()
        repo.list_workspaces.side_effect = [
            ([], None),
            ([workspace], None),
        ]
        repo.claim_workspace_for_deletion.return_value = workspace
        repo.delete_workspace_if_claimed.return_value = True
        controller = _make_controller(workspace_repo=repo)

        with _patch_clients(_make_jobs_client([]), _make_mock_files_client([]), _make_models_client([])):
            await controller._async_step()

        failed_calls = [
            call
            for call in repo.list_workspaces.await_args_list
            if call.kwargs["filter_op"].value == WorkspaceDeletionStage.FAILED
        ]
        assert len(failed_calls) == 1
        assert failed_calls[0].kwargs["sort"] == "updated_at"
        assert failed_calls[0].kwargs["page_size"] == 1
        repo.claim_workspace_for_deletion.assert_awaited_once_with(
            name="test-workspace",
            expected_stage=WorkspaceDeletionStage.FAILED,
            expected_updated_at=workspace.updated_at,
            lease_expired_before=None,
        )
        repo.delete_workspace_if_claimed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_failed_workspace_inside_backoff(self):
        workspace = _make_workspace(deletion_stage=WorkspaceDeletionStage.FAILED)
        repo = AsyncMock()
        repo.list_workspaces.side_effect = [
            ([], None),
            ([workspace], None),
            ([], None),
        ]
        controller = _make_controller(workspace_repo=repo)

        await controller._async_step()

        repo.claim_workspace_for_deletion.assert_not_called()
        repo.delete_workspace_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_reclaims_deleting_workspace_after_lease_expires(self):
        stale_timestamp = datetime.now(tz=timezone.utc) - timedelta(seconds=120)
        stale = _make_workspace(
            updated_at=stale_timestamp,
            deletion_stage=WorkspaceDeletionStage.DELETING,
        )
        claimed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        repo = AsyncMock()
        repo.list_workspaces.side_effect = [
            ([], None),
            ([], None),
            ([stale], None),
        ]
        repo.claim_workspace_for_deletion.return_value = claimed
        repo.delete_workspace_if_claimed.return_value = True
        controller = _make_controller(workspace_repo=repo)

        with _patch_clients(_make_jobs_client([]), _make_mock_files_client([]), _make_models_client([])):
            await controller._async_step()

        claim = repo.claim_workspace_for_deletion.await_args
        assert claim.kwargs["expected_stage"] == WorkspaceDeletionStage.DELETING
        assert claim.kwargs["expected_updated_at"] == stale_timestamp
        assert claim.kwargs["lease_expired_before"] < claimed.updated_at.replace(tzinfo=None)
        repo.delete_workspace_if_claimed.assert_awaited_once_with(
            name=stale.name,
            expected_updated_at=claimed.updated_at,
        )

    @pytest.mark.asyncio
    async def test_leaves_recent_deleting_workspace_alone(self):
        recent = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        repo = AsyncMock()
        repo.list_workspaces.side_effect = [
            ([], None),
            ([], None),
            ([recent], None),
        ]
        controller = _make_controller(workspace_repo=repo)

        await controller._async_step()

        repo.claim_workspace_for_deletion.assert_not_called()
        repo.delete_workspace_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_known_deletion_stages_are_queried(self):
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([], None)
        controller = _make_controller(workspace_repo=repo)

        await controller._async_step()

        queried_stages = [call.kwargs["filter_op"].value for call in repo.list_workspaces.await_args_list]
        assert queried_stages == [
            WorkspaceDeletionStage.PENDING,
            WorkspaceDeletionStage.FAILED,
            WorkspaceDeletionStage.DELETING,
        ]

    @pytest.mark.asyncio
    async def test_renews_lease_while_cleanup_is_in_progress(self):
        pending = _make_workspace()
        claimed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        renewed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        cleanup_release = asyncio.Event()
        renewal_release = asyncio.Event()
        renewal_completed = asyncio.Event()
        renewal_sleep_count = 0

        async def block_cleanup(**_kwargs):
            await cleanup_release.wait()
            return _MockAsyncPaginatedResponse([])

        async def controlled_sleep(_delay: float) -> None:
            nonlocal renewal_sleep_count
            renewal_sleep_count += 1
            if renewal_sleep_count == 1:
                await renewal_release.wait()
            else:
                await asyncio.Event().wait()

        async def renew(**_kwargs):
            renewal_completed.set()
            return renewed

        jobs_client = _make_jobs_client([])
        jobs_client.list_jobs.side_effect = block_cleanup
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([pending], None)
        repo.claim_workspace_for_deletion.return_value = claimed
        repo.renew_workspace_deletion_lease.side_effect = renew
        repo.delete_workspace_if_claimed.return_value = True
        controller = _make_controller(workspace_repo=repo)

        with (
            _patch_clients(jobs_client, _make_mock_files_client([]), _make_models_client([])),
            patch("nhx.core.entities.controllers.workspace_cleanup.asyncio.sleep", side_effect=controlled_sleep),
        ):
            cleanup_task = asyncio.create_task(controller._async_step())
            renewal_release.set()
            await renewal_completed.wait()
            cleanup_release.set()
            await cleanup_task

        repo.renew_workspace_deletion_lease.assert_awaited_once_with(
            name=pending.name,
            expected_updated_at=claimed.updated_at,
        )
        repo.delete_workspace_if_claimed.assert_awaited_once_with(
            name=pending.name,
            expected_updated_at=renewed.updated_at,
        )

    @pytest.mark.asyncio
    async def test_lost_lease_prevents_final_state_changes(self):
        pending = _make_workspace()
        claimed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        cleanup_started = asyncio.Event()
        cleanup_cancelled = asyncio.Event()
        renewal_release = asyncio.Event()
        renewal_completed = asyncio.Event()

        async def block_cleanup(**_kwargs):
            cleanup_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_cancelled.set()
                raise

        async def controlled_sleep(_delay: float) -> None:
            await renewal_release.wait()

        async def lose_lease(**_kwargs):
            renewal_completed.set()
            return None

        jobs_client = _make_jobs_client([])
        jobs_client.list_jobs.side_effect = block_cleanup
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([pending], None)
        repo.claim_workspace_for_deletion.return_value = claimed
        repo.renew_workspace_deletion_lease.side_effect = lose_lease
        controller = _make_controller(workspace_repo=repo)

        with (
            _patch_clients(jobs_client, _make_mock_files_client([]), _make_models_client([])),
            patch("nhx.core.entities.controllers.workspace_cleanup.asyncio.sleep", side_effect=controlled_sleep),
        ):
            cleanup_task = asyncio.create_task(controller._async_step())
            await cleanup_started.wait()
            renewal_release.set()
            await renewal_completed.wait()
            await cleanup_task

        assert cleanup_cancelled.is_set()
        repo.fail_workspace_deletion_if_claimed.assert_not_called()
        repo.delete_workspace_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_renewal_error_cancels_cleanup_and_fails_current_claim(self):
        pending = _make_workspace()
        claimed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        cleanup_started = asyncio.Event()
        cleanup_cancelled = asyncio.Event()
        renewal_release = asyncio.Event()

        async def block_cleanup(**_kwargs):
            cleanup_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_cancelled.set()
                raise

        async def controlled_sleep(_delay: float) -> None:
            await renewal_release.wait()

        jobs_client = _make_jobs_client([])
        jobs_client.list_jobs.side_effect = block_cleanup
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([pending], None)
        repo.claim_workspace_for_deletion.return_value = claimed
        repo.renew_workspace_deletion_lease.side_effect = RuntimeError("database unavailable")
        repo.fail_workspace_deletion_if_claimed.return_value = True
        controller = _make_controller(workspace_repo=repo)

        with (
            _patch_clients(jobs_client, _make_mock_files_client([]), _make_models_client([])),
            patch("nhx.core.entities.controllers.workspace_cleanup.asyncio.sleep", side_effect=controlled_sleep),
        ):
            cleanup_task = asyncio.create_task(controller._async_step())
            await cleanup_started.wait()
            renewal_release.set()
            await cleanup_task

        assert cleanup_cancelled.is_set()
        repo.fail_workspace_deletion_if_claimed.assert_awaited_once_with(
            name=pending.name,
            expected_updated_at=claimed.updated_at,
        )
        repo.delete_workspace_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_stalled_renewal_is_cancelled_before_the_lease_expires(self, monkeypatch: pytest.MonkeyPatch):
        from nhx.core.entities.controllers import workspace_cleanup as workspace_cleanup_module

        # Equal values leave no time after the renewal interval, so a call that
        # does not return is cancelled instead of running past the lease.
        monkeypatch.setattr(workspace_cleanup_module, "_DELETION_LEASE_TIMEOUT_SECONDS", 20.0)
        monkeypatch.setattr(workspace_cleanup_module, "_DELETION_LEASE_RENEW_INTERVAL_SECONDS", 20.0)
        pending = _make_workspace()
        claimed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        cleanup_started = asyncio.Event()
        cleanup_cancelled = asyncio.Event()
        renewal_release = asyncio.Event()

        async def block_cleanup(**_kwargs):
            cleanup_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_cancelled.set()
                raise

        async def controlled_sleep(_delay: float) -> None:
            await renewal_release.wait()

        async def hang_renewal(**_kwargs):
            await asyncio.Event().wait()

        jobs_client = _make_jobs_client([])
        jobs_client.list_jobs.side_effect = block_cleanup
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([pending], None)
        repo.claim_workspace_for_deletion.return_value = claimed
        repo.renew_workspace_deletion_lease.side_effect = hang_renewal
        repo.fail_workspace_deletion_if_claimed.return_value = True
        controller = _make_controller(workspace_repo=repo)

        with (
            _patch_clients(jobs_client, _make_mock_files_client([]), _make_models_client([])),
            patch("nhx.core.entities.controllers.workspace_cleanup.asyncio.sleep", side_effect=controlled_sleep),
        ):
            cleanup_task = asyncio.create_task(controller._async_step())
            await cleanup_started.wait()
            renewal_release.set()
            await cleanup_task

        assert cleanup_cancelled.is_set()
        repo.fail_workspace_deletion_if_claimed.assert_awaited_once_with(
            name=pending.name,
            expected_updated_at=claimed.updated_at,
        )
        repo.delete_workspace_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_workspace_is_preferred_over_failed(self):
        pending = _make_workspace(name="pending-workspace")
        failed = _make_workspace(
            name="failed-workspace",
            updated_at=datetime.now(tz=timezone.utc) - timedelta(seconds=120),
        )
        repo = AsyncMock()

        async def list_workspaces(**kwargs):
            if kwargs["filter_op"].value == WorkspaceDeletionStage.PENDING:
                return ([pending], None)
            return ([failed], None)

        repo.list_workspaces.side_effect = list_workspaces
        repo.claim_workspace_for_deletion.return_value = pending
        repo.delete_workspace_if_claimed.return_value = True
        controller = _make_controller(workspace_repo=repo)

        with _patch_clients(_make_jobs_client([]), _make_mock_files_client([]), _make_models_client([])):
            await controller._async_step()

        queried_stages = [call.kwargs["filter_op"].value for call in repo.list_workspaces.await_args_list]
        assert queried_stages == [WorkspaceDeletionStage.PENDING]
        repo.delete_workspace_if_claimed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_after_model_list_503(self):
        pending = _make_workspace()
        failed = _make_workspace(
            updated_at=datetime.now(tz=timezone.utc) - timedelta(seconds=120),
            deletion_stage=WorkspaceDeletionStage.FAILED,
        )
        repo = AsyncMock()
        repo.list_workspaces.side_effect = [
            ([pending], None),
            ([], None),
            ([failed], None),
        ]
        repo.claim_workspace_for_deletion.side_effect = [pending, failed]
        repo.fail_workspace_deletion_if_claimed.return_value = True
        repo.delete_workspace_if_claimed.return_value = True
        models_client = _make_models_client([])
        models_client.list_models = AsyncMock(
            side_effect=[Exception("HTTP 503: no healthy upstream"), _MockAsyncPaginatedResponse([])]
        )
        controller = _make_controller(workspace_repo=repo)

        with _patch_clients(_make_jobs_client([]), _make_mock_files_client([]), models_client):
            await controller._async_step()
            await controller._async_step()

        repo.fail_workspace_deletion_if_claimed.assert_awaited_once()
        repo.delete_workspace_if_claimed.assert_awaited_once()

    @pytest.mark.parametrize("kind", ["deployment", "model", "adapter", "fileset"])
    @pytest.mark.asyncio
    async def test_item_delete_failure_keeps_the_workspace(self, kind: str):
        workspace = _make_workspace()
        claimed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.claim_workspace_for_deletion.return_value = claimed
        repo.fail_workspace_deletion_if_claimed.return_value = True
        first = _named("first")
        second = _named("second")
        model = _named("model")
        adapter = _named("adapter")
        fileset = _named("weights")
        models_client = _make_models_client()
        files_client = _make_mock_files_client([fileset])

        if kind == "deployment":
            models_client.list_deployments = AsyncMock(return_value=_MockAsyncPaginatedResponse([first, second]))
            models_client.delete_deployment = AsyncMock(side_effect=[Exception("fail"), None])
            attempted = models_client.delete_deployment
        elif kind == "adapter":
            models_client.list_adapters = AsyncMock(return_value=_MockAsyncPaginatedResponse([first, second]))
            models_client.list_models = AsyncMock(return_value=_MockAsyncPaginatedResponse([model]))
            models_client.delete_adapter = AsyncMock(side_effect=[Exception("fail"), None])
            models_client.delete_model = AsyncMock()
            attempted = models_client.delete_adapter
        elif kind == "model":
            models_client.list_adapters = AsyncMock(return_value=_MockAsyncPaginatedResponse([adapter]))
            models_client.list_models = AsyncMock(return_value=_MockAsyncPaginatedResponse([first, second]))
            models_client.delete_adapter = AsyncMock()
            models_client.delete_model = AsyncMock(side_effect=[Exception("fail"), None])
            attempted = models_client.delete_model
        else:
            files_client.list_filesets = AsyncMock(return_value=_MockAsyncPaginatedResponse([first, second]))
            files_client.delete_fileset = AsyncMock(side_effect=[Exception("fail"), None])
            attempted = files_client.delete_fileset

        controller = _make_controller(workspace_repo=repo)
        with _patch_clients(_make_jobs_client([]), files_client, models_client):
            await controller._async_step()

        assert [call.kwargs["name"] for call in attempted.await_args_list] == ["first", "second"]
        if kind == "adapter":
            models_client.delete_model.assert_awaited_once_with(name="model", workspace="test-workspace")
        if kind == "model":
            models_client.delete_adapter.assert_awaited_once_with(name="adapter", workspace="test-workspace")
        if kind != "fileset":
            files_client.delete_fileset.assert_not_called()
        repo.delete_workspace_if_claimed.assert_not_called()
        repo.fail_workspace_deletion_if_claimed.assert_awaited_once_with(
            name=workspace.name,
            expected_updated_at=claimed.updated_at,
        )

    @pytest.mark.parametrize("kind", ["deployment", "model", "adapter", "fileset"])
    @pytest.mark.asyncio
    async def test_not_found_item_delete_still_deletes_the_workspace(self, kind: str):
        workspace = _make_workspace()
        claimed = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.claim_workspace_for_deletion.return_value = claimed
        repo.delete_workspace_if_claimed.return_value = True
        item = _named("already-gone")
        models_client = _make_models_client()
        files_client = _make_mock_files_client()

        if kind == "deployment":
            models_client.list_deployments = AsyncMock(return_value=_MockAsyncPaginatedResponse([item]))
            models_client.delete_deployment = AsyncMock(side_effect=_not_found())
        elif kind == "adapter":
            models_client.list_adapters = AsyncMock(return_value=_MockAsyncPaginatedResponse([item]))
            models_client.delete_adapter = AsyncMock(side_effect=_not_found())
        elif kind == "model":
            models_client.list_models = AsyncMock(return_value=_MockAsyncPaginatedResponse([item]))
            models_client.delete_model = AsyncMock(side_effect=_not_found())
        else:
            files_client.list_filesets = AsyncMock(return_value=_MockAsyncPaginatedResponse([item]))
            files_client.delete_fileset = AsyncMock(side_effect=_not_found())

        controller = _make_controller(workspace_repo=repo)
        with _patch_clients(_make_jobs_client([]), files_client, models_client):
            await controller._async_step()

        repo.delete_workspace_if_claimed.assert_awaited_once_with(
            name=workspace.name,
            expected_updated_at=claimed.updated_at,
        )
        repo.fail_workspace_deletion_if_claimed.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_deletes_the_deployment_that_survived_the_failed_attempt(self):
        pending = _make_workspace()
        failed = _make_workspace(
            updated_at=datetime.now(tz=timezone.utc) - timedelta(seconds=120),
            deletion_stage=WorkspaceDeletionStage.FAILED,
        )
        claimed_pending = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        claimed_retry = _make_workspace(deletion_stage=WorkspaceDeletionStage.DELETING)
        removed = _named("removed")
        remaining = _named("remaining")
        repo = AsyncMock()
        repo.list_workspaces.side_effect = [
            ([pending], None),
            ([], None),
            ([failed], None),
        ]
        repo.claim_workspace_for_deletion.side_effect = [claimed_pending, claimed_retry]
        repo.fail_workspace_deletion_if_claimed.return_value = True
        repo.delete_workspace_if_claimed.return_value = True
        models_client = _make_models_client()
        models_client.list_deployments = AsyncMock(
            side_effect=[
                _MockAsyncPaginatedResponse([removed, remaining]),
                _MockAsyncPaginatedResponse([remaining]),
            ]
        )
        models_client.delete_deployment = AsyncMock(
            side_effect=[None, Exception("HTTP 503: no healthy upstream"), None]
        )
        controller = _make_controller(workspace_repo=repo)

        with _patch_clients(_make_jobs_client([]), _make_mock_files_client([]), models_client):
            await controller._async_step()
            repo.delete_workspace_if_claimed.assert_not_called()
            await controller._async_step()

        assert [call.kwargs["name"] for call in models_client.delete_deployment.await_args_list] == [
            "removed",
            "remaining",
            "remaining",
        ]
        repo.fail_workspace_deletion_if_claimed.assert_awaited_once_with(
            name=pending.name,
            expected_updated_at=claimed_pending.updated_at,
        )
        repo.delete_workspace_if_claimed.assert_awaited_once_with(
            name=pending.name,
            expected_updated_at=claimed_retry.updated_at,
        )

    @pytest.mark.asyncio
    async def test_cleanup_failure_increments_error_counter(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.claim_workspace_for_deletion.return_value = workspace
        repo.fail_workspace_deletion_if_claimed.return_value = True

        jobs_client = MagicMock()
        jobs_client.list_jobs = AsyncMock(side_effect=Exception("boom"))

        controller = _make_controller(workspace_repo=repo)

        with _patch_jobs_client(jobs_client), patch.object(controller._cleanup_errors, "add") as mock_add:
            await controller._async_step()
            mock_add.assert_called_once_with(1, attributes={"error_type": "cleanup_failed"})


class TestWorkspaceCleanupJobs:
    @pytest.mark.asyncio
    async def test_cancels_running_jobs_before_deleting(self):
        workspace = _make_workspace()
        running_job = MagicMock()
        running_job.name = "running-job"
        running_job.status = "active"

        jobs_client = _make_jobs_client([running_job])

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            await controller._cleanup_jobs(workspace)

        jobs_client.cancel_job.assert_awaited_once_with(
            name="running-job",
            workspace="test-workspace",
        )
        jobs_client.delete_job.assert_awaited_once_with(
            name="running-job",
            workspace="test-workspace",
        )

    @pytest.mark.asyncio
    async def test_deletes_completed_jobs_without_cancelling(self):
        workspace = _make_workspace()
        completed_job = MagicMock()
        completed_job.name = "completed-job"
        completed_job.status = "completed"

        jobs_client = _make_jobs_client([completed_job])

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            await controller._cleanup_jobs(workspace)

        jobs_client.cancel_job.assert_not_awaited()
        jobs_client.delete_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_attempts_remaining_jobs_then_raises_on_job_delete_failure(self):
        workspace = _make_workspace()
        job1 = MagicMock()
        job1.name = "fail-job"
        job1.status = "completed"
        job2 = MagicMock()
        job2.name = "ok-job"
        job2.status = "completed"

        jobs_client = _make_jobs_client([job1, job2])
        jobs_client.delete_job = AsyncMock(side_effect=[Exception("fail"), None])

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            with pytest.raises(WorkspaceJobCleanupError):
                await controller._cleanup_jobs(workspace)

        assert jobs_client.delete_job.await_count == 2

    @pytest.mark.asyncio
    async def test_non_terminal_job_after_cancel_is_not_treated_as_deleted(self, monkeypatch: pytest.MonkeyPatch):
        from nhx.core.entities.controllers import workspace_cleanup as workspace_cleanup_module

        monkeypatch.setattr(workspace_cleanup_module, "_JOB_TERMINAL_WAIT_TIMEOUT_SECONDS", 0.0)
        workspace = _make_workspace()
        jobs_client = _make_jobs_client([_make_job("still-running-job", status="active")])
        jobs_client.get_job_status = AsyncMock(return_value=_make_response(_make_job_status("active")))

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            with pytest.raises(WorkspaceJobCleanupError):
                await controller._cleanup_jobs(workspace)

        jobs_client.cancel_job.assert_awaited_once_with(name="still-running-job", workspace="test-workspace")
        jobs_client.delete_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wait_for_terminal_job_emits_heartbeat_between_polls(self, monkeypatch: pytest.MonkeyPatch):
        from nhx.core.entities.controllers import workspace_cleanup as workspace_cleanup_module

        monkeypatch.setattr(workspace_cleanup_module, "_JOB_TERMINAL_WAIT_TIMEOUT_SECONDS", 60.0)
        monkeypatch.setattr(workspace_cleanup_module, "_JOB_TERMINAL_WAIT_POLL_SECONDS", 1.0)
        jobs_client = _make_jobs_client([])
        jobs_client.get_job_status = AsyncMock(
            side_effect=[
                _make_response(_make_job_status("active")),
                _make_response(_make_job_status("cancelled")),
            ]
        )
        controller = _make_controller()

        async def _mark_sleep(_delay: float) -> None:
            return None

        with (
            patch.object(controller, "emit_heartbeat") as emit_heartbeat,
            patch("nhx.core.entities.controllers.workspace_cleanup.asyncio.sleep", side_effect=_mark_sleep),
        ):
            await controller._wait_for_terminal_job(jobs_client, "test-workspace", "active-job")

        emit_heartbeat.assert_called_once()
        assert jobs_client.get_job_status.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_on_list_failure(self):
        workspace = _make_workspace()
        jobs_client = MagicMock()
        jobs_client.list_jobs = AsyncMock(side_effect=Exception("unavailable"))

        controller = _make_controller()

        with pytest.raises(Exception, match="unavailable"):
            with _patch_jobs_client(jobs_client):
                await controller._cleanup_jobs(workspace)


class TestWorkspaceCleanupDeployments:
    @pytest.mark.asyncio
    async def test_deletes_deployments(self):
        workspace = _make_workspace()
        deployment = MagicMock()
        deployment.name = "test-deployment"

        models_client = _make_models_client([deployment])
        controller = _make_controller()
        with patch.object(AsyncModelsClient, "from_client", return_value=models_client):
            await controller._cleanup_deployments(workspace)

        models_client.delete_deployment.assert_awaited_once_with(
            name="test-deployment",
            workspace="test-workspace",
        )

    @pytest.mark.asyncio
    async def test_attempts_remaining_deployments_then_raises(self):
        workspace = _make_workspace()
        dep1 = MagicMock()
        dep1.name = "dep1"
        dep2 = MagicMock()
        dep2.name = "dep2"

        models_client = _make_models_client([dep1, dep2])
        models_client.delete_deployment = AsyncMock(side_effect=[Exception("fail"), None])

        controller = _make_controller()
        with patch.object(AsyncModelsClient, "from_client", return_value=models_client):
            with pytest.raises(WorkspaceResourceCleanupError):
                await controller._cleanup_deployments(workspace)

        assert models_client.delete_deployment.await_count == 2


class TestWorkspaceCleanupModelsAndAdapters:
    @pytest.mark.asyncio
    async def test_deletes_adapters_before_models(self):
        workspace = _make_workspace()
        adapter = MagicMock()
        adapter.name = "test-adapter"
        model = MagicMock()
        model.name = "test-model"
        models_client = _make_models_client(models=[model], adapters=[adapter])
        call_order: list[str] = []
        models_client.delete_adapter = AsyncMock(side_effect=lambda **_: call_order.append("adapter"))
        models_client.delete_model = AsyncMock(side_effect=lambda **_: call_order.append("model"))
        controller = _make_controller()

        with patch.object(AsyncModelsClient, "from_client", return_value=models_client):
            await controller._cleanup_models_and_adapters(workspace)

        assert call_order == ["adapter", "model"]
        models_client.delete_adapter.assert_awaited_once_with(
            name="test-adapter",
            workspace="test-workspace",
        )
        models_client.delete_model.assert_awaited_once_with(
            name="test-model",
            workspace="test-workspace",
        )

    @pytest.mark.asyncio
    async def test_attempts_remaining_models_and_adapters_then_raises(self):
        workspace = _make_workspace()
        adapter1 = MagicMock()
        adapter1.name = "adapter1"
        adapter2 = MagicMock()
        adapter2.name = "adapter2"
        model1 = MagicMock()
        model1.name = "model1"
        model2 = MagicMock()
        model2.name = "model2"
        models_client = _make_models_client(
            models=[model1, model2],
            adapters=[adapter1, adapter2],
        )
        models_client.delete_adapter = AsyncMock(side_effect=[Exception("fail"), None])
        models_client.delete_model = AsyncMock(side_effect=[Exception("fail"), None])
        controller = _make_controller()

        with patch.object(AsyncModelsClient, "from_client", return_value=models_client):
            with pytest.raises(WorkspaceResourceCleanupError):
                await controller._cleanup_models_and_adapters(workspace)

        assert models_client.delete_adapter.await_count == 2
        assert models_client.delete_model.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_on_list_failure(self):
        workspace = _make_workspace()
        models_client = _make_models_client()
        models_client.list_adapters = AsyncMock(side_effect=Exception("unavailable"))
        controller = _make_controller()

        with pytest.raises(Exception, match="unavailable"):
            with patch.object(AsyncModelsClient, "from_client", return_value=models_client):
                await controller._cleanup_models_and_adapters(workspace)


class TestWorkspaceCleanupFilesets:
    @pytest.mark.asyncio
    async def test_deletes_filesets(self):
        workspace = _make_workspace()
        fileset = MagicMock()
        fileset.name = "test-fileset"

        mock_files = _make_mock_files_client([fileset])
        controller = _make_controller()

        with patch.object(AsyncFilesClient, "from_client", return_value=mock_files):
            await controller._cleanup_filesets(workspace)

        mock_files.delete_fileset.assert_awaited_once_with(
            name="test-fileset",
            workspace="test-workspace",
        )

    @pytest.mark.asyncio
    async def test_attempts_remaining_filesets_then_raises(self):
        workspace = _make_workspace()
        fs1 = MagicMock()
        fs1.name = "fs1"
        fs2 = MagicMock()
        fs2.name = "fs2"

        mock_files = _make_mock_files_client([fs1, fs2])
        mock_files.delete_fileset = AsyncMock(side_effect=[Exception("fail"), None])
        controller = _make_controller()

        with patch.object(AsyncFilesClient, "from_client", return_value=mock_files):
            with pytest.raises(WorkspaceResourceCleanupError):
                await controller._cleanup_filesets(workspace)

        assert mock_files.delete_fileset.await_count == 2


class TestJobCancellationBranches:
    """Tests for job cancellation logic — covers branches that were previously dead code."""

    @pytest.mark.asyncio
    async def test_cancels_pending_jobs(self):
        workspace = _make_workspace()
        jobs_client = _make_jobs_client([_make_job("pending-job", status="pending")])

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            await controller._cleanup_jobs(workspace)

        jobs_client.cancel_job.assert_awaited_once_with(name="pending-job", workspace="test-workspace")
        jobs_client.delete_job.assert_awaited_once_with(name="pending-job", workspace="test-workspace")

    @pytest.mark.asyncio
    async def test_cancels_created_jobs(self):
        workspace = _make_workspace()
        jobs_client = _make_jobs_client([_make_job("created-job", status="created")])

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            await controller._cleanup_jobs(workspace)

        jobs_client.cancel_job.assert_awaited_once()
        jobs_client.delete_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_cancel_terminal_jobs(self):
        workspace = _make_workspace()
        jobs = [
            _make_job("done", status="completed"),
            _make_job("failed", status="error"),
            _make_job("stopped", status="cancelled"),
        ]
        jobs_client = _make_jobs_client(jobs)

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            await controller._cleanup_jobs(workspace)

        jobs_client.cancel_job.assert_not_awaited()
        assert jobs_client.delete_job.await_count == 3

    @pytest.mark.asyncio
    async def test_cancel_failure_still_deletes(self):
        """Regression: cancel() throwing must not prevent delete()."""
        workspace = _make_workspace()
        jobs_client = _make_jobs_client([_make_job("flaky-job", status="active")])
        jobs_client.cancel_job = AsyncMock(side_effect=Exception("cancel failed"))

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            await controller._cleanup_jobs(workspace)

        jobs_client.cancel_job.assert_awaited_once()
        jobs_client.delete_job.assert_awaited_once_with(name="flaky-job", workspace="test-workspace")

    @pytest.mark.asyncio
    async def test_mixed_statuses(self):
        workspace = _make_workspace()
        jobs = [
            _make_job("active-job", status="active"),
            _make_job("done-job", status="completed"),
            _make_job("pending-job", status="pending"),
        ]
        jobs_client = _make_jobs_client(jobs)

        controller = _make_controller()
        with _patch_jobs_client(jobs_client):
            await controller._cleanup_jobs(workspace)

        cancel_calls = [c.kwargs["name"] for c in jobs_client.cancel_job.call_args_list]
        assert set(cancel_calls) == {"active-job", "pending-job"}
        assert jobs_client.delete_job.await_count == 3
