# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_platform_plugin.client.errors import ConflictError
from nmp.core.entities.controllers.workspace_cleanup import WorkspaceCleanup, WorkspaceJobCleanupError
from nmp.core.entities.entities import Workspace, WorkspaceDeletionStage


def _make_workspace(name: str = "test-workspace") -> Workspace:
    now = datetime.now(tz=timezone.utc)
    return Workspace(
        id="ws-123",
        name=name,
        description="test",
        created_at=now,
        updated_at=now,
    )


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

    Production routes jobs calls through ``client_from_platform(sdk, AsyncJobsClient)``
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
    ``client_from_platform(sdk, AsyncModelsClient)``.
    """
    models_client = MagicMock()
    models_client.list_deployments = AsyncMock(return_value=_MockAsyncPaginatedResponse(deployments or []))
    models_client.delete_deployment = AsyncMock()
    models_client.list_models = AsyncMock(return_value=_MockAsyncPaginatedResponse(models or []))
    models_client.delete_model = AsyncMock()
    models_client.list_adapters = AsyncMock(return_value=_MockAsyncPaginatedResponse(adapters or []))
    models_client.delete_adapter = AsyncMock()
    return models_client


_CLIENT_FROM_PLATFORM_PATCH = "nmp.core.entities.controllers.workspace_cleanup.client_from_platform"


def _patch_jobs_client(jobs_client: MagicMock):
    """Patch ``client_from_platform`` to dispatch by requested client class.

    Returns *jobs_client* for ``AsyncJobsClient`` and safe empty mocks for the
    deployment/fileset clients. Dispatching by class (rather than returning
    *jobs_client* for every ``client_from_platform`` call) keeps ``_async_step``
    tests correct even when ``_cleanup_jobs`` succeeds and execution proceeds to
    ``_cleanup_deployments`` and ``_cleanup_filesets``.
    """
    return _patch_clients(jobs_client, _make_mock_files_client([]), _make_models_client([]))


def _patch_clients(jobs_client: MagicMock, files_client: MagicMock, models_client: MagicMock | None = None):
    """Patch ``client_from_platform`` to dispatch by requested client class.

    ``_async_step`` cleans up jobs, deployments, models/adapters, and filesets,
    so it calls ``client_from_platform`` for ``AsyncJobsClient``,
    ``AsyncModelsClient``, and ``AsyncFilesClient`` — return the matching mock.
    """
    from nemo_platform_plugin.files.client import AsyncFilesClient
    from nemo_platform_plugin.jobs.client import AsyncJobsClient
    from nemo_platform_plugin.models.client import AsyncModelsClient

    models = models_client if models_client is not None else _make_models_client([])

    def _dispatch(_sdk, client_cls):
        if client_cls is AsyncFilesClient:
            return files_client
        if client_cls is AsyncJobsClient:
            return jobs_client
        if client_cls is AsyncModelsClient:
            return models
        raise AssertionError(f"unexpected client class: {client_cls!r}")

    return patch(_CLIENT_FROM_PLATFORM_PATCH, side_effect=_dispatch)


def _make_job(name: str, status: str = "completed") -> MagicMock:
    job = MagicMock()
    job.name = name
    job.status = status
    return job


def _make_controller(
    workspace_repo: AsyncMock | None = None,
    nmp_sdk: MagicMock | None = None,
) -> WorkspaceCleanup:
    if workspace_repo is None:
        workspace_repo = AsyncMock()
    if nmp_sdk is None:
        nmp_sdk = MagicMock()

    return WorkspaceCleanup(
        nmp_sdk=nmp_sdk,
        workspace_repository=workspace_repo,
    )


_FILES_CLIENT_PATCH = "nmp.core.entities.controllers.workspace_cleanup.client_from_platform"


class TestWorkspaceCleanupStep:
    def test_step_skips_when_stop_signal_set(self):
        import threading

        stop = threading.Event()
        stop.set()
        repo = AsyncMock()
        controller = WorkspaceCleanup(
            nmp_sdk=MagicMock(),
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
            nmp_sdk=MagicMock(),
            workspace_repository=repo,
            loop=loop,
        )

        assert controller._loop is loop
        controller.step()
        assert controller.is_healthy
        repo.list_workspaces.assert_called_once()
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

        repo.mark_workspace_for_deletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_workspace_deletion(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.mark_workspace_for_deletion.return_value = True

        sdk = MagicMock()
        mock_files = _make_mock_files_client([])
        controller = _make_controller(workspace_repo=repo, nmp_sdk=sdk)

        with _patch_clients(_make_jobs_client([]), mock_files, _make_models_client([])):
            await controller._async_step()

        repo.mark_workspace_for_deletion.assert_any_call(
            name="test-workspace",
            deletion_stage=WorkspaceDeletionStage.DELETING,
        )
        repo.delete_workspace.assert_awaited_once_with(name="test-workspace")

    @pytest.mark.asyncio
    async def test_deletes_models_and_adapters_before_filesets(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.mark_workspace_for_deletion.return_value = True

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
        repo.delete_workspace.assert_awaited_once_with(name="test-workspace")

    @pytest.mark.asyncio
    async def test_workspace_already_being_processed(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.mark_workspace_for_deletion.return_value = False

        controller = _make_controller(workspace_repo=repo)

        await controller._async_step()

        repo.delete_workspace.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_failure_marks_workspace_failed(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.mark_workspace_for_deletion.return_value = True

        jobs_client = MagicMock()
        jobs_client.list_jobs = AsyncMock(side_effect=Exception("jobs service down"))

        controller = _make_controller(workspace_repo=repo)

        with _patch_jobs_client(jobs_client):
            await controller._async_step()

        repo.mark_workspace_for_deletion.assert_any_call(
            name="test-workspace",
            deletion_stage=WorkspaceDeletionStage.FAILED,
        )
        repo.delete_workspace.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_failure_increments_error_counter(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.mark_workspace_for_deletion.return_value = True

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
        from nmp.core.entities.controllers import workspace_cleanup as workspace_cleanup_module

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
        from nmp.core.entities.controllers import workspace_cleanup as workspace_cleanup_module

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
            patch("nmp.core.entities.controllers.workspace_cleanup.asyncio.sleep", side_effect=_mark_sleep),
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
        with patch(_CLIENT_FROM_PLATFORM_PATCH, return_value=models_client):
            await controller._cleanup_deployments(workspace)

        models_client.delete_deployment.assert_awaited_once_with(
            name="test-deployment",
            workspace="test-workspace",
        )

    @pytest.mark.asyncio
    async def test_continues_on_individual_deployment_failure(self):
        workspace = _make_workspace()
        dep1 = MagicMock()
        dep1.name = "dep1"
        dep2 = MagicMock()
        dep2.name = "dep2"

        models_client = _make_models_client([dep1, dep2])
        models_client.delete_deployment = AsyncMock(side_effect=[Exception("fail"), None])

        controller = _make_controller()
        with patch(_CLIENT_FROM_PLATFORM_PATCH, return_value=models_client):
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

        with patch(_CLIENT_FROM_PLATFORM_PATCH, return_value=models_client):
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
    async def test_continues_on_individual_model_and_adapter_failure(self):
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

        with patch(_CLIENT_FROM_PLATFORM_PATCH, return_value=models_client):
            await controller._cleanup_models_and_adapters(workspace)

        assert models_client.delete_adapter.await_count == 2
        assert models_client.delete_model.await_count == 2

    @pytest.mark.asyncio
    async def test_refused_model_delete_blocks_workspace_delete(self):
        workspace = _make_workspace()
        repo = AsyncMock()
        repo.list_workspaces.return_value = ([workspace], None)
        repo.mark_workspace_for_deletion.return_value = True
        shared = MagicMock()
        shared.name = "shared-base"
        other = MagicMock()
        other.name = "other"
        models_client = _make_models_client(models=[shared, other])
        refusal = ConflictError(httpx.Response(409, json={"detail": "has child entities in other workspaces"}))
        models_client.delete_model = AsyncMock(side_effect=[refusal, None])
        controller = _make_controller(workspace_repo=repo)

        with _patch_clients(_make_jobs_client([]), _make_mock_files_client([]), models_client):
            await controller._async_step()

        assert models_client.delete_model.await_count == 2
        repo.delete_workspace.assert_not_called()
        repo.mark_workspace_for_deletion.assert_any_call(
            name="test-workspace",
            deletion_stage=WorkspaceDeletionStage.FAILED,
        )

    @pytest.mark.asyncio
    async def test_raises_on_list_failure(self):
        workspace = _make_workspace()
        models_client = _make_models_client()
        models_client.list_adapters = AsyncMock(side_effect=Exception("unavailable"))
        controller = _make_controller()

        with pytest.raises(Exception, match="unavailable"):
            with patch(_CLIENT_FROM_PLATFORM_PATCH, return_value=models_client):
                await controller._cleanup_models_and_adapters(workspace)


class TestWorkspaceCleanupFilesets:
    @pytest.mark.asyncio
    async def test_deletes_filesets(self):
        workspace = _make_workspace()
        fileset = MagicMock()
        fileset.name = "test-fileset"

        mock_files = _make_mock_files_client([fileset])
        controller = _make_controller()

        with patch(_FILES_CLIENT_PATCH, return_value=mock_files):
            await controller._cleanup_filesets(workspace)

        mock_files.delete_fileset.assert_awaited_once_with(
            name="test-fileset",
            workspace="test-workspace",
        )

    @pytest.mark.asyncio
    async def test_continues_on_individual_fileset_failure(self):
        workspace = _make_workspace()
        fs1 = MagicMock()
        fs1.name = "fs1"
        fs2 = MagicMock()
        fs2.name = "fs2"

        mock_files = _make_mock_files_client([fs1, fs2])
        mock_files.delete_fileset = AsyncMock(side_effect=[Exception("fail"), None])
        controller = _make_controller()

        with patch(_FILES_CLIENT_PATCH, return_value=mock_files):
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
