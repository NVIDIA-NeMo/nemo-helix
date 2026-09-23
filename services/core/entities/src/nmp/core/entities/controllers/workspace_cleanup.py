# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading

from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.models.client import AsyncModelsClient
from nmp.common.api.filter import ComparisonOperation, FilterOperator
from nmp.common.controller.controller import Controller, HeartbeatMixin
from nmp.common.observability import start_span_with_ctx
from nmp.core.entities.app.ctx import WorkspaceCleanupContext
from nmp.core.entities.app.repository.workspace import WorkspaceRepositoryInterface
from nmp.core.entities.entities import Workspace, WorkspaceDeletionStage
from opentelemetry import metrics, trace

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
logger = logging.getLogger(__name__)

_TERMINAL_JOB_STATUSES: frozenset[PlatformJobStatus] = frozenset(
    {PlatformJobStatus.COMPLETED, PlatformJobStatus.ERROR, PlatformJobStatus.CANCELLED}
)
_JOB_TERMINAL_WAIT_TIMEOUT_SECONDS = 300.0
_JOB_TERMINAL_WAIT_POLL_SECONDS = 2.0


class WorkspaceJobCleanupError(RuntimeError):
    """Raised when workspace cleanup cannot delete every job."""


class WorkspaceModelCleanupError(RuntimeError):
    """Raised when a model refuses deletion, so the workspace delete must not cascade over it."""


def _job_status_value(status: object) -> PlatformJobStatus | str:
    value = getattr(status, "value", status)
    try:
        return PlatformJobStatus(str(value))
    except ValueError:
        return str(value or "")


def _job_status_is_terminal(status: object) -> bool:
    return _job_status_value(status) in _TERMINAL_JOB_STATUSES


class WorkspaceCleanup(HeartbeatMixin, Controller):
    def __init__(
        self,
        nmp_sdk: AsyncNeMoPlatform,
        workspace_repository: WorkspaceRepositoryInterface,
        stop_signal: threading.Event | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._nmp_sdk = nmp_sdk
        self._workspace_repository = workspace_repository
        self._stop_signal = stop_signal
        self._is_healthy = False
        self._loop = loop or asyncio.new_event_loop()

        self._cleanup_total = meter.create_counter(
            name="nmp.entities.workspace.cleanup.total",
            description="Total number of workspace cleanup attempts",
        )
        self._cleanup_errors = meter.create_counter(
            name="nmp.entities.workspace.cleanup.errors",
            description="Number of workspace cleanup errors",
        )

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def step(self):
        if self._stop_signal and self._stop_signal.is_set():
            logger.debug("Stop signal received, skipping cleanup step")
            return
        logger.debug("Running workspace cleanup routine")
        try:
            self._loop.run_until_complete(self._async_step())
            self._is_healthy = True
        except Exception as e:
            logger.error(f"Workspace cleanup step failed: {e}", exc_info=True)
            self._is_healthy = False

    async def _async_step(self):
        with tracer.start_as_current_span("workspace_cleanup/fetch_pending"):
            cleanup_filter = ComparisonOperation(
                operator=FilterOperator.EQ,
                field="deletion_stage",
                value=WorkspaceDeletionStage.PENDING,
            )
            workspaces, _ = await self._workspace_repository.list_workspaces(
                filter_op=cleanup_filter,
                page_size=1,
            )
            if not workspaces:
                logger.debug("No workspaces pending deletion")
                return

        workspace = workspaces[0]
        self._cleanup_total.add(1)

        with start_span_with_ctx(
            tracer,
            "workspace_cleanup/delete_workspace",
            WorkspaceCleanupContext(workspace_name=workspace.name),
        ):
            logger.info(f"Processing workspace deletion: {workspace.name}")

            try:
                success = await self._workspace_repository.mark_workspace_for_deletion(
                    name=workspace.name,
                    deletion_stage=WorkspaceDeletionStage.DELETING,
                )
                if not success:
                    logger.info(f"Workspace already being processed: {workspace.name}")
                    return

                await self._cleanup_jobs(workspace)
                self.emit_heartbeat()
                await self._cleanup_deployments(workspace)
                self.emit_heartbeat()
                # Models and adapters can hold fileset refs in this workspace.
                # Delete them before filesets so same-workspace teardown is not
                # blocked by the fileset DELETE 409 referential guard.
                await self._cleanup_models_and_adapters(workspace)
                self.emit_heartbeat()
                await self._cleanup_filesets(workspace)
                self.emit_heartbeat()

                await self._workspace_repository.delete_workspace(name=workspace.name)
                logger.info(f"Successfully deleted workspace: {workspace.name}")

            except Exception as e:
                logger.error(
                    f"Failed to cleanup workspace {workspace.name}: {e}",
                    exc_info=True,
                )
                self._cleanup_errors.add(
                    1,
                    attributes={"error_type": "cleanup_failed"},
                )
                await self._workspace_repository.mark_workspace_for_deletion(
                    name=workspace.name,
                    deletion_stage=WorkspaceDeletionStage.FAILED,
                )

    @tracer.start_as_current_span("workspace_cleanup/cleanup_jobs")
    async def _cleanup_jobs(self, workspace: Workspace) -> None:
        logger.info(f"Cleaning up jobs for workspace: {workspace.name}")
        try:
            jobs_client = client_from_platform(self._nmp_sdk, AsyncJobsClient)
            jobs = [job async for job in (await jobs_client.list_jobs(workspace=workspace.name)).items()]

            cleanup_errors: list[Exception] = []
            for job in jobs:
                try:
                    if not _job_status_is_terminal(job.status):
                        cancel_succeeded = False
                        try:
                            logger.info(f"Cancelling job: {job.name}")
                            await jobs_client.cancel_job(
                                name=job.name,
                                workspace=workspace.name,
                            )
                            cancel_succeeded = True
                        except Exception as e:
                            logger.warning(f"Failed to cancel job {job.name}: {e}")
                        if cancel_succeeded:
                            await self._wait_for_terminal_job(jobs_client, workspace.name, job.name)

                    logger.info(f"Deleting job: {job.name}")
                    await jobs_client.delete_job(
                        name=job.name,
                        workspace=workspace.name,
                    )
                except NotFoundError:
                    logger.info(f"Job already deleted: {job.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete job {job.name}: {e}")
                    cleanup_errors.append(e)
                finally:
                    self.emit_heartbeat()

            if cleanup_errors:
                raise WorkspaceJobCleanupError(
                    f"Failed to delete {len(cleanup_errors)} job(s) while cleaning workspace {workspace.name}"
                )

        except Exception as e:
            logger.error(f"Failed to cleanup jobs for workspace {workspace.name}: {e}")
            raise

    async def _wait_for_terminal_job(self, jobs_client: AsyncJobsClient, workspace: str, job_name: str) -> None:
        """Wait until a cancelled job is terminal before hard deletion."""
        deadline = asyncio.get_running_loop().time() + _JOB_TERMINAL_WAIT_TIMEOUT_SECONDS
        last_status: PlatformJobStatus | str = ""
        while True:
            try:
                response = await jobs_client.get_job_status(name=job_name, workspace=workspace)
            except NotFoundError:
                return
            status_info = response.data()
            last_status = _job_status_value(status_info.status)
            if last_status in _TERMINAL_JOB_STATUSES:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for job {job_name} in workspace {workspace} to reach a terminal status; "
                    f"last status was {last_status}"
                )
            self.emit_heartbeat()
            await asyncio.sleep(_JOB_TERMINAL_WAIT_POLL_SECONDS)

    @tracer.start_as_current_span("workspace_cleanup/cleanup_deployments")
    async def _cleanup_deployments(self, workspace: Workspace) -> None:
        logger.info(f"Cleaning up deployments for workspace: {workspace.name}")
        try:
            models_client = client_from_platform(self._nmp_sdk, AsyncModelsClient)
            deployments_response = await models_client.list_deployments(workspace=workspace.name)
            deployments = [deployment async for deployment in deployments_response.items()]

            for deployment in deployments:
                try:
                    logger.info(f"Deleting deployment: {deployment.name}")
                    await models_client.delete_deployment(
                        name=deployment.name,
                        workspace=workspace.name,
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete deployment {deployment.name}: {e}")
                finally:
                    self.emit_heartbeat()

        except Exception as e:
            logger.error(f"Failed to list deployments for workspace {workspace.name}: {e}")
            raise

    @tracer.start_as_current_span("workspace_cleanup/cleanup_models_and_adapters")
    async def _cleanup_models_and_adapters(self, workspace: Workspace) -> None:
        logger.info(f"Cleaning up models and adapters for workspace: {workspace.name}")
        try:
            models_client = client_from_platform(self._nmp_sdk, AsyncModelsClient)
            adapters_response = await models_client.list_adapters(workspace=workspace.name)
            adapters = [adapter async for adapter in adapters_response.items()]
            models_response = await models_client.list_models(workspace=workspace.name)
            models = [model async for model in models_response.items()]

            for adapter in adapters:
                try:
                    logger.info(f"Deleting adapter: {adapter.name}")
                    await models_client.delete_adapter(
                        name=adapter.name,
                        workspace=workspace.name,
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete adapter {adapter.name}: {e}")
                finally:
                    self.emit_heartbeat()

            refused: list[str] = []
            for model in models:
                try:
                    logger.info(f"Deleting model: {model.name}")
                    await models_client.delete_model(
                        name=model.name,
                        workspace=workspace.name,
                    )
                except ConflictError as e:
                    logger.warning(f"Model {model.name} refused deletion: {e.detail}")
                    refused.append(model.name)
                except Exception as e:
                    logger.warning(f"Failed to delete model {model.name}: {e}")
                finally:
                    self.emit_heartbeat()

            if refused:
                raise WorkspaceModelCleanupError(
                    f"Model(s) {', '.join(refused)} in workspace {workspace.name} refused deletion"
                )

        except Exception as e:
            logger.error(f"Failed to clean up models or adapters for workspace {workspace.name}: {e}")
            raise

    @tracer.start_as_current_span("workspace_cleanup/cleanup_filesets")
    async def _cleanup_filesets(self, workspace: Workspace) -> None:
        logger.info(f"Cleaning up filesets for workspace: {workspace.name}")
        try:
            files = client_from_platform(self._nmp_sdk, AsyncFilesClient)
            filesets_response = await files.list_filesets(workspace=workspace.name)

            async for fileset in filesets_response.items():
                try:
                    logger.info(f"Deleting fileset: {fileset.name}")
                    await files.delete_fileset(
                        name=fileset.name,
                        workspace=workspace.name,
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete fileset {fileset.name}: {e}")
                finally:
                    self.emit_heartbeat()

        except Exception as e:
            logger.error(f"Failed to list filesets for workspace {workspace.name}: {e}")
            raise
