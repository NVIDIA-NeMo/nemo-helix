# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for workspace cleanup controller.

Verifies that deleting a workspace triggers asynchronous cleanup of all
associated resources via the background cleanup controller:
- Running/pending jobs (cancelled then deleted via Jobs API)
- Completed jobs (deleted via Jobs API)
- Filesets (explicitly deleted via Files API by the controller)
- Deployment configs and deployments (deployment cleanup requires GPU;
  config entities are cascade-deleted with the workspace)
- Models, projects, guardrail configs (cascade-deleted via FK with workspace)

These tests run with and without auth to verify both flows:
- No auth: workspace returns 404 immediately (deletion_stage set)
- Auth: user loses access immediately (role bindings deleted synchronously)
"""

import time
import uuid
from collections.abc import Callable

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.guardrail.client import GuardrailClient
from nemo_helix_plugin.guardrail.types import CreateGuardrailConfigRequest
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
)
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
)
from nemo_helix_plugin.projects.client import ProjectsClient
from nemo_helix_plugin.projects.types import CreateProjectRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import as_user, unique_email
from nhx.testing.e2e import E2EBackend, wait_for_platform_job
from nhx.testing.pytest_outcomes import pytest_skip

CLEANUP_TIMEOUT = 60.0
CLEANUP_POLL_INTERVAL = 2.0
JOB_SOURCE = "workspace-cleanup-test"

pytestmark = [pytest.mark.skip("Skipping workspace cleanup tests due to orphaned resources issue.")]


def _assert_cleanup_logs(
    backend: E2EBackend | None,
    workspace_name: str,
    expected_cancelled: list[str] | None = None,
    expected_deleted_jobs: list[str] | None = None,
    expected_deleted_filesets: list[str] | None = None,
) -> None:
    """Assert the cleanup controller logged expected operations.

    The e2e test stack does not include an OTLP collector, so traces and
    metrics are not queryable at runtime. Container logs serve as the
    observable record of cleanup behaviour — the controller emits a
    structured INFO line for every cancel/delete it performs.
    """
    if backend is None:
        pytest_skip("Backend is not supported for this test since it is running on a Kubernetes cluster")
        return
    logs = backend.get_logs(tail=None)
    if logs is None:
        pytest_skip("Backend does not support log retrieval")
        return

    stdout, stderr = logs
    combined = stdout + "\n" + stderr

    assert f"Successfully deleted workspace: {workspace_name}" in combined, (
        f"Expected workspace '{workspace_name}' to be successfully deleted in logs"
    )

    for job_name in expected_cancelled or []:
        assert f"Cancelling job: {job_name}" in combined, f"Expected job '{job_name}' to be cancelled in cleanup logs"

    for job_name in expected_deleted_jobs or []:
        assert f"Deleting job: {job_name}" in combined, f"Expected job '{job_name}' to be deleted in cleanup logs"

    for fileset_name in expected_deleted_filesets or []:
        assert f"Deleting fileset: {fileset_name}" in combined, (
            f"Expected fileset '{fileset_name}' to be deleted in cleanup logs"
        )


def _wait_for_workspace_deleted(
    client: NemoClient,
    workspace_name: str,
    timeout: float = CLEANUP_TIMEOUT,
    poll_interval: float = CLEANUP_POLL_INTERVAL,
) -> None:
    """Wait for a workspace to be fully deleted by the cleanup controller.

    Polls by attempting to create a workspace with the same name:
    - 201: old row is gone, cleanup complete (probe workspace is deleted)
    - 409: name still taken, cleanup still in progress
    - anything else: unexpected error, fail immediately

    Raises:
        TimeoutError: If the workspace is not cleaned up within the timeout.
        AssertionError: If the probe request returns an unexpected status code.
    """
    workspaces = WorkspacesClient.from_client(client)
    start = time.time()
    while time.time() - start < timeout:
        try:
            workspace = workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
            if workspace:
                # Old workspace fully removed. Clean up the probe.
                try:
                    workspaces.delete_workspace(name=workspace_name)
                except Exception:
                    pass
                return
        except ConflictError:
            time.sleep(poll_interval)

    raise TimeoutError(
        f"Workspace '{workspace_name}' was not fully deleted within {timeout}s. "
        "Ensure the workspace cleanup controller is running."
    )


def _create_job(
    client: NemoClient,
    workspace: str,
    job_name: str,
    image: Callable[[str], str],
    sleep_seconds: int = 120,
) -> None:
    """Create a platform job that sleeps for the given duration."""
    JobsClient.from_client(client).create_job(
        workspace=workspace,
        body=CreateHelixJobRequest(
            name=job_name,
            source=JOB_SOURCE,
            spec={"cleanup": "test"},
            platform_spec=HelixJobSpec(
                steps=[
                    HelixJobStep(
                        name="cleanup-step",
                        executor=CPUExecutionProviderSpec(
                            provider="cpu",
                            container=ContainerSpec(
                                image=image("nhx-tasks"),
                                command=["sh", "-c", f"sleep {sleep_seconds}"],
                            ),
                        ),
                    ),
                ],
            ),
        ),
    )


def _try_create_deployment(client: NemoClient, workspace: str) -> bool:
    """Best-effort deployment creation for testing the deployment cleanup path.

    Creates a NIM deployment config and deployment. If deployment creation fails
    (e.g., no GPU available), returns False. The deployment config entity will
    still be cascade-deleted with the workspace.

    Returns True if a deployment was successfully created.
    """
    config_name = "cleanup-deploy-config"
    models = ModelsClient.from_client(client)
    try:
        models.create_deployment_config(
            workspace=workspace,
            body=CreateModelDeploymentConfigRequest(
                name=config_name,
                engine=Engine.NIM,
                model_spec=ModelDeploymentConfigModelSpec(model_name="meta/llama-3.2-1b-instruct"),
                executor_config=ContainerExecutorConfig(
                    gpu=1,
                    image_name="nvcr.io/nim/meta/llama-3.2-1b-instruct",
                    image_tag="1.8.6",
                ),
            ),
        )
    except Exception:
        return False

    try:
        models.create_deployment(
            workspace=workspace,
            body=CreateModelDeploymentRequest(name="cleanup-deploy", config=config_name),
        )
        return True
    except Exception:
        return False


def _create_workspace_resources(client: NemoClient, workspace: str, prefix: str) -> None:
    """Create a fileset plus the model, project and guardrail entities that cascade-delete with the workspace."""
    FilesClient.from_client(client).create_fileset(
        workspace=workspace, body=CreateFilesetRequest(name=f"{prefix}-fileset")
    )
    ModelsClient.from_client(client).create_model(
        workspace=workspace, body=CreateModelEntityRequest(name=f"{prefix}-model")
    )
    ProjectsClient.from_client(client).create_project(
        workspace=workspace, body=CreateProjectRequest(name=f"{prefix}-project")
    )
    GuardrailClient.from_client(client).create_guardrail_config(
        workspace=workspace, body=CreateGuardrailConfigRequest(name=f"{prefix}-guardrail")
    )


class TestWorkspaceCleanup:
    """Verify workspace deletion triggers cleanup without auth."""

    def test_cleanup_of_completed_jobs_and_filesets(
        self,
        client: NemoClient,
        image: Callable[[str], str],
        backend: E2EBackend,
    ):
        """Workspace deletion cleans up completed jobs, filesets, and cascade-deleted entities.

        1. Create workspace with a completed job, a fileset, a model,
           a project, a guardrail config, and optionally a NIM deployment
        2. Delete the workspace (marks as pending for async cleanup)
        3. Verify workspace returns 404 immediately
        4. Wait for cleanup controller to fully remove the workspace
        5. Verify cleanup operations in container logs
        """
        workspace_name = f"e2e-cleanup-{uuid.uuid4().hex[:8]}"
        workspaces = WorkspacesClient.from_client(client)
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        try:
            # Create a short job and wait for completion
            _create_job(client, workspace_name, "cleanup-test-job", image, sleep_seconds=1)
            completed_job = wait_for_platform_job(client, "cleanup-test-job", workspace_name)
            assert completed_job.status == "completed", f"Job did not complete: {completed_job.status}"

            # Create a fileset plus entities that are cascade-deleted with the workspace (via FK)
            _create_workspace_resources(client, workspace_name, "cleanup-test")

            # Best-effort: create a NIM deployment (exercises deployment cleanup if GPU available)
            _try_create_deployment(client, workspace_name)

            # Verify resources exist before deletion
            job = JobsClient.from_client(client).get_job(name="cleanup-test-job", workspace=workspace_name).data()
            assert job, "Job should exist before cleanup"

            filesets = FilesClient.from_client(client).list_filesets(workspace=workspace_name).items()
            assert any(f.name == "cleanup-test-fileset" for f in filesets), "Fileset should exist before cleanup"

            model = (
                ModelsClient.from_client(client).get_model(name="cleanup-test-model", workspace=workspace_name).data()
            )
            assert model, "Model should exist before cleanup"

            project = (
                ProjectsClient.from_client(client)
                .get_project(name="cleanup-test-project", workspace=workspace_name)
                .data()
            )
            assert project, "Project should exist before cleanup"

            guardrail = (
                GuardrailClient.from_client(client)
                .get_guardrail_config(name="cleanup-test-guardrail", workspace=workspace_name)
                .data()
            )
            assert guardrail, "Guardrail config should exist before cleanup"

            # Delete workspace - marks as PENDING for async cleanup
            workspaces.delete_workspace(name=workspace_name)

            # Workspace should immediately return 404 (deletion_stage set)
            with pytest.raises(NotFoundError):
                workspaces.get_workspace(name=workspace_name)

            # Wait for the cleanup controller to fully remove the workspace
            _wait_for_workspace_deleted(client, workspace_name)

            # Verify the controller performed expected cleanup operations
            _assert_cleanup_logs(
                backend,
                workspace_name,
                expected_deleted_jobs=["cleanup-test-job"],
                expected_deleted_filesets=["cleanup-test-fileset"],
            )

        except Exception:
            try:
                workspaces.delete_workspace(name=workspace_name)
            except Exception:
                pass
            raise

    def test_cleanup_cancels_running_jobs(
        self,
        client: NemoClient,
        image: Callable[[str], str],
        backend: E2EBackend,
    ):
        """Workspace deletion cancels running jobs before cleanup.

        This is the critical path: the controller must cancel active jobs
        before deleting them, otherwise K8s resources are orphaned.

        1. Create workspace with a long-running platform job
        2. Wait for the job to reach 'active' status
        3. Delete the workspace
        4. Verify workspace returns 404 immediately
        5. Wait for cleanup controller to fully remove the workspace
           (which requires the controller to cancel the running job first)
        6. Verify cancel + delete in container logs
        """
        workspace_name = f"e2e-cleanup-running-{uuid.uuid4().hex[:8]}"
        workspaces = WorkspacesClient.from_client(client)
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        job_name = "long-running-cleanup-job"

        try:
            _create_job(client, workspace_name, job_name, image)

            # Wait for the job to actually be running
            active_job = wait_for_platform_job(
                client,
                job_name,
                workspace_name,
                status_to_check="active",
                timeout=60.0,
            )
            assert active_job.status == "active", f"Job did not become active: {active_job.status}"

            # Delete workspace while job is still running
            workspaces.delete_workspace(name=workspace_name)

            with pytest.raises(NotFoundError):
                workspaces.get_workspace(name=workspace_name)

            # The cleanup controller must cancel the running job, then delete
            # the job, then delete the workspace. If cancellation is broken,
            # this will time out because the job delete will fail or hang.
            _wait_for_workspace_deleted(client, workspace_name)

            # Verify the controller cancelled the active job before deleting it
            _assert_cleanup_logs(
                backend,
                workspace_name,
                expected_cancelled=[job_name],
                expected_deleted_jobs=[job_name],
            )

        except Exception:
            # Best-effort cleanup: try to cancel the job and delete workspace
            try:
                JobsClient.from_client(client).cancel_job(workspace=workspace_name, name=job_name)
            except Exception:
                pass
            try:
                workspaces.delete_workspace(name=workspace_name)
            except Exception:
                pass
            raise

    def test_cleanup_handles_pending_jobs(
        self,
        client: NemoClient,
        image: Callable[[str], str],
        backend: E2EBackend,
    ):
        """Workspace deletion handles jobs that haven't started running yet.

        Creates a job and immediately deletes the workspace. The job may still
        be in 'created' or 'pending' status when the controller processes it.
        The controller should attempt cancellation (non-terminal) then delete.
        """
        workspace_name = f"e2e-cleanup-pending-{uuid.uuid4().hex[:8]}"
        workspaces = WorkspacesClient.from_client(client)
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        job_name = "pending-cleanup-job"

        try:
            _create_job(client, workspace_name, job_name, image)

            # Delete workspace immediately — don't wait for the job to start
            workspaces.delete_workspace(name=workspace_name)

            with pytest.raises(NotFoundError):
                workspaces.get_workspace(name=workspace_name)

            _wait_for_workspace_deleted(client, workspace_name)

            # Job was non-terminal when cleanup ran, so controller should
            # have attempted cancellation before deletion
            _assert_cleanup_logs(
                backend,
                workspace_name,
                expected_cancelled=[job_name],
                expected_deleted_jobs=[job_name],
            )

        except Exception:
            try:
                JobsClient.from_client(client).cancel_job(workspace=workspace_name, name=job_name)
            except Exception:
                pass
            try:
                workspaces.delete_workspace(name=workspace_name)
            except Exception:
                pass
            raise

    def test_cleanup_mixed_resources(
        self,
        client: NemoClient,
        image: Callable[[str], str],
        backend: E2EBackend,
    ):
        """Workspace deletion cleans up a mix of running jobs, completed jobs, filesets, and entities.

        Exercises the full cleanup sequence: jobs (cancel+delete) → deployments → files → entities (FK cascade).
        """
        workspace_name = f"e2e-cleanup-mixed-{uuid.uuid4().hex[:8]}"
        workspaces = WorkspacesClient.from_client(client)
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        running_job_name = "mixed-running-job"

        try:
            # Create a short job and wait for completion
            _create_job(client, workspace_name, "mixed-completed-job", image, sleep_seconds=1)
            completed_job = wait_for_platform_job(client, "mixed-completed-job", workspace_name)
            assert completed_job.status == "completed"

            # Create a standalone fileset plus entities that are cascade-deleted with the workspace (via FK)
            _create_workspace_resources(client, workspace_name, "mixed")

            # Create a long-running job and wait for it to be active
            _create_job(client, workspace_name, running_job_name, image)
            active_job = wait_for_platform_job(
                client,
                running_job_name,
                workspace_name,
                status_to_check="active",
                timeout=60.0,
            )
            assert active_job.status == "active"

            # Best-effort deployment
            _try_create_deployment(client, workspace_name)

            # Delete workspace with all resource types present
            workspaces.delete_workspace(name=workspace_name)

            with pytest.raises(NotFoundError):
                workspaces.get_workspace(name=workspace_name)

            _wait_for_workspace_deleted(client, workspace_name)

            # Verify: running job was cancelled, both jobs deleted, fileset deleted
            _assert_cleanup_logs(
                backend,
                workspace_name,
                expected_cancelled=[running_job_name],
                expected_deleted_jobs=["mixed-completed-job", running_job_name],
                expected_deleted_filesets=["mixed-fileset"],
            )

        except Exception:
            try:
                JobsClient.from_client(client).cancel_job(workspace=workspace_name, name=running_job_name)
            except Exception:
                pass
            try:
                workspaces.delete_workspace(name=workspace_name)
            except Exception:
                pass
            raise


@pytest.mark.feature("auth")
class TestWorkspaceCleanupWithAuth:
    """Verify workspace deletion revokes access and triggers cleanup with auth."""

    def test_deletion_revokes_access_and_cleans_up(
        self,
        client: NemoClient,
        backend: E2EBackend,
    ):
        """With auth, workspace deletion immediately revokes user access.

        1. Create workspace as a user (becomes admin)
        2. Create resources in the workspace
        3. Delete the workspace
        4. Verify user can no longer see the workspace (role bindings deleted)
        5. Wait for cleanup controller to fully remove the workspace
        6. Verify fileset cleanup in container logs
        """
        admin_email = unique_email("cleanup-admin")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        workspace_name = f"e2e-cleanup-{uuid.uuid4().hex[:8]}"

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        try:
            # Create resources as the admin user so the requests carry that user's auth headers.
            # Job cleanup is already verified in the no-auth test above.
            FilesClient.from_client(admin).create_fileset(
                workspace=workspace_name, body=CreateFilesetRequest(name="cleanup-auth-fileset")
            )
            ModelsClient.from_client(admin).create_model(
                workspace=workspace_name, body=CreateModelEntityRequest(name="cleanup-auth-model")
            )
            ProjectsClient.from_client(admin).create_project(
                workspace=workspace_name, body=CreateProjectRequest(name="cleanup-auth-project")
            )
            _try_create_deployment(admin, workspace_name)

            # Delete workspace
            admin_workspaces.delete_workspace(name=workspace_name)

            # User should immediately lose access (role bindings deleted synchronously)
            workspace_names = [ws.name for ws in admin_workspaces.list_workspaces().items()]
            assert workspace_name not in workspace_names, (
                "Workspace should not appear in user's workspace list after deletion"
            )

            # Workspace should be inaccessible via direct retrieval (403 or 404)
            with pytest.raises((NotFoundError, PermissionDeniedError)):
                admin_workspaces.get_workspace(name=workspace_name)

            # Wait for cleanup controller to finish (use the base client for the probe)
            _wait_for_workspace_deleted(client, workspace_name)

            # Verify fileset was cleaned up by the controller
            _assert_cleanup_logs(
                backend,
                workspace_name,
                expected_deleted_filesets=["cleanup-auth-fileset"],
            )

        except Exception:
            try:
                admin_workspaces.delete_workspace(name=workspace_name)
            except Exception:
                pass
            raise
