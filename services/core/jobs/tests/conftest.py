# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.capabilities import reset_capability_cache
from nemo_helix_plugin.jobs.api_factory import ContainerSpec as FactoryContainerSpec
from nemo_helix_plugin.jobs.api_factory import CPUExecutionProviderSpec as FactoryCPUExecutionProviderSpec
from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable as FactoryEnvironmentVariable
from nemo_helix_plugin.jobs.api_factory import HelixJobSpec as FactoryHelixJobSpec
from nemo_helix_plugin.jobs.api_factory import HelixJobStep as FactoryHelixJobStep
from nemo_helix_plugin.jobs.api_factory import job_route_factory
from nhx.common.config import Configuration, ImagePullSecret, HelixConfig
from nhx.common.entities.client import EntityClient
from nhx.common.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nhx.common.jobs.file_manager import FileStorageType, TmpDirPath
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.common.service.dependencies import get_entity_client
from nhx.core.jobs.api.dependencies import dep_dispatcher
from nhx.core.jobs.api.v2.jobs.endpoints import router
from nhx.core.jobs.api.v2.jobs.rerun import router as rerun_router
from nhx.core.jobs.api.v2.jobs.schemas import (
    CreateHelixJobRequest,
    HelixJobResponse,
    HelixJobStepWithContext,
)
from nhx.core.jobs.app.dispatcher import JobDispatcher
from nhx.core.jobs.app.providers import ContainerSpec, CPUExecutionProvider
from nhx.core.jobs.app.schemas import (
    HelixJobEnvironmentVariable,
    HelixJobStepSpec,
)
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.config import JobsServiceConfig
from nhx.core.jobs.controllers.backends.registry import BackendKey, BackendRegistry
from nhx.core.jobs.controllers.backends.subprocess import SubprocessJobBackend
from nhx.core.jobs.controllers.backends.test import (
    MockDockerCPUJobBackend,
    MockDockerGPUJobBackend,
    MockKubernetesCPUJobBackend,
    MockKubernetesGPUJobBackend,
)
from nhx.core.jobs.entities import (
    HelixJob,
    HelixJobAttempt,
    HelixJobResult,
)
from nhx.testing import create_test_client, subprocess_job_executor_patch
from nhx.testing.blockbuster import blockbuster_fixture
from pydantic import BaseModel
from pytest import fixture

# Enable BlockBuster to detect blocking calls in async code
blockbuster = blockbuster_fixture(autouse=True)

# ============================================================================
# Pytest Hooks
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_capability_cache() -> Iterator[None]:
    """Prevent probe_docker process cache from poisoning jobs unit tests."""
    reset_capability_cache()
    yield
    reset_capability_cache()


@pytest.fixture
def docker_probe_unavailable():
    """Patch probe_docker as unavailable on merge + registry import paths."""
    from nemo_helix_plugin.capabilities import ProbeResult

    result = ProbeResult(available=False, detail="down")
    with (
        patch("nhx.core.jobs.controllers.backends.config.probe_docker", return_value=result),
        patch("nhx.core.jobs.controllers.backends.registry.probe_docker", return_value=result),
    ):
        yield result


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Auto-marks tests based on their location:
    - Tests in e2e/ directories get the 'e2e' marker
    - Tests in integration/ directories get the 'integration' marker
    - Tests without category markers get the 'unit' marker
    """
    # Category markers that determine test type
    category_markers = {"unit", "e2e", "integration", "regression", "canary", "slow", "skip_in_ci"}

    for item in items:
        # Get current marker names
        marker_names = {marker.name for marker in item.iter_markers()}

        # Auto-mark tests in e2e directories
        if "/e2e/" in str(item.fspath):
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
                marker_names.add("e2e")

        # Auto-mark tests in integration directories
        elif "/integration/" in str(item.fspath):
            if "integration" not in marker_names:
                item.add_marker(pytest.mark.integration)
                marker_names.add("integration")

        # Auto-mark tests without category markers as unit tests
        if not marker_names.intersection(category_markers):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="function")
def mock_store():
    """Create an EntityClient for testing backed by in-memory storage.

    Uses create_test_client with EntitiesService in-memory backend
    to avoid issues with nested ASGITransport contexts.
    """
    # Include projects used in tests (proj-1234 used by test_hello_world_jobs_list)
    projects = ["default/test-project", "default/proj-1234"]
    with create_test_client(client_type=EntityClient, projects=projects) as client:
        yield client


@pytest_asyncio.fixture()
async def mock_dispatcher(mock_store, mock_nhx_client) -> JobDispatcher:
    """Create a JobDispatcher instance for testing with mock EntityStore."""
    return JobDispatcher(
        store=mock_store,
        sdk=mock_nhx_client,
    )


@pytest.fixture
def sample_platform_job_request() -> CreateHelixJobRequest:
    return CreateHelixJobRequest(
        name="test-job",
        description=TestConstants.DESC_TEST,
        project=TestConstants.PROJECT,
        source=TestConstants.SOURCE,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=TestConstants.PLATFORM_SPEC,
        ownership=TestConstants.OWNERSHIP_BASIC,
        custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
    )


@pytest.fixture
def sample_platform_job(sample_platform_job_request: CreateHelixJobRequest) -> HelixJobResponse:
    """Create a sample HelixJobResponse for testing."""
    pjr = sample_platform_job_request
    return HelixJobResponse(
        id="job-123",
        attempt_id="attempt-1",
        name="test-job",
        workspace="default",
        fileset="test-logs-fileset",
        description=pjr.description,
        project=pjr.project,
        source=pjr.source,
        spec=pjr.spec,
        platform_spec=pjr.platform_spec,
        status=HelixJobStatus.CREATED,
        ownership=pjr.ownership,
        custom_fields=pjr.custom_fields,
    )


@pytest.fixture
def complex_platform_job() -> HelixJob:
    """Create a complex HelixJob with nested JSON structures for comprehensive testing."""
    return HelixJob(
        name="complex-job",
        workspace="default",
        fileset="test-logs-fileset",
        source=TestConstants.SOURCE,
        spec=TestConstants.SPEC_COMPLEX,
        platform_spec=TestConstants.PLATFORM_SPEC,
        custom_fields=TestConstants.CUSTOM_FIELDS_COMPLEX,
    )


@pytest.fixture
def sample_platform_job_result():
    return HelixJobResult(
        name="test-job-result-name",
        workspace="default",
        job="test-job-id",
        artifact_url="default/test-fileset#path",
        artifact_storage_type=FileStorageType.FILESET,
    )


@pytest.fixture
def sample_platform_job_attempt():
    """Create a sample HelixJobAttempt for testing."""
    return HelixJobAttempt(
        name="test-attempt",
        workspace="default",
        job="test-job-id",
        seq=0,
        status=HelixJobStatus.CREATED,
        status_details=TestConstants.STATUS_DETAILS_QUEUED,
        spec=TestConstants.SPEC_BASIC,
        platform_spec=TestConstants.PLATFORM_SPEC,
    )


@pytest.fixture
def sample_job_dict():
    """Create a sample job dictionary matching test_registry.py structure."""
    return {
        "name": "docker-test-job",
        "workspace": "default",
        "source": "curl-test",
        "fileset": "test-logs-fileset",
        "spec": {"parameters": {"test_param": "test_value"}},
        "platform_spec": {
            "steps": [
                {
                    "name": "docker-step-cpu-1",
                    "executor": {
                        "provider": "cpu",
                        "profile": "default",
                        "container": {"image": "ubuntu:latest", "command": ["c1", "c2"], "entrypoint": ["a1", "a2"]},
                    },
                    "environment": [{"name": "TEST_ENV", "value": "test_value"}],
                },
                {
                    "name": "docker-step-gpu",
                    "executor": {
                        "provider": "gpu",
                        "profile": "default",
                        "container": {"image": "ubuntu:latest", "command": ["c1", "c2"], "entrypoint": ["a1", "a2"]},
                        "resources": {"num_gpus": 2},
                    },
                    "environment": [{"name": "TEST_ENV", "value": "test_value"}],
                },
                {
                    "name": "docker-step-no-command-or-entrypoint",
                    "executor": {
                        "provider": "cpu",
                        "profile": "default",
                        "container": {"image": "ubuntu:latest"},
                    },
                    "environment": [{"name": "TEST_ENV", "value": "test_value"}],
                },
            ]
        },
    }


@fixture
def _mock_files_client():
    """Create a mock AsyncFilesClient for testing."""
    mock_files = AsyncMock()
    mock_fileset = MagicMock()
    mock_fileset.name = "test-fileset-id"
    mock_response = MagicMock()
    mock_response.data.return_value = mock_fileset
    mock_files.create_fileset.return_value = mock_response
    mock_files.delete_fileset.return_value = None
    return mock_files


# Controller modules that import ``client_from_platform`` to build a typed Jobs
# client. The fixture patches it in each so the shared ``mock_jobs`` client is
# returned for ``JobsClient`` requests.
#
# The backends (docker/subprocess/kubernetes_job) build their client once in
# ``JobBackend.__init__`` (base module) and reuse it via ``self._jobs``, so they
# no longer import ``client_from_platform`` directly — patching ``base`` covers
# them. ``common`` has a standalone helper that still builds its own client.
_JOBS_CLIENT_CONTROLLER_MODULES = (
    "nhx.core.jobs.controllers.scheduler",
    "nhx.core.jobs.controllers.reconciler",
    "nhx.core.jobs.controllers.diagnostics",
    "nhx.core.jobs.controllers.backends.base",
    "nhx.core.jobs.controllers.backends.kubernetes.common",
)


@fixture
def mock_jobs_client():
    """Mock of the typed ``JobsClient`` used by the controllers.

    Methods return ``.data()``/``.items()``-aware responses so call sites like
    ``client_from_platform(sdk, JobsClient).get_job_step(...).data()`` work. Tests
    set ``.return_value`` on the individual methods and assert against them.
    """
    return MagicMock()


@fixture
def mock_nhx_client(_mock_files_client, mock_jobs_client):
    """Create a flexible mock of NeMoHelix for testing.

    ``client_from_platform`` is patched in the dispatcher (returns the files client)
    and in every controller module that builds a typed Jobs client. The controller
    patches dispatch on the requested client type: ``JobsClient`` requests resolve to
    ``mock_jobs_client``; anything else falls back to the files client.
    """
    mock_client = MagicMock()
    mock_client.beta = MagicMock()

    from nemo_helix_plugin.jobs.client import JobsClient

    def _dispatch(_sdk, client_type):
        if client_type is JobsClient:
            return mock_jobs_client
        return _mock_files_client

    patchers = [patch("nhx.core.jobs.app.dispatcher.client_from_platform", return_value=_mock_files_client)]
    patchers += [
        patch(f"{module}.client_from_platform", side_effect=_dispatch) for module in _JOBS_CLIENT_CONTROLLER_MODULES
    ]
    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield mock_client


@fixture
def mock_result_manager(tmp_path: Path):
    m = MagicMock()
    m._tmp_dir = tmp_path / "mydir"
    m._tmp_dir.mkdir()
    m._path = m._tmp_dir / "myfile.txt"
    m._path.write_text("hello world")

    async def download_artifact(artifact_url: str, local_dir: str | Path | None = None) -> TmpDirPath:
        return TmpDirPath(path=m._path, tmp_dir=m._tmp_dir)

    m.download_artifact = download_artifact

    return m


@pytest.fixture
def test_step_pending() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.PENDING)


@pytest.fixture
def test_step_active() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.ACTIVE)


@pytest.fixture
def test_step_completed() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.COMPLETED)


@pytest.fixture
def test_step_error() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.ERROR)


@pytest.fixture
def test_step_pausing() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.PAUSING)


@pytest.fixture
def test_step_paused() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.PAUSED)


@pytest.fixture
def test_step_resuming() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.RESUMING)


@pytest.fixture
def test_step_cancelling() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.CANCELLING)


@pytest.fixture
def test_step_cancelled() -> HelixJobStepWithContext:
    return create_step_with_status(HelixJobStatus.CANCELLED)


def create_step_with_status(status: HelixJobStatus) -> HelixJobStepWithContext:
    return HelixJobStepWithContext(
        id="test-step-id",
        job="test-job-id",
        workspace="default",
        attempt_id="test-job-attempt-id",
        name="test-step",
        fileset="test-logs-fileset",
        step_spec=HelixJobStepSpec(
            name="test-step",
            executor=CPUExecutionProvider(
                provider="cpu", profile="default", container=ContainerSpec(image="test-image")
            ),
            config={},
            environment=[
                HelixJobEnvironmentVariable(name="ENV_VAR", value="test_value"),
                HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/var/test"),
                HelixJobEnvironmentVariable(name=EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, value="/var/tmp"),
            ],
        ),
        status=status,
        created_at=datetime.datetime.now(datetime.timezone.utc),
        updated_at=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.fixture
def mock_platform_config() -> HelixConfig:
    """Real HelixConfig for controller tests (get_service_url, loopback_address, to_shared_envvars work correctly)."""
    return HelixConfig(  # type: ignore[abstract]
        base_url="http://localhost:8080",
        files_url="http://localhost:8080",
        image_pull_secrets=[ImagePullSecret(name="global-pull-secret")],
        loopback_address=None,
    )


@pytest.fixture
def job_config_with_many_profiles() -> JobsServiceConfig:
    # Define the YAML content as a variable
    yaml_content = """# This is the default configuration for the Jobs microservice.
platform:
  base_url: "http://localhost:8080"
  runtime: "kubernetes"

executor_defaults:
  kubernetes_job:
    storage:
      pvc_name: test_jobs_storage
  docker:
    storage:
      volume_name: test_jobs_storage

jobs:
  # Executor profiles configuration. The subprocess/default entry mirrors what
  # ships in `packages/nhx_platform/config/local.yaml` and opts the documented
  # `cpu/default` plugin steps into the cpu→subprocess translation in the Jobs
  # API (see `translate_cpu_container_steps_to_subprocess`). Tests that submit
  # jobs through the core /apis/jobs/v2/workspaces/{ws}/jobs endpoint with a
  # `cpu/default` step will get rewritten to `subprocess/default` before
  # validation, matching production deployment behavior.
  executors:
    - provider: subprocess
      profile: default
      backend: subprocess
      config:
        working_directory: /tmp/nhx-subprocess-jobs
    - provider: cpu
      profile: default
      backend: docker
      config:
        storage:
          volume_name: test_jobs_storage
    - provider: gpu
      profile: docker_gpu
      backend: docker
      config:
        storage:
          volume_name: test_jobs_storage
    - provider: cpu
      profile: k8s
      backend: kubernetes_job
      config:
        storage:
          pvc_name: test_jobs_storage
    - provider: gpu
      profile: default
      backend: kubernetes_job
      config:
        storage:
          pvc_name: test_jobs_storage
"""

    # Create a temporary file and write the YAML content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as temp_file:
        temp_file.write(yaml_content)
        temp_file.flush()

        # Load the configuration from the temporary file
        jobs_config: JobsServiceConfig = Configuration.get_service_config_from_file(temp_file.name, JobsServiceConfig)
        return jobs_config


@pytest.fixture
def job_config_kubernetes() -> JobsServiceConfig:
    # Define the YAML content as a variable
    yaml_content = """# This is the default configuration for the Jobs microservice.
platform:
  base_url: "http://localhost:8080"
  runtime: "kubernetes"

jobs:
  # Executor profiles configuration
  executors: []
"""

    # Create a temporary file and write the YAML content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as temp_file:
        temp_file.write(yaml_content)
        temp_file.flush()

        # Load the configuration from the temporary file
        jobs_config: JobsServiceConfig = Configuration.get_service_config_from_file(temp_file.name, JobsServiceConfig)
        return jobs_config


@pytest.fixture
def backend_registry(mock_nhx_client, job_config_with_many_profiles) -> BackendRegistry:
    """Create a backend registry with test configuration."""
    return BackendRegistry.from_config(
        nhx_sdk=mock_nhx_client,
        profiles=job_config_with_many_profiles.executors,
        # Mock the backends. Register the real SubprocessJobBackend to satisfy
        # the subprocess/default executor that ships in
        # `job_config_with_many_profiles` (added so test_client picks up the
        # subprocess profile and the cpu→subprocess translation in the Jobs
        # API fires consistently with production deployments).
        backends={
            BackendKey("cpu", "docker"): MockDockerCPUJobBackend,
            BackendKey("gpu", "docker"): MockDockerGPUJobBackend,
            BackendKey("cpu", "kubernetes_job"): MockKubernetesCPUJobBackend,
            BackendKey("gpu", "kubernetes_job"): MockKubernetesGPUJobBackend,
            BackendKey("subprocess", "subprocess"): SubprocessJobBackend,
        },
    )


class HelloWorldJobConfig(BaseModel):
    config: str | object
    target: str


def hello_world_job_config(
    workspace: str,
    input_spec: HelloWorldJobConfig,
    output_spec: HelloWorldJobConfig,
    entity_client: EntityClient,
    job_name: str | None,
    sdk,
) -> FactoryHelixJobSpec:
    return FactoryHelixJobSpec(
        steps=[
            FactoryHelixJobStep(
                name="hello-world-step-1",
                executor=FactoryCPUExecutionProviderSpec(
                    provider="cpu",
                    profile="default",
                    container=FactoryContainerSpec(
                        image="my-registry/hello-world-job:local",
                        command=["python", "-m", "hello_world"],
                        entrypoint=["--target", output_spec.target],
                    ),
                ),
                config=output_spec.model_dump(),
                environment=[FactoryEnvironmentVariable(name="ENV_VAR", value="test_value")],
            ),
        ]
    )


@pytest_asyncio.fixture
async def test_client(mock_dispatcher, mock_store, job_config_with_many_profiles) -> AsyncGenerator[AsyncClient, None]:
    # Mock the config.executors to have the test execution profiles, including
    # subprocess/default for cpu/default to subprocess/default translation.
    from nhx.common.auth.middleware import AuthorizationMiddleware
    from nhx.common.service.dependencies import get_sdk_client

    with subprocess_job_executor_patch(job_config_with_many_profiles.executors):
        app = FastAPI()

        # Add auth middleware with auth disabled - this sets up auth_client_context
        app.add_middleware(AuthorizationMiddleware)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:

            async def override_get_dispatcher():
                return mock_dispatcher

            def override_get_entity_client():
                return mock_store

            # Create SDK for dependency injection
            test_sdk = AsyncNeMoHelix(base_url=ac.base_url, http_client=ac)

            app.dependency_overrides[dep_dispatcher] = override_get_dispatcher
            app.dependency_overrides[get_entity_client] = override_get_entity_client
            app.dependency_overrides[get_sdk_client] = lambda: test_sdk

            # Mount under /apis/jobs so SDK requests (e.g. /apis/jobs/v2/workspaces/default/jobs) hit the app
            api_prefix = "/apis/jobs"
            app.include_router(rerun_router, prefix=api_prefix)
            app.include_router(router, prefix=api_prefix)

            factory_router = job_route_factory(
                service_name="hello-world",
                job_type="HelloWorld",
                job_input=HelloWorldJobConfig,
                platform_job_config_compiler=hello_world_job_config,
            )
            app.include_router(
                factory_router,
                prefix=f"{api_prefix}/v2/workspaces/{{workspace}}/hello-world",
                tags=["Hello World Microservice"],
            )

            yield ac


@pytest.fixture
def test_sdk(test_client: AsyncClient) -> AsyncNeMoHelix:
    # Disable retries to prevent duplicate entity creation attempts on transient errors
    return AsyncNeMoHelix(base_url=test_client.base_url, http_client=test_client, max_retries=0)
