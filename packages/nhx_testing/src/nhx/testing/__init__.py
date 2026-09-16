# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing utilities for NeMo Helix services.

This package provides:

CLI testing:
- NemoRun / NhxRun: Type alias for a callable that runs the NeMo CLI
- assert_exit_0: Assert that a CLI invocation succeeded
- get_repo_root: Return the repository root using git
- run_nemo_local / run_nhx_local: Run NeMo CLI from repo root with optional platform URL injection

API testing:
- create_test_client: Helper for creating FastAPI test clients with in-memory storage
- ClientContext: Container for all client types returned by create_test_client
- TEST_USER_EMAIL, TEST_ADMIN_EMAIL: Constants for test principals
- subprocess_job_executor_patch: Opt into cpu/default to subprocess/default translation

Utilities:
- short_unique_name: Helper for generating unique names with length constraints
- unique_email: Helper for generating unique test user emails
- as_user: Helper for creating SDK client authenticated as a specific user
- as_service_for: Helper for creating SDK client authenticated as a service principal
- grant_workspace_role: Helper for granting workspace roles in auth-enabled tests
- add_mock_provider: Helper for adding mock providers to IGW model cache
- MockProviderResponse: Wrapper to configure dynamic mock LLM responses
- wait_for_model_entity: Poll until a model entity is available in IGW's model cache

Task testing:
- task_harness: Task integration test context manager
- TaskContext: Context for running tasks with SDK access
- TaskResult: Result of a task execution

Docker testing:
- DockerTestContext: Context for Docker integration tests with cleanup support
- create_docker_client: Helper for creating validated Docker clients
- build_mock_nim_image: Helper for building mock NIM images for testing
- get_worker_port_range: Helper for pytest-xdist port range allocation

Notebook testing:
- run_notebooks: Run documentation notebooks with @nemo-nb: process marker
- execute_notebook: Execute a single notebook (.md or .ipynb) via papermill
- create_temp_venv_with_kernel: Create an isolated venv with a Jupyter kernel
- cleanup_temp_venv_and_kernel: Remove a temporary venv and kernel spec
"""

from .client import TEST_ADMIN_EMAIL, TEST_USER_EMAIL, ClientContext, SDKTestClientAdapter, create_test_client
from .docker import (
    DEFAULT_RETRY_CONFIG,
    MOCK_NIM_NGINX_CONF,
    MODELS_CONTROLLER_MANAGED_LABEL,
    DockerRetryConfig,
    DockerTestContext,
    build_mock_nim_image,
    build_mock_sidecar_image,
    cleanup_model_deployment_containers,
    create_docker_client,
    ensure_mock_nim_image,
    ensure_mock_sidecar_image,
    get_worker_port_range,
)
from .jobs import subprocess_job_executor_patch
from .notebooks import (
    cleanup_temp_venv_and_kernel,
    create_temp_venv_with_kernel,
    execute_notebook,
    run_notebooks,
)
from .tasks import TaskContext, TaskResult, task_harness
from .utils import (
    MockProviderResponse,
    NemoRun,
    add_mock_provider,
    as_service_for,
    as_user,
    assert_exit_0,
    get_repo_root,
    grant_workspace_role,
    igw_mock_provider_mode,
    run_nemo_local,
    short_unique_name,
    unique_email,
    wait_for_model_entity,
)

__all__ = [
    # CLI testing
    "NemoRun",
    "assert_exit_0",
    "get_repo_root",
    "run_nemo_local",
    # API testing
    "create_test_client",
    "ClientContext",
    "SDKTestClientAdapter",
    "TEST_USER_EMAIL",
    "TEST_ADMIN_EMAIL",
    "subprocess_job_executor_patch",
    # Utilities
    "short_unique_name",
    "unique_email",
    "as_service_for",
    "as_user",
    "grant_workspace_role",
    "add_mock_provider",
    "igw_mock_provider_mode",
    "MockProviderResponse",
    "wait_for_model_entity",
    # Task testing
    "task_harness",
    "TaskContext",
    "TaskResult",
    # Docker testing
    "DockerRetryConfig",
    "DEFAULT_RETRY_CONFIG",
    "DockerTestContext",
    "create_docker_client",
    "build_mock_nim_image",
    "ensure_mock_nim_image",
    "build_mock_sidecar_image",
    "ensure_mock_sidecar_image",
    "cleanup_model_deployment_containers",
    "get_worker_port_range",
    "MOCK_NIM_NGINX_CONF",
    "MODELS_CONTROLLER_MANAGED_LABEL",
    # Notebook testing
    "create_temp_venv_with_kernel",
    "cleanup_temp_venv_and_kernel",
    "execute_notebook",
    "run_notebooks",
]
