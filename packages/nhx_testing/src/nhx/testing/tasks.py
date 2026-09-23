# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task integration testing utilities for NeMo Helix.

This module provides the test_task_harness async context manager for testing
task modules in isolation with mocked platform services via ASGI transport.

Example:
    from nhx.hello_world.service import HelloWorldService
    from nhx.hello_world.tasks import hello_world
    from nhx.testing import task_harness

    @pytest.mark.asyncio
    async def test_hello_world_task():
        async with task_harness(
            hello_world,
            HelloWorldService,
            config={"config_name": "my-config"},
            env={"NEMO_JOB_WORKSPACE": "test-ws"},
        ) as ctx:
            # Setup: create test data via SDK
            await ctx.sdk.hello_world.configs.create(
                workspace="test-ws",
                name="my-config",
                message="Hello from test!",
            )

            # Run the task
            result = ctx.run_task()

            # Assertions
            assert result.exit_code == 0
            assert "Hello from test!" in result.stdout
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import io
import json
import os
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from types import ModuleType
from typing import Any, AsyncGenerator

from nemo_helix import AsyncNeMoHelix
from nemo_helix._client import NeMoHelix
from nhx.common.auth import NHX_PRINCIPAL_ENVVAR, Principal
from nhx.common.service import Service
from nhx.testing.access_log import AccessLog
from nhx.testing.client import ClientContext, create_test_client


@dataclass
class TaskResult:
    """Result of a task execution."""

    exit_code: int
    stdout: str
    stderr: str
    exception: Exception | None = None


@dataclass
class TaskContext:
    """Context for running a task with access to the test SDK."""

    sdk: NeMoHelix
    async_sdk: AsyncNeMoHelix
    _module: ModuleType
    access_log: AccessLog | None = None
    auth_enabled: bool = False

    def run_task(self, args: list[str] | None = None) -> TaskResult:
        """Execute the task module and capture results.

        If called from within an async context (event loop running), runs the task
        in a separate thread to avoid nested event loop issues when tasks use
        asyncio.run() internally.

        The SDK is injected into the task's run() function if it accepts an 'sdk' parameter.

        Args:
            args: Optional list of CLI arguments to pass to the task's run() function.
                  If the task's run() accepts args, they will be passed through.

        Returns:
            TaskResult with exit_code, stdout, stderr, and any exception
        """
        # Determine which SDK to inject based on task signature
        sig = inspect.signature(self._module.run)
        sdk_param = sig.parameters.get("sdk")

        # Determine the right SDK type to inject
        injected_sdk: NeMoHelix | AsyncNeMoHelix | None = None
        if sdk_param is not None:
            # Check the type annotation to determine sync vs async SDK
            if sdk_param.annotation is not inspect.Parameter.empty:
                annotation_str = str(sdk_param.annotation)
                if "AsyncNeMoHelix" in annotation_str:
                    injected_sdk = self.async_sdk
                else:
                    injected_sdk = self.sdk
            else:
                # Default to sync SDK if no annotation
                injected_sdk = self.sdk

        def _execute_task():
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            exit_code = 0
            exception = None

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Build kwargs based on what parameters the run() function accepts
                    kwargs = {}
                    if "args" in sig.parameters:
                        kwargs["args"] = args
                    if sdk_param is not None:
                        kwargs["sdk"] = injected_sdk

                    exit_code = self._module.run(**kwargs)
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 1
            except Exception as e:
                exception = e
                exit_code = 1

            return TaskResult(
                exit_code=exit_code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                exception=exception,
            )

        # Check if we're in an async context (event loop running)
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False

        if has_loop:
            # Run in thread to allow task to use asyncio.run() without nested loop issues
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_task)
                return future.result()
        else:
            # No event loop - run directly
            return _execute_task()


@asynccontextmanager
async def task_harness(
    module: ModuleType,
    *services: type[Service],
    config: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    auth_enabled: bool = False,
    access_log: bool = False,
    workspace: str | None = None,
) -> AsyncGenerator[TaskContext, None]:
    """Async context manager for testing task modules with mocked platform services.

    Uses create_test_client internally to set up a FastAPI app with the specified
    services using ASGI in-process transport.

    Args:
        module: Python module reference with a run() -> int function
        *services: Service classes to load (e.g., HelloWorldService)
        config: Config dict passed via NEMO_JOB_STEP_CONFIG env var
        env: Additional environment variables to set during task execution.
             If NEMO_JOB_WORKSPACE is set, that workspace will be auto-created.
             If NHX_PRINCIPAL is set (JSON with id, email, groups matching the
             Principal model) and auth_enabled=True, the SDK will include auth
             headers for that principal.
        auth_enabled: Enable authorization middleware with embedded PDP.
        access_log: Enable request capture for test verification. When True,
                   ctx.access_log will be available to inspect all HTTP requests.

    Yields:
        TaskContext with sdk, async_sdk, run_task(), and optionally access_log
    """
    from nhx.common.jobs.constants import TASK_CONFIG_ENVVAR

    config = config or {}
    env = env or {}

    # Save original env values
    keys_to_modify = {TASK_CONFIG_ENVVAR, "NHX_URL"} | set(env.keys())
    original_env: dict[str, str | None] = {key: os.environ.get(key) for key in keys_to_modify}

    # Set test env values
    os.environ[TASK_CONFIG_ENVVAR] = json.dumps(config)
    os.environ["NHX_URL"] = "http://localhost:8080"
    for key, value in env.items():
        os.environ[key] = value

    try:
        with create_test_client(
            *services,
            client_type=ClientContext,
            workspaces=[env.get("NEMO_JOB_WORKSPACE", "default")],
            auth_enabled=auth_enabled,
            access_log=access_log,
            workspace=workspace,
        ) as ctx:
            # Configure SDK with auth headers from NHX_PRINCIPAL env var if set.
            # This simulates how production tasks get auth context propagated.
            sdk = ctx.sdk
            async_sdk = ctx.async_sdk
            principal_json = os.environ.get(NHX_PRINCIPAL_ENVVAR)
            if principal_json and auth_enabled:
                principal = Principal.model_validate_json(principal_json)
                auth_headers = principal.get_headers()
                sdk = sdk.with_options(set_default_headers=auth_headers)
                async_sdk = async_sdk.with_options(set_default_headers=auth_headers)

            yield TaskContext(
                sdk=sdk,
                async_sdk=async_sdk,
                _module=module,
                access_log=ctx.access_log,
                auth_enabled=auth_enabled,
            )
    finally:
        # Restore original env
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
