# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for platform main module."""

import os
from typing import List

import pytest
from fastapi import APIRouter
from nhx.common.jobs.config import get_task_config
from nhx.common.jobs.constants import TASK_CONFIG_ENVVAR
from nhx.common.service import RouterConfig, Service
from nhx.platform import main as platform_main
from nhx.platform.main import load_service, run_task


class MockService(Service):
    """Mock service for testing."""

    def __init__(self):
        super().__init__(name="mock", module_name="nhx.mock")

    def get_routers(self) -> List[RouterConfig]:
        return [RouterConfig(APIRouter(), tag="Mock", description="Mock")]


class TestLoadService:
    """Tests for load_service function."""

    def test_load_service_invalid_path_format(self):
        """Test load_service raises with invalid path format."""
        with pytest.raises(ValueError, match="must be in format"):
            load_service("test", "invalid_path")

    def test_load_service_module_not_found(self):
        """Test load_service raises ImportError for missing module."""
        with pytest.raises(ImportError):
            load_service("test", "nonexistent.module:service")

    def test_load_service_attribute_not_found(self):
        """Test load_service raises AttributeError for missing attribute."""
        with pytest.raises(AttributeError):
            load_service("test", "nhx.platform.main:nonexistent")

    def test_load_service_not_service_instance(self):
        """Test load_service raises TypeError for non-Service."""
        with pytest.raises(TypeError, match="must be an instance of Service"):
            load_service("test", "nhx.platform.main:logger")

    def test_load_service_success(self):
        """Test load_service successfully loads a service."""
        # Use the hello-world service for testing
        service = load_service("hello-world", "nhx.hello_world.main:service")

        assert isinstance(service, Service)
        assert service.name == "hello-world"


class TestRunTask:
    """Tests for the task compatibility runner."""

    def test_run_task_config_is_available_to_task_config_loader(self, monkeypatch):
        """``--config`` should populate the env var read by ``get_task_config``."""
        captured_config = {}

        def fake_run_module(module_name: str, run_name: str):
            captured_config.update(get_task_config())
            return {}

        monkeypatch.delenv(TASK_CONFIG_ENVVAR, raising=False)
        monkeypatch.delenv("NEMO_JOB_STEP_CONFIG", raising=False)
        monkeypatch.setattr(platform_main.runpy, "run_module", fake_run_module)

        try:
            result = run_task(
                "nhx.fake.tasks.verify_config",
                [],
                '{"workspace":"auth-idp-ws-test"}',
            )

            assert result == 0
            assert captured_config == {"workspace": "auth-idp-ws-test"}
            assert os.environ[TASK_CONFIG_ENVVAR] == '{"workspace":"auth-idp-ws-test"}'
            assert os.environ["NEMO_JOB_STEP_CONFIG"] == '{"workspace":"auth-idp-ws-test"}'
        finally:
            os.environ.pop(TASK_CONFIG_ENVVAR, None)
            os.environ.pop("NEMO_JOB_STEP_CONFIG", None)


class TestStartupFailureExitCode:
    """Tests for deprecated service startup behavior."""

    def test_service_startup_exits_with_code_1_and_guidance(self, tmp_path):
        """Service startup should fail fast and direct callers to `nemo services run`."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "nhx.platform.main", "run", "--services", "files"],
            capture_output=True,
            timeout=30,
        )

        stderr = result.stderr.decode()
        assert result.returncode == 2
        assert "invalid choice" in stderr
        assert "choose from 'task'" in stderr or "choose from task" in stderr
