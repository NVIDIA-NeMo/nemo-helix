# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HelloWorldService."""

from unittest.mock import AsyncMock, MagicMock

from nemo_helix_plugin.client.client import AsyncNemoClient
from nhx.common.entities import EntityClient
from nhx.hello_world.api.v2.jobs.endpoints import compile_hello_world_job
from nhx.hello_world.api.v2.jobs.schemas import HelloWorldJobConfig
from nhx.hello_world.service import HelloWorldService


class TestHelloWorldService:
    """Tests for HelloWorldService class."""

    def test_service_name(self):
        """Test service has correct name."""
        service = HelloWorldService()
        assert service.name == "hello-world"

    def test_service_module_name(self):
        """Test service has correct module name."""
        service = HelloWorldService()
        assert service.module_name == "nhx.hello_world"

    def test_service_title(self):
        """Test service title."""
        service = HelloWorldService()
        assert service.title == "Hello World Service"

    def test_get_routers(self):
        """Test get_routers returns routers."""
        service = HelloWorldService()
        routers = service.get_routers()

        assert len(routers) == 3
        tags = {r.tag for r in routers}
        assert tags == {"Hello", "Jobs", "Messages"}

    def test_app_creation(self):
        """Test app is created successfully."""
        service = HelloWorldService()
        app = service.app

        assert app is not None
        assert app.title == "Hello World Service"


class TestCompileHelloWorldJob:
    """Tests for compile_hello_world_job function."""

    def test_compile_hello_world_job_signature(self):
        """Test that compile_hello_world_job accepts all required parameters."""
        workspace = "test-workspace"
        job_config = HelloWorldJobConfig(message="Test message")
        entity_client = MagicMock(spec=EntityClient)
        job_name = "test-job"
        sdk = AsyncMock(spec=AsyncNemoClient)

        # This test verifies the function signature is correct
        # If any parameter is missing, this will raise TypeError
        result = compile_hello_world_job(
            workspace=workspace,
            original_spec=job_config,
            transformed_spec=job_config,
            entity_client=entity_client,
            job_name=job_name,
            sdk=sdk,
        )

        # Verify return type - HelixJobSpec is a Pydantic model that can be accessed as dict
        assert result is not None
        # Access as dict (Pydantic models serialize to dict when accessed)
        steps = result["steps"] if isinstance(result, dict) else result.steps
        assert len(steps) == 1

        step = steps[0]
        # Access step fields as dict
        step_dict = step if isinstance(step, dict) else step.model_dump()
        assert step_dict["name"] == "hello-world"
        assert step_dict["executor"]["provider"] == "cpu"
        assert step_dict["executor"]["profile"] == "default"
        assert "nemo-helix" in step_dict["executor"]["container"]["entrypoint"]
        assert "nhx.hello_world.tasks.hello_world" in step_dict["executor"]["container"]["command"]

    def test_compile_hello_world_job_with_none_job_name(self):
        """Test that compile_hello_world_job works with None job_name."""
        workspace = "test-workspace"
        job_config = HelloWorldJobConfig(message="Test message")
        entity_client = MagicMock(spec=EntityClient)
        sdk = AsyncMock(spec=AsyncNemoClient)

        # Test with None job_name
        result = compile_hello_world_job(
            workspace=workspace,
            original_spec=job_config,
            transformed_spec=job_config,
            entity_client=entity_client,
            job_name=None,
            sdk=sdk,
        )

        assert result is not None
        steps = result["steps"] if isinstance(result, dict) else result.steps
        assert len(steps) == 1

    def test_compile_hello_world_job_config_included(self):
        """Test that the job config is included in the platform spec."""
        workspace = "test-workspace"
        job_config = HelloWorldJobConfig(message="Custom message")
        entity_client = MagicMock(spec=EntityClient)
        sdk = AsyncMock(spec=AsyncNemoClient)

        result = compile_hello_world_job(
            workspace=workspace,
            original_spec=job_config,
            transformed_spec=job_config,
            entity_client=entity_client,
            job_name="test-job",
            sdk=sdk,
        )

        # Verify config is included
        steps = result["steps"] if isinstance(result, dict) else result.steps
        step = steps[0]
        step_dict = step if isinstance(step, dict) else step.model_dump()
        assert step_dict["config"] is not None
        assert step_dict["config"]["message"] == "Custom message"
