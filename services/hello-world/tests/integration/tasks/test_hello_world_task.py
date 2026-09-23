# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the hello-world task."""

import pytest
from nhx.core.files.service import FilesService
from nhx.hello_world.tasks import hello_world
from nhx.testing import task_harness


class TestHelloWorldTask:
    """Integration tests for the hello-world task module."""

    @pytest.mark.asyncio
    async def test_task_writes_default_message(self):
        """Test that task writes default message to file API."""
        async with task_harness(
            hello_world,
            FilesService,
            config={},
            env={
                "NEMO_JOB_ID": "test-job-123",
                "NEMO_JOB_WORKSPACE": "test-workspace",
            },
        ) as ctx:
            result = ctx.run_task()

            assert result.exit_code == 0, f"Task failed: stdout={result.stdout}, stderr={result.stderr}"
            assert "Successfully wrote message: Hello World" in result.stdout
            assert result.exception is None

            # Verify the file was uploaded to the auto-created fileset
            content = ctx.sdk.files.download_content(
                workspace="test-workspace",
                fileset="hello-world-test-job-123",
                remote_path="message.txt",
            )
            assert content.decode("utf-8") == "Hello World"

    @pytest.mark.asyncio
    async def test_task_writes_custom_message(self):
        """Test that task writes custom message from config."""
        async with task_harness(
            hello_world,
            FilesService,
            config={
                "message": "Hello from integration test!",
            },
            env={
                "NEMO_JOB_ID": "custom-msg-job",
                "NEMO_JOB_WORKSPACE": "test-workspace",
            },
        ) as ctx:
            result = ctx.run_task()

            assert result.exit_code == 0, f"Task failed: stdout={result.stdout}, stderr={result.stderr}"
            assert "Successfully wrote message: Hello from integration test!" in result.stdout
            assert result.exception is None

            # Verify the file was uploaded
            content = ctx.sdk.files.download_content(
                workspace="test-workspace",
                fileset="hello-world-custom-msg-job",
                remote_path="message.txt",
            )
            assert content.decode("utf-8") == "Hello from integration test!"
