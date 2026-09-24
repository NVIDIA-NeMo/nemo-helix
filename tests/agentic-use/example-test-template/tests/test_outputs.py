# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Use this file to define pytest tests that verify the outputs of the task.

This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.

TODO: Replace the example test below with your actual verification logic.
"""

import os

from nemo_helix_plugin.client.client import NemoClient


# TODO: Rename this test function to describe what it verifies
def test_todo_replace_with_descriptive_name():
    """
    TODO: Add a docstring describing what this test verifies.

    Example: Test that the expected resource was successfully created.
    """
    # TODO: Get any needed environment variables
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")

    # TODO: Create a typed client and perform verification
    client = NemoClient(base_url=nhx_base_url)  # noqa: F841

    # TODO: Replace with actual verification logic
    # Example: Check that a resource was created
    #
    # response = WorkspacesClient.from_client(client).list_workspaces()
    # workspace_names = [ws.name for ws in response.items()]
    # assert "expected-workspace-name" in workspace_names, (
    #     f"Expected workspace was not created! Found: {workspace_names}"
    # )

    # TODO: Remove this placeholder assertion
    assert False, "TODO: Implement actual test verification logic"


# TODO: Add additional test functions if needed
# def test_another_verification():
#     """Verify another aspect of the task completion."""
#     pass
