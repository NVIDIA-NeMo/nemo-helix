# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Use this file to define pytest tests that verify the outputs of the task.

This file will be copied to /tests/test_outputs.py and run by the /tests/test.sh file
from the working directory.
"""

import os

from nemo_helix_plugin.workspaces.client import WorkspacesClient


def test_workspace_created() -> None:
    """Test that the harbor-test-workspace was successfully created."""
    # Get NeMo Helix API base URL from environment
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")

    # Create SDK client and list workspaces
    client = WorkspacesClient(base_url=nhx_base_url)
    response = client.list_workspaces()

    # Extract workspace names from the SDK response
    workspace_names = [ws.name for ws in response.items()]

    # Verify that harbor-test-workspace EXISTS
    assert "harbor-test-workspace" in workspace_names, (
        f"Workspace 'harbor-test-workspace' was not created! Found workspaces: {workspace_names}"
    )

    print("Test passed: harbor-test-workspace was successfully created")
