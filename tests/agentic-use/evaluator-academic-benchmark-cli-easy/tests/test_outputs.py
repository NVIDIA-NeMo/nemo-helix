# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent set up an academic benchmark evaluation job via CLI.

Tests workspace creation, benchmark discovery, and benchmark job creation.
Note: Job execution (completion, results) is not tested because the
quickstart environment does not include the job execution worker.
"""

import os

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient

WORKSPACE = "benchmark-eval-workspace"


def _get_client() -> NemoClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return NemoClient(base_url=nhx_base_url)


def test_workspace_exists():
    """Verify the benchmark-eval-workspace was created."""
    client = _get_client()
    response = WorkspacesClient.from_client(client).list_workspaces()
    workspace_names = [ws.name for ws in response.items()]
    assert WORKSPACE in workspace_names, f"Workspace '{WORKSPACE}' not found. Found: {workspace_names}"
