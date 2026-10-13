# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests that verify the platform is reachable and core APIs respond.

These are intentionally minimal — they validate the e2e harness works and
that services are up. Add more substantive tests in separate files.
"""

import uuid

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest


def test_health_ready(client: NemoClient):
    """GET /status returns 200 with healthy status when all services are up."""
    resp = client._client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_live(client: NemoClient):
    """GET /status returns 200 (platform is reachable)."""
    resp = client._client.get("/status")
    assert resp.status_code == 200


def test_create_and_delete_workspace(client: NemoClient):
    """Workspace create and delete round-trips through the platform."""
    workspaces = WorkspacesClient.from_client(client)
    name = f"e2e-smoke-{uuid.uuid4().hex[:8]}"
    ws = workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    try:
        assert ws.name == name
    finally:
        workspaces.delete_workspace(name=name).data()


def test_list_workspaces(client: NemoClient, workspace: str):
    """Listing workspaces returns at least the test workspace."""
    page = WorkspacesClient.from_client(client).list_workspaces()
    names = [w.name for w in page.items()]
    assert workspace in names
