# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests that verify the platform is reachable and core APIs respond.

These are intentionally minimal — they validate the e2e harness works and
that services are up. Add more substantive tests in separate files.
"""

import uuid

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest


def test_health_ready(sdk: NeMoHelix):
    """GET /status returns 200 with healthy status when all services are up."""
    resp = sdk._client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_live(sdk: NeMoHelix):
    """GET /status returns 200 (platform is reachable)."""
    resp = sdk._client.get("/status")
    assert resp.status_code == 200


def test_create_and_delete_workspace(sdk: NeMoHelix):
    """Workspace create and delete round-trips through the platform."""
    workspaces = client_from_platform(sdk, WorkspacesClient)
    name = f"e2e-smoke-{uuid.uuid4().hex[:8]}"
    ws = workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
    try:
        assert ws.name == name
    finally:
        workspaces.delete_workspace(name=name).data()


def test_list_workspaces(sdk: NeMoHelix, workspace: str):
    """Listing workspaces returns at least the test workspace."""
    page = client_from_platform(sdk, WorkspacesClient).list_workspaces()
    names = [w.name for w in page.items()]
    assert workspace in names
