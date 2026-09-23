# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Workspaces API (Entity Store).

Wraps the endpoint functions from ``workspaces.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.

Usage::

    client = WorkspacesClient(base_url="...", workspace="default")
    ws = client.get_workspace(name="ml-team").data()
    created = client.create_workspace(body=CreateWorkspaceRequest(name="team")).data()
"""

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.workspaces import endpoints


class _WorkspacesMethods:
    get_workspace = method(endpoints.get_workspace)
    list_workspaces = method(endpoints.list_workspaces)
    create_workspace = method(endpoints.create_workspace)
    update_workspace = method(endpoints.update_workspace)
    delete_workspace = method(endpoints.delete_workspace)
    list_workspace_members = method(endpoints.list_workspace_members)
    create_workspace_member = method(endpoints.create_workspace_member)
    update_workspace_member = method(endpoints.update_workspace_member)
    delete_workspace_member = method(endpoints.delete_workspace_member)


class WorkspacesClient(_WorkspacesMethods, NemoClient):
    """Sync client for the Workspaces API."""


class AsyncWorkspacesClient(_WorkspacesMethods, AsyncNemoClient):
    """Async client for the Workspaces API."""
