# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest


@contextmanager
def managed_admin_workspace(admin_client: NemoClient, workspace_name: str) -> Iterator[str]:
    workspaces = WorkspacesClient.from_client(admin_client)
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
    try:
        yield workspace_name
    finally:
        workspaces.delete_workspace(name=workspace_name).data()


def job_exists_in_pages(items: Iterator[Any], job_name: str) -> bool:
    return any(item.name == job_name for item in items)
