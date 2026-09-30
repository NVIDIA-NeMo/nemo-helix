# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task that exercises workload-auth by reading a workspace through the typed client."""

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.common.client_factory import get_task_nemo_client
from nhx.common.jobs.config import get_task_config
from pydantic import BaseModel


class WorkloadWorkspaceGetConfig(BaseModel):
    """Configuration for the workload workspace read task."""

    workspace: str


def run(*, client: NemoClient | None = None) -> int:
    """Read the configured workspace using the typed client workload identity path."""
    try:
        config = get_task_config(WorkloadWorkspaceGetConfig)
        client = client or get_task_nemo_client("jobs")
        workspace = WorkspacesClient.from_client(client).get_workspace(name=config.workspace).data()
        print(f"Successfully retrieved workspace: {workspace.name}")
        return 0
    except Exception as exc:
        print(f"Workload workspace retrieval failed: {exc}")
        return 1
