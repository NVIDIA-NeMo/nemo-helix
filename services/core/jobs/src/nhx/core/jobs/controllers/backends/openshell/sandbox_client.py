# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell SDK client for the jobs backend.

Imports the optional ``openshell`` package at module load, so the backend
imports this module lazily (see ``backend._ensure_openshell``).
"""

from __future__ import annotations

from openshell import SandboxClient
from openshell._proto import datamodel_pb2 as dm  # ty: ignore[unresolved-import]
from openshell._proto import openshell_pb2 as pb  # ty: ignore[unresolved-import]


class JobsSandboxClient(SandboxClient):
    """``SandboxClient`` plus a GetSandbox that returns the raw ``Sandbox`` proto.

    The SDK's ``SandboxRef`` does not carry ``status.conditions``, which the
    backend needs for a terminal sandbox's transition time (TTL cleanup) and for
    the error message of a sandbox that failed before its main process ran.
    Everything else goes through the public SDK surface.
    """

    def get_sandbox(self, name: str, *, workspace: str) -> pb.Sandbox:
        response = self._stub.GetSandbox(
            pb.GetSandboxRequest(workspace_scope=dm.WorkspaceSelector(workspace=workspace), name=name),
            timeout=self._timeout,
        )
        return response.sandbox
