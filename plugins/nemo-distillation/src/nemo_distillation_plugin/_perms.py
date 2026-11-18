# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permission vocabulary for distillation workflow routes."""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class WorkflowPerms(PermissionSet, namespace="distillation.workflows"):
    CREATE = perm("Create a distillation workflow")
    LIST = perm("List distillation workflows")
    READ = perm("Read a distillation workflow")
    CANCEL = perm("Cancel a distillation workflow")
    RETRY = perm("Retry a failed distillation workflow node")
    RENDER = perm("Render a distillation workflow template")
