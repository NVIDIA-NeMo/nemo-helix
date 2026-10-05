# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from nhx.common.observability import BaseContext


@dataclass
class WorkspaceCleanupContext(BaseContext):
    otel_prefix: str = "workspace.cleanup"

    workspace_name: str | None = None
    deletion_stage: str | None = None
