# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello World task package.

This task accesses a fileset in the configured workspace.

This task is used for E2E testing of auth propagation. It attempts to retrieve a fileset and
reports whether access was granted or denied.
"""

from nhx.hello_world.tasks.access_fileset.run import run

__all__ = ["run"]
