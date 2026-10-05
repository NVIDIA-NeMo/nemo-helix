# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for running the workload workspace read task as a module."""

from nhx.hello_world.tasks.workload_workspace_get.run import run

if __name__ == "__main__":
    raise SystemExit(run())
