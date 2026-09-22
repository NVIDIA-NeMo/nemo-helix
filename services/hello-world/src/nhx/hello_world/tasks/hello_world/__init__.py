# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello World task package.

This task writes a greeting message to the file API using the configured workspace and fileset.
"""

from nhx.hello_world.tasks.hello_world.run import run

__all__ = ["run"]
