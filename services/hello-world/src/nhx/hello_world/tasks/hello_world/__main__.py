# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for running the hello-world task as a module."""

import sys

from nhx.hello_world.tasks.hello_world import run

if __name__ == "__main__":
    sys.exit(run())
