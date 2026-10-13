# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for running platform seed as a standalone task (e.g. CLI or K8s Job)."""

import sys

from nhx.platform_seed.tasks.seed import run


def main() -> None:
    """Synchronous entry point for scripts and Jobs (e.g. platform-seed script)."""
    sys.exit(run())


if __name__ == "__main__":
    main()
