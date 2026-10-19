# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for model_spec task.

Usage:
    python -m nhx.core.models.tasks.model_spec
"""

import sys

from .run import run

if __name__ == "__main__":
    sys.exit(run())
