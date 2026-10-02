# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path


def get_project_dir() -> Path:
    return Path(os.getenv("CI_PROJECT_DIR", os.getcwd()))


def get_sdk_dir() -> Path:
    """Return the checked-in Python SDK tree under ``sdk/python``."""
    return get_project_dir() / "sdk" / "python" / "nemo-helix"
