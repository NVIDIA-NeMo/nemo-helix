# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills directory exposure — registered under ``nemo.skills``.

Returns the path to the ``skills/`` directory inside this package so the
platform can discover and load skill markdown files shipped with the plugin.
"""

from __future__ import annotations

from pathlib import Path


def skills_dir() -> Path:
    """Return the directory containing this plugin's skills."""
    return Path(__file__).parent / "skills"
