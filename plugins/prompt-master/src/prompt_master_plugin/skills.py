# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skill discovery for the Prompt Master plugin."""

from pathlib import Path


def skills_dir() -> Path:
    """Return the directory containing plugin-owned agent skills."""
    return Path(__file__).parent / "skills"
