# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def get_skills_path() -> Path:
    """Return the directory containing Agent Hardener plugin skills."""
    return Path(__file__).parent / "skills"
