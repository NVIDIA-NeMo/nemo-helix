# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def get_skills_path() -> Path:
    """Directory holding the UserSim plugin skills."""
    return Path(__file__).parent / "skills"
