# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List


def generate_key(value: str | List[str]) -> str:
    """Generate a key based on the provided prefix and value."""
    if isinstance(value, str):
        return value
    return "-".join(value)
